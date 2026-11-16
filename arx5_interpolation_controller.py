import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
import zerorpc
import arx5_interface as arx5

# The ARX SDK reports poses in the URDF eef_link frame (x = tool forward).
# UMI datasets, policies, and everything above this controller use the TCP
# frame (z = approach direction). The two are related by a constant rotation:
#     R_tcp = R_ee @ EE_TO_TCP_ROT
#     R_ee  = R_tcp @ EE_TO_TCP_ROT.T
# Same matrices as ee2tcp/tcp2ee in the SDK ZMQ client.
EE_TO_TCP_ROT = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

# no further use , just remain symbol
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5

def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def _apply_dynamic_latency(
        pose_interp,
        waypoint_buffer,
        t_now,
        matching_dt=0.01,
        latency_precision=0.02,
        max_latency=1.2,
        smoothing_time=0.4,
        verbose=False):
    """Align a buffered action chunk to the trajectory being executed.

    Same idea as real-stanford/umi-arx: search for the latency that best
    matches the shape of the new chunk to the current trajectory, then
    rebuild the interpolator from that phase. Waypoints before ``t_now`` are
    replaced with the current trajectory and the first ``smoothing_time``
    seconds are blended, so stale actions are not executed as jumps.

    The buffered timestamps only need to be increasing; only relative times
    (``input_times - input_times[0]``) are used, so wall-clock and monotonic
    domains do not need to match here.
    """
    input_poses = np.array(
        [cmd['target_pose'] for cmd in waypoint_buffer], dtype=np.float64)
    input_times = np.array(
        [float(cmd['target_time']) for cmd in waypoint_buffer], dtype=np.float64)
    input_gripper_pos = np.array(
        [float(cmd['target_gripper_width'][0]) for cmd in waypoint_buffer],
        dtype=np.float64)

    # Current executed trajectory, sampled around now.
    pose_samples = np.zeros((3, 6))
    pose_samples[0] = pose_interp(t_now - matching_dt)
    pose_samples[1] = pose_interp(t_now)
    pose_samples[2] = pose_interp(t_now + matching_dt)

    input_pose_interp = PoseTrajectoryInterpolator(
        times=input_times - input_times[0],
        poses=input_poses
    )

    # Position error matters more than orientation for choosing the phase.
    error_weights = np.array([1, 1, 1, 0.1, 0.1, 0.1])
    latencies = np.arange(matching_dt, max_latency, latency_precision)
    errors = []
    for latency in latencies:
        input_pose_samples = np.zeros((3, 6))
        input_pose_samples[0] = input_pose_interp(latency - matching_dt)
        input_pose_samples[1] = input_pose_interp(latency)
        input_pose_samples[2] = input_pose_interp(latency + matching_dt)
        errors.append(np.sum(
            np.abs(input_pose_samples - pose_samples) * error_weights))
    best_latency = float(latencies[np.argmin(errors)])

    new_times = input_times - input_times[0] + t_now - best_latency
    smoothed_poses = input_poses.copy()
    for i in range(len(new_times)):
        if new_times[i] < t_now:
            smoothed_poses[i] = pose_interp(new_times[i])
        elif new_times[i] < t_now + smoothing_time:
            alpha = (new_times[i] - t_now) / smoothing_time
            smoothed_poses[i] = (
                (1 - alpha) * pose_interp(new_times[i])
                + alpha * input_poses[i]
            )

    if verbose:
        print(f"[ArxPositionalController] dynamic latency: {best_latency:.3f}s, "
              f"error: {min(errors):.3f}")

    new_pose_interp = PoseTrajectoryInterpolator(
        times=new_times, poses=smoothed_poses)
    extended_gripper_pos = np.zeros_like(input_poses)
    extended_gripper_pos[:, 0] = input_gripper_pos
    new_gripper_interp = PoseTrajectoryInterpolator(
        times=new_times, poses=extended_gripper_pos)
    return new_pose_interp, new_gripper_interp, best_latency

class ArxInterface:
    def __init__(self, robot_type='X5_umi', can_id='can0', USE_MULTITHREADING=True, frequency = 100):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(robot_type)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", robot_config.joint_dof
        )
        controller_config.controller_dt = 1.0 / frequency

        if USE_MULTITHREADING:
            # Will create another thread that communicates with the arm, so each send_recv_once() will take no time
            # for the main thread to execute. Otherwise (without background send/recv), send_recv_once() will block the
            # main thread until the arm responds (usually 2ms).
            controller_config.background_send_recv = True
        else:
            controller_config.background_send_recv = False

        self.arx5_cartesian_controller = arx5.Arx5CartesianController(
            robot_config, controller_config, can_id
        )
        self.robot_config = self.arx5_cartesian_controller.get_robot_config()
        self.controller_config = self.arx5_cartesian_controller.get_controller_config()

        # The SDK initializes the arm in damping mode (kp=0, kd=default) so it
        # can be moved by hand. Without setting the position gains, EEF commands
        # are solved but never tracked, and the arm appears not to move.
        self.arx5_cartesian_controller.set_gain(
            arx5.Gain(
                self.controller_config.default_kp,
                self.controller_config.default_kd,
                self.controller_config.default_gripper_kp,
                self.controller_config.default_gripper_kd,
            )
        )
        self.use_multithreading = USE_MULTITHREADING
        self.robot_type = robot_type
        self.can_id = can_id

        if robot_type not in ('X5_umi', 'X5_iphumi', 'L5_umi'):
            print(
                f"[ArxInterface] WARNING: robot_type={robot_type!r} does not put "
                "eef_link at the UMI TCP. TCP positions will be offset by the "
                "eef_link-to-TCP distance; use X5_umi/L5_umi for UMI data."
            )

        np.set_printoptions(precision=3, suppress=True)
        self.arx5_cartesian_controller.set_log_level(arx5.LogLevel.DEBUG)

    def get_ee_pose(self):
        # SDK: eef_link pose, xyz + KDL RPY (radians).
        # KDL RPY is Rz(yaw) @ Ry(pitch) @ Rx(roll), scipy's lowercase xyz.
        # Convert the rotation to the UMI TCP frame (z = approach direction)
        # so obs/policy/actions above this boundary are all TCP. With
        # robot_type X5_umi/L5_umi the eef_link origin is the TCP, so the
        # position is unchanged.
        pose_6d = np.array(
            self.arx5_cartesian_controller.get_eef_state().pose_6d().copy()
        )
        R_ee = st.Rotation.from_euler(
            'xyz', pose_6d[3:], degrees=False
        ).as_matrix()
        R_tcp = R_ee @ EE_TO_TCP_ROT
        return np.concatenate([
            pose_6d[:3],
            st.Rotation.from_matrix(R_tcp).as_rotvec(),
        ])

    def get_joint_positions(self):
        return np.array(self.arx5_cartesian_controller.get_joint_state().pos().copy())

    def get_joint_velocities(self):
        return np.array(self.arx5_cartesian_controller.get_joint_state().vel().copy())

    def get_gripper_widthes(self):
        return np.array([self.arx5_cartesian_controller.get_joint_state().gripper_pos], dtype=np.float64)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        # Arx5CartesianController has no joint-command interface. The old
        # implementation fed the 6 joint angles into EEFState as if they were
        # a Cartesian pose, which would command a meaningless motion. Refuse
        # instead; keep init_joints=False for the UMI eval.
        raise NotImplementedError(
            "Joint-space initialization is not supported by the Cartesian "
            "controller; keep init_joints=False."
        )

    def start_cartesian_impedance(self):
        pass

    def get_command_times(self, lead_time):
        """Return one future instant in host monotonic and SDK-relative clocks."""
        before = time.monotonic()
        sdk_now = self.arx5_cartesian_controller.get_timestamp()
        after = time.monotonic()
        host_now = (before + after) / 2
        return host_now + lead_time, sdk_now + lead_time

    def update_desired_ee_pose(self, pos, gripper_width, sdk_target_time):
        """Send a UMI TCP pose (xyz + rotation vector) to the SDK.

        The SDK expects an eef_link pose (x = tool forward, KDL RPY), so the
        TCP rotation is converted back to the EE frame here (inverse of
        get_ee_pose). With robot_type X5_umi/L5_umi the eef_link origin is
        the TCP, so the position is unchanged.
        """
        pose = np.asarray(pos, dtype=np.float64)
        R_tcp = st.Rotation.from_rotvec(pose[3:]).as_matrix()
        R_ee = R_tcp @ EE_TO_TCP_ROT.T
        pose_rpy = pose.copy()
        pose_rpy[3:] = st.Rotation.from_matrix(R_ee).as_euler(
            'xyz', degrees=False
        )
        eef_cmd = arx5.EEFState(pose_rpy, float(gripper_width[0]))
        # Never send an expired target or let timestamp=0 select SDK defaults.
        if sdk_target_time <= self.arx5_cartesian_controller.get_timestamp():
            # Skip a stale sample; the next control cycle samples the updated time.
            return False
        eef_cmd.timestamp = sdk_target_time
        self.arx5_cartesian_controller.set_eef_cmd(eef_cmd)
        return True

    def terminate_current_policy(self):
        self.arx5_cartesian_controller.reset_to_home()

    def close(self):
        pass

class ArxInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager,
        robot_type='X5_umi',
        robot_can='can0',
        frequency=1000,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        command_lead_time=None,
        max_pos_speed=0.25,
        max_rot_speed=0.16
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        max_pos_speed: m/s, Cartesian speed limit for scheduled waypoints
        max_rot_speed: rad/s, rotational speed limit for scheduled waypoints
        """
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="ArxPositionalController")
        self.robot_type = robot_type
        self.robot_can = robot_can
        self.frequency = frequency
        # Two control cycles by default: 20 ms at 100 Hz. This is a scheduling
        # margin, not a measured hardware latency or a tracking guarantee.
        self.command_lead_time = (
            2.0 / frequency if command_lead_time is None else float(command_lead_time)
        )
        if not np.isfinite(self.command_lead_time) or self.command_lead_time <= 0:
            raise ValueError("command_lead_time must be finite and positive")
        self.launch_timeout = launch_timeout
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_gripper_width': np.zeros((1,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
            ('ActualGripperWidth','get_gripper_widthes')
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(6)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
            elif 'gripper' in func_name:
                example[key] = np.zeros(1, dtype=np.float64)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ArxPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, gripper_width, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        gripper_width = np.array(gripper_width)
        assert pose.shape == (6,)
        assert gripper_width.shape == (1,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'target_gripper_width': gripper_width,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, gripper_width, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        gripper_width = np.array(gripper_width)
        assert gripper_width.shape == (1,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_gripper_width': gripper_width,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def add_waypoint(self, pose, gripper_width, target_time):
        """Buffer a waypoint for the next update_trajectory() call."""
        pose = np.array(pose)
        assert pose.shape == (6,)
        gripper_width = np.array(gripper_width)
        assert gripper_width.shape == (1,)

        message = {
            'cmd': Command.ADD_WAYPOINT.value,
            'target_pose': pose,
            'target_gripper_width': gripper_width,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def update_trajectory(self):
        """Align all buffered waypoints to the current trajectory at once."""
        message = {
            'cmd': Command.UPDATE_TRAJECTORY.value
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start polymetis interface
        robot = ArxInterface(self.robot_type, self.robot_can, USE_MULTITHREADING=True, frequency=self.frequency)

        try:
            if self.verbose:
                print(f"[ArxPositionalController] Connect to robot: {self.robot_can}")
            
            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            curr_gripper_width = robot.get_gripper_widthes()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            gripper_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[float(curr_gripper_width[0]),0,0,0,0,0]]
            )

            # waypoints buffered between ADD_WAYPOINT and UPDATE_TRAJECTORY
            waypoint_buffer = []

            # start franka cartesian impedance policy
            robot.start_cartesian_impedance()

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # Sample and command the SAME future instant in the two clocks.
                sample_time, sdk_target_time = robot.get_command_times(
                    self.command_lead_time
                )
                tip_pose = pose_interp(sample_time)
                gripper_width = np.array([gripper_interp(sample_time)[0]])
                robot.update_desired_ee_pose(
                    tip_pose, gripper_width, sdk_target_time
                )

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )

                        target_gripper_width= float(command['target_gripper_width'][0])
                        gripper_interp = gripper_interp.drive_to_waypoint(
                            pose=[target_gripper_width,0,0,0,0,0],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[ArxPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )

                        target_gripper_width= float(command['target_gripper_width'][0])
                        gripper_interp = gripper_interp.schedule_waypoint(
                            pose=[target_gripper_width,0,0,0,0,0],
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.ADD_WAYPOINT.value:
                        if len(waypoint_buffer) > 0:
                            if (command['target_time']
                                    <= waypoint_buffer[-1]['target_time']):
                                if self.verbose:
                                    print(
                                        "[ArxPositionalController] Waypoint time "
                                        f"{command['target_time']:.3f} is not in "
                                        "the future, skipping")
                                continue
                        waypoint_buffer.append(command)
                    elif cmd == Command.UPDATE_TRAJECTORY.value:
                        if len(waypoint_buffer) == 0:
                            if self.verbose:
                                print(
                                    "[ArxPositionalController] No new waypoints "
                                    "to update trajectory")
                            continue
                        pose_interp, gripper_interp, _ = (
                            _apply_dynamic_latency(
                                pose_interp=pose_interp,
                                waypoint_buffer=waypoint_buffer,
                                t_now=t_now,
                                verbose=self.verbose
                            )
                        )
                        waypoint_buffer = []
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[ArxPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[ArxPositionalController] Disconnected from robot: {self.robot_can}")
