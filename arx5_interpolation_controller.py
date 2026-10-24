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

# no further use , just remain symbol
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2

class ArxInterface:
    def __init__(self, robot_type='X5', can_id='can0', USE_MULTITHREADING=True, frequency = 100):
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

        np.set_printoptions(precision=3, suppress=True)
        self.arx5_cartesian_controller.set_log_level(arx5.LogLevel.DEBUG)

    def get_ee_pose(self):
        # SDK: xyz + KDL RPY (radians). UMI: xyz + rotation vector.
        # KDL RPY is Rz(yaw) @ Ry(pitch) @ Rx(roll), scipy's lowercase xyz.
        arx5_ee_pose = np.array(self.arx5_cartesian_controller.get_eef_state().pose_6d().copy())
        arx5_ee_pose[3:] = st.Rotation.from_euler(
            'xyz', arx5_ee_pose[3:], degrees=False
        ).as_rotvec()
        return arx5_ee_pose

    def get_joint_positions(self):
        return np.array(self.arx5_cartesian_controller.get_joint_state().pos().copy())

    def get_joint_velocities(self):
        return np.array(self.arx5_cartesian_controller.get_joint_state().vel().copy())

    def get_gripper_widthes(self):
        return np.array([self.arx5_cartesian_controller.get_joint_state().gripper_pos], dtype=np.float64)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        cartesian_waypoints = positions

        waypoint_interval_s = time_to_go
        interpolate_interval_s = 0.1
        interpolated_waypoints = []
        for i in range(len(cartesian_waypoints) - 1):
            start = cartesian_waypoints[i]
            end = cartesian_waypoints[i + 1]

            num_interpolations = int(waypoint_interval_s / interpolate_interval_s)
            for j in range(num_interpolations):
                interpolated_waypoint = start + (end - start) * j / num_interpolations
                interpolated_waypoints.append(interpolated_waypoint)

        for waypoint in interpolated_waypoints:
            eef_state = self.arx5_cartesian_controller.get_eef_state()
            eef_cmd = arx5.EEFState(waypoint, 0.08)
            eef_cmd.timestamp = eef_state.timestamp + interpolate_interval_s
            self.arx5_cartesian_controller.set_eef_cmd(eef_cmd)

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
        """Send a UMI rotation-vector pose with an explicit SDK arrival time.

        This only converts rotation representation. The SDK eef frame must
        already coincide with the TCP used by the caller.
        """
        pose_rpy = np.asarray(pos, dtype=np.float64).copy()
        pose_rpy[3:] = st.Rotation.from_rotvec(pose_rpy[3:]).as_euler(
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
        robot_type='X5',
        robot_can='can0',
        frequency=1000,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        command_lead_time=None
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

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
