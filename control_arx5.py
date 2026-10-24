# %%
import sys
import os

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from real_arx5.arx5_interpolation_controller import ArxInterpolationController


# %%
@click.command()
@click.option('-rh', '--robot_type', default='X5')
@click.option('-gp', '--robot_can', type=str, default='can0')
@click.option('-f', '--frequency', type=float, default=30)
def main(robot_type,robot_can,frequency):
    max_gripper_width = 0.08
    gripper_speed = 0.01
    max_pos_speed = 0.25
    max_rot_speed = 0.6

    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
        ArxInterpolationController(
            shm_manager=shm_manager,
            robot_type=robot_type,
            robot_can=robot_can,
            frequency=100,
            verbose=False
        ) as controller, \
        Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print('Ready! ')
            state = controller.get_state()

            target_pose = state["ActualTCPPose"]
            gripper_target_pos = state["ActualGripperWidth"]
            t_start = time.monotonic()

            iter_idx = 0
            stop = False
            while not stop:
                state = controller.get_state()

                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    # if key_stroke != None:
                    #     print(key_stroke)
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                        break
                precise_wait(t_sample)
                sm_state = sm.get_motion_state_transformed()

                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * \
                        st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()

                dpos = 0
                if sm.is_button_pressed(0):
                    # close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dpos = gripper_speed / frequency
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                controller.schedule_waypoint(target_pose, gripper_target_pos, 
                    t_command_target-time.monotonic()+time.time())

                precise_wait(t_cycle_end)
                iter_idx += 1
                # print(1/(time.time() -s))

# %%
if __name__ == '__main__':
    main()