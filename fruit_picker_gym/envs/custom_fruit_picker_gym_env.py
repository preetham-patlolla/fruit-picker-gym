import gym
from gym import spaces

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

from typing import Any

import logging

logging.basicConfig(filename="fruit_picker_env_logs.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class FruitPickerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Connect to the UI for visualization
        p.connect(p.GUI)

        # Reset the 3D OpenGL debug visualizer camera distance (between eye and camera
        # target position), camera yaw and pitch and camera target position
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-60, cameraPitch=-40,
                                     cameraTargetPosition=[0.55,-0.35,0.2])

        # Define the action space
        self.action_space = spaces.Box(np.array([-1] * 8), np.array([1] * 8))
        logger.debug(f"Initial action space: {self.action_space}")

        # Define the observation space
        self.observation_space = spaces.Box(np.array([-1] * 8), np.array([1] * 8))
        logger.debug(f"Initial action space: {self.observation_space}")

    @staticmethod
    def reward_func(state_object: Any, trayUid: Any, name: str, penalty_factor: int = 2) -> (int, bool):
        """
        This is a static method within the environment class to define and curate the reward function.
        :param state_object: <Any> Current state of an object.
        :param trayUid: <Any> PyBullet Id for the tray in which the object is supposed to be put.
        :param penalty_factor: <Any> Factor by which the -ve reward is amplified compared to the +ve reward.
        :return: <tuple> Returns a tuple containing the reward as int and a bool status (done/not_done).
        """

        positive_reward: int = 1
        negative_reward: int = -(positive_reward * penalty_factor)

        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = p.getAABB(trayUid)
        if ((min_x < state_object[0] < max_x) and (min_y < state_object[1] < max_y)
                and (min_z < state_object[2] < max_z)):
            reward = positive_reward
            done = True
        else:
            reward = negative_reward
            done = False

        logger.debug(f"Reward for {name} --> {state_object}: {reward}; Done status: {done}")

        return reward, done

    def step(self, action: Any) -> Any:
        """
        This method decides what happens in each step of the simulation
        :param action: <Any> Takes in the action and updates the environment accordingly
        :return: <Any> Returns a tuple of observation, final_reward, status (done/not_done)
                 and diagnostic info.
        """

        # Configure the built-in OpenGL visualizer for better rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Convert the orientation from Euler to Quaternion
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        logger.debug(f"Orientation: {orientation}")

        # Factor for smoother inverse kinematics output
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers_pose = action[3]

        dr = 0.5
        da = action[4] * dr
        db = action[5] * dr
        dc = action[6] * dr

        # Get the current pose of the end-effector
        # As per the URDF file of the Franka Emika Panda robot, the end-effector link index 11
        currentPose = p.getLinkState(self.pandaUid, 11)

        # Extract the position ignoring the orientation from the pose obtained above
        currentPosition = currentPose[0]
        currentOrientation = currentPose[1]

        logger.debug(f"Current pose: {currentPosition}")
        logger.debug(f"Current orientation: {currentOrientation}")

        # Compute new positions as per the changes in the corresponding dimensions
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]

        newOrientation = [currentOrientation[0] + da,
                          currentOrientation[1] + db,
                          currentOrientation[2] + dc]

        logger.debug(f"New position: {newPosition}")

        # Get the joint pose via inverse kinematics based on the new position and current orientation
        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, newOrientation)
        logger.debug(f"Joint poses for the above new position: {jointPoses}")

        # Simulate the motors to reach the given target value
        p.setJointMotorControlArray(self.pandaUid, list(range(11)), p.POSITION_CONTROL,
                                    list(jointPoses) + 2 * [fingers_pose])

        # Perform one step of the simulation
        p.stepSimulation()
        logger.debug(f"One more simulation step")

        # Get the states of the all the objects after performing one simulation step
        state_object1, _ = p.getBasePositionAndOrientation(self.objectUid1)
        state_object2, _ = p.getBasePositionAndOrientation(self.objectUid2)
        state_object3, _ = p.getBasePositionAndOrientation(self.objectUid3)
        state_object4, _ = p.getBasePositionAndOrientation(self.objectUid4)
        state_object5, _ = p.getBasePositionAndOrientation(self.objectUid5)
        state_object6, _ = p.getBasePositionAndOrientation(self.objectUid6)

        # Get the robot state after performing one simulation step
        state_robot = p.getLinkState(self.pandaUid, 11)
        logger.debug(f"Robot state: {state_robot}")

        # Get the state of the robot's fingers
        state_fingers = p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)
        logger.debug(f"Fingers state: {state_fingers}")

        # Define and update the reward

        # For Strawberries
        reward1, done1 = self.reward_func(state_object1, self.trayUidStrawberries,
                                          penalty_factor=3, name="strawberry_1")
        reward2, done2 = self.reward_func(state_object2, self.trayUidStrawberries,
                                          penalty_factor=3, name="strawberry_2")
        reward3, done3 = self.reward_func(state_object3, self.trayUidStrawberries,
                                          penalty_factor=3, name="strawberry_3")

        # For Bananas
        reward4, done4 = self.reward_func(state_object4, self.trayUidBananas,
                                          penalty_factor=3, name="banana_1")
        reward5, done5 = self.reward_func(state_object5, self.trayUidBananas,
                                          penalty_factor=3, name="banana_2")
        reward6, done6 = self.reward_func(state_object6, self.trayUidBananas,
                                          penalty_factor=3, name="banana_3")

        # Compute final reward and final done status
        final_reward = reward1 + reward2 + reward3 + reward4 + reward5 + reward6
        done = (done1 and done2 and done3 and done4 and done5 and done6)

        # Diagnostic information
        info = {"state_object1": state_object1, "state_object2": state_object2, "state_object3": state_object3,
                "state_object4": state_object4, "state_object5": state_object5, "state_object6": state_object6}

        logger.debug(f"Diagnostic info of all the objects: {info}")

        observation = state_robot + state_fingers
        logger.debug(f"Final observation --> robot_state + fingers_state: {observation}")

        logger.debug(f"Final reward: {final_reward}")
        logger.debug(f"Done status: {done}")

        return observation, final_reward, done, info

    def reset(self, seed: Any = None, options: Any = None) -> tuple:
        """
        This method resets all the parameters and spaces before the start of the simulation
        :param seed: <Any> Used to reproduce the results
        :param options: <Any>
        :return: <Any> Returns the observation in the beginning of the simulation
        """

        logger.debug(f"Begin resetting simulation and the environment")
        p.resetSimulation()

        # Disable rendering while loading all the parameters
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Set gravity for the environment
        p.setGravity(0, 0, -10)

        # Get the root path for the URDF files
        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        # Rest poses for the robot
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        # Add the Franka Panda Robot
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)

        for i in range(9):
            # Reset joints to the rest poses
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        # Add the table and put the robot on the table
        tableUidSource = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),
                                    basePosition=[0.0, -0.4, -0.65], globalScaling=1.0)

        # Add the tray on top of the table
        self.trayUidMix = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                     basePosition=[-0.05, -0.6, 0], globalScaling=1.2)

        # Table 2
        tableUidDestination = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),
                                         basePosition=[0.0, 0.6, -0.65], globalScaling=1.0)

        # Tray for bananas
        self.trayUidBananas = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                         basePosition=[-0.4, 0.45, 0], globalScaling=1.2)

        # Tray for strawberries
        self.trayUidStrawberries = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                              basePosition=[0.4, 0.45, 0], globalScaling=1.2)

        # Randomly place the objects within the boundaries of the mix tray leveraging the following random generator
        state_object_base = [random.uniform(-0.2, 0.15), random.uniform(-0.85, -0.5), 0.001]

        current_path = os.path.abspath(os.path.dirname(__file__))
        self.objectUid1 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbStrawberry/strawberry.urdf"),
                                     basePosition=state_object_base, globalScaling=2.0)
        self.objectUid2 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbStrawberry/strawberry.urdf"),
                                     basePosition=state_object_base, globalScaling=2.0)
        self.objectUid3 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbStrawberry/strawberry.urdf"),
                                     basePosition=state_object_base, globalScaling=2.0)
        self.objectUid4 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbBanana/banana.urdf"),
                                     basePosition=state_object_base)
        self.objectUid5 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbBanana/banana.urdf"),
                                     basePosition=state_object_base)
        self.objectUid6 = p.loadURDF(os.path.join(current_path, "../../additional_objects/YcbBanana/banana.urdf"),
                                     basePosition=state_object_base)

        # Get the state of the end-effector and extract only the pose (ignore orientation)
        state_robot = p.getLinkState(self.pandaUid, 11)

        # Get the state of the fingers
        state_fingers = p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)

        # Observation space = end-effector state + fingers state
        observation = state_robot + state_fingers

        # Turning visualizer on after loading all the parameters
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        logger.debug(f"Finished resetting the environment")
        return observation, {}

    def render(self, mode: str = "human") -> Any:
        """
        This method renders the simulation frame by frame as an image
        :param mode: <str> Used to control the rendering mode
        :return: <Any> RGB image of the environment at a given step
        """

        logger.debug(f"Started rendering the image")
        # Place the camera ta a desired position and orientation in the environment
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.25,0.1,0.2],
                                                          distance=1.75,
                                                          yaw=0,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)

        # Get the image from the camera with specified dimensions
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]

        logger.debug(f"Extracted the rgb image")
        return rgb_array

    def close(self) -> None:
        """
        Used to disconnect from the physics server of PyBullet
        :return: <None> Returns nothing
        """
        p.disconnect()
