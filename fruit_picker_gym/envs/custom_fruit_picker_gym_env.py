import gymnasium
from gymnasium import spaces

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

from typing import Any

import logging

from fruit_picker_gym.collision import CollisionDetector

logging.basicConfig(filename="fruit_picker_env_logs.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class FruitPickerEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Connect to the UI for visualization
        p.connect(p.GUI)

        # Reset the 3D OpenGL debug visualizer camera distance (between eye and camera
        # target position), camera yaw and pitch and camera target position
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-60, cameraPitch=-40,
                                     cameraTargetPosition=[0.55,-0.35,0.2])

        # Define the action space
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        logger.debug(f"Initial action space: {self.action_space}")

        # Define the observation space
        self.observation_space = spaces.Box(np.array([-1] * 5), np.array([1] * 5))
        logger.debug(f"Initial action space: {self.observation_space}")


    def detect_grasping(self, object) -> bool:
        finger1_contact = p.getContactPoints(bodyA=self.pandaUid, bodyB=object, linkIndexA=9)
        finger2_contact = p.getContactPoints(bodyA=self.pandaUid, bodyB=object, linkIndexA=10)

        if finger1_contact and finger2_contact:
            if finger1_contact[0][0] and finger2_contact[0][0]:
                return True
            else:
                return False
        else:
            return False

    def detect_positive_approach(self, objectUid, threshold) -> (bool, float):
        distance = p.getClosestPoints(bodyA=self.pandaUid, bodyB=objectUid, distance=2.0, linkIndexA=9)
        if distance:
            if distance[0][8] < threshold:
                return True, distance[0][8]
            else:
                return False, threshold
        else:
            return False, threshold

    def detect_collisions(self) -> bool:
        return self.collision_detector.in_collision()



    def reward_func(self, objectUid: Any, state_object: Any, robot_state: Any, trayUid: Any,
                    state_fingers: Any, name: str) -> (float, bool):
        """
        This is a static method within the environment class to define and curate the reward function.
        :param state_fingers: <Any> State of the robot fingers
        :param name: <str> Name of the fruit/object
        :param state_object: <Any> Current state of an object.
        :param trayUid: <Any> PyBullet ID for the tray in which the object is supposed to be put.
        :return: <tuple> Tuple of reward and done status
        """
        overall_reward = 0.0

        # Positive rewards:
        ultimate_positive_reward = 20.0
        grasping_reward= 10.0
        max_positive_approach_reward = 5.0

        # Negative rewards:
        collision_penalty = -30.0
        negative_reward = -20.0

        # Get the boundaries for the destination tray
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = p.getAABB(trayUid)

        if robot_state[-1] < -0.01:
            logger.debug(f"Collision detected. Resetting the environment")
            return collision_penalty, False, True

        # Reward for grasping
        if self.detect_grasping(objectUid):
            overall_reward += grasping_reward
            logger.debug(f"Object {name} has been grasped. Achieved a reward of {grasping_reward}")
            done = False
        elif state_object[2] > 1.0:
            overall_reward += (grasping_reward + 2)
            logger.debug(f"Object {name} has been grasped and lifted. Achieved a reward of {grasping_reward + 2}")
            done = False

        if self.detect_positive_approach(objectUid, threshold=0.2)[0]:
            distance = self.detect_positive_approach(objectUid, threshold=0.2)[1]
            rew = min((0.2/(distance*distance)), max_positive_approach_reward)
            overall_reward += rew
            logger.debug(f"Positive approach has been detected for the object {name}. Achieved a reward of {rew}")
            done = False

        # If the fruit is within the boundaries of the destination tray,
        # reward the agent with the ultimate positive reward
        if ((min_x < state_object[0] < max_x) and (min_y < state_object[1] < max_y)
                and (min_z < state_object[2] < max_z)):
            overall_reward += ultimate_positive_reward
            logger.debug(f"Object {name} target reached. Achieved a reward of {ultimate_positive_reward}")
            done = True
        else:
            overall_reward += negative_reward
            logger.debug(f"Did nothing. Penalty of {negative_reward} has been imposed")
            done = False

        logger.debug(f"Reward for {name} --> {state_object}: {overall_reward}; Done status: {done}")

        return overall_reward, done, False

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
        fingers = action[3]

        # Get the current pose of the end-effector
        # As per the URDF file of the Franka Emika Panda robot, the end-effector link index 11
        currentPose = p.getLinkState(self.pandaUid, 11)

        # Extract the position ignoring the orientation from the pose obtained above
        currentPosition = currentPose[0]

        logger.debug(f"Current pose: {currentPosition}")

        # Compute new positions as per the changes in the corresponding dimensions
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]

        logger.debug(f"New position: {newPosition}")

        # Get the joint pose via inverse kinematics based on the new position and current orientation
        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, newPosition, orientation)
        logger.debug(f"Joint poses for the above new position: {jointPoses}")

        # Simulate the motors to reach the given target value
        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL,
                                    list(jointPoses[:7])+2*[fingers])

        # Perform one step of the simulation
        p.stepSimulation()
        logger.debug(f"One more simulation step")

        # Get the states of the all the objects after performing one simulation step
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)

        # Get the robot state after performing one simulation step
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        logger.debug(f"Robot state: {state_robot}")

        # Get the state of the robot's fingers
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        logger.debug(f"Fingers state: {state_fingers}")

        # Define and update the reward

        # For Strawberries
        reward, done, truncate = self.reward_func(self.objectUid, state_object, state_robot, self.trayUidStrawberries,
                                          state_fingers, name="strawberry_1")

        # Diagnostic information
        info = {"state_object1": state_object}

        logger.debug(f"Diagnostic info of all the objects: {info}")

        observation = state_robot + state_fingers
        logger.debug(f"Final observation --> robot_state + fingers_state: {observation}")

        logger.debug(f"Final reward: {reward}")
        logger.debug(f"Done status: {done}")

        return observation, reward, done, truncate, info

    def reset(self, seed = None, options = None) -> tuple:
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

        # p.setTimeStep(1/60)

        # Get the root path for the URDF files
        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        # Rest poses for the robot
        rest_poses = [-1.57, 0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        # Add the Franka Panda Robot

        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        for i in range(7):
            # Reset joints to the rest poses
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        # Add the table and put the robot on the table
        self.tableUidSource = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),
                                    basePosition=[0.0, -0.4, -0.65], globalScaling=1.0)

        # Add the tray on top of the table
        self.trayUidMix = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                     basePosition=[-0.05, -0.6, 0], globalScaling=1.0)

        # Table 2
        self.tableUidDestination = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),
                                         basePosition=[0.0, 0.6, -0.65], globalScaling=1.0)

        # Tray for bananas
        self.trayUidBananas = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                         basePosition=[-0.4, 0.45, 0], globalScaling=1.0)

        # Tray for strawberries
        self.trayUidStrawberries = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),
                                              basePosition=[0.4, 0.45, 0], globalScaling=1.0)

        # Randomly place the objects within the boundaries of the mix tray leveraging the following random generator
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = p.getAABB(self.trayUidMix)
        state_object_base = [random.uniform(min_x+0.3, max_x-0.3), random.uniform(min_y+0.3, max_y-0.3), 0.05]

        current_path = os.path.abspath(os.path.dirname(__file__))

        fruits = ["../../additional_objects/YcbPear/pear.urdf", "../../additional_objects/YcbStrawberry/strawberry.urdf"]
        fruit = random.choice(fruits)

        # Dummy object
        objectUid1 = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"),
                                    basePosition=[0.1, -0.8, 0.05], globalScaling=1.0)

        # Actual object which the robot is expected to pick
        self.objectUid = p.loadURDF(os.path.join(current_path, fruit),
                                basePosition=state_object_base, globalScaling=1.0)

        # Initialize collision detector:
        self.collision_detector = CollisionDetector(0,
                                                    [(self.pandaUid, 0), (self.pandaUid, 1),
                                                     (self.pandaUid, 2), (self.pandaUid, 3),(self.pandaUid, 4),
                                                     (self.pandaUid, 5), (self.pandaUid, 6),(self.pandaUid, 7),
                                                     (self.pandaUid, 8), (self.pandaUid, 9), (self.pandaUid, 10),
                                                     (self.pandaUid, self.trayUidMix), (self.pandaUid, self.trayUidBananas),
                                                     (self.pandaUid, self.trayUidStrawberries)])


        # Reset base velocity
        p.resetBaseVelocity(self.objectUid, [0,0,0], [0,0,0])

        # Get the state of the end-effector and extract only the pose (ignore orientation)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]

        # Get the state of the fingers
        state_fingers = p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0]

        # Observation space = end-effector state + fingers state
        observation = state_robot + state_fingers

        # Turning visualizer on after loading all the parameters
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        logger.debug(f"Finished resetting the environment")
        return observation, {}

    def render(self) -> Any:
        """
        This method renders the simulation frame by frame as an image
        :param mode: <str> Used to control the rendering mode
        :return: <Any> RGB image of the environment at a given step
        """

        logger.debug(f"Started rendering the image")
        # Place the camera ta a desired position and orientation in the environment
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.1,-0.6,0.2],
                                                          distance=0.7,
                                                          yaw=0,
                                                          pitch=-90,
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
