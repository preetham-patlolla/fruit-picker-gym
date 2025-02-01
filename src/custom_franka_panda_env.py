import os
import math
import pybullet as p
import pybullet_data

# Start the simulation graphical user interface
p.connect(p.GUI)

# Reset camera position and angle so that the simulation view starts close to the setup instead of starting far away
p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-60, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

# Set gravity
p.setGravity(0,0,-9.8)

print(f"Path to pybullet_data: {pybullet_data.getDataPath()}")

# Add the Franka Panda Robot
pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True, basePosition=[1,0,0])

# Add the table and put the robot on the table
tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),basePosition=[0.5,-0.4,-0.65], globalScaling=1.0)

# Add the tray on top of the table
trayUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "tray/traybox.urdf"),basePosition=[0.4,-0.4,0], globalScaling=1.3)

# Add the conveyor belt next to the table
conveyorUid = p.loadURDF(r"C:\MyProjects\RoboticsProject\additional_objects\conveyor\conveyor_belt.urdf", basePosition=[3.2,0.65,-0.225])

# Add a random object in the bin
objectUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf"), basePosition=[0.4,-0.4,0.01])
objectUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf"), basePosition=[0.43,-0.42,0.01])
objectUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf"), basePosition=[0.46,-0.46,0.01])
objectUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf"), basePosition=[0.37,-0.37,0.01])
objectUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf"), basePosition=[0.34,-0.34,0.01])


while True:
    p.stepSimulation()