# 
# http://wiki.ros.org/urdf/Tutorials
# http://wiki.ros.org/urdf/Tutorials/Building%20a%20Visual%20Robot%20Model%20with%20URDF%20from%20Scratch
# 
# 

import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
#p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)

#boxId = p.loadURDF("sphere2.urdf",startPos, startOrientation)
#boxId = p.loadURDF("racecar/racecar.urdf",startPos, startOrientation)
#boxId = p.loadURDF("lego/lego.urdf",startPos, startOrientation)
#boxId = p.loadURDF("./wheel.urdf",startPos, startOrientation)
boxId = p.loadURDF("./test.urdf",startPos, startOrientation)

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
