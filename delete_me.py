#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import MjSim, MjViewer, functions
import mujoco_py
import math
import os
import numpy as np

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.00001" />
    <worldbody>
        <body name="robot" pos="0 0 .17">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="1.0" pos="0 0 0" rgba="1 0 0 1" size="0.15" type="sphere"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        </body>
        <body name="floor" pos="0 0 0">
            <geom condim="3" size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <motor gear="1.0" joint="slide0"/>
        <motor gear="1.0" joint="slide1"/>
        <motor gear="1.0" joint="slide2"/>
    </actuator>
</mujoco>
"""

render = True
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim_virtual = MjSim(model)
sim = MjSim(model)
def update_virtual_data(sim, sim_virtual):
    for i in range(len(sim.data.qpos)):
        sim_virtual.data.qpos[i] = sim.data.qpos[i]
    for i in range(len(sim.data.qvel)):
        sim_virtual.data.qvel[i] = sim.data.qvel[i]
    for i in range(len(sim.data.qacc)):
        sim_virtual.data.qacc[i] = sim.data.qacc[i]
    return sim_virtual

tlim = 100
if render:
    viewer = MjViewer(sim)
    tlim = float('Inf')
t = 0
print(f"qpos0 has length {len(model.qpos0)}")
print(f"qpos0 is {model.qpos0}")
print(f"Joint Names is {model.joint_names}")
print(f"qpos_addr is {model.jnt_qposadr}")
print(f"Joint types are {model.jnt_type}")

update_virtual_data(sim, sim_virtual)
while t<tlim:
    sim.data.ctrl[0] = -10
    sim.data.ctrl[1] = 0
    sim.data.ctrl[2] = 0
    t += 1
    sim.step()

    functions.mj_inverse(sim.model, sim_virtual.data)
    twodecimals = ["%.2E" % v for v in sim_virtual.data.qfrc_inverse]
    print(f"qfrc inverse is {twodecimals}")
    #print(f"First 4 entries of efc force are {sim_virtual.data.efc_force[:4]}")
    update_virtual_data(sim, sim_virtual)
    #print(f"Attributes of model: {model.__dir__()}")
    if render:
        viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
