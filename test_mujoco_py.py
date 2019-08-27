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
    <option timestep="0.005" />
    <worldbody>
        <body name="robot" pos="0 0 1.2">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="1.0" pos="0.1 0.1 0" rgba="1 0 0 1" size="0.15" type="sphere"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        </body>
        <body mocap="true" name="mocap" pos="0.5 0.5 0.5">
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.1 0.1 0.1" type="box"></geom>
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.2 0.2 0.05" type="box"></geom>
		</body>
        <body name="cylinder" pos="0.1 0.1 0.2">
            <geom mass="1" size="0.15 0.15" type="cylinder"/>
            <joint axis="1 0 0" name="cylinder:slidex" type="slide"/>
            <joint axis="0 1 0" name="cylinder:slidey" type="slide"/>
        </body>
        <body name="box" pos="-0.8 0 0.2">
            <geom mass="0.1" size="0.15 0.15 0.15" type="box"/>
        </body>
        <body name="floor" pos="0 0 0.025">
            <geom condim="3" size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <motor gear="2000.0" joint="slide0"/>
        <motor gear="2000.0" joint="slide1"/>
        <motor gear="2000.0" joint="slide2"/>
    </actuator>
</mujoco>
"""
MODEL_XML_2 = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="floor" pos="0 0 0">
            <geom condim="3" size="3 3 0.02" rgba="0 1 0 1" type="box"/>
        </body>
        <body name="torso" pos="0 0 0.2" euler="0 0 90">
            <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
            <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" range="-30 30" type="free"/>
            <body name="front_left_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
                <body name="aux_1" pos="0.2 0.2 0">
                    <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
                    <body pos="0.2 0.2 0">
                        <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="front_right_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
                <body name="aux_2" pos="-0.2 0.2 0">
                    <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
                    <body pos="-0.2 0.2 0">
                        <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="back_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
                <body name="aux_3" pos="-0.2 -0.2 0">
                    <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
                    <body pos="-0.2 -0.2 0">
                        <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            <body name="right_back_leg" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
                <body name="aux_4" pos="0.2 -0.2 0">
                    <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
                    <body pos="0.2 -0.2 0">
                        <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    </actuator>
</mujoco>

"""

qpos_desired =[-6.35834486e-18,  1.94761792e-04,  2.91630355e-01,  7.07106781e-01,
               -1.21467660e-05,  1.21467660e-05,  7.07106781e-01,  1.79360869e-03,
               3.39601866e-01,  1.79349124e-03, -3.39640114e-01, -1.79349124e-03
               -3.39640114e-01, -1.79360869e-03, 3.39601866e-01]

render = True
model = mujoco_py.load_model_from_xml(MODEL_XML_2)
sim_virtual = MjSim(model)
#sim_virtual.data.qpos = qpos_desired
#assert sim_virtual.data.qpos == qpos_desired
sim = MjSim(model)
tlim = 50
if render:
    viewer = MjViewer(sim)
    tlim = float('Inf')
t = 0
print(f"qpos0 has length {len(model.qpos0)}")
print(f"qpos0 is {model.qpos0}")
print(f"Joint Names is {model.joint_names}")
print(f"qpos_addr is {model.jnt_qposadr}")
print(f"Joint types are {model.jnt_type}")

while t<tlim:
    sim.data.ctrl[0] = -0.01
    sim.data.ctrl[1] = 0
    sim.data.ctrl[2] = -0.01
    sim.data.ctrl[3] = 0
    sim.data.ctrl[4] = 0.01
    sim.data.ctrl[5] = 0
    sim.data.ctrl[6] = 0.01
    sim.data.ctrl[7] = 0
    print(f"Shape of qpos_desired {np.asarray(qpos_desired).shape}")
    print(f"Shape of qvel {np.asarray(sim.data.qvel).shape}")
    print(f"Shape of qacc {np.asarray(sim.data.qacc).shape}")
    print(f"Shape of mocap_pos {np.asarray(sim.data.mocap_pos).shape}")
    print(f"Shape of mocap_quat {np.asarray(sim.data.mocap_quat).shape}")
    #inverse_output = functions.mj_inverse(model,
     #                                     np.concatenate((qpos_desired,
      #                                                    sim.data.qvel,
       #                                                   sim.data.qacc,)))
        #                                                  #sim.data.mocap_pos,
         #                                                 #sim.data.mocap_quat)))
    print(f"qpos is {sim.data.qpos}")
    t += 1
    sim.step()
    if render:
        viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
