from mujoco_py import MjSim, MjViewer, functions
import mujoco_py
import math
import os
import numpy as np

MODEL_XML = """<?xml version="1.0" ?>
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

model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = MjSim(model)
tlim = 50
t = 0

while t<tlim:
    sim.data.ctrl[0] = -0.01
    sim.data.ctrl[1] = 0
    sim.data.ctrl[2] = -0.01
    sim.data.ctrl[3] = 0
    sim.data.ctrl[4] = 0.01
    sim.data.ctrl[5] = 0
    sim.data.ctrl[6] = 0.01
    sim.data.ctrl[7] = 0
    #######
    inverse_output = functions.mj_inverse(model, np.array(0))
    print(inverse_output)
    #######
    t += 1
    sim.step()
