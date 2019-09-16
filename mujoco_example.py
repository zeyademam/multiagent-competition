from mujoco_py import MjSim, MjViewer, functions
import mujoco_py
import math
import os
import numpy as np

MODEL_XML_2 = """<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.00001" />
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
      <motor joint="hip_1" gear="1"/>
      <motor joint="ankle_1" gear="1"/>
      <motor joint="hip_2" gear="1"/>
      <motor joint="ankle_2" gear="1"/>
      <motor joint="hip_3" gear="1"/>
      <motor joint="ankle_3" gear="1"/>
      <motor joint="hip_4" gear="1"/>
      <motor joint="ankle_4" gear="1"/>
    </actuator>
</mujoco>

"""

def update_virtual_data(sim, sim_virtual):
    for i in range(len(sim.data.qpos)):
        sim_virtual.data.qpos[i] = sim.data.qpos[i]
    for i in range(len(sim.data.qvel)):
        sim_virtual.data.qvel[i] = sim.data.qvel[i]
    for i in range(len(sim.data.qacc)):
        sim_virtual.data.qacc[i] = sim.data.qacc[i]
    return sim_virtual


"""XML contains an ant standing on floor."""
model = mujoco_py.load_model_from_xml(MODEL_XML_2)
sim_virtual = MjSim(model)
sim = MjSim(model)
viewer = MjViewer(sim)
tlim = 10**5
t = 0
print(f"qpos0 has length {len(model.qpos0)}")
print(f"qpos0 is {model.qpos0}")
print(f"Joint Names are {model.joint_names}")
print(f"Joint types are {model.jnt_type}")

update_virtual_data(sim, sim_virtual)
while t<tlim:
    sim.data.ctrl[0] = -100
    sim.data.ctrl[1] = 0
    sim.data.ctrl[2] = -100
    sim.data.ctrl[3] = 0
    sim.data.ctrl[4] = 100
    sim.data.ctrl[5] = 0
    sim.data.ctrl[6] = 100
    sim.data.ctrl[7] = 0
    t += 1
    sim.step()

    functions.mj_inverse(sim.model, sim_virtual.data)
    if t==1:
        print(f"The length of qfrc_inverse is {len(sim_virtual.data.qfrc_inverse)}")
    print(f"qfrc_inverse: {sim_virtual.data.qfrc_inverse}")
    update_virtual_data(sim, sim_virtual)
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
