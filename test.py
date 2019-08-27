import gym
import gym_compete
import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import random

from policy import LSTMPolicy, MlpPolicyValue

path = os.path.join(".", "agent-zoo", "run-to-goal", "ants", "agent1_parameters-v1.pkl")

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    sess = tf.Session()

    env = gym.make("run-to-goal-ants-v0")
    obs = env.reset()
    env.seed(0)

    print(f"Shape of Observations {np.asarray(obs).shape}")
    print(f"Observations of First agent \n {obs[0]}")
    print(f"Observations of Second agent \n {obs[1]}")

    with sess.as_default():
        policy = MlpPolicyValue(scope='lstm', reuse=False, ob_space=env.observation_space.spaces[0],
                                ac_space=env.action_space.spaces[0], hiddens=[64, 64], normalize=True)
        sess.run(tf.variables_initializer(tf.global_variables()))
        params = load_from_file(param_pkl_path=path)
        setFromFlat(policy.get_variables(), params)
        act = policy.act(stochastic=True, observation=obs[0])
    print("*"*10)
    print(f"Shape of action {np.asarray(act)[0].shape}")
    print(f"Actions of First Agent\n {act}")
    ant0 = env.agents[0]
    print(f"The output of get_qpos has length: {len(ant0.get_qpos())}")
    print(f"The output of get_qvel has length: {len(ant0.get_qvel())}")
    print(f"The output of get_other_qpos has length: {len(ant0.get_other_qpos())}")
