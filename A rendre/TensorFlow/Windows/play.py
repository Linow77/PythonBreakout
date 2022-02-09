from environment import BreakoutEnv
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_environment

from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

#-- Create Python Environment --#
env = BreakoutEnv(visualize=True) #add fps parameters if needed
# print('Action Spec:', env.action_spec())

# # check if python environement is correct
# print("validate python environment")
# utils.validate_py_environment(env, episodes=5)

#Variables for testing environment
num_episode = 10
reward = 0

# #-- Test  with Python Environment pygame --#
# for i in range(num_episode):
#     time_step = env.reset()
#     #while episode not done
#     while not time_step.is_last():
#         #tensorFlow environment
#         action = np.random.randint(0,3)
#         time_step = env.step(action)
#         reward += time_step.reward
#     print("Episode "+str(i+1)+"/"+str(num_episode)+ " done : " + str(env.score) + 
#     "/" + str(env.wall.breakableBricks)+" bricks")

# print("reward:",reward)

#-- Convert in Tensor Environment --#
tf_env = tf_py_environment.TFPyEnvironment(env)

# # Check if TensorFlow environment is correct
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())

#-- Test  with Tensor Environment pygame --#
for i in range(num_episode):
    time_step = tf_env.reset()
    #while episode not done
    while not time_step.is_last():
        #tensorFlow environment
        action = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        time_step = tf_env.step(action)
        reward += time_step.reward
    print("Episode "+str(i+1)+"/"+str(num_episode)+ " done : " + str(env.score) + 
    "/" + str(env.wall.breakableBricks)+" bricks")

print("reward:",reward.numpy())




