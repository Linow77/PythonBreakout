import pygame
from breakout_gym import BreakoutEnv2
import numpy as np

from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
import tensorflow as tf

# limit the clock
clock = pygame.time.Clock()
fps = 60

# keep window open
running = True


# ─── FUNCTIONS FOR USER INPUT ──────── #


def key_to_action(key, paddle, width):
    # move left
    if key[pygame.K_LEFT] and paddle.rect.left > 0:
        return 0

    # move right
    elif key[pygame.K_RIGHT] and paddle.rect.right < width:
        return 2
    else:
        return 1

def choose_action(paddle, ball):
    if(ball.speedy > 0):
        if(ball.rect.x > paddle.rect.x + paddle.paddleWidth):
            return 2
        elif (ball.rect.x + ball.rad*2 < paddle.rect.x):
            return 0
        else :
            return 1
    else :
        return 1


clock = pygame.time.Clock()
env = BreakoutEnv2()
print('Action Spec:', env.action_spec())

discrete_action_env = wrappers.ActionDiscretizeWrapper(env, num_actions=3)
print('Discretized Action Spec:', discrete_action_env.action_spec())

tf_env = tf_py_environment.TFPyEnvironment(env)
# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 100
transitions = []
reward = 0

time_step = env.reset()
print(time_step)
cumulative_reward = time_step.reward

print(env.gameOver)
breakout_action = np.array(0, dtype=np.int32)


#while env.gameOver != 1:
for i in range(num_steps):

    #print(env.gameOver)
    #print(env.gameRunning)

    # limit clock
    clock.tick(fps)

    #tensorflow environement    
    action = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    #transitions.append([time_step, action, next_time_step])
    print(action)
    time_step = tf_env.step(action)
    #print(time_step)
    reward += time_step.reward
    

    #python environement
    #time_step = env.step(breakout_action)
    #print(time_step)
    #cumulative_reward += time_step.reward
    

    # get quit event
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            env.gameOver = 1

    #action = choose_action(env.paddle, env.ball)
    #action = env.action_space.sample()
    #reward, done, info = env.step(action)

    # move the ball
    env.ball.move(env.paddle.rect, env.wall.bricks)

    env.render()  # make pygame render calls to window
    pygame.display.update()  # update window

pygame.quit()
