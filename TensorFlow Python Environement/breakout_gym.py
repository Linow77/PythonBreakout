import pygame
import random as r
import math
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tf_agents.environments import py_environment

from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent

# Allow to use tf_random
tf.compat.v1.enable_eager_execution()

class wall():
    def __init__(self,instanceEnv):
        ## Env Variables ##
        self.env = instanceEnv

        ## Wall Variables ##
        # Same random seed for every creation of wall
        r.seed(1000) #1000
        # number of brick per row and per column
        self.rows = 6
        self.cols = 5
        #Sizes
        self.wallHeight = (self.env.height - 100) // 2
        self.brickHeight = self.wallHeight // self.rows

        #Table for storing bricks
        self.bricks = []
        #Number of breakable bricks
        self.breakableBricks = 0
        
        # border of the bricks (gap between bricks)
        self.border = 3
        self.brickWidth = self.env.width // self.cols

        #Save of the wall in order to reset the episode
        self.save = []

    def createBricks(self):
        for rowNumber in range(self.rows):
            for colNumber in range(self.cols):
                # Create Rectancle for each brick
                brick = pygame.Rect(
                    colNumber*self.brickWidth, 100+rowNumber*self.brickHeight, self.brickWidth, self.brickHeight)
                
                # 25% unbreakable bricks
                type = r.randint(0,3)
                if type != 0:
                    self.breakableBricks+=1

                # Store bricks inside table
                self.bricks.append(
                    (colNumber, rowNumber, type, brick))

                colNumber += 1
            rowNumber += 1
        #Save the wall 
        self.save = self.bricks.copy()

    def resetBricks(self):
        self.bricks = self.save.copy()      

    def printWall(self,screen):
        for brick in self.bricks:
            # check for unbreakable bricks
            if(brick[2] == 0):
                color = self.env.color1
            else:
                color = self.env.color2
            
            # print bricks
            pygame.draw.rect(screen, (color),
                             ((brick[3].x + self.border), (brick[3].y + self.border), self.brickWidth - 2*self.border, self.brickHeight - 2*self.border))

class paddle():
    def __init__(self,instanceEnv):
        ## Env Variables ##
        self.env = instanceEnv

        ## Paddle Variables ##
        # Sizes
        self.paddleWidth = self.env.width // 5
        self.paddleHeight = 10
        # Initial Position
        self.x = (self.env.width - self.paddleWidth)/2  
        self.y = self.env.height - 60
        #Create teh rectangle of the paddle
        self.rect = pygame.Rect(
            self.x, self.y, self.paddleWidth, self.paddleHeight)
        #Speed of the paddle
        self.speed = 8

    def printPaddle(self,screen):
        pygame.draw.rect(screen, self.env.color3, self.rect)     

    def move(self):
        # get key pressed
        key = pygame.key.get_pressed()

        # move left
        if key[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed

        # move right
        if key[pygame.K_RIGHT] and self.rect.right < self.env.width:
            self.rect.x += self.speed

class ball():
    def __init__(self,instanceEnv):
        ## Env Variables ##
        self.env = instanceEnv

        ## Ball Variables ##
        # Initial speed of the ball on each axes
        self.ballSpeed = 2.5  # on Axes at the beginning
        self.speedx = self.ballSpeed
        self.speedy = -self.ballSpeed

        # Global Velocity of the ball
        self.velocity = math.sqrt(2*(self.ballSpeed**2))

        # Radius of the ball
        self.rad = 10

        # Initial position of the ball
        self.x = self.env.width // 2 - self.rad  # init position of the rectangle
        self.y = self.env.height - 90            # init position of the rectangle

        # Create a rectangle around the ball
        self.rect = pygame.Rect(
            self.x, self.y, 2*self.rad, 2*self.rad)

        #Horizontal Direction (from the left to the right = 1)
        self.directionH = 1

        #Angle of the colision between the ball and the paddle (or the wall)
        self.angle = -1
        #Angle of the redirection after colision
        self.newAngle = -1

        #Save the nature of the colision (wall, screen, or paddle)
        self.lastCollision = ""

        # margin Error for Colisions
        self.margin = 5

    def printBall(self,screen):
        pygame.draw.circle(screen, self.env.color3, (self.rect.x +
                                            self.rad, self.rect.y + self.rad), self.rad)
        
    def move(self, paddleRect, wallBricks):
        #reset lastCollision
        self.lastCollision = ""

        # Add speed every tick to the ball's coordinates
        self.x += self.speedx
        self.y += self.speedy

        # Update the position of the ball
        self.rect.x = self.x
        self.rect.y = self.y

        #-- Check for screen borders --#
        # left and right borders
        if self.rect.right > self.env.width or self.rect.left < 0:
            self.speedx *= -1
            #save collision for reward
            self.lastCollision = "sideBorder"

        # top border
        elif self.rect.top < 0:
            self.speedy *= -1
            #save collision for reward
            self.lastCollision = "topBorder"

        # bottom border
        elif self.rect.bottom > self.env.height:
            #save collision for reward
            self.lastCollision = "endBorder"

            # reset ball position
            self.x = self.env.width // 2 - self.rad
            self.y = self.env.height - 90
            self.rect.x = self.x
            self.rect.y = self.y
            self.speedx = self.ballSpeed
            self.speedy = -self.ballSpeed

            # reset paddle position
            paddleRect.rect.x = (self.env.width - paddleRect.paddleWidth)/2  # init position

            # get a new ball if availabe
            if self.env.balls > 0:
                # delete a ball
                self.env.balls -= 1

            else:  # no ball left
                self.env._episode_ended = True

        #-- End Check for screen borders --#

        #-- Check for collisions between ball and paddle --#
        elif self.rect.y > paddleRect.rect.y - 2*self.rad:
            if self.rect.colliderect(paddleRect):

                # TOP COLLISION
                # check if ball is on top of the trail (between the 5px margin)
                if(abs(self.rect.bottom - paddleRect.rect.y) < self.margin and self.speedy > 0):

                    #save collision for reward
                    self.lastCollision = "top"

                    # calculate angle between the ball's path and the trail
                    if self.speedx == 0:
                        self.angle = 90
                    else:
                        self.angle = abs(math.degrees(
                            math.atan(self.speedy/abs(self.speedx))))

                    if(self.speedx > 0):
                        self.directionH = 1  # from the left
                    else:
                        self.directionH = -1  # from the right

                    # change x if not on the middle of the paddle

                    # LEFT PART
                    if self.rect.right >= paddleRect.rect.x and self.rect.left < paddleRect.rect.x + (0.2 * paddleRect.paddleWidth):

                        if self.directionH == 1:  # from the left
                            # increase angle by 35%
                            if self.angle < 30:
                                self.newAngle = self.angle*1.35

                            elif self.angle > 60:  # increase angle by 25%
                                self.newAngle = self.angle*1.25

                            else:  # increase angle by 30%
                                self.newAngle = self.angle*1.30

                        else:  # from the right
                            # reduce angle by 35%
                            if self.angle < 30:
                                self.newAngle = self.angle*0.65

                            elif self.angle > 60:  # reduce angle by 25%
                                self.newAngle = self.angle*0.75

                            else:  # reduce angle by 30%
                                self.newAngle = self.angle*0.70

                        self.speedx = - \
                            (math.cos(math.radians(self.newAngle)) * self.velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * self.velocity)

                    # MIDDLE LEFT PART
                    elif self.rect.right >= paddleRect.rect.x + (0.2 *paddleRect.paddleWidth) and self.rect.left < paddleRect.rect.x + (0.4 *paddleRect.paddleWidth):

                        if self.directionH == 1:  # from the left
                            # increase angle by 20%
                            if self.angle < 30:
                                self.newAngle = self.angle*1.2

                            elif self.angle > 60:  # increase angle by 10%
                                self.newAngle = self.angle*1.1

                            else:  # increase angle by 15%
                                self.newAngle = self.angle*1.15

                        else:  # from the right
                            # reduce angle by 20%
                            if self.angle < 30:
                                self.newAngle = self.angle*0.8

                            elif self.angle > 60:  # reduce angle by 10%
                                self.newAngle = self.angle*0.9

                            else:  # reduce angle by 15%
                                self.newAngle = self.angle*0.85

                        self.speedx = - \
                            (math.cos(math.radians(self.newAngle)) * self.velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * self.velocity)

                    elif self.rect.right >= paddleRect.rect.x + (0.4 * paddleRect.paddleWidth) and self.rect.left < paddleRect.rect.x + (0.6 * paddleRect.paddleWidth):
                        # angle is not changed
                        self.speedy *= -1

                    # MIDLE RIGHT PART
                    elif self.rect.right >= paddleRect.rect.x + (0.6 * paddleRect.paddleWidth) and self.rect.left < paddleRect.rect.x + (0.8 * paddleRect.paddleWidth):

                        if self.directionH == 1:  # from the left
                            # reduce angle by 20%
                            if self.angle < 30:
                                self.newAngle = self.angle*0.8

                            elif self.angle > 60:  # reduce angle by 10%
                                self.newAngle = self.angle*0.9

                            else:  # send the ball in opposite direction
                                # reduce angle by 15%
                                self.newAngle = self.angle*0.85

                        else:  # from the right
                            if self.angle < 30:
                                # increase angle by 20%
                                self.newAngle = self.angle*1.2

                            elif self.angle > 60:  # increase angle by 10%
                                self.newAngle = self.angle*1.1

                            else:  # increase angle by 15%
                                self.newAngle = self.angle*1.15

                        self.speedx = \
                            (math.cos(math.radians(self.newAngle)) * self.velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * self.velocity)

                    # RIGHT PART
                    elif self.rect.right >= paddleRect.rect.x + (0.8 * paddleRect.paddleWidth) and self.rect.left < paddleRect.rect.x + paddleRect.paddleWidth:
                        if self.directionH == 1:  # from the left
                            # reduce angle by 35%
                            if self.angle < 30:
                                self.newAngle = self.angle*0.65

                            elif self.angle > 60:  # reduce angle by 25%
                                self.newAngle = self.angle*0.75

                            else:  # send the ball in opposite direction
                                # reduce angle by 30%
                                self.newAngle = self.angle*0.70

                        else:  # from the right
                            if self.angle < 30:
                                # increase angle by 35%
                                self.newAngle = self.angle*1.35

                            elif self.angle > 60:  # increase angle by 25%
                                self.newAngle = self.angle*1.25

                            else:  # increase angle by 30%
                                self.newAngle = self.angle*1.30

                        self.speedx = \
                            (math.cos(math.radians(self.newAngle)) * self.velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * self.velocity)

                # SIDES COLLISIONS
                else:  # collision with side
                    #save collision for reward
                    self.lastCollision = "side"
                    if (abs(self.rect.left - paddleRect.rect.right) < self.margin):
                        # check direction of the ball
                        if self.speedx < 0:
                            self.speedx *= -1
                        else:  # if same direction as the paddle don't reverse direction but increase speed
                            self.speedx += 3

                    elif (abs(self.rect.right - paddleRect.rect.left) < self.margin):
                        # check direction of the ball
                        if self.speedx > 0:
                            self.speedx *= -1
                        else:  # if same direction as the paddle don't reverse direction but increase speed
                            self.speedx -= 3

        #-- End Check for collisions between ball and paddle --#

        #-- Check for collisions between ball and Bricks --#
        elif self.rect.y > 0 and self.rect.y < (self.env.height // 2) + 100:
            for brick in wallBricks:
                if(self.rect.colliderect(brick[3])):
                    # check if collision is on top or at bottom of the brick
                    if ((abs(self.rect.top - brick[3].bottom) < self.margin and self.speedy < 0) or
                            (abs(self.rect.bottom - brick[3].top) < self.margin and self.speedy > 0)):
                        # top or bottom
                        self.speedy *= -1

                    elif ((abs(self.rect.left - brick[3].right) < self.margin and self.speedx < 0) or
                          (abs(self.rect.right - brick[3].left) < self.margin and self.speedx > 0)):
                        # right or left
                        self.speedx *= -1

                    # delete the brick if breakable
                    if(brick[2] != 0):
                        wallBricks.remove(brick)
                        #add score
                        self.env.score+=1

    def gravity(self):
        if (self.speedy >= 0 and self.speedy < 1):  # stuck horizontally
            self.speedy = 1
        elif (self.speedy < 0 and self.speedy > -1):  # stuck horizontally
            self.speedy = -1

class BreakoutEnv2(py_environment.PyEnvironment):
  def __init__(self, visualize, fps=10000):
    ## Init Variables ##
    # screen size
    self.width = 640
    self.height = 600   

    # remaining balls
    self.balls = 2 

    ## Init object ##
    self.wall = wall(self)
    self.paddle = paddle(self)
    self.ball = ball(self)

    # Score #
    self.score = 0

    ## Definition of actions and observation 
    #three actions : move left, move right or stand still
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
    #observation on ball's position and paddle's position
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,2), dtype=np.int32, minimum=0, maximum=self.width, name='observation')
    #observation store inside self._state
    self._state = [self.paddle.rect.x,self.ball.rect.x]

    #Check if episode is ended
    self._episode_ended = False

    #Variables for Pygame
    self.visualize = visualize 
    #Activate the visualisation
    if(visualize==True):
        # Create the screen
        self.screen = None

        # Create the clock
        self.clock = pygame.time.Clock()
        self.fps = fps

        # title
        pygame.display.set_caption('Breakout')

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self): #a faire
    self._episode_ended = False
    #Reset wall
    self.wall.resetBricks()
    #reset remainig balls
    self.balls = 2
    #reset score
    self.score = 0

    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):
    #Check if last action ended the episode
    if self._episode_ended:
      #restart a new episode
      return self.reset()

    #Move the ball
    self.ball.move(self.paddle, self.wall.bricks)

    # Apply action
    if(action == 0 ) : #left
        if(self.paddle.rect.x - self.paddle.speed < 0):
            self.paddle.rect.x = 0
        else:
            self.paddle.rect.x -= self.paddle.speed
        self.state = self.paddle.rect.x
    elif action == 1 : #stand still
        pass
    else : # right
        if(self.paddle.rect.x + self.paddle.paddleWidth + self.paddle.speed > self.width):
            self.paddle.rect.x = self.width - self.paddle.paddleWidth
        else:
            self.paddle.rect.x += self.paddle.speed
        self.state = self.paddle.rect.x     

    #reward for beiing at same position of the ball
    if(self.ball.rect.x > self.paddle.rect.x and (self.ball.rect.x + self.ball.rad*2) < (self.paddle.rect.x + self.paddle.paddleWidth)):
        reward = 1
    else : 
        reward = -1

    if(self.visualize == True):
        #limit the clock
        self.clock.tick(self.fps)
        self.render()

    # Check if game is done
    if self._episode_ended == True: 
        return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
        return ts.transition(np.array([self._state], dtype=np.int32), reward)

  def render(self):  
    # done only once
    def init_pygame(self):
        # Colors
        self.color1 = (36, 32, 56)  # unbreakable bricks
        self.color2 = (144, 103, 198)  # breakable bricks
        self.color3 = (123, 123, 213)  # padd and ball
        self.color4 = (202, 196, 206)  # background color
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        # create bricks walls
        self.wall.createBricks()

    def render(self):
        # render wall
        self.wall.printWall(self.screen)
        # render paddle
        self.paddle.printPaddle(self.screen)
        # render ball
        self.ball.printBall(self.screen)

    # init pygame
    if self.screen is None:
        init_pygame(self)

    # print background
    self.screen.fill(self.color4)
    render(self)
    pygame.display.update()  # update window

#-- Create Python Environment --#
env = BreakoutEnv2(visualize=True) #add fps parameters if needed
# print('Action Spec:', env.action_spec())

# check if python environement is correct
# print("validate python environment")
# utils.validate_py_environment(env, episodes=5)


#-- Convert in Tensor Environment --#
tf_env = tf_py_environment.TFPyEnvironment(env)
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())

#Variables
num_episode = 10
reward = 0

#-- Test  with Python Environment pygame --#
# for i in range(num_episode):
#     time_step = env.reset()
#     #while episode not done
#     while not time_step.is_last():
#         #tensorFlow environment
#         action = np.random.randint(0,3)
#         time_step = env.step(action)
#         reward += time_step.reward
#     print("Episode "+str(i+1)+"/"+str(num_episode)+ " done")
# print("reward:",reward)

#-- Test  with Tensor Environment pygame --#
for i in range(num_episode):
    time_step = tf_env.reset()
    #while episode not done
    while not time_step.is_last():
        #tensorFlow environment
        action = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        time_step = tf_env.step(action)
        reward += time_step.reward
    print("Episode "+str(i+1)+"/"+str(num_episode)+ " done : " + str(env.score) + "/" + str(env.wall.breakableBricks)+" bricks")
print("reward:",reward.numpy())


#-- Training --#
# print("-- Training --")
# fc_layer_params = (100,)

# learning_rate = 1e-3 # @param {type:"number"}

# train_env = tf_py_environment.TFPyEnvironment(env)

# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     fc_layer_params=fc_layer_params)


# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# train_step_counter = tf.Variable(0)

# tf_agent = reinforce_agent.ReinforceAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     actor_network=actor_net,
#     optimizer=optimizer,
#     normalize_returns=True,
#     train_step_counter=train_step_counter)
# tf_agent.initialize()


# permettre la fermeture de la fenetre avec la croix
#     # get quit event
#     get_event = pygame.event.get()
#     for event in get_event:
#         if event.type == pygame.QUIT:
#             env.gameOver = 1



# states = tf_env.time_step_spec()
# actions = tf_env.action_spec()
# print("observations: "+str(states))
# print("actions: "+str(actions))


# def build_model(states, actions):
#     model = Sequential()   
#     model.add(Dense(24, activation='relu', input_shape=states))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model


# model = build_model(states, actions)
# model.summary()
# print(model.output_shape)


# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=10000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                   nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
#     return dqn

# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)


# scores = dqn.test(env, nb_episodes=10, visualize=False)
# print(np.mean(scores.history['episodes reward']))
