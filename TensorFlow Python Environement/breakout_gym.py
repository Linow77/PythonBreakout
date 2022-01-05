import gym
from gym.spaces import Discrete
from gym.spaces.box import Box
import pygame
import random as r
import math
import numpy as np



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

# -- Variables Declaration -- #
# screen size
width = 640
height = 600
screen = pygame.display.set_mode((width, height))

# title
pygame.display.set_caption('Breakout')


# Colors
color1 = (36, 32, 56)  # unbreakable bricks
color2 = (144, 103, 198)  # breakbale bricks
color3 = (123, 123, 213)  # padd and ball
color4 = (202, 196, 206)  # background color

# BRICK VARIABLES
# number of brick
rows = 6
cols = 5
# border of brick
border = 3
brickWidth = width // cols


# BALL VARIABLES
# remaining balls
balls = 2
ballSpeed = 2.5  # on Axes at the beginning
velocity = math.sqrt(2*(ballSpeed**2))
ballPositionY = height - 90

# paddle VARIABLES
paddlePositionY = height - 60
paddleWidth = brickWidth

# margin Error
margin = 5
# Same random seed for every launched
r.seed(1000)

# Start Game Text
pygame.font.init()
myfont = pygame.font.SysFont('Calibri', 25)
startTextSize = myfont.size("Press any key to start")
startText = myfont.render("Press any key to start", True, color2)
gameOverTextSize = myfont.size("Game Over")
gameOverText = myfont.render("Game Over", True, color1)

# game launched ?
gameRunning = 1
gameOver = 0

class wall():
    def __init__(self):
        self.width = width
        self.height = (height - 100) // 2
        self.brickWidth = brickWidth
        self.brickHeight = self.height // rows
        self.bricks = []

    def createBricks(self):
        rowNumber = 0
        for row in range(rows):

            colNumber = 0
            for col in range(cols):

                brick = pygame.Rect(
                    colNumber*self.brickWidth, 100+rowNumber*self.brickHeight, self.brickWidth, self.brickHeight)
                # store brick
                # 25% unbreakable bricks
                self.bricks.append(
                    (colNumber, rowNumber, r.randint(0, 3), brick))

                colNumber += 1
            rowNumber += 1

    def printWall(self):
        for brick in self.bricks:
            # check for unbreakable bricks
            if(brick[2] == 0):
                color = color1
            else:
                color = color2
            # print border bricks
            pygame.draw.rect(screen, (color4),
                             brick[3])
            # print bricks
            pygame.draw.rect(screen, (color),
                             ((brick[3].x + border), (brick[3].y + border), self.brickWidth - 2*border, self.brickHeight - 2*border))

class paddle():
    def __init__(self):
        self.paddleWidth = paddleWidth
        self.paddleHeight = 10
        self.x = (width - self.paddleWidth)/2  # init position
        self.y = paddlePositionY
        self.rect = pygame.Rect(
            self.x, self.y, self.paddleWidth, self.paddleHeight)
        self.speed = 8

    def printPaddle(self):
        pygame.draw.rect(screen, color3, self.rect)

    def move(self):
        # get key pressed
        key = pygame.key.get_pressed()

        # move left
        if key[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed

        # move right
        if key[pygame.K_RIGHT] and self.rect.right < width:
            self.rect.x += self.speed

class ball():
    def __init__(self, ballSpeed):
        self.rad = 10
        self.x = width // 2 - self.rad  # init position of the rectangle
        self.y = ballPositionY          # init position of the rectangle
        self.speedx = ballSpeed
        self.speedy = -ballSpeed
        self.rect = pygame.Rect(
            self.x, self.y, 2*self.rad, 2*self.rad)
        self.directionH = 1
        self.angle = -1
        self.newAngle = -1
        self.ballSpeed = ballSpeed
        self.lastCollision = ""

    def printBall(self):
        pygame.draw.circle(screen, color3, (self.rect.x +
                                            self.rad, self.rect.y + self.rad), self.rad)

    def move(self, paddleRect, wallBricks):
        global balls
        global gameRunning

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
        if self.rect.right > width or self.rect.left < 0:
            self.speedx *= -1
            #save collision for reward
            self.lastCollision = "sideBorder"

        # top border
        elif self.rect.top < 0:
            self.speedy *= -1
            #save collision for reward
            self.lastCollision = "topBorder"

        # bottom border
        elif self.rect.bottom > height:
            #save collision for reward
            self.lastCollision = "endBorder"

            # reset ball position
            self.x = width // 2 - self.rad
            self.y = ballPositionY
            self.rect.x = self.x
            self.rect.y = self.y
            self.speedx = self.ballSpeed
            self.speedy = -self.ballSpeed

            # reset paddle position
            paddleRect.x = (width - paddleWidth)/2  # init position

            # get a new ball if availabe
            if balls > 0:
                print("ball lost")
                gameRunning = 0
                # delete a ball
                balls -= 1

            else:  # no ball left
                global gameOver
                gameOver = 1

        #-- End Check for screen borders --#

        #-- Check for collisions between ball and paddle --#
        elif self.rect.y > paddlePositionY - 2*self.rad:
            if self.rect.colliderect(paddleRect):

                # TOP COLLISION
                # check if ball is on top of the trail (between the 5px margin)
                if(abs(self.rect.bottom - paddlePositionY) < margin and self.speedy > 0):

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
                    if self.rect.right >= paddleRect.x and self.rect.left < paddleRect.x + (0.2 * paddleWidth):

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
                            (math.cos(math.radians(self.newAngle)) * velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * velocity)

                    # MIDDLE LEFT PART
                    elif self.rect.right >= paddleRect.x + (0.2 * paddleWidth) and self.rect.left < paddleRect.x + (0.4 * paddleWidth):

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
                            (math.cos(math.radians(self.newAngle)) * velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * velocity)

                    elif self.rect.right >= paddleRect.x + (0.4 * paddleWidth) and self.rect.left < paddleRect.x + (0.6 * paddleWidth):
                        # angle is not changed
                        self.speedy *= -1

                    # MIDLE RIGHT PART
                    elif self.rect.right >= paddleRect.x + (0.6 * paddleWidth) and self.rect.left < paddleRect.x + (0.8 * paddleWidth):

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
                            (math.cos(math.radians(self.newAngle)) * velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * velocity)

                    # RIGHT PART
                    elif self.rect.right >= paddleRect.x + (0.8 * paddleWidth) and self.rect.left < paddleRect.x + paddleWidth:
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
                            (math.cos(math.radians(self.newAngle)) * velocity)
                        self.speedy = - \
                            (math.sin(math.radians(self.newAngle)) * velocity)

                # SIDES COLLISIONS
                else:  # collision with side
                    #save collision for reward
                    self.lastCollision = "side"
                    if (abs(self.rect.left - paddleRect.right) < margin):
                        # check direction of the ball
                        if self.speedx < 0:
                            self.speedx *= -1
                        else:  # if same direction as the paddle don't reverse direction but increase speed
                            self.speedx += 3

                    elif (abs(self.rect.right - paddleRect.left) < margin):
                        # check direction of the ball
                        if self.speedx > 0:
                            self.speedx *= -1
                        else:  # if same direction as the paddle don't reverse direction but increase speed
                            self.speedx -= 3

        #-- End Check for collisions between ball and paddle --#

        #-- Check for collisions between ball and Bricks --#
        elif self.rect.y > 0 and self.rect.y < (height // 2) + 100:
            for brick in wallBricks:
                if(self.rect.colliderect(brick[3])):
                    # check if collision is on top or at bottom of the brick
                    if ((abs(self.rect.top - brick[3].bottom) < margin and self.speedy < 0) or
                            (abs(self.rect.bottom - brick[3].top) < margin and self.speedy > 0)):
                        # top or bottom
                        self.speedy *= -1

                    elif ((abs(self.rect.left - brick[3].right) < margin and self.speedx < 0) or
                          (abs(self.rect.right - brick[3].left) < margin and self.speedx > 0)):
                        # right or left
                        self.speedx *= -1

                    # delete the brick if breakable
                    if(brick[2] != 0):
                        wallBricks.remove(brick)

        #-- End Check for collisions between ball and Bricks --#

    def gravity(self):
        if (self.speedy >= 0 and self.speedy < 1):  # stuck horizontally
            self.speedy = 1
        elif (self.speedy < 0 and self.speedy > -1):  # stuck horizontally
            self.speedy = -1


class BreakoutEnv(gym.Env):
    def __init__(self, env_config={}):
        global gameOver
        #init object
        self.wall = wall()
        self.paddle = paddle()
        self.ball = ball(ballSpeed)
        #Set screen variable for pygame
        self.screen = None
        self.width = width
        #Actions, go left, right or stand still
        self.action_space = Discrete(3)
        #Observations about the position of the paddle and the ball on axis X
        self.observation_space = Box(low=np.array([0]), high=np.array([self.width]))
        #Set start position of the paddle and the ball
        self.state = self.paddle.rect.x
        self.ball_position = self.ball.x
        self.gameOver = gameOver

        self.reset()
        # exit()

    def step(self, action):
        # Apply action
        if(action == 0) : #left
            self.paddle.rect.x -= self.paddle.speed
            self.state = self.paddle.rect.x
        elif action == 1 : #stand still
            pass
        else : # right
            self.paddle.rect.x += self.paddle.speed
            self.state = self.paddle.rect.x

        #calculate reward
        #reward for touching ball
        '''if(self.ball.lastCollision != ""):
            print("nouvelle colision:"+self.ball.lastCollision)

        if(self.ball.lastCollision == "top"):
            reward = 1
        elif (self.ball.lastCollision == "endBorder"):
            reward = -1
        else : #touch walls
            reward = 0'''
        
        #reward for beiing at same position of the ball
        if(self.ball.rect.x > self.paddle.rect.x and (self.ball.rect.x + self.ball.rad*2) < (self.paddle.rect.x + self.paddle.paddleWidth)):
            reward = 1

        else : 
            reward = -1
            
        
        # Check if game is done
        
        if gameOver == 1: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def reset(self):
        # reset the environment to initial state
        self.state = self.paddle.rect.x
        self.ball_position = self.ball.rect.x
        global balls
        balls =2
        global gameOver
        gameOver = 0
        print("reset")

        return self.state

    def render(self):
        
        # done only once
        def init_pygame(self):
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            print("render")
            # create bricks walls
            self.wall.createBricks()

        def render(self):
            env.ball.move(env.paddle.rect, env.wall.bricks)
            # render wall
            self.wall.printWall()
            # render paddle
            self.paddle.printPaddle()
            # render ball
            self.ball.printBall()

        # init pygame
        if self.screen is None:
            init_pygame(self)

        # print background
        self.screen.fill(color4)
        render(self)
        pygame.display.update()  # update window

class BreakoutEnv2(py_environment.PyEnvironment):

  def __init__(self):
    global gameOver
    #init object
    self.wall = wall()
    self.paddle = paddle()
    self.ball = ball(ballSpeed)
    #Set screen variable for pygame
    self.screen = None
    self.width = width
    self.ball_position = self.ball.x
    self.gameOver = gameOver

    #three actions : move left, move right or stand still
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
    #observation on ball's position and paddle's position
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,2), dtype=np.int32, minimum=0, maximum=self.width, name='observation')

    self._state = [self.paddle.rect.x,self.ball.rect.x]
    self._episode_ended = False

    

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self): #a faire
    self._state = [self.paddle.rect.x,self.ball.rect.x]
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

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
        if(self.paddle.rect.x + self.paddle.paddleWidth + self.paddle.speed > width):
            self.paddle.rect.x = width - self.paddle.paddleWidth
        else:
            self.paddle.rect.x += self.paddle.speed
        self.state = self.paddle.rect.x     
    
    # Check if game is done
    if gameOver == 1: 
        self._episode_ended = True
    else:
        self._episode_ended = False

    #reward for beiing at same position of the ball
    if(self.ball.rect.x > self.paddle.rect.x and (self.ball.rect.x + self.ball.rad*2) < (self.paddle.rect.x + self.paddle.paddleWidth)):
        reward = 1
    else : 
        reward = -1
    return ts.transition(np.array([self._state], dtype=np.int32), reward)

  def render(self):  
    # done only once
    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        print("render")
        # create bricks walls
        self.wall.createBricks()

    def render(self):
        # render wall
        self.wall.printWall()
        # render paddle
        self.paddle.printPaddle()
        # render ball
        self.ball.printBall()

    # init pygame
    if self.screen is None:
        init_pygame(self)

    # print background
    self.screen.fill(color4)
    render(self)
    pygame.display.update()  # update window

'''
env = BreakoutEnv2()

#print(env.observation_space.sample())
print("validate environement")
utils.validate_py_environment(env, episodes=5)



states = env.observation_space.shape
actions = env.action_space.n
print("observations: "+str(states))
print("actions: "+str(actions))

episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    clock = pygame.time.Clock() 
    
    while not done:

        clock.tick(60)
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
        env.render()
        
    print('Episode:{} Score:{}'.format(episode, score))

'''
'''
def build_model(states, actions):
    model = Sequential()   
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
model.summary()
print(model.output_shape)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)


scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episodes reward']))
'''