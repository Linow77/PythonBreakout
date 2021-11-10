import gym
from gym.spaces import Discrete
from gym.spaces.box import Box
import pygame
import random as r
import math


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
gameRunning = 0
gameover = 0


class Environment:
    def __init__(self, game):
        self.game = game


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


class BreakoutEnv(gym.Env):
    def __init__(self, env_config={}):
        # self.parse_env_config(env_config)
        self.screen = None
        self.action_space = Discrete(2)
        self.observation_space = Box(0, 255, [600, 640, 3])

        self.env = Environment(self)
        self.wall = wall()
        # self.rocket = Rocket(self, self.env)
        self.spectator = None

        self.reset()
        # exit()

    def reset(self):
        # reset the environment to initial state
        print("coucou")

        return "observation"

    def step(self, action):
        # perform one step in the game logic

        return "observation, reward, done, info"

    def render(self):

        def init_pygame(self):
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            print("render")
            # create bricks walls
            self.wall.createBricks()

        def render_wall(self):
            self.wall.printWall()

        if self.screen is None:
            init_pygame(self)

        # print background
        self.screen.fill(color4)
        render_wall(self)
