import pygame
import random as r

pygame.init()

# screen size
width = 640
height = 480
screen = pygame.display.set_mode((width, height))

# title
pygame.display.set_caption('Breakout')

# limit the clock
clock = pygame.time.Clock()
fps = 60

# Colors
bgColor = (202, 194, 193)
BrickColor = (36, 156, 177)  # breakbale blue
uBrickColor = (14, 14, 14)  # unbreakable black
trayColor = (156, 178, 102)

# BRICK VARIABLES
# number of brick
rows = 6
cols = 5
# border of brick
border = 3

# BALL VARIABLES
# remaining balls
balls = 2
ballSpeed = 4

# Same seed for every launched
r.seed(1000)

# keep window open
running = True

# wall containing all the bricks


class wall():
    def __init__(self):
        self.width = width
        self.height = height // 2
        self.brickWidth = width // cols
        self.brickHeight = self.height // rows
        self.bricks = []

    def createBricks(self):
        rowNumber = 0

        for row in range(rows):

            colNumber = 0
            for col in range(cols):
                # 25% unbreakable bricks
                self.bricks.append((colNumber, rowNumber, r.randint(0, 3)))
                colNumber += 1
            rowNumber += 1

    def printWall(self):
        for brick in self.bricks:
            # check for unbreakable bricks
            if(brick[2] == 0):
                color = uBrickColor
            else:
                color = BrickColor
            # print border bricks
            pygame.draw.rect(screen, (bgColor),
                             (brick[0]*self.brickWidth, brick[1]*self.brickHeight, self.brickWidth, self.brickHeight))
            # print bricks
            pygame.draw.rect(screen, (color),
                             ((brick[0]*self.brickWidth + border), (brick[1]*self.brickHeight + border), self.brickWidth - 2*border, self.brickHeight - 2*border))


class tray():
    def __init__(self):
        self.trayWidth = width / cols
        self.trayHeight = 20
        self.x = (width - self.trayWidth)/2
        self.y = height - 40
        self.rect = pygame.Rect(
            self.x, self.y, self.trayWidth, self.trayHeight)
        self.speed = 8

    def printTray(self):
        pygame.draw.rect(screen, trayColor, self.rect)

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
        self.width = 10
        self.height = 10
        self.x = (width - self.width) // 2
        self.y = (height - 50)
        self.speedx = ballSpeed
        self.speedy = -ballSpeed

    def print(self):
        pygame.draw.circle(screen, trayColor, (self.x, self.y), self.width)

    def move(self, ballSpeed):
        global balls
        self.x += self.speedx
        self.y += self.speedy

        # check for borders
        # left and right borders
        if self.x > width - self.width or self.x - self.width < 0:
            self.speedx *= -1

        # top border
        if self.y - self.height < 0:
            self.speedy *= -1

        # bottom border (it trail touch ball)
        if self.y > height - 50:
            # check if trail touch

            # if no trail
            # get a new ball
            if balls > 0:
                # delete a ball
                balls -= 1
                # reset ball position
                self.x = (width - self.width) // 2
                self.y = (height - 50)
                self.speedx = ballSpeed
                self.speedy = -ballSpeed
            else:
                print("gameOver")


bricksWall = wall()
bricksWall.createBricks()
playerTray = tray()
# print(bricksWall.bricks)
playerBall = ball(ballSpeed)

while running:
    # limit clock
    clock.tick(fps)
    # print background
    screen.fill(bgColor)

    # print Wall
    bricksWall.printWall()

    # print Tray
    playerTray.printTray()

    # print ball
    playerBall.print()

    # move
    playerTray.move()
    playerBall.move(ballSpeed)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
