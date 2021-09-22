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
trayColor = (123, 123, 213)

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
ballSpeed = 3

# Same seed for every launched
r.seed(1000)

# keep window open
running = True

# Start Game Text
pygame.font.init()
myfont = pygame.font.SysFont('Calibri', 25)
startTextSize = myfont.size("Press any key to start")
startText = myfont.render("Press any key to start", True, trayColor)
gameOverTextSize = myfont.size("Game Over")
gameOverText = myfont.render("Game Over", True, trayColor)

# game launched ?
gameRunning = 0
gameover = 0

# wall containing all the bricks


class wall():
    def __init__(self):
        self.width = width
        self.height = height // 2
        self.brickWidth = brickWidth
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
        self.trayWidth = brickWidth
        self.trayHeight = 20
        self.x = (width - self.trayWidth)/2  # init position
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
        self.rad = 10
        self.x = width // 2 - self.rad  # init position of the rectangle
        self.y = (height - 70)          # init position of the rectangle
        self.speedx = ballSpeed
        self.speedy = -ballSpeed
        self.rect = pygame.Rect(
            self.x, self.y, 2*self.rad, 2*self.rad)

    def print(self):
        #pygame.draw.rect(screen, (123, 123, 213), self.rect)
        pygame.draw.circle(screen, trayColor, (self.rect.x +
                           self.rad, self.rect.y + self.rad), self.rad)

    def move(self, ballSpeed, trayRect):
        global balls
        global gameRunning
        # move the rectangle containing the ball
        self.rect.x += self.speedx
        self.rect.y += self.speedy

        #-- Check for screen borders --#
        # left and right borders
        if self.rect.right > width or self.rect.left < 0:
            self.speedx *= -1

        # top border
        if self.rect.top < 0:
            self.speedy *= -1

        # bottom border
        if self.rect.bottom > height:
            # reset ball position

            self.rect.x = width // 2 - self.rad
            self.rect.y = (height - 70)
            self.speedx = ballSpeed
            self.speedy = -ballSpeed

            # reset tray position
            trayRect.x = (width - brickWidth)/2  # init position

            # get a new ball if availabe
            if balls > 0:
                print("ball lost")
                gameRunning = 0
                # delete a ball
                balls -= 1

            else:  # no ball left
                global gameover
                gameover = 1

        #-- Check for screen borders --#

        #-- Check for collisions between ball and Tray --#
        # if ball is at trail height or under
        if self.rect.bottom >= height - 40:
            # check if ball is on top of the trail (between the 5px margin)
            if self.rect.bottom >= height - 40 and self.rect.bottom < height - 35:
                # check if the ball can touch the trail
                if self.rect.right >= trayRect.left and self.rect.left < trayRect.right:
                    # resend ball
                    self.speedy *= -1
                    # change direction on with a specific angle
                    print("create redirection on x")

            else:  # ball is on the side of the trail (the ball is lost)
                # check for collision between ball and side of the trail (5px margin)

                # left collision (ball) on right side of the trail
                if self.rect.left <= trayRect.right and self.rect.left > trayRect.right - 5:
                    self.speedx *= -1

                # right collision (ball) on left side of the trail
                if self.rect.right >= trayRect.left and self.rect.right < trayRect.left + 5:
                    self.speedx *= -1

        #-- Check for collisions between ball and Tray --#


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

    if gameRunning == 0:
        # print message to start the game
        screen.blit(
            startText, ((width - startTextSize[0])//2, (height + 60) // 2))
    elif gameover:
        screen.blit(
            gameOverText, ((width - gameOverTextSize[0])//2, (height + 60) // 2))
    else:
        # move
        playerTray.move()
        playerBall.move(ballSpeed, playerTray.rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            gameRunning = 1

    pygame.display.update()
