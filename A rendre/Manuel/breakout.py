import pygame
import random as r
import math
pygame.init()


# -- Variables Declaration -- #
# screen size
width = 640
height = 600
screen = pygame.display.set_mode((width, height))

# title
pygame.display.set_caption('Breakout')

# limit the clock
clock = pygame.time.Clock()
fps = 60

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

#Number of breakable bricks
breakableBricks = 0

#Number of brick destroyed
score=0

# keep window open
running = True

# Start Game Text
pygame.font.init()
myfont = pygame.font.SysFont('Calibri', 25)
startTextSize = myfont.size("Press any key to start")
startText = myfont.render("Press any key to start", True, color2)
gameOverTextSize = myfont.size("Game Over")
gameOverText = myfont.render("Game Over", True, color1)
gameWinTextSize = myfont.size("You Win !")
gameWinText = myfont.render("You Win !", True, color2)

# game launched ?
gameRunning = 0
gameover = 0
#win ?
win= False

# -- Class Declaration -- #


class wall():
    def __init__(self):
        ## Wall Variables ##
        self.width = width
        self.height = (height - 100) // 2
        self.brickWidth = brickWidth
        self.brickHeight = self.height // rows
        #Table for storing bricks
        self.bricks = []
        

    def createBricks(self):
        for rowNumber in range(rows):
            for colNumber in range(cols):
                # Create Rectancle for each brick
                brick = pygame.Rect(
                    colNumber*self.brickWidth, 100+rowNumber*self.brickHeight, self.brickWidth, self.brickHeight)
                
                # 25% unbreakable bricks
                type = r.randint(0,3)
                if type != 0:
                    global breakableBricks
                    breakableBricks+=1

                # Store bricks inside table
                self.bricks.append(
                    (colNumber, rowNumber, type, brick))

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

    def printpaddle(self):
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

    def printball(self):
        pygame.draw.circle(screen, color3, (self.rect.x +
                                            self.rad, self.rect.y + self.rad), self.rad)

    def move(self, ballSpeed, paddleRect, wallBricks):
        global balls
        global gameRunning

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

        # top border
        elif self.rect.top < 0:
            self.speedy *= -1

        # bottom border
        elif self.rect.bottom > height:
            # reset ball position

            self.x = width // 2 - self.rad
            self.y = ballPositionY
            self.rect.x = self.x
            self.rect.y = self.y
            self.speedx = ballSpeed
            self.speedy = -ballSpeed

            # reset paddle position
            paddleRect.x = (width - paddleWidth)/2  # init position

            # get a new ball if availabe
            if balls > 0:
                gameRunning = 0
                # delete a ball
                balls -= 1

            else:  # no ball left
                global gameover
                gameover = 1

        #-- End Check for screen borders --#

        #-- Check for collisions between ball and paddle --#
        elif self.rect.y > paddlePositionY - 2*self.rad:
            if self.rect.colliderect(paddleRect):

                # TOP COLLISION
                # check if ball is on top of the trail (between the 5px margin)
                if(abs(self.rect.bottom - paddlePositionY) < margin and self.speedy > 0):

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
                        global score
                        score+=1
                        
                        #check if game is done
                        if score == breakableBricks :
                            global win
                            win=1

        #-- End Check for collisions between ball and Bricks --#

    def gravity(self):
        if (self.speedy >= 0 and self.speedy < 1):  # stuck horizontally
            self.speedy = 1
        elif (self.speedy < 0 and self.speedy > -1):  # stuck horizontally
            self.speedy = -1


# -- Execution -- #
bricksWall = wall()
bricksWall.createBricks()
playerpaddle = paddle()

playerBall = ball(ballSpeed)

while running:
    # limit clock
    clock.tick(fps)
    # print background
    screen.fill(color4)

    # print Wall
    bricksWall.printWall()

    # print paddle
    playerpaddle.printpaddle()

    # print ball
    playerBall.printball()

    # print available ball
    for ball in range(balls):
        pygame.draw.circle(screen, color3,
                           (width - ball*30 - 20, height - 20), 10)

    if gameRunning == 0:
        # print message to start the game
        screen.blit(
            startText, ((width - startTextSize[0])//2, (height + 160) // 2))
    elif gameover:
        screen.blit(
            gameOverText, ((width - gameOverTextSize[0])//2, (height + 160) // 2))
    elif win :
        screen.blit(
            gameWinText, ((width - gameWinTextSize[0])//2, (height + 160) // 2))
    else:

        # move ball and paddle
        playerpaddle.move()
        playerBall.move(ballSpeed, playerpaddle.rect, bricksWall.bricks)
        # add gravity to prevent ball stuck
        playerBall.gravity()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            gameRunning = 1

    pygame.display.update()
