import pygame
import random as r
import math

pygame.init()


# screen size
width = 640
height = 500
screen = pygame.display.set_mode((width, height))

# title
pygame.display.set_caption('Breakout')

# limit the clock
clock = pygame.time.Clock()
fps = 60

# Colors
color1 = (36, 32, 56)  # unbreakable bricks
color2 = (144, 103, 198)  # breakbale bricks
color3 = (123, 123, 213)
color4 = (202, 196, 206)

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
ballSpeed = 3  # on Axes at the beginning
velocity = math.sqrt(2*(ballSpeed**2))
ballPositionY = height - 90

# TRAY VARIABLES
trayPositionY = height - 60
trayWidth = brickWidth

# margin Error
margin = 5
# Same random seed for every launched
r.seed(1000)

# keep window open
running = True

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

                brick = pygame.Rect(
                    colNumber*self.brickWidth, rowNumber*self.brickHeight, self.brickWidth, self.brickHeight)
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


class tray():
    def __init__(self):
        self.trayWidth = trayWidth
        self.trayHeight = 10
        self.x = (width - self.trayWidth)/2  # init position
        self.y = trayPositionY
        self.rect = pygame.Rect(
            self.x, self.y, self.trayWidth, self.trayHeight)
        self.speed = 8

    def printTray(self):
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

    def move(self, ballSpeed, trayRect, wallBricks):
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
        elif self.rect.top < 0:
            self.speedy *= -1

        # bottom border
        elif self.rect.bottom > height:
            # reset ball position

            self.rect.x = width // 2 - self.rad
            self.rect.y = ballPositionY
            self.speedx = ballSpeed
            self.speedy = -ballSpeed

            # reset tray position
            trayRect.x = (width - trayWidth)/2  # init position

            # get a new ball if availabe
            if balls > 0:
                #print("ball lost")
                gameRunning = 0
                # delete a ball
                balls -= 1

            else:  # no ball left
                global gameover
                gameover = 1

        #-- End Check for screen borders --#

        #-- Check for collisions between ball and Tray --#
        elif self.rect.colliderect(trayRect):

            # check if ball is on top of the trail (between the 5px margin)
            # if (abs(self.rect.bottom - trayPositionY) < margin and self.speedy > 0):
            if self.rect.bottom >= trayPositionY and self.rect.bottom < trayPositionY + 5 and self.speedy > 0:
                #print("AVANT vitesse x :"+str(self.speedx))
                #print("AVANT vitesse y :"+str(self.speedy))

                # calculate angle between the ball's path and the trail
                if self.speedx == 0:
                    self.angle = 90
                else:
                    self.angle = abs(math.degrees(
                        math.atan(self.speedy/abs(self.speedx))))
                # #print("angle:"+str(self.angle))
                if(self.speedx > 0):
                    self.directionH = 1  # from the left
                else:
                    self.directionH = -1  # from the right

                #print("direction H:"+str(self.directionH))
                self.speedy *= -1

                # change x if not on the middle of the tray

                # LEFT PART
                if self.rect.right >= trayRect.x and self.rect.left < trayRect.x + (0.2 * trayWidth):

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

                    self.speedx = -ballSpeed / \
                        (math.tan(math.radians(self.newAngle)))

                # MIDDLE LEFT PART
                elif self.rect.right >= trayRect.x + (0.2 * trayWidth) and self.rect.left < trayRect.x + (0.4 * trayWidth):

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

                    self.speedx = -ballSpeed / \
                        (math.tan(math.radians(self.newAngle)))

                    # MIDDLE RIGHT PART

                # MIDLE RIGHT PART
                elif self.rect.right >= trayRect.x + (0.6 * trayWidth) and self.rect.left < trayRect.x + (0.8 * trayWidth):

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

                    self.speedx = ballSpeed / \
                        (math.tan(math.radians(self.newAngle)))

                # RIGHT PART
                elif self.rect.right >= trayRect.x + (0.8 * trayWidth) and self.rect.left < trayRect.x + trayWidth:

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

                    self.speedx = ballSpeed / \
                        (math.tan(math.radians(self.newAngle)))

                #print("vitesse x :"+str(self.speedx))
                #print("vitesse y :"+str(self.speedy))
            # SIDE COLLISIONS
            else:  # collision with side
                # #print("side")
                # if (abs(self.rect.left - trayRect.right) < margin):
                if self.rect.left <= trayRect.right and self.rect.left > trayRect.right - 5:
                    # check direction of the ball
                    if self.speedx < 0:
                        self.speedx *= -1
                    else:  # if same direction as the tray don't reverse direction but increase speed
                        self.speedx += 3

                # elif (abs(self.rect.right - trayRect.left) < margin):
                elif self.rect.right >= trayRect.left and self.rect.right < trayRect.left + 5:
                    # check direction of the ball
                    if self.speedx > 0:
                        self.speedx *= -1
                    else:  # if same direction as the tray don't reverse direction but increase speed
                        self.speedx -= 3
        #print("vitesse : x="+str(self.speedx) + " y="+str(self.speedy))
        #-- End Check for collisions between ball and Tray --#

        #-- Check for collisions between ball and Bricks --#

        else:
            for brick in wallBricks:
                if(self.rect.colliderect(brick[3])):
                    # check if collision is on top or at bottom of the brick
                    if (abs(self.rect.top - brick[3].bottom) < margin and self.speedy < 0):
                        # if (self.rect.top <= brick[3].bottom and self.rect.top > brick[3].bottom - 5):
                        # top
                        self.speedy *= -1
                    elif (abs(self.rect.bottom - brick[3].top) < margin and self.speedy > 0):
                        # elif (self.rect.bottom >= brick[3].top and self.rect.bottom < brick[3].top + 5):
                        # bottom
                        self.speedy *= -1
                    elif (abs(self.rect.left - brick[3].right) < margin and self.speedx < 0):
                        # right
                        self.speedx *= -1
                    elif (abs(self.rect.right - brick[3].left) < margin and self.speedx > 0):
                        # left
                        self.speedx *= -1

                    # delete the brick if breakable
                    if(brick[2] != 0):
                        wallBricks.remove(brick)

        #-- End Check for collisions between ball and Bricks --#

    def gravity(self):
        if (self.speedy > 0 and self.speedy < 1):  # stuck horizontally
            self.speedy = 1


bricksWall = wall()
bricksWall.createBricks()
playerTray = tray()
# #print(bricksWall.bricks)
playerBall = ball(ballSpeed)

while running:
    # limit clock
    clock.tick(fps)
    # print background
    screen.fill(color4)

    # print Wall
    bricksWall.printWall()

    # print Tray
    playerTray.printTray()

    # print ball
    playerBall.printball()

    # print available ball
    for ball in range(balls):
        pygame.draw.circle(screen, color3,
                           (width - ball*30 - 20, height - 20), 10)

    if gameRunning == 0:
        # print message to start the game
        screen.blit(
            startText, ((width - startTextSize[0])//2, (height + 60) // 2))
    elif gameover:
        screen.blit(
            gameOverText, ((width - gameOverTextSize[0])//2, (height + 60) // 2))
    else:

        # move ball and tray
        playerTray.move()
        playerBall.move(ballSpeed, playerTray.rect, bricksWall.bricks)
        # add gravity to prevent ball stuck
        playerBall.gravity()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            gameRunning = 1

    pygame.display.update()
