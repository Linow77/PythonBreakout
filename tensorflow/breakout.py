import pygame
from breakout_gym import BreakoutEnv

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
env = BreakoutEnv()
print(env.gameOver)
while env.gameOver != 1:
    print(env.gameOver)
    #print(env.gameRunning)
    # limit clock
    clock.tick(fps)

    # get quit event
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            env.gameOver = 1

    # get pressed key
    #key = pygame.key.get_pressed()
    # get the action
    #action = key_to_action(key, env.paddle, env.width)

    #action = choose_action(env.paddle, env.ball)
    action = env.action_space.sample()
    reward, done, info = env.step(action)

    # move the ball
    env.ball.move(env.paddle.rect, env.wall.bricks)

    env.render()  # make pygame render calls to window
    pygame.display.update()  # update window
pygame.quit()
