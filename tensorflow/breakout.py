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
        return -1

    # move right
    elif key[pygame.K_RIGHT] and paddle.rect.right < width:
        return 1
    else:
        return 0


clock = pygame.time.Clock()
env = BreakoutEnv()
while running:
    # limit clock
    clock.tick(fps)

    # get quit event
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            running = False

    # get pressed key
    key = pygame.key.get_pressed()

    # get the action
    action = key_to_action(key, env.paddle, env.width)

    env.step(action)

    # move the ball
    env.ball.move(env.paddle.rect, env.wall.bricks)

    env.render()  # make pygame render calls to window
    pygame.display.update()  # update window
pygame.quit()
