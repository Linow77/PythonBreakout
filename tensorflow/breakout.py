import pygame
from breakout_gym import BreakoutEnv

# limit the clock
clock = pygame.time.Clock()
fps = 60

# keep window open
running = True


# ─── FUNCTIONS FOR USER INPUT ──────── #


def event_to_action(eventlist):
    global running
    for event in eventlist:
        if event.type == pygame.QUIT:
            running = False


clock = pygame.time.Clock()
env = BreakoutEnv()
while running:
    # limit clock
    clock.tick(fps)

    # get event of the user
    get_event = pygame.event.get()
    event_to_action(get_event)
    # action = # [...], get action
    # env.step(action) # calculate game step
    env.render()  # make pygame render calls to window
    pygame.display.update()  # update window
pygame.quit()
