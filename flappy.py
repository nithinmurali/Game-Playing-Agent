
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np
import agent


def preprocess(image):
    x_t = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return np.stack((x_t, x_t, x_t, x_t), axis=2)


def play_flappy():
    flappy_agent = agent.Agent(actions=2)
    flappy_game = game.GameState()

    # play game
    while True:
        # get the action to be done from the agent
        action = flappy_agent.act()
        # apply that action to game
        next_observation, reward, terminal = flappy_game.frame_step(action)
        # feed the next state to the agent
        flappy_agent.observe(preprocess(next_observation), reward, terminal)


if __name__ == '__main__':
    play_flappy()
