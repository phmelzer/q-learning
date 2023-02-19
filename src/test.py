import gym
import utils
import numpy as np
from logging import config
import logging
from IPython.display import clear_output
from time import sleep


config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")


def test():
    # load environment
    env = gym.make(config["env_name"], render_mode="human")

    # load model / q-table
    q_table = np.load("../models/q_table.npy")
    logger.info("Model loaded")

    timesteps_list = []
    penalties_list = []
    scores_list = []

    for episode in range(config["test_episodes"]):
        logger.info("Episode: {}".format(episode+1))
        timesteps, penalties, reward, score = 0, 0, 0, 0
        done = False
        state = env.reset()[0]
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = (env.step(action)[:4])

            if reward == -10:
                penalties += 1

            score += reward

            if config["render_test"]:
                clear_output(wait=True)
                env.render()
                logger.info("Timesteps: {}, Action: {}, Penalties: {}, Reward: {}".format(timesteps, action, penalties,
                                                                                          reward))
                sleep(0.5)

            timesteps += 1

        timesteps_list.append(timesteps)
        avg_timesteps = np.mean(timesteps_list[-100:])
        penalties_list.append(penalties)
        avg_penalties = np.mean(penalties_list[-100:])
        scores_list.append(score)
        avg_score = np.mean(scores_list[-100:])

        logger.info("Timesteps: {}, Avg_Timesteps: {}, Penalties: {}, Avg_Penalties: {}, Score: {},"
                    " Avg_Score: {}".format(timesteps, avg_timesteps, penalties, avg_penalties, score,
                                            avg_score))

    logger.info("Test finished")


if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("test")
    test()
