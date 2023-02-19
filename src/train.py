from agent import Agent
import gym
import utils
import numpy as np
from logging import config
import logging
from IPython.display import clear_output
from time import sleep


config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")


def train():
    # load gym environment
    env = gym.make(config["env_name"])

    # initialize agent
    agent = Agent(learning_rate=config["learning_rate"], discount_factor=config["discount_factor"],
                  epsilon=config["eps"],
                  observation_space=env.observation_space.n, nb_actions=env.action_space.n, env=env)

    if config["save_training"]:
        # create lists for plotting and visualization
        episodes_list = []
        timesteps_list = []
        penalties_list = []
        scores_list = []

    logger.info("Training starts")
    for episode in range(config["training_episodes"]):
        timesteps, penalties, reward, score = 0, 0, 0, 0
        done = False
        state = env.reset()[0]
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = (env.step(action)[:4])
            agent.learn(state, action, reward, next_state)

            if reward == -10:
                penalties += 1

            if config["render_training"]:
                clear_output(wait=True)
                env.render()
                logger.info("Timesteps: {}, Action: {}, Penalties: {}, Score: {}".format(timesteps, action, penalties,
                                                                                         score))
                sleep(0.5)

            state = next_state
            timesteps += 1
            score += reward

        logger.info("Episode: {}, Timesteps: {}, Penalties: {}, Score: {}".format(episode, timesteps,
                                                                                  penalties, score))
        if config["save_training"]:
            episodes_list.append(episode)
            timesteps_list.append(timesteps)
            penalties_list.append(penalties)
            scores_list.append(score)

    logger.info("Training finished")

    if config["save_training"]:
        utils.save_training_data(episodes_list, timesteps_list, penalties_list, scores_list, config["env_name"])
        logger.info("Training data saved")
        np.save("../models/q_table.npy", agent.q_table)
        logger.info("Model saved")


if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("train")
    train()
