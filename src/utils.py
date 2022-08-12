import os
import yaml
import pandas as pd


def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config


def load_logging_config(filename):
    with open(filename, "rt") as f:
        logging_config = yaml.safe_load(f.read())
        f.close()
    return logging_config


def save_training_data(episodes_list, timesteps_list, penalties_list, scores_list, env):
    training_data_dict = {"episode": episodes_list, "timesteps": timesteps_list, "penalties": penalties_list,
                          "score": scores_list}
    df = pd.DataFrame(training_data_dict)
    df.to_csv("../data/training/{}_q_learning_training_data.csv".format(env))
