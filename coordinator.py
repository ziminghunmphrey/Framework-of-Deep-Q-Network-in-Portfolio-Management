import sys
import os
current_path = os.path.abspath(__file__)
sys.path.append(OLPS_path)
from environments import PortfolioEnv
import pandas as pd
from dqn_agent import Dqn_agent


df_train = pd.read_csv('data_file_name')


class Coordinator:

    def __init__(self, config, name):

        name = name
        asset_num = config['env']['asset_num']
        feature_num = config['env']['feature_num']
        window_length = int(config['env']['window_length'])
        input = config['env']['input']
        norm = config['env']['norm']
        trading_cost = config['env']['trading_cost']
        trade_period = config['env']['trading_period']
        expan_coe = config['env']['expan_coe']

        network_config = config['net']

        self.total_training_step = config['train']['steps']
        self.replay_period = config['train']['replay_period']
        self.reward_scale = config['train']['reward_scale']
        learning_rate = config['train']['learning_rate']
        epsilon = config['train']['epsilon']
        epsilon_decay_period = config['train']['epsilon_decay_period']
        start_date = config['train']['start_date']
        end_date = config['train']['end_date']
        date_range_s = config['train']['date_range_s']
        date_range_e = config['train']['date_range_e']
        steps_per_episode = config['train']['steps_per_episode']
        division = config['train']['division']
        gamma = config['train']['discount']
        batch_size = config['train']['batch_size']
        memory_size = config['train']['memory_size']
        upd_tar_prd = config['train']['upd_tar_prd']
        save = config['train']['save']
        save_period = config['train']['save_period']

        self.config = config

        self.agent = Dqn_agent(asset_num,
                               division,
                               feature_num,
                               gamma,
                               epsilon=epsilon,
                               learning_rate=learning_rate,
                               network_topology=network_config,
                               update_tar_period=upd_tar_prd,
                               epsilon_decay_period=epsilon_decay_period,
                               memory_size=memory_size,
                               batch_size=batch_size,
                               history_length=window_length,
                               save=save,
                               save_period=save_period,
                               name=name)

        self.env_train = PortfolioEnv(df_train,
                                      start_date,
                                      end_date,
                                      date_range_s,
                                      date_range_e,
                                      steps=steps_per_episode,
                                      trading_cost=trading_cost,
                                      window_length=window_length,
                                      trade_period=trade_period,
                                      input=input,
                                      norm=norm,
                                      expan_coe=expan_coe)


    def train(self):

        training_step = 0
        self.rewards = []

        while training_step < self.total_training_step:
            observation = self.env_train.reset()
            while True:
                action_idx, action, fc_input = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env_train._step(action)
                self.rewards.append(reward)
                reward *= self.reward_scale
                self.agent.store(observation, action_idx, reward, observation_)
                observation = observation_
                if self.agent.start_replay():
                    if self.agent.memory_cnt() % self.replay_period == 0:
                        self.agent.replay()
                        training_step = self.agent.get_training_step()
                if done:
                    break


    def restore(self, name):
        self.agent.restore(name)
