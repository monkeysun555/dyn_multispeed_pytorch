import os 
import random
import numpy as np
import argparse
import torch
import env as Env
from config import Config
from reply_buffer import Reply_Buffer
from agent import Agent
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--massive', dest='massive', help='massive testing',
                        const=True, default=False, type=str)
    parser.add_argument('-e', '--episode', dest='episode', help='episode of checkpoint',
                        default=None, type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', help='version of model',
                        default=None, type=str, required=True)
    # parser.add_argument('-t', '--train', dest='train', help='train policy or not',
    #                     default=True, type=bool)
    args = parser.parse_args()
    return args
args = parse_args() 

def main():
    massive = args.massive
    model_e = args.episode
    model_v = args.version

    env = Env.Live_Streaming(testing=True)
    _, action_dims = env.get_action_info()
    # reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_dims)
    model_path = './logs_' + model_v + '/model-' + model_e + '.pth' 
    agent.restore(model_path)

    if massive:
        while True:
            # Start testing
            env_end = env.reset()
            if env_end:
                break
            env.act(0, 3)   # Default
            state = env.get_state()
            # state = np.stack([[obs for _ in range(4)]], axis=0)
            total_reward = 0.0
            while not env.streaming_finish():
                if model_v == 0:
                    action_1, action_2 = agent.take_action(np.array([state]))
                    # print(action_1, action_2)
                    reward = env.act(action_1, action_2)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward

                elif model_v == 1:
                    action = agent.take_action(np.array([state]))
                    action_1 = int(action/action_dims[1])
                    action_2 = action%action_dims[1]
                    reward = env.act(action_1, action_2)
                    # print(reward)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward




    else:
        # Single testing
        env_end = env.reset()
        env.act(0, 3)   # Default
        state = env.get_state()
        # state = np.stack([[obs for _ in range(4)]], axis=0)
        total_reward = 0.0

if __name__ == '__main__':
    main()