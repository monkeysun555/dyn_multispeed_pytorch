import os 
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
                        default=False, action='store_true')
    parser.add_argument('-e', '--episode', dest='episode', help='episode of checkpoint',
                        default=None, type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', help='version of model',
                        default=None, type=int, required=True)
    # parser.add_argument('-t', '--train', dest='train', help='train policy or not',
    #                     default=True, type=bool)
    args = parser.parse_args()
    return args
args = parse_args() 

def main():
    massive = args.massive
    model_e = args.episode
    model_v = args.version
    # check results log path
    if not os.path.exists(Config.massive_result_files):
         os.makedirs(Config.massive_result_files) 
    # print(massive, model_e, model_v)

    env = Env.Live_Streaming(testing=True, massive=massive)
    _, action_dims = env.get_action_info()
    # reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_dims)
    agent.set_epsilon_for_testing()
    model_path = './logs_' +str(model_v) + '/model-' + model_e + '.pth' 
    agent.restore(model_path, model_v)

    if massive:
        while True:
            # Start testing
            env_end = env.reset(testing=True)
            if env_end:
                break
            testing_start_time = env.get_server_time()
            tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
            log_path = Config.massive_result_files + trace_name 
            log_file = open(log_path, 'w')
            env.act(0, 3)   # Default
            state = env.get_state()
            total_reward = 0.0
            while not env.streaming_finish():
                if model_v == 0:
                    action_1, action_2 = agent.take_action(np.array([state]))
                    # print(action_1, action_2)
                    reward = env.act(action_1, action_2,log_file)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward
                    # print(action_1, action_2, reward)
                elif model_v == 1:
                    action = agent.take_action(np.array([state]))
                    action_1 = int(action/action_dims[1])
                    action_2 = action%action_dims[1]
                    reward = env.act(action_1, action_2,log_file)
                    # print(reward)
                    state_new = env.get_state()
                    state = state_new
                    total_reward += reward            
            print('File: ', trace_name, ' reward is: ', total_reward) 
            # Get initial latency of player and how long time is used. and tp/time trace
            testing_duration = env.get_server_time() - testing_start_time
            tp_record, time_record = get_tp_time_trace_info(tp_trace, time_trace, starting_idx, testing_duration + env.player.get_buffer())
            log_file.write('\t'.join(str(tp) for tp in tp_record))
            log_file.write('\n')
            log_file.write('\t'.join(str(time) for time in time_record))
            # log_file.write('\n' + str(IF_NEW))
            log_file.write('\n' + str(testing_start_time))
            log_file.write('\n')
            log_file.close()
    else:
        # # Single testing
        # env_end = env.reset()
        # env.act(0, 3)   # Default
        # state = env.get_state()
        # total_reward = 0.0
        pass

if __name__ == '__main__':
    main()
