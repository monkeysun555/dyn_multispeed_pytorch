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

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--test', dest='test', help='do testing',
#                         default=None, type=str)
#     # parser.add_argument('-t', '--train', dest='train', help='train policy or not',
#     #                     default=True, type=bool)
#     args = parser.parse_args()
#     return args
# args = parse_args() 

def main():
    # restore = args.restore
    # Load env
    env = Env.Live_Streaming()
    _, action_dims = env.get_action_info()
    reply_buffer = Reply_Buffer(Config.reply_buffer_size)
    agent = Agent(action_dims)
    reward_logs = []
    loss_logs = []

    # restore model
    # if restore:
    #    agent.restore(restore)

    for episode in range(1, Config.total_episode+1):
        # reset env
        env_end = env.reset()
        env.act(0, 3)   # Default
        state = env.get_state()
        total_reward = 0.0

        # Update epsilon
        agent.update_epsilon_by_epoch(episode)
        while not env.streaming_finish():
            if Config.model_version == 0:
                action_1, action_2 = agent.take_action(np.array([state]))
                # print(action_1, action_2)
                reward = env.act(action_1, action_2)
                # print(reward)
                state_new = env.get_state()
                total_reward += reward
                action_onehots = []
                action_1_onehot = np.zeros(action_dims[0])
                action_2_onehot = np.zeros(action_dims[1])
                action_1_onehot[action_1] = 1
                action_2_onehot[action_2] = 1
                # print(env.streaming_finish())
                reply_buffer.append((state, action_1_onehot, action_2_onehot, reward, state_new, env.streaming_finish()))
                state = state_new
            elif Config.model_version == 1:                
                action = agent.take_action(np.array([state]))
                action_1 = int(action/action_dims[1])
                action_2 = action%action_dims[1]
                reward = env.act(action_1, action_2)
                # print(reward)
                state_new = env.get_state()
                total_reward += reward
                action_onehot = np.zeros(action_dims[0]*action_dims[1])
                action_onehot[action] = 1
                # print(env.streaming_finish())
                reply_buffer.append((state, action_onehot, reward, state_new, env.streaming_finish()))
                state = state_new

        # sample batch from reply buffer
        if episode < Config.observe_episode:
            continue

        # update target network
        if episode % Config.update_target_frequency == 0:
            agent.update_target_network()

        if Config.model_version == 0:
            batch_state, batch_actions_1, batch_actions_2, batch_reward, batch_state_new, batch_over = reply_buffer.sample()
            # update policy network
            loss = agent.update_Q_network_v1(batch_state, batch_actions_1, batch_actions_2, batch_reward, batch_state_new, batch_over)

        elif Config.model_version == 1:
            batch_state, batch_actions, batch_reward, batch_state_new, batch_over = reply_buffer.sample()
            loss = agent.update_Q_network_v2(batch_state, batch_actions, batch_reward, batch_state_new, batch_over)
        
        loss_logs.extend([[episode, loss]])
        reward_logs.extend([[episode, total_reward]])

        # save model
        if episode % Config.save_logs_frequency == 0:
            print("episode:", episode)
            agent.save(episode, Config.logs_path)
            np.save(os.path.join(Config.logs_path, 'loss.npy'), np.array(loss_logs))
            np.save(os.path.join(Config.logs_path, 'reward.npy'), np.array(reward_logs))

        # print reward and loss
        if episode % Config.show_loss_frequency == 0: 
            if Config.model_version == 0:
                print('Episode: {} Reward: {:.3f} Loss: {:.3f} and {:.3f}' .format(episode, total_reward, loss[0], loss[1]))
            elif Config.model_version == 1:
                print('Episode: {} Reward: {:.3f} Loss: {:.3f}' .format(episode, total_reward, loss[0]))
        agent.update_epsilon_by_epoch(episode)

if __name__ == "__main__":
    main()



