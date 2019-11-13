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


# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def main():
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
        env.reset()
        env.act(0, 3)	# Default
        state = env.get_state()
        # state = np.stack([[obs for _ in range(4)]], axis=0)
        total_reward = 0.0

        # Update epsilon
        agent.update_epsilon_by_epoch(episode)
        while not env.streaming_finish():
            action_1, action_2 = agent.take_action(state)
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
            reply_buffer.append((state, [action_1_onehot, action_2_onehot], reward, state_new, env.streaming_finish()))
            state = state_new

        # update target network
        if episode % Config.update_target_frequency == 0:
            agent.update_target_network()

        # sample batch from reply buffer
        batch_state, batch_actions, batch_reward, batch_state_new, batch_over = reply_buffer.sample()

        # update policy network
        loss = agent.update_Q_network(batch_state, batch_actions, batch_reward, batch_state_new, batch_over)

        loss_logs.extend([[episode, loss]])
        reward_logs.extend([[episode, total_reward]])

        # save model
        if episode % Config.save_logs_frequency == 0:
            agent.save(episode, Config.logs_path)
            np.save(os.path.join(Config.logs_path, 'loss.npy'), np.array(loss_logs))
            np.save(os.path.join(Config.logs_path, 'reward.npy'), np.array(reward_logs))

        # print reward and loss
        if episode % Config.show_loss_frequency == 0: 
            print('Episode: {} Reward: {:.3f} Loss: {:.3f}' .format(episode, total_reward, loss))

        agent.update_epsilon()

if __name__ == "__main__":
    main()



