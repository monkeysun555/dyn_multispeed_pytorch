from collections import deque
from config import Config
import random
import numpy as np


class Reply_Buffer:
    def __init__(self, buffer_size):
        self.buffer = deque()
        self.buffer_size = buffer_size

    def append(self, item):
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(item)

    def sample(self, batch_size=Config.sampling_batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_state, batch_action_1, batch_action_2, batch_reward, batch_state_new, batch_over = [], [], [], [], [], []
        
        for b in batch:
            batch_state.append(b[0])
            batch_action_1.append(b[1])
            batch_action_2.append(b[2])
            batch_reward.append(b[3])
            batch_state_new.append(b[4])
            batch_over.append(float(b[5]))
        
        batch_state = np.stack(batch_state)
        batch_action_1 = np.stack(batch_action_1)
        batch_action_2 = np.stack(batch_action_2)
        batch_reward = np.stack(batch_reward)
        batch_state_new = np.stack(batch_state_new)
        batch_over = np.stack(batch_over)
        return batch_state, batch_action_1, batch_action_2, batch_reward, batch_state_new, batch_over