from collections import deque
import numpy as np


class BufferManager:

    def __init__(self, seq_len):

        self.buffer = deque(maxlen=seq_len)


    def add(self, v, i, t):

        self.buffer.append([v, i, t])


    def is_ready(self):

        return len(self.buffer) == self.buffer.maxlen


    def get_sequence(self):

        return np.array(self.buffer)