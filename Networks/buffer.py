import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, algorithm):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float16)
        if algorithm == 'DDPG':
            self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float16)  # +1 is for the discrete action
        elif algorithm == 'Hybrid':
            self.action_memory = np.zeros((self.mem_size, n_actions + 1), dtype=np.float16)
        # ToDo: HardCoded --> for the case of multiple discrete actions, change the code to be more flexible
        self.reward_memory =  np.zeros((self.mem_size), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float16)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward_mati, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward_mati
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# class PDQN_ReplayBuffer():
#     def __init__(self, total_size, input_shape, n_actions, sample_size=0):
#         self.mem_size = total_size
#         self.sample_size = sample_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float16)
#         self.action_memory = np.zeros((self.mem_size, n_actions + 1), dtype=np.float16)  # +1 is for the discrete action
#         # ToDo: HardCoded --> for the case of multiple discrete actions, change the code to be more flexible
#         self.reward_memory =  np.zeros((self.mem_size), dtype=np.float16)
#         self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float16)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
#
#         # full grad information (full sample of the memory)
#         self.state_memory_loader = np.zeros((self.sample_size, input_shape), dtype=np.float16)
#         self.action_memory_loader = np.zeros((self.sample_size, n_actions + 1), dtype=np.float16)  # +1 is for the discrete action
#         self.reward_memory_loader = np.zeros((self.sample_size), dtype=np.float16)
#         self.new_state_memory_loader = np.zeros((self.sample_size, input_shape), dtype=np.float16)
#         self.terminal_memory_loader = np.zeros(self.sample_size, dtype=np.bool)
#
#     def store_transition(self, state, action, reward_mati, state_, done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward_mati
#         self.new_state_memory[index] = state_
#         self.terminal_memory[index] = done
#
#         self.mem_cntr += 1
#
#     def loader_sample(self):
#         max_mem = min(self.mem_cntr, self.mem_size)
#         batch = np.random.choice(max_mem, self.sample_size, replace=False)
#
#         self.state_memory_loader = self.state_memory[batch]
#         self.action_memory_loader = self.action_memory[batch]
#         self.reward_memory_loader = self.reward_memory[batch]
#         self.new_state_memory_loader = self.new_state_memory[batch]
#         self.terminal_memory_loader = self.terminal_memory[batch]
#
#
#     def sample_buffer(self, batch_size):
#
#         batch = np.random.choice(self.sample_size, batch_size, replace=False)
#
#         states = self.state_memory_loader[batch]
#         actions = self.action_memory_loader[batch]
#         rewards = self.reward_memory_loader[batch]
#         states_ = self.new_state_memory_loader[batch]
#         dones = self.terminal_memory_loader[batch]
#
#         return states, actions, rewards, states_, dones