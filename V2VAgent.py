from base_funcs.PDQN_agent import Agent as PDQN_Agent
from base_funcs.DDPG_agent import Agent as DDPG_Agent

class hybrid_v2v_agent(PDQN_Agent):
    def __init__(self, id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                 Fed_Comm, folder):
        super().__init__(id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                         Fed_Comm, folder)

    def train(self):

        self.learn()
        self.clone_model_paramenter(self.actor_model.parameters(), self.local_actor_model)
        self.clone_model_paramenter(self.critic_model.parameters(), self.local_critic_model)

class DDPG_v2v_agent(DDPG_Agent):
    def __init__(self, id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                 Fed_Comm, folder):
        super().__init__(id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                         Fed_Comm, folder)

    def train(self):

        self.learn()
        self.clone_model_paramenter(self.actor_model.parameters(), self.local_actor_model)
        self.clone_model_paramenter(self.critic_model.parameters(), self.local_critic_model)
