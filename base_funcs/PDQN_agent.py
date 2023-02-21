from copy import deepcopy
import numpy as np
import os
import torch
import torch as T
from Networks.PDQN_Networks import PDQN_ActorNetwork, PDQN_CriticNetwork
from Networks.noise import OUActionNoise
from torch.autograd import Variable
import torch.nn.functional as F
from Networks.buffer import ReplayBuffer
from Utils import soft_update, hard_update

class Agent:
    def __init__(self, id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                 Fed_Comm, folder):

        torch.manual_seed(np.random.randint(1, 100))
        self.actor_model = PDQN_ActorNetwork(actor_net.actor_lr, actor_net.state_size, actor_net.hidden_layers,
                                             actor_net.activation, actor_net.squashing_function, actor_net.init_type,
                                             actor_net.action_size)
        self.critic_model = PDQN_CriticNetwork(critic_net.critic_lr, critic_net.state_size, critic_net.hidden_layers,
                                               critic_net.activation, critic_net.squashing_function, critic_net.init_type,
                                               critic_net.action_size)
        self.target_actor_model = deepcopy(self.actor_model)
        self.target_critic_model = deepcopy(self.critic_model)
        self.id = id
        self.clip_grad = 10
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = 0.1
        self.eps_decay = 5e-5
        self.tau = taus
        self.batch_size = batch_size
        self.full_data = full_data * batch_size
        self.n_actions = self.actor_model.action_size
        self.n_states = self.actor_model.state_size
        self.min_power = 0
        self.algorithm = algorithm
        self.Fed_Comm = Fed_Comm
        self.folder = folder
        self.create_model_path()

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        self.memory = ReplayBuffer(mem_size, self.n_states, self.n_actions, algorithm)

        hard_update(self.actor_model, self.target_actor_model)
        self.target_actor_model.eval()
        hard_update(self.critic_model, self.target_critic_model)
        self.target_critic_model.eval()

        # saving the gradients of the local models and the server
        self.local_actor_model = deepcopy(list(self.actor_model.parameters()))
        self.local_critic_model = deepcopy(list(self.critic_model.parameters()))

        self.server_Trx_actor_grad = deepcopy(list(self.actor_model.parameters()))
        self.server_Trx_critic_grad = deepcopy(list(self.critic_model.parameters()))

        self.pre_local_actor_grad = deepcopy(list(self.actor_model.parameters()))
        self.pre_local_critic_grad = deepcopy(list(self.critic_model.parameters()))

    def set_passthrough_resenet_weight_biases(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_model.action_passthrough_layer

        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.actor_model.device)

        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.actor_model.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update(self.actor_model, self.target_actor_model)

    def choose_action(self, observation, option):
        #  Choosing action: combination of DDPG and DQN
        with torch.no_grad():
            observation_ = np.expand_dims(observation, axis=0)
            state = T.tensor(observation_, dtype=T.float).to(self.actor_model.device)
            continuous_acts = self.actor_model.forward(state).to(
                self.actor_model.device)  # continuous parameters of the discrete actions
            # epsilon Greedy; similar to DQN policy
            rnd = np.random.uniform(low=0, high=1)
            if rnd < self.epsilon:
                discrete_act = np.random.choice(self.n_actions)
                # continuous_acts = torch.tensor(np.random.uniform(-1, 1))
            else:
                # select maximum action
                Q_a = self.critic_model.forward(state, continuous_acts)
                Q_a = Q_a.detach().cpu().data.numpy()
                discrete_act = np.argmax(Q_a)

            # Adding noise to the selected actions
            continuous_acts = continuous_acts.cpu().data.numpy().flatten()
            if option == 'train':
                continuous_acts[discrete_act] = continuous_acts[discrete_act] + self.noise()[discrete_act]
            continuous_act_param = continuous_acts[discrete_act]

        return [discrete_act, continuous_act_param, continuous_acts]

    def learn(self):

        assert self.memory.mem_cntr >= self.full_data

        states, total_actions, rewards, states_, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor_model.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor_model.device)
        total_actions = T.tensor(total_actions, dtype=T.float).to(self.actor_model.device)
        Dis_acts = total_actions[:, 0].long()
        Con_acts = total_actions[:, 1:]
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor_model.device)
        done = T.tensor(done).to(self.actor_model.device)

        # ---------------------- optimizing the Critic-network ----------------------
        with torch.no_grad():
            target_actions_predicted = self.target_actor_model.forward(states_)
            pred_Q_a = self.target_critic_model.forward(states_, target_actions_predicted)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = rewards + (1 - done.int()) * self.gamma * Qprime

        q_values = self.critic_model(states, Con_acts)
        y_predicted = q_values.gather(1, Dis_acts.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = F.mse_loss(y_predicted, y_expected)

        self.critic_model.critic_optimizer.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.clip_grad)
        self.critic_model.critic_optimizer.step()

        # ---------------------- optimizing the Actor-network ----------------------
        with torch.no_grad():
            action_params = self.actor_model(states)
        action_params.requires_grad = True
        Q = self.critic_model(states, action_params)
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.critic_model.zero_grad()
        Q_loss.backward()
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_model(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, inplace=True)
        out = -torch.mul(delta_a, action_params)
        self.actor_model.zero_grad()
        out.backward(torch.ones(out.shape).to(self.actor_model.device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.clip_grad)
        self.actor_model.actor_optimizer.step()

        soft_update(self.critic_model, self.target_critic_model, self.tau[0])
        soft_update(self.actor_model, self.target_actor_model, self.tau[1])

        self.decrement_epsilon()

        return loss_Q, out

    def _invert_gradients(self, grad, vals, inplace=True):

        max_p = torch.ones(self.n_actions)  #ToDo: hard coded --> based on the Tanh output; but it can take any value
        min_p = -1*max_p
        range = max_p - min_p

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        range = range.cpu()

        grad = grad.cpu()
        vals = vals.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / range)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / range)[~index]

        return grad

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_decay \
            if self.epsilon > self.eps_min else self.eps_min

    def set_parameters(self, server_actor_model, server_critic_model):
        # actor
        for old_actor_param, server_actor_param, local_actor_param in zip(self.actor_model.parameters(),
                                                                          server_actor_model.parameters(),
                                                                          self.local_actor_model):
            old_actor_param.data = server_actor_param.data.clone()
            local_actor_param.data = server_actor_param.data.clone()
            if(server_actor_param.grad != None):
                if(old_actor_param.grad == None):
                    old_actor_param.grad = torch.zeros_like(server_actor_param.grad)

                if(local_actor_param.grad == None):
                    local_actor_param.grad = torch.zeros_like(server_actor_param.grad)

                old_actor_param.grad.data = server_actor_param.grad.data.clone()
                local_actor_param.grad.data = server_actor_param.grad.data.clone()

        # critic
        for old_critic_param, server_critic_param, local_critic_param in zip(self.critic_model.parameters(),
                                                                                 server_critic_model.parameters(),
                                                                                 self.local_critic_model):
            old_critic_param.data = server_critic_param.data.clone()
            local_critic_param.data = server_critic_param.data.clone()
            if(server_critic_param.grad != None):
                if(old_critic_param.grad == None):
                    old_critic_param.grad = torch.zeros_like(server_critic_param.grad)

                if(local_critic_param.grad == None):
                    local_critic_param.grad = torch.zeros_like(server_critic_param.grad)

                old_critic_param.grad.data = server_critic_param.grad.data.clone()
                local_critic_param.grad.data = server_critic_param.grad.data.clone()

    def get_actor_parameters(self):
        return self.actor_model.parameters()

    def get_critic_parameters(self):
        return self.critic_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
            if (param.grad != None):
                if (clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()

        return clone_param

    def create_model_path(self):
        if self.Fed_Comm:
            self.intermediary_model_path = os.path.join(self.folder, "agent_models")
            if not os.path.exists(self.intermediary_model_path):
                os.makedirs(self.intermediary_model_path)
        else:
            self.intermediary_model_path = os.path.join(self.folder, "agent_models")
            if not os.path.exists(self.intermediary_model_path):
                os.makedirs(self.intermediary_model_path)

    def save_agent_model(self):
        torch.save(self.actor_model, os.path.join(self.intermediary_model_path, "actor_model_" + str(self.id) + ".pt"))
        torch.save(self.critic_model, os.path.join(self.intermediary_model_path, "critic_model_" + str(self.id) + ".pt"))
        torch.save(self.target_actor_model, os.path.join(self.intermediary_model_path, "target_actor_model_" + str(self.id) + ".pt"))
        torch.save(self.target_critic_model, os.path.join(self.intermediary_model_path, "target_critic_model_" + str(self.id) + ".pt"))

    def load_agent_model(self, actor_model, critic_model, extend):
        if extend:
            self.actor_model = actor_model
            self.critic_model = critic_model
        else:
            model_actor_path = os.path.join(self.intermediary_model_path, "actor_model_" + str(self.id) + ".pt")
            model_critic_path = os.path.join(self.intermediary_model_path, "critic_model_" + str(self.id) + ".pt")
            model_target_actor_path = os.path.join(self.intermediary_model_path, "target_actor_model_" + str(self.id) + ".pt")
            model_target_critic_path = os.path.join(self.intermediary_model_path, "target_critic_model_" + str(self.id) + ".pt")
            self.actor_model = torch.load(model_actor_path)
            self.critic_model = torch.load(model_critic_path)
            self.target_actor_model = torch.load(model_target_actor_path)
            self.target_critic_model = torch.load(model_target_critic_path)

        # return self.actor_model, self.critic_model
