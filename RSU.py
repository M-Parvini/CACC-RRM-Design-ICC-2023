import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import torch
import scipy
import numpy as np
from base_funcs.server import Server
import Utils.utils as OTTO
from V2VAgent import hybrid_v2v_agent
from V2VAgent import DDPG_v2v_agent
import shap


class RSU_server(Server):
    def __init__(self, actor_model, critic_model, n_agents, N_veh, env, HS_CACC, epsilon, gamma, taus, n_train, n_test,
    batch_size, mem_size, Platoon_speed, Headway, safety_distance, sim_time, full_data, algorithm, Fed_Comm, save_path,
                folder, extend, model_exp):
        super().__init__(actor_model, critic_model, n_agents, Fed_Comm, folder)

        # Initialize
        self.n_state = None if (algorithm == 'opt_fair') or (algorithm == 'opt_non_fair') or (algorithm == 'random') \
            else actor_model.state_size
        self.n_action = None if (algorithm == 'opt_fair') or (algorithm == 'opt_non_fair') or (algorithm == 'random') \
            else actor_model.action_size
        self.chosen_acts = 2 # for calculating the reward values// It has nothing to do with the action size
        self.n_agents = n_agents
        self.N_veh = N_veh
        self.size_platoon = env.size_platoon
        self.env = env
        self.HS_CACC = HS_CACC
        self.n_train = n_train
        self.n_test = n_test
        self.MATI_quantity = self.HS_CACC.MATI
        self.headway = Headway
        self.Headway_gap = Platoon_speed / 3.6 * Headway
        self.initial_velocity = Platoon_speed / 3.6
        self.safety_distance = safety_distance
        self.sim_time = sim_time
        self.update_cntr = 0
        self.max_power = self.env.max_power
        self.algorithm = algorithm
        self.Fed_Comm = Fed_Comm
        self.save_path = save_path
        self.folder = folder
        self.extend = extend
        self.explain = model_exp
        # Algorithm initialization
        if algorithm == 'DDPG':
            for i in range(self.n_agents):
                id = 'agent_' + str(i)
                user = DDPG_v2v_agent(id, gamma, mem_size, epsilon, taus, batch_size, actor_model, critic_model,
                                      full_data, algorithm, Fed_Comm, folder)
                self.users.append(user)

        elif algorithm == 'Hybrid':
            for i in range(self.n_agents):
                id = 'agent_' + str(i)
                user = hybrid_v2v_agent(id, gamma, mem_size, epsilon, taus, batch_size, actor_model, critic_model,
                                        full_data, algorithm, Fed_Comm, folder)
                self.users.append(user)

        elif algorithm == 'opt_fair':
            self.fairness = True

        elif algorithm == 'opt_non_fair':
            self.fairness = False


    def reset(self):
        # Initialization
        self.done = False
        self.time = 0
        self.Trx_Cntr = np.zeros(self.n_agents)
        self.MATIs = np.zeros(self.n_agents, dtype=int) * self.MATI_quantity
        self.success_rate = np.zeros(self.n_agents, dtype=int)
        self.String_Stability = np.zeros(self.n_agents)
        self.acc_data = np.zeros(self.n_agents)
        self.y_initial = np.kron(np.ones([1, self.size_platoon]), np.array([0, self.initial_velocity, 0, 0]))
        self.Total_Output = self.y_initial.copy()
        self.Total_SS = self.String_Stability.copy()
        self.Total_Time = self.time
        self.y_initial = np.block([self.y_initial, np.zeros([1, self.N_veh])]).reshape(-1)
        self.V2V_dist = OTTO.V2V_Dist(self.N_veh, self.y_initial.copy(), self.headway, self.safety_distance)
        self.state_outage_old_all = [0] * self.n_agents
        self.state_outage_new_all = [0] * self.n_agents
        self.env.V2V_demand = np.ones(self.n_agents, dtype=np.float16) * self.env.V2V_demand_size
        self.env.V2V_MATI = self.env.MATI_bound * np.ones(self.n_agents, dtype=int)

    def v2v_passthrough_layer_ini(self):
        # Linear passthrough bias and weight initialization
        initial_weights = np.zeros((self.n_action, self.n_state))
        initial_bias = np.ones(self.n_action) * (-0.8)  # adding a small bias tomake the output comparable to the state
        for user in self.users:
            user.set_passthrough_resenet_weight_biases(initial_weights, initial_bias)

    def save_episodic_results(self, episode):
        total_output_path = os.path.join(self.save_path + '/total_outputs.mat')
        scipy.io.savemat(total_output_path, {'total_outputs': self.Total_Output})

        trx_path = os.path.join(self.save_path + '/trx_' + str(episode) + '_cntr.mat')
        scipy.io.savemat(trx_path, {'trx_' + str(episode) + '_cntr': self.Trx_Cntr})

        total_ss_path = os.path.join(self.save_path + '/total_ss.mat')
        scipy.io.savemat(total_ss_path, {'total_ss': self.Total_SS})

    def save_intermediary_results(self, episode, action_outage, V_rate, outage_reward, SINR, total_state_info, opt):

        with open(self.save_path + '/power_consumption_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(action_outage[:, 1].flatten())

        with open(self.save_path + '/total_state_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(total_state_info.flatten())

        with open(self.save_path + '/resource_allocation_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(action_outage[:, 0].flatten())

        with open(self.save_path + '/V2V_rate_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(V_rate)

        with open(self.save_path + '/outage_reward_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(outage_reward)

        with open(self.save_path + '/string_stability_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.String_Stability)

        with open(self.save_path + '/SINR_vals_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SINR)

        with open(self.save_path + '/time' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(np.array([self.time], np.float))

    def Save_models(self):
        self.save_server_model()
        for user in self.users:
            user.save_agent_model()

    def PDQN_Run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds
            #  Getting the initial states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            # If the time equals the MATI, then the agent will proceed with a new MATI
            #  In every MATI, a new packet must be transmitted, hence the success rate must be updated
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound

            #  Choosing the action for packet transmissions
            action_outage = np.zeros([self.n_agents, self.chosen_acts], dtype=int)  # power, resource block
            action_outage_list = []
            for i in range(self.n_agents):
                '''
                In this phase the agents will select their power and the resources and based on these information,
                the channel gains & data rates of the whole platoon (vehicles) will be determined.
                '''
                action = self.users[i].choose_action(self.state_outage_old_all[i], option)
                # action = OTTO.Clipping(action)  # No clipping --> gradient inverting
                action_outage_list.append(np.concatenate(([action[0]], action[2])).ravel())
                action_outage[i, 0] = int(action[0])  # chosen RB
                action_outage[i, 1] = OTTO.action_unwrapper(action[1], self.max_power)  # power selected by PL
            #                .
            #  State update: X = AX + Bu
            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            # print('Chosen MATIs: ', MATIs)
            print('Chosen RBs: ', action_outage[:, 0].flatten())
            print('Chosen powers: ', action_outage[:, 1].flatten())
            # print('remaining V2V payload: ', env.V2V_demand)
            print('exploration: ', self.users[0].epsilon)
            print('outage performance: ', outage_reward)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all), option)

            # taking the agents actions, states and reward and learning phase
            if option == 'train':
                for i in range(self.n_agents):
                    self.users[i].memory.store_transition(self.state_outage_old_all[i], action_outage_list[i],
                                                          outage_reward[i], self.state_outage_new_all[i], self.done)

                if self.users[0].memory.mem_cntr >= self.users[0].full_data:  # does not matter which user (same for all)
                    print('============== Training phase ===============')
                    for user in self.users:
                        user.train()
                    if self.Fed_Comm:
                        self.aggregate_parameters()
            # old observation = new_observation
            for i in range(self.n_agents):
                self.state_outage_old_all[i] = self.state_outage_new_all[i]

    def Optim_run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound


            #  Choosing the action for packet transmissions
            power_assigns, ch_assigns = OTTO.convex_opt_solution(self.env, self.fairness)

            element = int(self.n_agents/2)
            action_outage = np.zeros([self.n_agents, 2])  # power, resource block
            action_outage[:element, 0] = np.array(range(element))
            action_outage[element:, 0] = ch_assigns
            channels_ = action_outage[:, 0]
            for i in range(element):
                v2v_idx = np.where(channels_ == i)
                action_outage[i, 1] = OTTO.action_wrapper(power_assigns[0][i][int(v2v_idx[0][1]-element)])
                action_outage[v2v_idx[0][1], 1] = OTTO.action_wrapper(power_assigns[1][i][int(v2v_idx[0][1]-element)])

            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            # print('Chosen MATIs: ', MATIs)
            print('Chosen RBs: ', action_outage[:, 0].flatten())
            print('Chosen powers: ', action_outage[:, 1].flatten())
            # print('remaining V2V payload: ', env.V2V_demand)
            # print('exploration: ', self.users[0].epsilon)
            print('outage performance: ', outage_reward)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)

    def Rand_run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound


            #  Choosing the action for packet transmissions
            action_outage = np.zeros([self.n_agents, 2])  # power, resource block
            for i in range(self.n_agents):
                action_outage[i, 0] = np.random.randint(3)
                action_outage[i, 1] = 30

            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            # print('Chosen MATIs: ', MATIs)
            print('Chosen RBs: ', action_outage[:, 0].flatten())
            print('Chosen powers: ', action_outage[:, 1].flatten())
            # print('remaining V2V payload: ', env.V2V_demand)
            # print('exploration: ', self.users[0].epsilon)
            print('outage performance: ', outage_reward)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)

    # def DDPG_Run(self, episode, option):
    #
    #     self.reset()
    #     #  Renewing the channels --> path-loss + fast fading
    #     self.env.renew_channel(self.V2V_dist)
    #     self.env.renew_channels_fastfading()
    #
    #     while self.time < self.sim_time * 1000:  # changing the time milliseconds
    #         #  Getting the initial states
    #         for i in range(self.n_agents):
    #             state_outage = OTTO.get_outage_state(env=self.env, idx=i)
    #             self.state_outage_old_all[i] = state_outage
    #             # print('agent ' + str(i) + ' state', state_outage)
    #
    #         # If the time equals the MATI, then the agent will proceed with a new MATI
    #         #  In every MATI, a new packet must be transmitted, hence the success rate must be updated
    #         for i in range(self.n_agents):
    #             if self.time == self.MATIs[i]:
    #                 self.Trx_Cntr[i] += 1
    #                 # Which acceleration data we are transmitting?
    #                 self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
    #                 # get the new MATI: agent.get_mati[i]
    #                 self.MATIs[i] += self.MATI_quantity
    #                 self.success_rate[i] = 0
    #                 self.env.V2V_demand[i] = self.env.V2V_demand_size
    #                 self.env.V2V_MATI[i] = self.env.MATI_bound
    #
    #         #  Choosing the action for packet transmissions
    #         action_outage = np.zeros([self.n_agents, self.chosen_acts])  # power, resource block
    #         action_outage_list = []
    #         for i in range(self.n_agents):
    #             '''
    #             In this phase the agents will select their power and the resources and based on these information,
    #             the channel gains & data rates of the whole platoon (vehicles) will be determined.
    #             '''
    #             action = self.users[i].choose_action(self.state_outage_old_all[i], option)
    #             action = np.clip(action, -0.999, 0.999)
    #             action_outage_list.append(action)
    #             # action = OTTO.Clipping(action)  # No clipping --> gradient inverting
    #             action_outage[i, 0] = 0  # chosen RB
    #             # action_outage[i, 0] = ((action[0]+1)/2) * self.env.n_RB  # chosen RB
    #             action_outage[i, 1] = OTTO.action_unwrapper(action[0], self.max_power)  # power selected by PL
    #         #                .
    #         #  State update: X = AX + Bu
    #         T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
    #                                                  self.MATIs, self.acc_data)
    #
    #         y_initial_ = Y_new[-1]
    #         time_ = int(np.round((T_new[-1])))
    #         self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
    #         self.Total_Time = np.block([self.Total_Time, T_new[1:]])
    #         #  Calculating the reward for the outage and packet transmission part
    #         outage_reward, V_rate, Demand_R, success_rate_, SINR = \
    #             self.env.Outage_Reward(action_outage, self.success_rate, time_ - self.time)
    #
    #         #  calculating the new distance by subtracting the distance differences of V2V links
    #         V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
    #         print("-----------------------------------")
    #         print('Time before:', self.time)
    #         print('Time after:', time_)
    #         print('MATIS: ', self.MATIs)
    #         print('string stability: ', self.String_Stability)
    #
    #         self.time = time_
    #         self.y_initial = y_initial_
    #         self.V2V_dist = V2V_dist_
    #         self.success_rate = success_rate_
    #
    #         #  Renewing the channels --> path-loss + fast fading
    #         self.env.renew_channel(self.V2V_dist)
    #         self.env.renew_channels_fastfading()
    #
    #         ## Learning phase for the MATI neural networks
    #         for i in range(self.n_agents):
    #             if self.time == self.MATIs[i]:
    #                 # Some part of the
    #                 String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
    #                                                                             self.Total_Time.copy())
    #                 self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)
    #
    #         self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])
    #
    #         # Getting the new states
    #         for i in range(self.n_agents):
    #             state_outage = OTTO.get_outage_state(env=self.env, idx=i)
    #             self.state_outage_new_all[i] = state_outage
    #
    #         #  Changing the done to True in case of terminal state
    #         if self.time >= self.sim_time * 1000:
    #             self.done = True
    #
    #         # print('Chosen MATIs: ', MATIs)
    #         print('Chosen RBs: ', action_outage[:, 0].flatten())
    #         print('Chosen powers: ', action_outage[:, 1].flatten())
    #         # print('remaining V2V payload: ', env.V2V_demand)
    #         print('exploration: ', self.users[0].epsilon)
    #         print('outage performance: ', outage_reward)
    #
    #         # Save intermediary results
    #         self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR, option)
    #
    #         # taking the agents actions, states and reward and learning phase
    #         for i in range(self.n_agents):
    #             self.users[i].memory.store_transition(self.state_outage_old_all[i], action_outage_list[i],
    #                                                   outage_reward[i], self.state_outage_new_all[i], self.done)
    #
    #         # train
    #         if self.users[0].memory.mem_cntr >= self.users[0].full_data:  # does not matter which user (same for all)
    #             print('============== Training phase ===============')
    #             for user in self.users:
    #                 user.train()
    #             if self.Fed_Comm:
    #                 self.aggregate_parameters()
    #         # old observation = new_observation
    #         for i in range(self.n_agents):
    #             self.state_outage_old_all[i] = self.state_outage_new_all[i]

    def train(self, training):

        for episode in range(self.n_train):
            print("-------------Round number: ", episode, " -------------")
            if self.Fed_Comm:
                self.send_parameters()
            # self.v2v_passthrough_layer_ini()
            if self.algorithm == 'DDPG':
                self.DDPG_Run(episode, training)
            elif self.algorithm == 'Hybrid':
                self.PDQN_Run(episode, training)
            else:
                raise ValueError("Unknown algorithm " + str(self.algorithm))

            self.save_episodic_results(episode)
            self.Save_models()

    def test(self, testing):
        # if self.extend:
            # self.load_server_model()
            # common_actor, common_critic = self.users[1].load_agent_model(self.server_actor_model, self.server_critic_model, extend=False)

        for user in self.users:
            # user.load_agent_model(self.server_actor_model, self.server_critic_model, self.extend)
            user.load_agent_model(actor_model=None, critic_model=None, extend=self.extend)
            user.epsilon = user.eps_min

        for episode in range(self.n_test):

            if self.algorithm == 'DDPG':
                self.DDPG_Run(episode, testing)
            elif self.algorithm == 'Hybrid':
                self.PDQN_Run(episode, testing)
            elif self.algorithm == 'opt_fair':
                self.Optim_run(episode, 'fair')
            elif self.algorithm == 'opt_non_fair':
                self.Optim_run(episode, 'non_fair')
            elif self.algorithm == 'random':
                self.Rand_run(episode, 'random')
            else:
                raise ValueError("Unknown algorithm " + str(self.algorithm))

            self.save_episodic_results(episode)

    def explainable_AI(self):
        #
        i = 0
        # feature_names = ['Path-loss', 'Fast-fading, RB1', 'Fast-fading, RB2', 'Fast-fading, RB3',
        #                  'Interference', 'Remaining time', 'Remaining packet'],
        plt.rcParams["font.family"] = "Times New Roman"
        l = self.users[0].actor_model.state_size

        for user in self.users:
            # user.load_agent_model(self.server_actor_model, self.server_critic_model, self.extend)
            user.load_agent_model(actor_model=None, critic_model=None, extend=self.extend)
            user.epsilon = user.eps_min
            state = np.genfromtxt(os.path.join(self.save_path,'total_state_7_test.csv'), delimiter=',')
            # Choose of number 7 was as a mater of taste. replace it with any number
            state = torch.tensor(state[:,l*i:l*(i+1)]).float().to(user.actor_model.device)
            explainer = shap.DeepExplainer(user.actor_model, state)  # build DeepExplainer
            shap_values = explainer.shap_values(state)  # Calculate shap values
            shap.summary_plot(shap_values, features=state,
                              feature_names=['Fast-fading, RB1', 'Fast-fading, RB2', 'Fast-fading, RB3',
                                             'Interference'],
                              class_names=['RB-power (1)', 'RB-power (2)', 'RB-power (3)'], plot_type = 'bar', show=False)
            plt.legend(frameon=True, framealpha=1, loc='lower right')
            ax = plt.gca()
            ax.grid(b=True, which='major', axis='x', color='#000000', linestyle='--', linewidth=0.25)
            file_name = os.path.join(self.save_path, str(self.folder)+'_'+str(user.id)+'_explainer.pdf')
            plt.savefig(file_name, dpi=500)
            plt.close()
            i += 1
