import numpy as np
import math


class V2Vchannels:
    # Simulator of the V2V Channels
    '''
    3GPP 37.885 "Study on evaluation methodology of new Vehicle-to-Everything (V2X) use cases for LTE and NR; (Rel. 15)"
    '''
    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2  # GHz
        self.decorrelation_distance = 25
        self.shadow_std = 3

    def get_path_loss(self, distance, block):
        v2v_distance = abs(distance)
        if v2v_distance < 1:  # This condition will barely happen. Just to make sure we do not end up in NaN.
            v2v_distance = 1
        Path_loss = 32.4 + 20 * np.log10(v2v_distance) + 20 * np.log10(self.fc)
        if block != 1:  # there are cars in between --> car blockage 3GPP 37.885 shadowing
            Path_loss = Path_loss + np.random.normal(15 + max(0, 15 * np.log10(distance) - 30),
                                                     4.5)  # deviated from the standard mean != 9
        return Path_loss

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)


class Vehicle:

    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity


class Environ:

    def __init__(self, width, size_platoon, n_RB, BW, V2V_SIZE, Gap, safety_dist, velocity, N_m, threshold,
                 outage_prob, min_rate, max_power, MATI):
        # Road Configuration
        # 3GPP 37.885 "Study on evaluation methodology of new V2X use cases for LTE and NR". P.37
        up_lanes = [i for i in [4 / 2, 4 / 2 + 4, 4 / 2 + 8]]
        self.lanes = up_lanes
        self.width = width
        self.road_labels = ['lower', 'middle', 'upper']  # line of the highway
        self.velocity = velocity
        self.gap = Gap
        self.safety_dist = safety_dist
        self.n_RB = n_RB
        self.size_platoon = size_platoon
        self.N_Agents = int(self.size_platoon - 2)
        self.bandwidth = BW  # bandwidth per RB, 180,000 MHz
        self.V2V_demand_size = V2V_SIZE  # V2V payload: * Bytes every MATI
        self.Nakagami_m = N_m
        self.threshold = np.power(10, threshold / 10)  # dB watt = 10*log10(watt)
        self.out_prob = outage_prob
        self.min_rate = min_rate
        self.max_power = max_power
        self.MATI_bound = MATI

        self.V2Vchannels = V2Vchannels()
        self.vehicles = []

        self.V2V_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2V_pathloss = []

        self.sig2_dBm = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** ((self.sig2_dBm-30) / 10)
        self.v_length = 0

        self.V2V_data_rate = np.zeros([self.N_Agents])
        self.platoon_V2V_Interference_db = np.zeros([self.N_Agents])
        self.V2V_powers = np.zeros([self.N_Agents])
        self.V2V_RBs = np.zeros([self.N_Agents])

    def seed(self, seed_val):
        np.random.seed(seed_val)

    def add_new_platoon(self, start_position, start_direction, start_velocity, size_platoon):
        for i in range(size_platoon):
            self.vehicles.append(Vehicle([start_position[0], start_position[1] - i * (self.gap + self.safety_dist)],
                                          start_direction, start_velocity))

    def add_new_platoon_by_number(self, size_platoon, shadowing_dist):
        '''
        it is important to mention that the initial starting points of the platoons
        shall not affect the overall system performance.

        :param shadowing_dist:  disance between the vehicles
        :param size_platoon:    platoon size
        :return:
        '''
        ind = np.random.randint(len(self.lanes))
        start_position = [self.lanes[ind], np.random.randint(450, self.width/10)]  # position of platoon leader
        self.add_new_platoon(start_position, self.road_labels[ind], self.velocity, size_platoon)

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.delta_distance = shadowing_dist

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle within the platoon
        # ===============
        pass

    def renew_channel(self, v2v_dist):
        """ Renew slow fading channel """
        '''
        In calculating the Shadowing for two objects, we calculate the shadowing from the manhattan distance 
        between these two objects. 
        Manhattan Distance Formula:
        In a plane with p1 at (x1, y1) and p2 at (x2, y2), it is |x1 - x2| + |y1 - y2|.
        Shadowing @ time (n) is: S(n) = exp(-D/D_corr).*S(n-1)+sqrt{ (1-exp(-2*D/D_corr))}.*N_S(n)
        D is update distance matrix where D(i,j) is change in distance of link i to j from time n-1 to time n
        '''
        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 90 * np.identity(len(self.vehicles))
        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))

        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = \
                    self.V2Vchannels.get_path_loss(np.sum(v2v_dist[i:j]), j-i)

        # for i in range(len(self.vehicles)):
        #     for j in range(i + 1, len(self.vehicles)):
        #         self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = \
        #             self.V2Vchannels.get_shadowing(np.sum(shadow_dist[i:j]), self.V2V_Shadowing[i][j])
        #         self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = \
        #             self.V2Vchannels.get_path_loss(np.sum(v2v_dist[i:j]))

        # self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing
        self.V2V_channels_abs = self.V2V_pathloss
        self.V2V_channels_abs = self.V2V_channels_abs[1:, 1:]

    def renew_channels_fastfading(self):

        """ Renew fast fading channel """
        # scaling with sqrt(2) is meant to bring down the exponential distribution lambda to one.
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        # self.V2V_channels_with_fastfading = V2V_channels_with_fastfading
        # self.Nakagami_fast_fading = V2V_channels_with_fastfading
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))
        self.Nakagami_fast_fading = V2V_channels_with_fastfading - 10 * np.log10(
            np.random.gamma(self.Nakagami_m, 1/self.Nakagami_m, V2V_channels_with_fastfading.shape))
        # self.V2V_channels_with_fastfading = np.clip(self.V2V_channels_with_fastfading, 70, 110) #Normalization

    def Revenue_function(self, quantity, threshold, coeff):
        # SNR outage function definition in the paper
        flag = 0
        if (quantity >= threshold):
            flag = 1

        revenue = coeff * (quantity - threshold)


        return revenue, flag

    def Compute_V2V_data_rate(self, platoons_actions, Trx_interval):

        sub_selection = platoons_actions[:, 0].copy().astype('int').reshape(self.N_Agents, 1)
        power_selection = platoons_actions[:, 1].copy().reshape(self.N_Agents, 1) - 30 # dB
        self.V2V_powers = power_selection.flatten()
        self.V2V_RBs = sub_selection.flatten()
        # ------------ Compute Interference --------------------
        self.platoon_V2V_Interference = np.zeros([self.N_Agents])  # V2V interferences
        self.V2V_Interference_ = np.zeros([self.N_Agents])  # V2V interferences
        self.platoon_V2V_Signal = np.zeros([self.N_Agents])  # V2V signals
        self.V2V_Signal_ = np.zeros([self.N_Agents])  # V2V signals
        # structure of fast fading module: [Tx, Rx, RB]
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):  # rolls over the receivers
                for k in range(len(indexes)):  # rolls over the transmitters
                    # if indexes[j, 0] != indexes[k, 0] and indexes[k, 0] - indexes[j, 0] != 1:  # no self interference
                    if indexes[j, 0] != indexes[k, 0]:  # no self interference
                        self.platoon_V2V_Interference[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] -
                            self.V2V_channels_with_fastfading[indexes[k, 0], indexes[j, 0]+1, i] +
                            2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                        # self.V2V_Interference_[indexes[j, 0]] += \
                        #     10 ** ((power_selection[indexes[k, 0], 0] -
                        #             self.V2V_channels_abs[indexes[k, 0], indexes[j, 0] + 1] +
                        #             2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # computing the platoons intra-platoon signals
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                self.platoon_V2V_Signal[indexes[j, 0]] = 10 ** ((power_selection[indexes[j, 0], 0] -
                                                                  self.Nakagami_fast_fading[
                                                                      indexes[j, 0], indexes[j, 0]+1, i] +
                                                                  2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                # self.V2V_Signal_[indexes[j, 0]] = 10 ** ((power_selection[indexes[j, 0], 0] -
                #                                          self.V2V_channels_abs[indexes[j, 0], indexes[j, 0] + 1] +
                #                                          2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        SINR = np.divide(self.platoon_V2V_Signal, (self.platoon_V2V_Interference + self.sig2))
        # SINR_ = np.divide(self.V2V_Signal_, (self.V2V_Interference_ + self.sig2))
        SINR_ = 0
        V2V_Rate = np.log2(1 + SINR)
        self.V2V_data_rate = V2V_Rate.copy()
        self.intraplatoon_rate = V2V_Rate * Trx_interval * self.bandwidth / 1000
        self.platoon_V2V_Interference_db = 10 * np.log10(self.platoon_V2V_Interference.copy())
        self.platoon_V2V_Interference_db[self.platoon_V2V_Interference_db == -math.inf] = -129
        self.platoon_V2V_Interference_db[self.platoon_V2V_Interference_db == math.inf] = -129

        return V2V_Rate, self.intraplatoon_rate, SINR, SINR_

    def Outage_Reward(self, actions, success_rate, Trx_interval):

        success_rate_ = success_rate.copy()
        per_user_reward = np.zeros(self.N_Agents)
        action_temp = actions.copy()
        V_rate, intra_rate, SINR, SINR_ = self.Compute_V2V_data_rate(action_temp, Trx_interval)
        self.V2V_MATI -= Trx_interval
        SNR_th = self.threshold/(np.log(1/(1-self.out_prob)))

        for i in range(self.N_Agents):
            snr_rev, snr_flag = self.Revenue_function(SINR[i], SNR_th, 0.01)
            v2v_rev, v2v_flag = self.Revenue_function(V_rate[i], self.min_rate, 0.1)
            if snr_flag:
                self.V2V_demand[i] -= intra_rate[i]
                if self.V2V_demand[i] <= 0:
                    self.V2V_demand[i] = 0
            # per_user_reward[i] = np.tanh(snr_rev) + np.tanh(v2v_rev) - (self.V2V_demand[i] / self.V2V_demand_size)
            per_user_reward[i] = np.tanh(snr_rev) + np.tanh(v2v_rev)

        success_rate_[self.V2V_demand <= 0] = 1


        return per_user_reward, V_rate, self.V2V_demand, success_rate_, SINR

    def MATI_Reward_design_one(self, control_signals, time_idxs, selected_MATIs, mati_bound, idx):
        '''
        :epsilon: trying to avoid the 0 by 0 division
        :param control_signals: u = KX
        :param idx: ID of the vehicle for which we are trying to compute the string stability.
        :return: The time domain string stability

                                      L2_norm[u_i]
                 string_Stability = ----------------
                                     L2_norm[u_{i-1}]
        '''
        epsilon = 1e-10

        str_stable = (self.integration_mati(control_signals[idx+1], time_idxs) + epsilon) / \
                     (self.integration_mati(control_signals[idx], time_idxs) + epsilon)

        if str_stable <= 1:
            mati_reward = 1 - np.exp(0.1 * str_stable)
        else:
            mati_reward = 1 - np.exp(2 * str_stable) + 6.2839

        if mati_reward <= -47:
            mati_reward = -47
        mati_reward = mati_reward - (5/(selected_MATIs))

        return mati_reward, str_stable

    def MATI_Reward_design_two(self, control_signals, time_idxs, idx):
        '''
        :epsilon: trying to avoid the 0 by 0 division
        :param control_signals: u = KX
        :param idx: ID of the vehicle for which we are trying to compute the string stability.
        :return: The time domain string stability

                                      L2_norm[u_i]
                 string_Stability = ----------------
                                     L2_norm[u_{i-1}]
        '''
        epsilon = 1e-10
        str_stable = (self.integration_mati(control_signals[idx+1], time_idxs)) / \
                     (self.integration_mati(control_signals[0], time_idxs) + epsilon)


        return str_stable

    def integration_mati(self, y, x):

        value = np.trapz(y**2, x)
        value = value ** (1/2)

        return value

    def new_random_game(self, V2V_dist, shadow_dist):

        # make a new game
        self.vehicles = []
        self.add_new_platoon_by_number(self.size_platoon, shadow_dist)
        self.renew_channel(V2V_dist)  # V2V_dist = shadow dist ---> in the beginning
        self.renew_channels_fastfading()
        self.V2V_demand = self.V2V_demand_size * np.ones(int(self.size_platoon-2), dtype=np.float16)
        self.V2V_MATI = self.MATI_bound * np.ones(int(self.size_platoon-2), dtype=int)
