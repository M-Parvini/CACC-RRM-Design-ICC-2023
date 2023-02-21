import numpy as np
import scipy.special as sc
from scipy.optimize import linear_sum_assignment
import os
import torch
from Utils import soft_update
'''
Some functions to handle the chore tasks
'''


def Clipping(val):

    clipped = np.clip(val, -0.999, 0.999)
    return clipped


def get_outage_state(env, idx):
    """ Get states related to the outage from the environment """
    """
    Normalization:
    large scale fading is around -95dB to -70dB; for that we run the environment and looked at the histogram plot.
    Small scale fading is around -110dB to -70dB; --> histogram plot
    """

    MIN = -130
    Rng = 100

    V2V_abs = (-env.V2V_channels_abs[idx, idx + 1] - MIN)/(Rng)

    V2V_fast = (-env.V2V_channels_with_fastfading[idx, idx + 1, :] - MIN)/(Rng)

    Remaining_mati = env.V2V_MATI[idx]/env.MATI_bound

    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])

    V2V_interference = np.asarray([(env.platoon_V2V_Interference_db[idx] - MIN)/(Rng)])

    return np.concatenate((np.reshape(V2V_abs, -1), np.reshape(V2V_fast, -1), V2V_interference, V2V_load_remaining,
                           np.reshape(Remaining_mati, -1)), axis=0)

# def get_outage_state(env, idx):
#     """ Get states related to the outage from the environment """
#     """
#     Normalization:
#     large scale fading is around -95dB to -70dB; for that we run the environment and looked at the histogram plot.
#     Small scale fading is around -110dB to -70dB; --> histogram plot
#     """
#
#     MIN = -130
#     Rng = 100
#
#     V2V_fast = (-env.V2V_channels_with_fastfading[idx, idx + 1, :] - MIN)/(Rng)
#
#     # V2V_interference = np.asarray([(env.platoon_V2V_Interference_db[idx] - MIN)/(Rng)])
#
#     # return np.concatenate((np.reshape(V2V_fast, -1), V2V_interference), axis=0)
#     return np.reshape(V2V_fast, -1)

def get_mati_state(vals, n_veh, acceleration, SS, idx):
    '''Get states related to the mati from the environment'''
    control_error = abs(vals[-n_veh:][idx])
    state_size = 4
    user_state = vals[idx*state_size:(idx+1)*state_size]
    user_state[1] = user_state[1]/38  # speed value normalization
    user_total_state = np.block([user_state, control_error, acceleration[idx], SS[idx]])
    # return np.array([control_error, MATIs[idx]/(sim_time*1000), time_/(sim_time*1000), s_rate[idx]])
    return abs(user_total_state)

def V2V_Dist(N_veh, y_ini, headwayVals, d_r0):

    Ini_V2V_dist = y_ini[4:-N_veh]  #
    Ini_V2V_dist = Ini_V2V_dist[np.array(range(N_veh)) * 4 + 1] * headwayVals + d_r0
    distance = Ini_V2V_dist  # is the change in distance of link i to j (necessary to calculate the shadowing, D_ij)
    return distance

def create_path(objs, directory):
    paths = []
    for i in range(len(objs)):
        paths.append(os.mkdir(os.path.join(directory, "model/" + objs[i])))

    return paths

def action_wrapper(element):
    return 10*np.log10((element/1e-3))

def action_unwrapper(act, max_power):
    '''
    :param act: NN output value
    :param max_power: maximum power
    :return: scaled version of the action between 1 and 30
    '''
    return np.clip(((act + 1) / 2) * max_power, 0, max_power)
    # return np.round(np.clip(((act + 1) / 2) * max_power, 0, max_power))

def FD_merge(nets, n_agents, taus, omega=0.5):
    # only one task == outage::0
    # full update :: Also known as Federated Averaging
    # actor
    with torch.no_grad():
        omega_ = (1-omega)/(n_agents-1)
        for agent_no in range(n_agents):
            agent_state_dict = dict(nets[agent_no][0].actor.named_parameters())
            for name in agent_state_dict:
                agent_state_dict[name] = omega * agent_state_dict[name].clone()
                for idx in range(n_agents):
                    idx_state_vals = dict(nets[idx][0].actor.named_parameters())
                    if agent_no != idx:
                        agent_state_dict[name] += omega_ * idx_state_vals[name].clone()
            soft_update(nets[agent_no][0].actor, nets[agent_no][0].target_actor, taus[1])

    # critic
    with torch.no_grad():
        omega_ = (1-omega)/(n_agents-1)
        for agent_no in range(n_agents):
            agent_state_dict = dict(nets[agent_no][0].critic.named_parameters())
            for name in agent_state_dict:
                agent_state_dict[name] = omega * agent_state_dict[name].clone()
                for idx in range(n_agents):
                    idx_state_vals = dict(nets[idx][0].critic.named_parameters())
                    if agent_no != idx:
                        agent_state_dict[name] += omega_ * idx_state_vals[name].clone()
            soft_update(nets[agent_no][0].critic, nets[agent_no][0].target_critic, taus[0])

def update_network_parameters(self, tau=None):
    actor_params = self.actor.named_parameters()
    critic_params = self.critic.named_parameters()
    target_actor_params = self.target_actor.named_parameters()
    target_critic_params = self.target_critic.named_parameters()

    critic_state_dict = dict(critic_params)
    actor_state_dict = dict(actor_params)
    target_critic_state_dict = dict(target_critic_params)
    target_actor_state_dict = dict(target_actor_params)

    for name in critic_state_dict:
        critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                  (1 - tau) * target_critic_state_dict[name].clone()

    for name in actor_state_dict:
        actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                 (1 - tau) * target_actor_state_dict[name].clone()

    self.target_critic.load_state_dict(critic_state_dict)
    self.target_actor.load_state_dict(actor_state_dict)
    # self.target_critic.load_state_dict(critic_state_dict, strict=False)
    # self.target_actor.load_state_dict(actor_state_dict, strict=False)


def convex_opt_solution(Env, fairness_opt):
    '''
    An extension to the optimization solution proposed by Le Liang;
        "Resource allocation for D2D-enabled Vehicular Communications; IEEE TCOM, 2017"
    '''
    v2v_channels = Env.V2V_pathloss[1:, 1:]
    n_links = Env.N_Agents
    max_power = Env.max_power
    fairness = fairness_opt
    bsAntGain = 8
    bsNoiseFigure = 5
    vehAntGain = 3
    vehNoiseFigure = 9
    gamma0 = Env.threshold
    p0 = Env.out_prob
    r0 = Env.min_rate
    sig2 = 10 ** (-144 / 10)
    Pd_max = Pc_max = P_max = 10 ** ((max_power - 30)/10)
    C_UEs = int(n_links / 2) # dividing the UEs into two equal groups
    D_UEs  = int(n_links - C_UEs)
    fade_margin = gamma0 / (np.log(1 / (1 - p0)))
    '''
    That is literally the drawback of the proposed solution by Le liang. Since the calculations were done for a single
    C_UE - D_UE pair, the simulations were designed for fulfilling the assumptions. Of course, the standardization is
    supporting Le's claim; hence no critiques to his solution.
    For this, we assume the first half of our links are cellular, and the rest are the D2D users to align with the 
    framework. The notations are changed likewise.
    For better understanding, refer to [https://github.com/le-liang/ResourceAllocationV2X] for more information.
    '''
    powers = []
    Power_Pd = np.zeros([C_UEs, D_UEs])
    Power_Pc = np.zeros([C_UEs, D_UEs])
    T_capacity = np.zeros([C_UEs, D_UEs])
    T_capacity1 = np.zeros([C_UEs, D_UEs])
    T_capacity2 = np.zeros([C_UEs, D_UEs])
    alpha_mk_ = v2v_channels[0:C_UEs, -C_UEs:] # interference from from vehicles to the last receivers
    alpha_k_ = np.zeros(D_UEs)
    alpha_mB_ = np.zeros(C_UEs)
    alpha_kB_ = v2v_channels[C_UEs:-1, 1:C_UEs+1]
    # finding the alpha_k_ and alpha_mB_ is a little tricky
    pointer_1 = 0
    pointer_2 = 1
    cntr = 0
    while cntr < C_UEs:
        alpha_mB_[cntr] = v2v_channels[pointer_1, pointer_2]
        pointer_1 += 1
        pointer_2 += 1
        cntr += 1

    cntr = 0
    while cntr < C_UEs:
        alpha_k_[cntr] = v2v_channels[pointer_1, pointer_2]
        pointer_1 += 1
        pointer_2 += 1
        cntr += 1

    for m in range(C_UEs):
        alpha_mB = 10**((-alpha_mB_[m]+ 2*vehAntGain-vehNoiseFigure)/10)
        for k in range(D_UEs):
            alpha_k = 10**((-alpha_k_[k]+ 2*vehAntGain-vehNoiseFigure)/10)
            alpha_mk = 10**((-alpha_mk_[m, k]+ 2*vehAntGain-vehNoiseFigure)/10)
            alpha_kB = 10**((-alpha_kB_[k, m]+ 2*vehAntGain-vehNoiseFigure)/10)
            # Feasibility check
            Feasibility_condition = ((alpha_mk*alpha_kB)/(alpha_mB*alpha_k)-((p0/(1-p0))/gamma0)**2)<=0.0
            # Feasibility_condition = True
            if Feasibility_condition:
                # print('Problem is feasible')
                # first calculation: V2V link with Cellular interference
                Pc_dmax = alpha_k * P_max / (gamma0 * alpha_mk) * (
                            np.exp(-gamma0 * sig2 / (Pd_max * alpha_k)) / (1 - p0) - 1)
                if Pc_dmax <= P_max:
                    Pd1_opt = P_max
                    Pc1_opt = Pc_dmax
                else:
                    # Bisection search to find Pd_cmax
                    epsi = 1e-5
                    Pd_left = -gamma0 * sig2 / (alpha_k * np.log(1 - p0))
                    Pd_right = Pd_max
                    tmpVeh = 0
                    while Pd_right - Pd_left > epsi:
                        tmpVeh = (Pd_left + Pd_right) / 2
                        if alpha_k * tmpVeh / (gamma0 * alpha_mk) * (
                                np.exp(-gamma0 * sig2 / (tmpVeh * alpha_k)) / (1 - p0) - 1) > P_max:
                            Pd_right = tmpVeh
                        else:
                            Pd_left = tmpVeh

                    Pd_cmax = tmpVeh
                    Pd1_opt = Pd_cmax
                    Pc1_opt = P_max

                # second calculation: Cellular link with V2V interference
                Pc_dmax = alpha_mB * Pd_max / (gamma0 * alpha_kB) * (
                        np.exp(-gamma0 * sig2 / (Pd_max * alpha_mB)) / (1 - p0) - 1)
                if Pc_dmax <= Pc_max:
                    Pc2_opt = Pd_max
                    Pd2_opt = Pc_dmax
                else:
                    # Bisection search to find Pd_cmax
                    epsi = 1e-5
                    Pd_left = -gamma0 * sig2 / (alpha_mB * np.log(1 - p0))
                    Pd_right = Pd_max
                    tmpVeh = 0
                    while Pd_right - Pd_left > epsi:
                        tmpVeh = (Pd_left + Pd_right) / 2
                        if alpha_mB * tmpVeh / (gamma0 * alpha_kB) * (
                                np.exp(-gamma0 * sig2 / (tmpVeh * alpha_mB)) / (1 - p0) - 1) > Pc_max:
                            Pd_right = tmpVeh
                        else:
                            Pd_left = tmpVeh

                    Pd_cmax = tmpVeh
                    Pc2_opt = Pd_cmax
                    Pd2_opt = Pc_max

                if Pd1_opt == Pd2_opt:
                    Pd_opt = Pd1_opt
                    pc_vals = np.sort(np.array([Pc1_opt, Pc2_opt]))
                    Pc_opt = (((Pd_opt * alpha_k)/fade_margin)-sig2)/alpha_mk
                    Pc_opt = np.floor(Pc_opt*100)/100
                    # Pc_opt = cp.Variable()
                    # objective = cp.Maximize(cp.log(1 + (Pc_opt * alpha_mB) / (Pd_opt * alpha_kB + sig2)))
                    # constraints = [Pc_opt >= pc_vals[0], Pc_opt <= pc_vals[1],
                    #                (Pd_opt * alpha_k) / (Pc_opt * alpha_mk + sig2) >= fade_margin]
                    # prob = cp.Problem(objective, constraints)
                    # result = prob.solve(qcp=True)
                    # Pc_opt = Pc_opt.value

                elif Pc2_opt == Pc1_opt:
                    Pc_opt = Pc1_opt
                    pc_vals = np.sort(np.array([Pc1_opt, Pc2_opt]))
                    Pd_opt = (((Pc_opt * alpha_mB)/fade_margin)-sig2)/alpha_kB
                    Pd_opt = np.floor(Pd_opt * 100) / 100
                else:
                    Pd_opt = P_max
                    Pc_opt = P_max

            else: # infeasible
                Pd_opt = 0.0  # dummy very small number to seize the transmission
                Pc_opt = 0.0

            Power_Pd[m, k] = Pd_opt
            Power_Pc[m, k] = Pc_opt

            # T_capacity1[m, k] = np.log2(1 + (Pc_opt * alpha_mB) / (Pd_opt * alpha_kB + sig2))
            # T_capacity2[m, k] = np.log2(1 + (Pd_opt * alpha_k) / (Pc_opt * alpha_mk + sig2))
            # if T_capacity1[m, k] >= r0 and T_capacity2[m, k] >= r0:
            #     T_capacity[m, k] = min(T_capacity1[m, k], T_capacity2[m, k])
            # else:
            #     T_capacity[m, k] = -2000
            # Optimal throughput calculation::Mutual throughput calculation
            # Pd over Pc
            a = Pc_opt * alpha_mB / sig2
            b = Pd_opt * alpha_kB / sig2
            if a != b:
                T_capacity1[m, k] = Compute_capacity(a, b)
            else:
                T_capacity1[m, k] = np.log2(1 + (Pc_opt * alpha_mB) / (Pd_opt * alpha_kB + sig2))
            # Pc over Pd
            a = Pd_opt * alpha_k / sig2
            b = Pc_opt * alpha_mk / sig2
            if a != b:
                T_capacity2[m, k] = Compute_capacity(a, b)
            else:
                T_capacity2[m, k] = np.log2(1 + (Pd_opt * alpha_k) / (Pc_opt * alpha_mk + sig2))
            # Validity Check:
            rate_condition = T_capacity1[m, k] >= r0 and T_capacity2[m, k] >= r0
            if rate_condition and (not fairness):
                T_capacity[m, k] = T_capacity1[m, k] + T_capacity2[m, k]
            elif rate_condition and fairness:
                T_capacity[m, k] = min(T_capacity1[m, k], T_capacity2[m, k])
            else:
                T_capacity[m, k] = -2000
    # Reuse pair matching
    capacity_eval = T_capacity.copy()
    if not fairness:
        xx, ch_assigns = linear_sum_assignment(-capacity_eval)
    else:
        ch_assigns = fair_allocation(capacity_eval)
    # xx2, ch_assigns2 = linear_sum_assignment(-T_capacity2)
    powers.append(Power_Pc)
    powers.append(Power_Pd)
    return powers, ch_assigns


def fair_allocation(capacityMat):
    M = capacityMat.shape[0]
    K = capacityMat.shape[1]
    costMat1D = np.reshape(capacityMat, M * K)
    sortVal = np.sort(costMat1D)
    minInd = 1
    maxInd = K * M
    assignment = np.ones(M)
    ch_assigns = np.ones(M)
    while (maxInd - minInd) > 1:
        mid = int(np.floor((minInd + maxInd) / 2))-1
        tmpMat = capacityMat
        for i in range(M):
            for j in range(K):
                if tmpMat[i, j] < sortVal[mid]:
                    tmpMat[i, j] = 1
                else:
                    tmpMat[i, j] = 0


        xx, ch_assigns = linear_sum_assignment(tmpMat)
        cost = tmpMat[xx, ch_assigns].sum()
        if int(cost) >= 0:
            maxInd = mid
        else:
            minInd = mid
            assignment = ch_assigns.copy()

    minCapacity = sortVal[minInd]

    return ch_assigns


def Compute_capacity(a, b):
    '''
    "Resource Allocation for D2D-Enabled Vehicular Communications"
      L Liang, GY Li, W Xu
      IEEE Transactions on Communications 65 (7), 3186 - 3197.
    '''
    output_args = 0
    # compute the CUE capacity according to the closed form expression;
    if a >= (1 / 700) and b >= (1 / 700):
        output_args = a / ((a - b) * np.log(2)) * (np.exp(1 / a) * sc.exp1(1 / a) - np.exp(1 / b) * sc.exp1(1 / b))
    elif a < (1 / 700) and b < (1 / 700):
        output_args = a / ((a - b) * np.log(2)) * (a - b)
    elif b < (1/700):
        output_args = a / ((a - b) * np.log(2)) * (np.exp(1 / a) * sc.exp1(1 / a) - b)
    elif a < (1/700):
        output_args = a / ((a - b) * np.log(2)) * (a - np.exp(1 / b) * sc.exp1(1 / b))

    return output_args