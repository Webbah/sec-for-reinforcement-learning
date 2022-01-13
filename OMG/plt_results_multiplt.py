import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.util import abc_to_dq0

save_results = True


def plot_stored_OMG_reults(interval_x=None, interval_y=None):
    if interval_x is None:
        interval_list_x = [7.1465, 7.15]    # testcase 1
        #interval_list_x = [7.1465, 7.1475]    # testcase 1
        #interval_list_x = [0, 0.0005]    # testcase 1
        interval_list_x = [1.145, 1.16]     #2
        interval_list_x = [2.5245,2.5275]   #3
        interval_list_x = [9.5072,9.509]   #4

    else:
        interval_list_x = interval_x

    if interval_y is None:
        interval_list_y = [80, 345]    # testcase 1
        interval_list_y = [-15, 205]    # testcase 2
        interval_list_y = [-30, 300]    # testcase 3
        interval_list_y = [-20, 220]    # testcase 4
    else:
        interval_list_y = interval_y

    folder_name = 'OMG/data'  # _deterministic'

    v_dc = 300

    number_of_steps = '_100000steps'

    df = pd.read_pickle(folder_name + '/new_pkls/PI' + number_of_steps)

    env_hist_PI = df['env_hist_PI']
    v_a_PI = env_hist_PI[0]['lc.capacitor1.v'].tolist()
    v_b_PI = env_hist_PI[0]['lc.capacitor2.v'].tolist()
    v_c_PI = env_hist_PI[0]['lc.capacitor3.v'].tolist()
    R_load_PI = (env_hist_PI[0]['r_load.resistor1.R'].tolist())
    phase_PI = env_hist_PI[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0_PI = abc_to_dq0(np.array([v_a_PI, v_b_PI, v_c_PI]), phase_PI)
    v_d_PI = (v_dq0_PI[0].tolist())
    v_q_PI = (v_dq0_PI[1].tolist())
    v_0_PI = (v_dq0_PI[2].tolist())

    i_a_PI = env_hist_PI[0]['lc.inductor1.i'].tolist()
    i_b_PI = env_hist_PI[0]['lc.inductor2.i'].tolist()
    i_c_PI = env_hist_PI[0]['lc.inductor3.i'].tolist()
    i_dq0_PI = abc_to_dq0(np.array([i_a_PI, i_b_PI, i_c_PI]), phase_PI)
    i_d_PI = (i_dq0_PI[0].tolist())
    i_q_PI = (i_dq0_PI[1].tolist())
    i_0_PI = (i_dq0_PI[2].tolist())

    reward_PI = df['Reward PI'][0]
    return_PI = df['Return PI'][0]
    kp_c = df['PI_Kp_c'][0]
    ki_c = df['PI_Ki_c'][0]
    kp_v = df['PI_Kp_v'][0]
    ki_v = df['PI_Ki_v'][0]

    u_dq0_PI = abc_to_dq0(np.array([df['PI_u_d'][0], df['PI_u_q'][0], df['PI_u_0'][0]]), phase_PI[1:])

    if v_dc is not None:
        u_d_PI = [sum(x*v_dc) for x in zip(u_dq0_PI[0].tolist())]
        u_q_PI = [sum(x * v_dc) for x in zip(u_dq0_PI[1].tolist())]
        u_0_PI = [sum(x * v_dc) for x in zip(u_dq0_PI[2].tolist())]
    else:
        u_d_PI = u_dq0_PI[0].tolist()
        u_q_PI = u_dq0_PI[1].tolist()
        u_0_PI = u_dq0_PI[2].tolist()

    model_names = ['model_OMG_DDPG_Actor.zip',
                   'model_OMG_SEC_DDPG.zip']
    ylabels = ['DDPG', 'DDPG-I']

    return_list_DDPG = []
    reward_list_DDPG = []

    ts = 1e-4  # if ts stored: take from db

    v_d_ref = [169.7] * len(v_0_PI)
    v_d_ref0 = [0] * len(v_0_PI)

    t_test = np.arange(0, round((len(v_0_PI)) * ts, 4), ts).tolist()
    #t_reward = np.arange(-ts, round((len(reward_PI)) * ts - ts, 4), ts).tolist()
    t_reward = np.arange(0, round((len(reward_PI)) * ts , 4), ts).tolist()


    # fig, axs = plt.subplots(len(model_names) + 4, len(interval_list_y),
    fig = plt.figure()

    ############## Subplots
    # fig = plt.figure(figsize=(10,12))  # a new figure window

    df_DDPG = pd.read_pickle(folder_name + '/' + model_names[0] + number_of_steps)

    return_list_DDPG.append(round(df_DDPG['Return DDPG'][0], 7))
    #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

    env_hist_DDPG = df_DDPG['env_hist_DDPG']

    v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
    v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
    v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
    i_a = env_hist_DDPG[0]['lc.inductor1.i'].tolist()
    i_b = env_hist_DDPG[0]['lc.inductor2.i'].tolist()
    i_c = env_hist_DDPG[0]['lc.inductor3.i'].tolist()
    phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
    i_dq0 = abc_to_dq0(np.array([i_a, i_b, i_c]), phase)
    v_d_DDPG = (v_dq0[0].tolist())
    v_q_DDPG = (v_dq0[1].tolist())
    v_0_DDPG = (v_dq0[2].tolist())
    i_d_DDPG = (i_dq0[0].tolist())
    i_q_DDPG = (i_dq0[1].tolist())
    i_0_DDPG = (i_dq0[2].tolist())

    if v_dc is not None:
        u_d_DDPG = [sum(x*v_dc) for x in zip(df_DDPG['ActionP0'][0])]
        u_q_DDPG = [sum(x * v_dc) for x in zip(df_DDPG['ActionP1'][0])]
        u_0_DDPG = [sum(x * v_dc) for x in zip(df_DDPG['ActionP2'][0])]
    else:
        u_d_DDPG = df_DDPG['ActionP0'][0]
        u_q_DDPG = df_DDPG['ActionP1'][0]
        u_0_DDPG = df_DDPG['ActionP2'][0]

    DDPG_reward = df_DDPG['Reward DDPG'][0]

    df_DDPG_I = pd.read_pickle(folder_name + '/' + model_names[1] + number_of_steps)

    return_list_DDPG.append(round(df_DDPG_I['Return DDPG'][0], 7))
    #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

    env_hist_DDPG_I = df_DDPG_I['env_hist_DDPG']

    v_a_I = env_hist_DDPG_I[0]['lc.capacitor1.v'].tolist()
    v_b_I = env_hist_DDPG_I[0]['lc.capacitor2.v'].tolist()
    v_c_I = env_hist_DDPG_I[0]['lc.capacitor3.v'].tolist()
    i_a_I = env_hist_DDPG_I[0]['lc.inductor1.i'].tolist()
    i_b_I = env_hist_DDPG_I[0]['lc.inductor2.i'].tolist()
    i_c_I = env_hist_DDPG_I[0]['lc.inductor3.i'].tolist()
    phase_I = env_hist_DDPG_I[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0_I = abc_to_dq0(np.array([v_a_I, v_b_I, v_c_I]), phase_I)
    i_dq0_I = abc_to_dq0(np.array([i_a_I, i_b_I, i_c_I]), phase_I)
    v_d_DDPG_I = (v_dq0_I[0].tolist())
    v_q_DDPG_I = (v_dq0_I[1].tolist())
    v_0_DDPG_I = (v_dq0_I[2].tolist())
    i_d_DDPG_I = (i_dq0_I[0].tolist())
    i_q_DDPG_I = (i_dq0_I[1].tolist())
    i_0_DDPG_I = (i_dq0_I[2].tolist())

    #u_dI_DDPG_I = df_DDPG_I['ActionI0'][0]
    #u_dP_DDPG_I = df_DDPG_I['ActionP0']

    u_d_DDPG_I = [sum(x) for x in zip(df_DDPG_I['integrator_sum0'][0], df_DDPG_I['ActionP0'][0])]
    u_q_DDPG_I = [sum(x) for x in zip(df_DDPG_I['integrator_sum1'][0], df_DDPG_I['ActionP1'][0])]
    u_0_DDPG_I = [sum(x) for x in zip(df_DDPG_I['integrator_sum2'][0], df_DDPG_I['ActionP2'][0])]

    if v_dc is not None:
        u_d_DDPG_I = [sum(x*300) for x in zip(u_d_DDPG_I)]
        u_q_DDPG_I = [sum(x*300) for x in zip(u_q_DDPG_I)]
        u_0_DDPG_I = [sum(x*300) for x in zip(u_0_DDPG_I)]

    DDPG_reward_I = df_DDPG_I['Reward DDPG'][0]


    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t_test, R_load_PI, 'g')
    axs[0].grid()
    axs[0].tick_params(axis='x', colors='w')
    axs[0].set_xlim([0, 10])
    axs[0].set_ylabel('$R_\mathrm{load}\,/\,\mathrm{\Omega}$')
    # axs[0].setxlabel(r'$t\,/\,\mathrm{s}$')

    axs[1].plot(t_test, v_d_PI, 'b', label='PI')
    axs[1].plot(t_test, v_q_PI, 'r')
    axs[1].plot(t_test, v_0_PI, 'g')
    axs[1].grid()
    axs[1].legend()
    axs[1].tick_params(axis='x', colors='w')
    axs[1].set_xlim([0, 10])
    axs[1].set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
    # axs[1].setxlabel(r'$t\,/\,\mathrm{s}$')

    axs[2].plot(t_test, v_d_DDPG_I, 'b', label='$\mathrm{SEC-DDPG}_\mathrm{}$')
    axs[2].plot(t_test, v_q_DDPG_I, 'r')
    axs[2].plot(t_test, v_0_DDPG_I, 'g')
    axs[2].grid()
    axs[2].legend()
    axs[2].set_xlim([0, 10])
    axs[2].set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
    axs[2].set_xlabel(r'$t\,/\,\mathrm{s}$')

    axs[3].plot(t_test, v_d_DDPG, 'b', label='$\mathrm{DDPG}_\mathrm{}$')
    axs[3].plot(t_test, v_q_DDPG, 'r')
    axs[3].plot(t_test, v_0_DDPG, 'g')
    axs[3].grid()
    axs[3].legend()
    axs[3].set_xlim([0, 10])
    axs[3].set_ylabel('$v_{\mathrm{dq0}}\,/\,\mathrm{V}$')
    axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')

    plt.show()

    fig = plt.figure()  # figsize =(6, 5))
    plt.plot(t_test, R_load_PI, 'g')
    plt.grid()
    plt.xlim([0, 10])
    plt.ylabel('$R_\mathrm{load}\,/\,\mathrm{\Omega}$')
    plt.xlabel(r'$t\,/\,\mathrm{s}$')
    plt.show()

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'
                                      r'\usepackage{amsmath,amssymb,mathtools}'
                                      r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                      r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
              'axes.labelsize': 12.5,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 12.5,
              'font.size': 12.5,  # was 10
              'legend.fontsize': 12.5,  # was 10
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': [4.5, 7.5],
              'font.family': 'serif',
              'lines.linewidth': 1.2
              }
    matplotlib.rcParams.update(params)

    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t_test, v_d_DDPG_I, 'b', label='$\mathrm{SEC}$')
    axs[0].plot(t_test, v_q_DDPG_I, 'r')
    axs[0].plot(t_test, v_0_DDPG_I, 'g')
    axs[0].plot(t_test, v_d_DDPG, '-.b', label='$\mathrm{DDPG}$')
    axs[0].plot(t_test, v_q_DDPG, '-.r')
    axs[0].plot(t_test, v_0_DDPG, '-.g')
    axs[0].plot(t_test, v_d_PI, '--b', label='$\mathrm{PI}$')
    axs[0].plot(t_test, v_q_PI, '--r')
    axs[0].plot(t_test, v_0_PI, '--g')
    axs[0].plot(t_test, v_d_ref, ':', color='gray', label='$v^*$', linewidth=2)
    axs[0].plot(t_test, v_d_ref0, ':', color='gray', linewidth=2)
    axs[0].grid()
    #axs[0].legend(ncol=3)
    axs[0].legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=4)
    axs[0].set_xlim(interval_list_x)
    axs[0].set_ylim(interval_list_y)
    # axs[0].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[0].tick_params(axis='x', colors='w')
    axs[0].set_ylabel("$v_{\mathrm{dq0}}\,/\,\mathrm{V}$")
    axs[0].tick_params(direction='in')

    axs[1].plot(t_test, i_d_DDPG_I, 'b', label='$i_\mathrm{d}$')
    axs[1].plot(t_test, i_q_DDPG_I, 'r', label='$i_\mathrm{q}$')
    axs[1].plot(t_test, i_0_DDPG_I, 'g', label='$i_\mathrm{0}$')
    axs[1].plot(t_test, i_d_DDPG, '-.b')
    axs[1].plot(t_test, i_q_DDPG, '-.r')
    axs[1].plot(t_test, i_0_DDPG, '-.g')
    axs[1].plot(t_test, i_d_PI, '--b')
    axs[1].plot(t_test, i_q_PI, '--r')
    axs[1].plot(t_test, i_0_PI, '--g')
    axs[1].grid()
    axs[1].set_xlim(interval_list_x)
    axs[1].set_ylim([-2, 12]) # 2
    axs[1].set_ylim([-5, 10.5]) # 3
    axs[1].set_ylim([-2, 10.5]) # 4
    #axs[1].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[1].tick_params(axis='x', colors='w')
    axs[1].set_ylabel("$i_{\mathrm{dq0}}\,/\,\mathrm{A}$")
    axs[1].tick_params(direction='in')
    fig.subplots_adjust(wspace=0, hspace=0.05)

    axs[2].plot(t_reward, u_d_DDPG_I, 'b')
    axs[2].plot(t_reward, u_q_DDPG_I, 'r')
    axs[2].plot(t_reward, u_0_DDPG_I, 'g')
    axs[2].plot(t_reward, u_d_DDPG, '-.b')
    axs[2].plot(t_reward, u_q_DDPG, '-.r')
    axs[2].plot(t_reward, u_0_DDPG, '-.g')
    axs[2].plot(t_reward, u_d_PI, '--b')
    axs[2].plot(t_reward, u_q_PI, '--r')
    axs[2].plot(t_reward, u_0_PI, '--g')
    axs[2].grid()
    axs[2].set_xlim(interval_list_x)
    axs[2].set_ylim([-0.15*300, 0.7*300]) # for testcase 2
    axs[2].set_ylim([-0.2*300, 0.8*300]) # for testcase 3
    axs[2].set_ylim([-0.15*300, 0.7*300]) # for testcase 4
    axs[2].tick_params(axis='x', colors='w')
    #axs[2].set_ylabel("$u_{\mathrm{dq0}}\,/\, v_\mathrm{DC}\,/\,2$")
    axs[2].set_ylabel("$u_{\mathrm{}}\,/\, \mathrm{V}$")
    axs[2].tick_params(direction='in')
    fig.subplots_adjust(wspace=0, hspace=0.05)

    axs[3].plot(t_test, R_load_PI, 'g')
    axs[3].grid()
    axs[3].set_xlim(interval_list_x)
    # axs[1].set_ylim(interval_list_y)
    axs[3].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[3].set_ylabel('$R_\mathrm{load}\,/\,\mathrm{\Omega}$')
    axs[3].tick_params(direction='in')
    fig.subplots_adjust(wspace=0, hspace=0.05)

    plt.show()

    if save_results:
        fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.pgf')
        fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.png')
        fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.pdf')

    plt.plot(t_reward, reward_PI, 'b', label=f'          PI: '
                                             f'{round(sum(reward_PI[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    plt.plot(t_reward, DDPG_reward, 'r', label=f'    DDPG: '
                                               f'{round(sum(DDPG_reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    plt.plot(t_reward, DDPG_reward_I, 'g', label=f'SEC-DDPG: '
                                                 f'{round(sum(DDPG_reward_I[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')

    plt.grid()
    plt.xlim(interval_list_x)
    # axs[1, i].set_ylim(interval_list_y[i])
    plt.legend()
    plt.xlabel(r'$t\,/\,\mathrm{s}$')

    plt.ylabel("Reward")
    plt.show()

plot_stored_OMG_reults()
