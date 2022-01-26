import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.util import abc_to_dq0

save_results = False


def plot_stored_OMG_dessca_reults(interval_x=None, interval_y=None):
    if interval_x is None:
        interval_list_x = [0, 1]   #4

    else:
        interval_list_x = interval_x

    if interval_y is None:
        interval_list_y = [-20, 220]    # testcase 4
    else:
        interval_list_y = interval_y

    folder_name = 'OMG/data/dessca_load'  # _deterministic'



    number_of_steps = '_10000steps'

    df = pd.read_pickle(folder_name + '/PI' + number_of_steps)

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


    reward_PI = df['Reward PI'][0]


    model_names = ['model_OMG_DDPG_Actor.zip',
                   'model_OMG_SEC_DDPG.zip']

    return_list_DDPG = []

    ts = 1e-4  # if ts stored: take from db

    v_d_ref = [169.7] * len(v_0_PI)
    v_d_ref0 = [0] * len(v_0_PI)

    t_test = np.arange(0, round((len(v_0_PI)) * ts, 4), ts).tolist()
    t_reward = np.arange(0, round((len(reward_PI)) * ts , 4), ts).tolist()


    df_DDPG = pd.read_pickle(folder_name + '/' + model_names[0] + number_of_steps)

    return_list_DDPG.append(round(df_DDPG['Return DDPG'][0], 7))
    #    reward_list_DDPG.append(df_DDPG['Reward DDPG'][0])

    env_hist_DDPG = df_DDPG['env_hist_DDPG']

    v_a = env_hist_DDPG[0]['lc.capacitor1.v'].tolist()
    v_b = env_hist_DDPG[0]['lc.capacitor2.v'].tolist()
    v_c = env_hist_DDPG[0]['lc.capacitor3.v'].tolist()
    phase = env_hist_DDPG[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
    v_dq0 = abc_to_dq0(np.array([v_a, v_b, v_c]), phase)
    v_d_DDPG = (v_dq0[0].tolist())
    v_q_DDPG = (v_dq0[1].tolist())
    v_0_DDPG = (v_dq0[2].tolist())


    DDPG_reward = df_DDPG['Reward DDPG'][0]

    df_DDPG_I = pd.read_pickle(folder_name + '/' + model_names[1] + number_of_steps)

    return_list_DDPG.append(round(df_DDPG_I['Return DDPG'][0], 7))

    env_hist_DDPG_I = df_DDPG_I['env_hist_DDPG']

    reward_cut_PI_list =[]
    reward_cut_DDPG_list = []
    reward_cut_DDPG_I_list = []

    for i in range(20):

        if i == 0:
            interval_list_x = [0.01, 0.05 - 0.005]
        else:
            interval_list_x = [0+i*0.05+0.01, 0.05+i*0.05-0.005]

        v_a_I = env_hist_DDPG_I[0]['lc.capacitor1.v'].tolist()
        v_b_I = env_hist_DDPG_I[0]['lc.capacitor2.v'].tolist()
        v_c_I = env_hist_DDPG_I[0]['lc.capacitor3.v'].tolist()
        phase_I = env_hist_DDPG_I[0]['inverter1.phase.0'].tolist()  # env_test.env.net.components[0].phase
        v_dq0_I = abc_to_dq0(np.array([v_a_I, v_b_I, v_c_I]), phase_I)
        v_d_DDPG_I = (v_dq0_I[0].tolist())
        v_q_DDPG_I = (v_dq0_I[1].tolist())
        v_0_DDPG_I = (v_dq0_I[2].tolist())

        DDPG_reward_I = df_DDPG_I['Reward DDPG'][0]

        """
        plt.plot(t_test, R_load_PI, 'g')
        plt.grid()
        plt.xlim([0, 1])
        plt.ylabel('$R_\mathrm{load}\,/\,\mathrm{\Omega}$')
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.show()
        """

        fig, axs = plt.subplots(2, 1)
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

        axs[1].plot(t_test, R_load_PI, 'g')
        axs[1].grid()
        axs[1].set_xlim(interval_list_x)
        # axs[1].set_ylim(interval_list_y)
        axs[1].set_xlabel(r'$t\,/\,\mathrm{s}$')
        axs[1].set_ylabel('$R_\mathrm{load}\,/\,\mathrm{\Omega}$')
        axs[1].tick_params(direction='in')
        fig.subplots_adjust(wspace=0, hspace=0.05)

        plt.show()

        if save_results:
            fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.pgf')
            fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.png')
            fig.savefig(f'{folder_name}/OMG_DDPG_SECDDPG_PI_compare4.pdf')

        reward_cut_PI = round(sum(reward_PI[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)
        reward_cut_DDPG = round(sum(DDPG_reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)
        reward_cut_DDPG_I = round(sum(DDPG_reward_I[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)
        plt.plot(t_reward, reward_PI, 'b', label=f'          PI: '
                                                 f'{reward_cut_PI}')
        plt.plot(t_reward, DDPG_reward, 'r', label=f'    DDPG: '
                                                   f'{reward_cut_DDPG}')
        plt.plot(t_reward, DDPG_reward_I, 'g', label=f'SEC-DDPG: '
                                                     f'{reward_cut_DDPG_I}')

        plt.grid()
        plt.xlim(interval_list_x)
        # axs[1, i].set_ylim(interval_list_y[i])
        plt.legend()
        plt.xlabel(r'$t\,/\,\mathrm{s}$')

        plt.ylabel("Reward")
        plt.show()
        asd = 0

        reward_cut_PI_list.append(reward_cut_PI)
        reward_cut_DDPG_list.append(reward_cut_DDPG)
        reward_cut_DDPG_I_list.append(reward_cut_DDPG_I)

    return [reward_cut_PI_list, reward_cut_DDPG_list, reward_cut_DDPG_I_list]

result = plot_stored_OMG_dessca_reults()

results = {
        'PI': result[0],
        'DDPG': result[1],
        'SEC': result[2],
    }

df = pd.DataFrame(results)
df.to_pickle("OMG_dessca_load.pkl")
