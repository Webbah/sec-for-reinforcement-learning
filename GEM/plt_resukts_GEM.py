import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_stored_GEM_reults(interval_x=None, interval_y=None):
    if interval_x is None:
        interval_list_x = [0.498, 0.505]
    else:
        interval_list_x = interval_x

    if interval_y is None:
        interval_list_y = [80, 345]
    else:
        interval_list_y = interval_y



    df_DDPG = pd.read_pickle('GEM/data/DDPG_data')
    df_DDPG_I = pd.read_pickle('GEM/data/SEC_DDPG_data')

    ts = 1e-4
    t_test = np.arange(0, len(df_DDPG['i_d_mess'][0]) * ts, ts).tolist()
    fig = plt.figure()

    reward_sec = df_DDPG_I['Reward_test'].tolist()[0]
    reward = df_DDPG['Reward_test'].tolist()[0]

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t_test, df_DDPG_I['i_q_mess'].tolist()[0], 'r', label='$\mathrm{SEC-DDPG}$')
    axs[0].plot(t_test, df_DDPG['i_q_mess'].tolist()[0], '--r', label='$\mathrm{DDPG}_\mathrm{}$')
    axs[0].plot(t_test, df_DDPG_I['i_q_ref'].tolist()[0], '--', color='gray')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlim(interval_list_x)
    axs[0].set_ylim([0.6, -0.5])
    # axs[0].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[0].tick_params(axis='x', colors='w')
    axs[0].set_ylabel("$i_{\mathrm{q}}\,/\,\mathrm{A}$")

    axs[1].plot(t_test, df_DDPG['i_d_mess'].tolist()[0], '--b', label='$\mathrm{DDPG}_\mathrm{}$')
    axs[1].plot(t_test, df_DDPG_I['i_d_mess'].tolist()[0], 'b', label='$\mathrm{DDPG}_\mathrm{I,pv}$')
    axs[1].plot(t_test, df_DDPG_I['i_d_ref'].tolist()[0], '--', color='gray')
    axs[1].grid()
    axs[1].set_xlim(interval_list_x)
    axs[1].set_ylim([-0.8, 0.05])
    axs[1].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[1].set_ylabel("$i_{\mathrm{d}}\,/\,\mathrm{A}$")
    plt.show()

    plt.plot(t_test, reward_sec, 'b', label=f'      SEC-DDPG: '
                                            f'{round(sum(reward_sec[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')

    plt.plot(t_test, reward, 'r', label=f'DDPG: '
                                        f'{round(sum(reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    plt.grid()
    plt.xlim(interval_list_x)
    plt.legend()
    plt.ylabel("Reward")
    plt.show()


#  plot_stored_GEM_reults()