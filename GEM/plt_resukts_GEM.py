import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_results = False

def plot_stored_GEM_reults(interval_x=None, interval_y=None):
    if interval_x is None:
        interval_list_x = [0.499, 0.506]  # 1
        #interval_list_x = [0, 1]
        #interval_list_x = [0.299, 0.305] # 2
        #interval_list_x = [0.949, 0.953] # 3


        #interval_list_x = [0.049, 0.052]
    else:
        interval_list_x = interval_x

    if interval_y is None:
        interval_list_y = [80, 345]
    else:
        interval_list_y = interval_y

    folder_name = 'GEM/data'

    df_DDPG = pd.read_pickle('GEM/data/DDPG_data')
    df_DDPG_I = pd.read_pickle('GEM/data/SEC_DDPG_data')
    df_PI = pd.read_pickle('GEM/data/GEM_PI_a4.pkl')

    ts = 1e-4
    t_test = np.arange(0, len(df_DDPG['i_d_mess'][0]) * ts, ts).tolist()

    t_PI_2 = np.arange(-ts, len(df_PI['i_d_mess']) * ts-ts, ts).tolist()
    t_reward = np.arange(-ts-ts, round((len(df_DDPG['v_d_mess'][0])) * ts - ts -ts, 4), ts).tolist()


    reward_sec = df_DDPG_I['Reward_test'].tolist()[0]
    reward = df_DDPG['Reward_test'].tolist()[0]
    reward_PI = df_PI['Reward'].tolist()

    if save_results:
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
                  'figure.figsize': [5.2, 5.625],#[4.5, 7.5],
                  'font.family': 'serif',
                  'lines.linewidth': 1.2
                  }
        matplotlib.rcParams.update(params)

    fig, axs = plt.subplots(3, 1)
    axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_q_mess'].tolist()[0]], 'r', label='$\mathrm{SEC}$')
    axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG['i_q_mess'].tolist()[0]], '-.r',
                label='$\mathrm{DDPG}_\mathrm{}$')
    axs[1].plot(t_test, [i * 160 * 1.41 for i in df_PI['i_q_mess'].tolist()], '--r',
                label='$\mathrm{PI}_\mathrm{}$')
    axs[1].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_q_ref'].tolist()[0]], ':', color='gray',
                label='$\mathrm{i}_\mathrm{q}^*$', linewidth=2)
    axs[1].plot(t_test, [i * 160 * 1.41 for i in df_PI['i_q_ref'].tolist()], ':', color='gray',
                label='$\mathrm{i}_\mathrm{q}^*$', linewidth=2)
    axs[1].grid()
    # axs[1].legend()
    axs[1].set_xlim(interval_list_x)
    axs[1].set_ylim([-0.5 * 160 * 1.41, 0.55 * 160 * 1.41]) # 1
    #axs[1].set_ylim([-0 * 160 * 1.41, 0.4 * 160 * 1.41]) # 2
    #axs[1].set_ylim([0.37 * 160 * 1.41, 0.52 * 160 * 1.41]) # 3
    # axs[0].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[1].tick_params(axis='x', colors='w')
    axs[1].set_ylabel("$i_{\mathrm{q}}\,/\,{\mathrm{A}}$")
    axs[1].tick_params(direction='in')

    axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_d_mess'].tolist()[0]], 'b',
                label='$\mathrm{SEC}_\mathrm{}$')
    axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG['i_d_mess'].tolist()[0]], '-.b',
                label='$\mathrm{DDPG}_\mathrm{}$')
    axs[0].plot(t_test, [i * 160 * 1.41 for i in df_PI['i_d_mess'].tolist()], '--b',
                label='$\mathrm{PI}_\mathrm{}$')
    axs[0].plot(t_test, [i * 160 * 1.41 for i in df_DDPG_I['i_d_ref'].tolist()[0]], ':', color='gray',
                label='$i_\mathrm{}^*$', linewidth=2)
    axs[0].grid()
    axs[0].legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=4)
    axs[0].set_xlim(interval_list_x)
    axs[0].set_ylim([-0.78 * 160 * 1.41, 0.05 * 160 * 1.41]) #
    axs[0].set_ylim([-0.78 * 160 * 1.41, 0.05 * 160 * 1.41]) # 1
    #axs[0].set_ylim([-0.9 * 160 * 1.41, 0.005 * 160 * 1.41]) # 2
    #axs[0].set_ylim([-1 * 160 * 1.41, -0.2 * 160 * 1.41]) # 3
    axs[0].tick_params(axis='x', colors='w')
    axs[0].set_ylabel("$i_{\mathrm{d}}\,/\,{\mathrm{A}}$")
    axs[0].tick_params(direction='in')
    fig.subplots_adjust(wspace=0, hspace=0.05)

    axs[2].plot(t_reward, [i * 200 for i in df_DDPG_I['v_q_mess'].tolist()[0]], 'r', label='$\mathrm{SEC}$')
    axs[2].plot(t_reward, [i * 200 for i in df_DDPG['v_q_mess'].tolist()[0]], '-.r',
                label='$\mathrm{DDPG}_\mathrm{}$')
    axs[2].plot(t_PI_2, [i * 200 for i in df_PI['v_q_mess'].tolist()], '--r',
                label='$\mathrm{PI}_\mathrm{}$')
    axs[2].plot(t_reward, [i * 200 for i in df_DDPG_I['v_d_mess'].tolist()[0]], 'b', label='$\mathrm{SEC}$')
    axs[2].plot(t_reward, [i * 200 for i in df_DDPG['v_d_mess'].tolist()[0]], '-.b',
                label='$\mathrm{DDPG}_\mathrm{}$')
    axs[2].plot(t_PI_2, [i * 200 for i in df_PI['v_d_mess'].tolist()], '--b',
                label='$\mathrm{PI}_\mathrm{}$')
    #axs[2].plot(t_reward, df_DDPG_I['v_q_mess'].tolist()[0], 'r', label='$\mathrm{SEC}$')
    #axs[2].plot(t_reward, df_DDPG['v_q_mess'].tolist()[0], '-.r',
    #            label='$\mathrm{DDPG}_\mathrm{}$')
    #axs[2].plot(t_reward, df_PI['v_q_mess'].tolist(), '--r',
    #            label='$\mathrm{PI}_\mathrm{}$')
   # axs[2].plot(t_reward, df_DDPG_I['v_d_mess'].tolist()[0], 'b', label='$\mathrm{SEC}$')
   # axs[2].plot(t_reward, df_DDPG['v_d_mess'].tolist()[0], '--b', label='$\mathrm{DDPG}_\mathrm{}$')
   # axs[2].plot(t_PI_3, df_PI['v_d_mess'].tolist(), '--b', label='$\mathrm{PI}_\mathrm{}$')
    axs[2].grid()
    # axs[1].legend()
    axs[2].set_xlim(interval_list_x)
    #axs[2].set_ylim([-100, 100])
    # axs[0].set_xlabel(r'$t\,/\,\mathrm{s}$')
    #axs[2].set_xlabel(r'$t\,/\,\mathrm{s}$')
    #axs[2].tick_params(axis='x', colors='w')
    axs[2].set_xlabel(r'$t\,/\,\mathrm{s}$')
    axs[2].set_ylabel("$v_{\mathrm{dq}}\,/\,{\mathrm{V}}$")
    #axs[2].set_ylabel("$u_{\mathrm{dq}}\,/\, v_\mathrm{DC}\,/\,2$")
    axs[2].tick_params(direction='in')

    """
    axs[3].plot(t_test, reward_sec, 'b', label=f'      SEC-DDPG: '
                                            f'{round(sum(reward_sec[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')

    axs[3].plot(t_test, reward, 'r', label=f'DDPG: '
                                        f'{round(sum(reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    axs[3].plot(t_PI_2, reward_PI, '--r', label=f'PI: '
                                           f'{round(sum(reward_PI[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    axs[3].grid()
    axs[3].set_xlim(interval_list_x)
    #axs[3].legend()
    axs[3].set_ylabel("Reward")
    plt.show()
    """

    plt.show()

    if save_results:
        fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq1.pgf')
        fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq1.png')
        fig.savefig(f'{folder_name}/GEM_DDPG_I_noI_idq1.pdf')

    plt.plot(t_test, reward_sec, 'b', label=f'      SEC-DDPG: '
                                            f'{round(sum(reward_sec[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')

    plt.plot(t_test, reward, 'r', label=f'DDPG: '
                                        f'{round(sum(reward[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    plt.plot(t_test, reward_PI, '--r', label=f'PI: '
                                        f'{round(sum(reward_PI[int(interval_list_x[0] / ts):int(interval_list_x[1] / ts)]) / ((interval_list_x[1] - interval_list_x[0]) / ts), 4)}')
    plt.grid()
    plt.xlim(interval_list_x)
    plt.legend()
    plt.ylabel("Reward")
    plt.show()



plot_stored_GEM_reults()