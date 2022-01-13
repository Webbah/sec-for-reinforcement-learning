import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gym_electric_motor as gem
from classic_controllers import Controller
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym_electric_motor.reference_generators import ConstReferenceGenerator, MultipleReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot

if __name__ == '__main__':

    """
        motor type:     'PMSM'      Permanent Magnet Synchronous Motor
                        'SynRM'     Synchronous Reluctance Motor

        control type:   'SC'         Speed Control
                        'TC'         Torque Control
                        'CC'         Current Control

        action_type:    'AbcCont'   Continuous Action Space in ABC-Coordinates
                        'Finite'    Discrete Action Space
    """



    d_generator = ConstReferenceGenerator('i_sd', 0)
    # q current changes dynamically
    q_generator = ConstReferenceGenerator('i_sq', 0)

    # The MultipleReferenceGenerator allows to apply these references simultaneously
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    # definition of the plotted variables
    #external_ref_plots = [ExternallyReferencedStatePlot(state) for state in
    #                      ['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq']]




    #psi_p = 0 if motor_type == 'SynRM' else 45e-3
    #limit_values = dict(omega=12e3 * np.pi / 30, torque=100, i=280, u=320)
    #nominal_values = dict(omega=10e3 * np.pi / 30, torque=95.0, i=240, epsilon=np.pi, u=300)
    #motor_parameter = dict(p=3, l_d=0.37e-3, l_q=1.2e-3, j_rotor=0.03883, r_s=18e-3, psi_p=psi_p)

    motor_type = 'PMSM'
    control_type = 'CC'
    action_type = 'AbcCont'  # 'AbcCont'

    env_id = action_type + '-' + control_type + '-' + motor_type + '-v0'
    # Set the electric parameters of the motor
    motor_parameter = dict(
        r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06)

    # Change the motor operational limits (important when limit violations can terminate and reset the environment)
    limit_values = dict(
        i=160 * 1.41,
        omega=12000 * np.pi / 30,
        u=450
    )

    # Change the motor nominal values
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}
    env = gem.make(env_id,
                   visualization=MotorDashboard(
                       state_plots=['i_sq', 'i_sd'],
                       action_plots='all',
                       reward_plot=True,
                       additional_plots=[MeanEpisodeRewardPlot()]
                   ),
                   motor=dict(limit_values=limit_values, nominal_values=nominal_values,
                              motor_parameter=motor_parameter),
                   converter=dict(
                       dead_time=True,
                   ),
                   supply=dict(
                       u_nominal=400
                   ),
                   load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
                   ode_solver='scipy.solve_ivp',
                   reward_function=dict(
                       # Set weighting of different addends of the reward function
                       reward_weights={'i_sq': 1, 'i_sd': 1},
                       # Exponent of the reward function
                       # Here we use a square root function
                       reward_power=0.5,
                   ),
                   constraints=(),
                   # constraints=(SquaredConstraint(('i_sq', 'i_sd')),),
                   # reference_generator=rg,
                   )

    """
    initialize the controller

    Args:
        environment                     gym-electric-motor environment
        external_ref_plots (optional)   plots of the environment, to plot all reference values
        automated_gain (optional)       if True (default), the controller will be tune automatically
        a (optional)                    tuning parameter of the Symmetrical Optimum (default: 4)

        stages (optional)               Each controller stage is defined in a dict. The key controller_type specifies
                                        which type of controller is used for the stage.  In addition, parameters of the
                                        controller can be passed like e.g. the p-gain and i-gain for the PI controller
                                        (see example below).
                                        The stages are grouped in an array in ascending order. For environments with
                                        current control only an array with the corresponding current controllers is
                                        needed (see example below).  For a higher-level torque or speed control, these
                                        controllers are passed in an additional controller. Note that the
                                        TorqueToCurrent controller is added automatically (see example below).

                                        Examples:

                                        controller stage:
                                        d_current_controller = {'controller_type': 'pi-controller', 'p_gain': 2,
                                                                'i_gain': 30}

                                        AbcCont currrent control:
                                            stages = [d_current_controller, q_current_controller]

                                        Finite current control:
                                            stages = [a_current_controller, b_current_controller, c_current_controller]

                                        AbcCont torque control:
                                            stages = [[d_current_controller, q_current_controller]]  (no overlaid
                                            controller, because the torque to current stage is added automatically)

                                        Finite torque control:      
                                            stages = [[a_current_controller, b_current_controller, c_current_controller]]

                                        AbcCont speed control:
                                            stages = [[d_current_controller, q_current_controller], [speed_controller]]

                                        Finite speed control:      
                                            stages = [[a_current_controller, b_current_controller, c_current_controller],
                                                      [speed_controller]]

        additionally for TC or SC:
        torque_control (optional)       mode of the torque controller, 'interpolate' (default), 'analytical' or 'online'
        plot_torque(optional)           plot some graphs of the torque controller (default: True)
        plot_modulation (optional)      plot some graphs of the modulation controller (default: False)

    """

    current_d_controller = {'controller_type': 'pi_controller', 'p_gain': 1, 'i_gain': 500}
    current_q_controller = {'controller_type': 'pi_controller', 'p_gain': 3, 'i_gain': 1400}
    speed_controller = {'controller_type': 'pi_controller', 'p_gain': 12, 'i_gain': 1300}

    current_controller = [current_d_controller, current_q_controller]
    overlaid_controller = [speed_controller]

    #stages = [current_controller, overlaid_controller]



    # Refs created with https://github.com/max-schenke/DESSCA
    i_d_refs = [ -0.5718831392706399, -0.11155989917458595, -0.8444233463864655, -0.19260596846844558,
                -0.48986342384598824,
                -0.08540375784816023, -0.6983532259844449, -0.3409346664209051, -0.9852563901175903,
                -0.019589794863040133,
                -0.3057052318511703, -0.010759738176742362, -0.7264074671265837, -0.7003086456948622,
                -0.5205127876117279,
                -0.0035883351279332454, -0.24656126983332566, -0.7385108721382044, -0.8711444379999949,
                -0.5322348905850738,
                -0.16443631057073907, -0.26335305001172343, -0.8339056052207534, -0.9840272325710973,
                -0.00099042967089491,
                -0.4276376345373605, -0.4392085789117308, -0.29885945214798054, -0.3526213053117569,
                -0.15544590095444902,
                -0.38133627476871246, -0.0007362814213280888, -0.13766159578201825, -0.6998437778149555,
                -0.02941718441323049,
                -0.14911600490992516, -0.8711008909873345, -0.5803207691231205, -0.3908087722441505,
                -0.30424273624679143,
                -0.6032911651567467, -0.6097285170523984, -0.23000688296189783, -0.009050042083058152,
                -0.13450601442490417,
                -0.8117883556545268, -0.7542685229940803, -0.4627233964160423, -0.23713451030767801, -0.580302276033946]
    i_q_refs = [ -0.3392001552090831, 0.9601935188371409, -0.3536698661685236, -0.7470423329656373, 0.7498405690613185,
                0.02118430489789434, 0.2733946954263321, 0.2919040855524663, 0.16184776106212195, 0.5033515631986878,
                -0.3472813053105329, -0.3978931436350608, 0.6856579757847681, -0.7061719805667996, 0.05173569323125849,
                -0.9859275339077078, 0.6511009114276964, -0.07964009848269302, 0.4872958851075428, 0.4244964715390715,
                0.3348234680253275, -0.02175414797059596, 0.1689424266837956, -0.15367806515850901, -0.6890239130635769,
                -0.5235888504056838, -0.18887320564466648, -0.9243752447874265, 0.9223611469482904,
                -0.47288531380037824,
                0.5419042725157753, 0.21808910731016923, -0.2114136814114341, -0.43862800579799827, 0.7610593015542114,
                -0.9580202514125911, -0.058327843098379906, -0.6351863815461574, 0.06422483040085132,
                -0.6157429182475818,
                0.6283510657507491, -0.1007305747146939, 0.9225787627793309, -0.15228745162185686, 0.6513516638638627,
                -0.5835510703463308, 0.46458552243856405, 0.25269729661377704, 0.1814216788492872, 0.2111335623928367]

    count = 0

    controller = Controller.make(env, automated_gain=True)  # , stages=stages)#, stages=stages, external_ref_plots=external_ref_plots, torque_control='analytical')

    i_d = []
    i_q = []
    i_d_ref = []
    i_q_ref = []
    omega = []
    T = []
    u_sd = []
    u_sq = []
    rewards = []

    state, reference = env.reset()
    reference = np.array([i_d_refs[count], i_q_refs[count]])
    """
    i_d.append(np.float64(state[5]))
    i_q.append(np.float64(state[6]))
    i_d_ref.append(np.float64(reference[0]))
    i_q_ref.append(np.float64(reference[1]))
    omega.append(np.float64(state[0]))
    T.append(np.float64(state[1]))
    u_sd.append(np.float64(state[10]))
    u_sq.append(np.float64(state[11]))
    """

    # simulate the environment
    for i in range(10000):#range(10001):
        env.render()


        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        i=i+1
        if i % 500 == 0:# & i > 0:
            count += 1
        reference = np.array([i_d_refs[count], i_q_refs[count]])

        i_d.append(np.float64(state[5]))
        i_q.append(np.float64(state[6]))
        i_d_ref.append(np.float64(reference[0]))
        i_q_ref.append(np.float64(reference[1]))
        omega.append(np.float64(state[0]))
        T.append(np.float64(state[1]))
        u_sd.append(np.float64(state[10]))
        u_sq.append(np.float64(state[11]))

        rew = -(np.sqrt(np.abs(reference[0]-np.float64(state[5]))) +
               np.sqrt(np.abs(reference[1]-np.float64(state[6]))))/2

        rewards.append(rew)

        if done:
        #if i % 2000 == 0:
            #i_d.append(np.float64(state[5]))
            #i_q.append(np.float64(state[6]))

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(i_d)
            axs[0].plot(i_d_ref, ':', color='gray', linewidth=1.5)
            axs[0].set_ylabel("$i_{\mathrm{d}}\,/\,i_\mathrm{lim}$")
            axs[0].grid()
            axs[1].plot(i_q)
            axs[1].set_ylabel("$i_{\mathrm{q}}\,/\,i_\mathrm{lim}$")
            axs[1].plot(i_q_ref, ':', color='gray', linewidth=1.5)
            axs[1].grid()
            plt.show()

            plt.plot(omega)
            #plt.plot(ref, ':', color='gray', linewidth=1.5)
            plt.ylabel("$\omega_{\mathrm{}}\,/\,}$")
            plt.grid()
            plt.show()

            plt.plot(T)
            #plt.plot(ref, ':', color='gray', linewidth=1.5)
            plt.ylabel("$T_{\mathrm{}}\, \,}$")
            plt.grid()
            plt.show()

            plt.plot(u_sd)
            plt.plot(u_sq, '--r')
            # plt.plot(ref, ':', color='gray', linewidth=1.5)
            plt.ylabel("$u_{\mathrm{sqd}}\, \,}$")
            plt.grid()
            plt.show()

            env.reset()
            controller.reset()

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(i_d)
    axs[0].plot(i_d_ref, ':', color='gray', linewidth=1.5)
    axs[0].set_ylabel("$i_{\mathrm{d}}\,/\,i_\mathrm{lim}$")
    axs[0].grid()
    axs[1].plot(i_q)
    axs[1].set_ylabel("$i_{\mathrm{q}}\,/\,i_\mathrm{lim}$")
    axs[1].plot(i_q_ref, ':', color='gray', linewidth=1.5)
    axs[1].grid()
    plt.show()

    plt.plot(omega)
    # plt.plot(ref, ':', color='gray', linewidth=1.5)
    plt.ylabel("$\omega_{\mathrm{}}\,/\,}$")
    plt.grid()
    plt.show()

    plt.plot(T)
    # plt.plot(ref, ':', color='gray', linewidth=1.5)
    plt.ylabel("$T_{\mathrm{}}\, \,}$")
    plt.grid()
    plt.show()

    plt.plot(u_sd)
    plt.plot(u_sq, '--r')
    # plt.plot(ref, ':', color='gray', linewidth=1.5)
    plt.ylabel("$u_{\mathrm{sqd}}\, \,}$")
    plt.grid()
    plt.show()

    plt.plot(rewards)
    # plt.plot(ref, ':', color='gray', linewidth=1.5)
    plt.ylabel("$reward_{\mathrm{}}\, \,}$")
    plt.grid()
    plt.show()
    #rewards.append(rew)
    results = {            "Reward": rewards,
                           "i_d_mess": i_d,
                           "i_q_mess": i_q,
                           "v_d_mess": u_sd,
                           "v_q_mess": u_sq,
                           "i_d_ref": i_d_ref,
                           "i_q_ref": i_q_ref}

    df = pd.DataFrame(results)
    df.to_pickle("GEM_PI_a4_3.pkl")


    env.close()
