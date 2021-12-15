import json
import os
import time

import sqlalchemy
from optuna.samplers import TPESampler

os.environ['PGOPTIONS'] = '-c statement_timeout=1000'

import optuna
import platform
import argparse
import sshtunnel
import numpy as np
# np.random.seed(0)
from GEM.util.config import cfg

from GEM.experiment_GEM import mongo_recorder, experiment_fit_DDPG
from OMG.util.scheduler import linear_schedule

model_path = 'experiments/hp_tune/trained_models/study_22_run_11534/'

PC2_LOCAL_PORT2PSQL = 11999
SERVER_LOCAL_PORT2PSQL = 6432
DB_NAME = 'optuna'
PC2_LOCAL_PORT2MYSQL = 11998
SERVER_LOCAL_PORT2MYSQL = 3306
STUDY_NAME = cfg['STUDY_NAME']  # 'DDPG_MRE_sqlite_PC2'

node = platform.uname().node


def ddpg_objective_fix_params(trial):
    file_congfig = open('DDPG_best_HP_set.json', )
    trial_config = json.load(file_congfig)

    number_learning_steps = 500000

    antiwindup_weight = trial_config["antiwindup_weight"]
    alpha_relu_actor = trial_config["alpha_relu_actor"]
    alpha_relu_critic = trial_config["alpha_relu_critic"]
    actor_hidden_size = trial_config["actor_hidden_size"]
    actor_number_layers = trial_config["actor_number_layers"]
    bias_scale = trial_config["bias_scale"]
    batch_size = trial_config["batch_size"]
    buffer_size = trial_config["buffer_size"]
    critic_hidden_size = trial_config["critic_hidden_size"]
    critic_number_layers = trial_config["critic_number_layers"]
    error_exponent = 0.5
    final_lr = trial_config["final_lr"]
    gamma = trial_config["gamma"]
    integrator_weight = trial_config["integrator_weight"]
    learning_rate = trial_config["learning_rate"]
    lr_decay_start = trial_config["lr_decay_start"]
    lr_decay_duration = trial_config["lr_decay_duration"]
    use_gamma_in_rew = 1
    noise_var = trial_config["noise_var"]
    noise_var_min = 0.0013
    noise_steps_annealing = int(0.25 * number_learning_steps)
    noise_theta = trial_config["noise_theta"]  # stiffness of OU
    number_past_vals = 5
    n_trail = str(trial.number)
    optimizer = trial_config["optimizer"]
    penalty_I_weight = trial_config["penalty_I_weight"]
    penalty_P_weight = trial_config["penalty_P_weight"]
    penalty_I_decay_start = trial_config["penalty_I_decay_start"]
    penalty_P_decay_start = trial_config["penalty_P_decay_start"]
    training_episode_length = trial_config["training_episode_length"]
    tau = trial_config["tau"]
    train_freq_type = "step"
    train_freq = trial_config["train_freq"]
    t_start_penalty_I = int(penalty_I_decay_start * number_learning_steps)
    t_start_penalty_P = int(penalty_P_decay_start * number_learning_steps)
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))


    weight_scale = trial_config["weight_scale"]

    learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                    t_start=t_start,
                                    t_end=t_end,
                                    total_timesteps=number_learning_steps)

    trail_config_mongo = {"Name": "Config",
                          "Node": node,
                          "Agent": "DDPG",
                          "Number_learning_Steps": number_learning_steps,
                          "Trial number": n_trail,
                          "Database name": cfg['STUDY_NAME'],
                          "Start time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                          "Info": "P10 setting, EU grid, HPs von Stuy 22 + 5 pastvals"
                                  "Reward design setzt sich aus MRE [0,1] und clipp-punishment [0,-1] zusammen",
                          }
    trail_config_mongo.update(trial.params)
    # mongo_recorder.save_to_mongodb('Trial_number_' + n_trail, trail_config_mongo)
    mongo_recorder.save_to_json('Trial_number_' + n_trail, trail_config_mongo)

    loss = experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                               # loss = experiment_fit_DDPG_custom(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                               alpha_relu_actor,
                               batch_size,
                               actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                               alpha_relu_critic,
                               noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                               training_episode_length, buffer_size,  # learning_starts,
                               tau, number_learning_steps, integrator_weight,
                               integrator_weight * antiwindup_weight, penalty_I_weight, penalty_P_weight,
                               train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
                               n_trail, number_past_vals)

    return loss


def ddpg_objective(trial):

    number_learning_steps = 500000
    actor_hidden_size = trial.suggest_int("actor_hidden_size", 10, 200)
    actor_number_layers = trial.suggest_int("actor_number_layers", 1, 4)
    antiwindup_weight = trial.suggest_float("antiwindup_weight", 0.00001, 1)
    alpha_relu_actor = trial.suggest_loguniform("alpha_relu_actor", 0.001, 0.5)
    alpha_relu_critic = trial.suggest_loguniform("alpha_relu_critic", 0.001, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    buffer_size = trial.suggest_int("buffer_size", int(20e4), number_learning_steps)
    bias_scale = trial.suggest_loguniform("bias_scale", 5e-5, 0.1)
    critic_hidden_size = trial.suggest_int("critic_hidden_size", 10, 300)
    critic_number_layers = trial.suggest_int("critic_number_layers", 1, 4)
    final_lr = trial.suggest_float("final_lr", 0.00001, 1)
    gamma = trial.suggest_float("gamma", 0.5, 0.9999)
    integrator_weight = trial.suggest_float("integrator_weight", 1 / 200, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 0.5e-2)  # 0.0002#
    lr_decay_start = trial.suggest_float("lr_decay_start", 0.00001, 1)
    lr_decay_duration = trial.suggest_float("lr_decay_duration", 0.00001,
                                            1)
    n_trail = str(trial.number)
    noise_var = trial.suggest_loguniform("noise_var", 0.01, 1)
    noise_var_min = 0.0013  # not used currently;  trial.suggest_loguniform("noise_var_min", 0.0000001, 2)
    noise_steps_annealing = int(
        0.25 * number_learning_steps)
    noise_theta = trial.suggest_loguniform("noise_theta", 1, 50)   # stiffness of OU
    error_exponent = 0.5
    training_episode_length = trial.suggest_int("training_episode_length", 1, 5000)
    tau = trial.suggest_loguniform("tau", 0.0001, 0.3)
    train_freq_type = "step"  # trial.suggest_categorical("train_freq_type", ["episode", "step"])
    train_freq = trial.suggest_int("train_freq", 1, 5000)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    penalty_I_weight = trial.suggest_float("penalty_I_weight", 100e-6, 2)
    penalty_P_weight = trial.suggest_float("penalty_P_weight", 100e-6, 2)
    penalty_I_decay_start = trial.suggest_float("penalty_I_decay_start", 0.00001, 1)
    penalty_P_decay_start = trial.suggest_float("penalty_P_decay_start", 0.00001, 1)
    t_start_penalty_I = int(penalty_I_decay_start * number_learning_steps)
    t_start_penalty_P = int(penalty_P_decay_start * number_learning_steps)
    t_start = int(lr_decay_start * number_learning_steps)
    t_end = int(np.minimum(lr_decay_start * number_learning_steps + lr_decay_duration * number_learning_steps,
                           number_learning_steps))
    use_gamma_in_rew = 1
    weight_scale = trial.suggest_loguniform("weight_scale", 5e-5, 0.2)


    learning_rate = linear_schedule(initial_value=learning_rate, final_value=learning_rate * final_lr,
                                    t_start=t_start,
                                    t_end=t_end,
                                    total_timesteps=number_learning_steps)
    number_past_vals = trial.suggest_int("number_past_vals", 0, 50)

    trail_config_mongo = {"Name": "Config",
                          "Node": node,
                          "Agent": "DDPG",
                          "Number_learning_Steps": number_learning_steps,
                          "Trial number": n_trail,
                          "Database name": cfg['STUDY_NAME'],
                          "Start time": time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime()),
                          "Optimierer/ Setting stuff": "DDPG HPO ohne Integrator, alle HPs fuer den I-Anteil "
                                                       "wurden daher fix gesetzt. Vgl. zu DDPG+I-Anteil"
                          }
    trail_config_mongo.update(trial.params)
    #  mongo_recorder.save_to_json('Trial_number_' + n_trail, trail_config_mongo)

    loss = experiment_fit_DDPG(learning_rate, gamma, use_gamma_in_rew, weight_scale, bias_scale,
                               alpha_relu_actor,
                               batch_size,
                               actor_hidden_size, actor_number_layers, critic_hidden_size, critic_number_layers,
                               alpha_relu_critic,
                               noise_var, noise_theta, noise_var_min, noise_steps_annealing, error_exponent,
                               training_episode_length, buffer_size,  # learning_starts,
                               tau, number_learning_steps, integrator_weight,
                               integrator_weight * antiwindup_weight, penalty_I_weight, penalty_P_weight,
                               train_freq_type, train_freq, t_start_penalty_I, t_start_penalty_P, optimizer,
                               n_trail, number_past_vals)

    return loss



def optuna_optimize_sqlite(objective, sampler=None, study_name='dummy'):
    parser = argparse.ArgumentParser(description='Train DDPG Single Inverter V-ctrl')
    parser.add_argument('-n', '--n_trials', default=50, required=False,
                        help='number of trials to execute', type=int)
    args = parser.parse_args()
    n_trials = args.n_trials or 100

    optuna_path = './optuna/'

    os.makedirs(optuna_path, exist_ok=True)

    study = optuna.create_study(study_name=study_name,
                                direction='maximize',
                                storage=f'sqlite:///{optuna_path}optuna.sqlite',
                                load_if_exists=True,
                                sampler=sampler
                                )
    study.optimize(objective, n_trials=n_trials)



#if __name__ == "__main__":

    #TPE_sampler = TPESampler(n_startup_trials=400)  # , constant_liar=True)

    #optuna_optimize_sqlite(ddpg_objective_fix_params, study_name=STUDY_NAME, sampler=TPE_sampler)
