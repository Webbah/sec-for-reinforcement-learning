# Steady-State Error Compensation for Reinforcement Learning Control

[Read the Paper](https://arxiv.org) Link will follow when published

Reference tracking problems are common in power grids as well as in automotive applications, and the utilization 
of RL controllers in such scenarios feature better performance than state-of-the-art methods during transients. 
However, they still exhibit non vanishing steady-state errors.
Therefore, the presented **S**teady-state **E**rror **C**ompensation tool will extend an established actor-critic 
based RL approach to compensate the steady-state error and close this gap.

Suggestions or experiences concerning applications of SEC are welcome!

## Citing
Detailed informations can be found in the article 
"Steady-State Error Compensation in Reference Tracking Problems with Reinforcement Learning Control".
Please cite it when using the provided code:

```
@misc{weber2021,
      title={Steady-State Error Compensation in Reference Tracking Problems with Reinforcement Learning Control}, 
      author={Daniel Weber and Maximilian Schenke and Oliver Wallscheid},
      year={2021}
}
```

## Usage
The ```jupyter notebook```  ```show_results.ipynp``` provides executable files to reproduce the plots and results shown in the paper.

Use
```angular2html
run_OMG_experiment()   

run_GEM_experiment()
```

to visualize the results.
Like shown in the notebook

```angular2html
optuna_optimize_sqlite(ddpg_objective, study_name=STUDY_NAME, sampler=TPE_sampler)
```

can be used to train DDPG agents.
In the respective (OMG/GEM / util) config file can be adjusted weather a standard DDPG (``` env_wrapper='no_I_term'``) 
or an SEC-DDPG (``` env_wrapper='past'``) should be trained.
The data is logged depending on the ```loglevel``` to the in ```meas_data_folder```defined folder.
Be aware that the calculation can be time consuming.