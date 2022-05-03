import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import Locator
import numpy as np


#study = optuna.load_studysad(study_name="DDPG_Lr_gamma_Anoise", storage="postgresql://optuna:angelfish-awful-2Circe-domed7@localhost:12345/optuna")
#study = optuna.load_study(study_name="PC2_DDPG_MRE_V_only", storage="sqlite:///Result_7")

study = optuna.load_study(study_name="PC2_DDPG_Vctrl_single_inv_22_newTestcase", storage=f"mysql://optuna:hPoPC2_2021dW_oK@131.234.124.80/optuna")

df = study.trials_dataframe()


fig = optuna.visualization.matplotlib.plot_slice(study, params=["gamma", "actor_hidden_size", "actor_number_layers"])
plt.title('ASDASD')
plt.show()
fig.savefig('HPO_slices.png')



asd  = 1

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

"""
plt.plot(df['number'], -df['value'], 'o', linewidth=0.1)
plt.yscale("log")
plt.xlabel('HPO sample')
plt.ylabel('- Return')
plt.grid()
plt.show()

plt.plot(df['number'], df['value'], 'o')
plt.xlabel('HPO sample')
plt.ylabel('Return')
plt.grid()
plt.show()
"""

plt.scatter(df['number'], df['value'], s=7)
plt.xlabel('HPO sample (#Trial)')
plt.ylabel('Objective Value')
plt.grid()
plt.show()

plt.scatter(df['number'], df['value'], s=7)
#plt.yscale("log")
plt.yscale("symlog")
plt.xlabel('HPO sample')
plt.ylabel('Return')
#plt.ylim([-2, 0])
plt.grid()
plt.show()



#df.to_pickle('PC2_DDPG_Vctrl_single_inv_22_newTestcase.pkl')

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))




plt.scatter(df['number'], df['value'], s=7)
plt.yscale('symlog', linthreshy=1e-1)

yaxis = plt.gca().yaxis
yaxis.set_minor_locator(MinorSymLogLocator(1e-1))

plt.show()
