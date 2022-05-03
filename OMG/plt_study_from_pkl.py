import matplotlib
import pandas as pd
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import Locator
import numpy as np

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
              'figure.figsize': [8, 6],
              'font.family': 'serif',
              'lines.linewidth': 1.2
              }
matplotlib.rcParams.update(params)

df = pd.read_pickle('PC2_DDPG_Vctrl_single_inv_22_newTestcase.pkl')

#max_val = df.rolling_max(df['value'], 1)
#max_val = df['value'].rolling(20).max().tolist()
#max_val[0] = max_val[1]

value = df['value'].to_numpy()
value[np.isnan(value)] = -0.8  # nur für die Max-linie!
#value[295] = -0.8

max_val = np.maximum.accumulate(df['value'].to_numpy())

class MinorSymLogLocator(Locator):
    
    def __init__(self, linthresh):
        
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



fig = plt.figure()
plt.scatter(df['number'], df['value'], s=7)
plt.plot(df['number'], max_val, 'orange')
plt.yscale('symlog', linthresh=0.0001)#, linthreshy=1e-1)
yaxis = plt.gca().yaxis
yaxis.set_minor_locator(MinorSymLogLocator(1e-2))
#plt.ylim([-1.5, -0.035])
plt.xlabel('HPO sample (Trial)')
plt.ylabel('Objective Value')
plt.xlim([0, 12227])
plt.grid()
plt.show()

fig.savefig('HPO_samples.png')



