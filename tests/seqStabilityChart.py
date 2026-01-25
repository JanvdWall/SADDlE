from SADDlE import generateStabilityTable, generalDDE
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

# setup the DDE that depends on the parameters a,b
dde = lambda a, b : generalDDE(1, [], [1], [0], np.array([[a]]), [], [lambda theta: np.array([[b]])])
# setup the ranges of the parameters
b = np.arange(-150, 50, dtype=np.double)
a = np.arange(-20, 20, dtype=np.double)
# calculate the stability table
table = generateStabilityTable(dde, (a, b), 20)

#plotting
cmap = colormaps['viridis']
plt.imshow(table, cmap=cmap)
plt.xlabel("b")
plt.ylabel("a")
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]
plt.legend(custom_lines, ['not stable', 'stable'])
plt.show()