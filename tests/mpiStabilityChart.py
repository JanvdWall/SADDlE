from SADDlE import generalDDE, mpiStabilityTable
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    dde = lambda a,b : generalDDE(2, [np.pi * 2], None, None, np.array([[0,1], [-a, 0]]), [np.array([[b, 0], [0,0]])], None)
    a = np.arange(-1, 1, 0.01, dtype=np.double)
    b = np.arange(-1, 3, 0.01, dtype=np.double)
    result = mpiStabilityTable(comm, dde, (a, b), 30)
    if rank==0:
        cmap = colormaps['viridis']
        plt.imshow(result, cmap=cmap)
        plt.xlabel("b")
        plt.ylabel("a")
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        plt.legend(custom_lines, ['not stable', 'stable'])
        plt.show()
