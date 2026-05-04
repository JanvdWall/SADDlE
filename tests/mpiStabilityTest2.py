from SADDlE import generalDDE, mpiStabilityTable
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps
import tikzplotlib

if __name__ == "__main__":
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    dde = lambda a,b : generalDDE(1, [], [1.0], [0.0], np.array([[a]]), [], [lambda theta: np.array([[b]])], None, False)
    a = np.arange(-20, 20, 0.25, dtype=np.double)
    b = np.arange(-150, 50, 0.25, dtype=np.double)
    times = []
    if rank==0:
        print("Starting calculation")

    for i in range(1):
        comm.barrier()
        time = MPI.Wtime()
        if rank==0:
            print(f"Run {i+1}/10")
        result = mpiStabilityTable(comm, dde, (a, b), 30)
        comm.barrier()
        time = MPI.Wtime() - time
        times.append(time)
    if rank==0:
        print(f"Calculation completed in {np.mean(times)} seconds (± {np.std(times)})")
        
        cmap = colormaps['viridis']
        plt.imshow(result, cmap=cmap, extent=(b[0], b[-1], a[-1], a[0]))
        plt.xlabel("b")
        plt.ylabel("a")
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        plt.legend(custom_lines, ['not stable', 'stable'])
        plt.show()
        """
        import sys
        sys.stdout.write(f"{np.mean(times)}")
        sys.stdout.flush()
        sys.exit(0)
        """