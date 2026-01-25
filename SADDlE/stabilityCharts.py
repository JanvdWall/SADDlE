import numpy as np
from generalDDE import generalDDE
from pseudoDiff import assembleAm, _generateMeshes, _calculateFirstRow
from typing import Callable
from utils import generateDiffMatrix
from mpi4py import MPI

def _stabilityListForParameters(dde, parameters_paired, M, varDelay: bool = True):
    stability = np.zeros(len(parameters_paired), dtype=np.bool)
    
    if varDelay:
        # if the delays are varying, we must calculate the whole matrix anew
        for i, pair in enumerate(parameters_paired):
                conDde = dde(*pair).convertToTheoreticalDDE()
                Am = assembleAm(conDde, M) 
                spec = np.linalg.eig(Am).eigenvalues.real
                stability[i] = all(spec < 0)
    else:
        #if the delays dont vary, only the first block-row must be recalculated, the rest of the matrix is recycled
        # create a dummy dde with the first parameters to calculate the bottom of the matrix
        conDde = dde(*parameters_paired[0]).convertToTheoreticalDDE()
        delays, meshes = _generateMeshes(conDde, M)
        dim = conDde.dimension
        numberDelays = len(delays)
        Id = np.identity(dim)    
        Am = np.zeros((dim*((numberDelays-1) * M) + 1, dim*((numberDelays-1) * M) + 1))
        
        #calculate the bottom of the matrix
        for i, mesh in enumerate(meshes):
            diff = generateDiffMatrix(mesh)
            AmK = np.kron(diff, Id)
            Am[(M*i + 1)*dim : (M*i + 1 + M)*dim, (i*M)*dim : (i*M + M + 1)*dim] = AmK[1:, :]
            
        # calculate the first block-row of the matrix for each pair
        for i, pair in enumerate(parameters_paired):
            Am[0:dim, :] = 0
            conDde = dde(*pair).convertToTheoreticalDDE()    
            _calculateFirstRow(Am, conDde, meshes, M, dim, delays)
                           
            spec = np.linalg.eig(Am).eigenvalues.real
            stability[i] = all(spec < 0)
            
    return stability

def generateStabilityTable(dde: Callable[..., generalDDE], parameters: tuple, M:int, varDelay: bool = True):
    numberOfParameters = len(parameters)
    parMesh = np.meshgrid(*parameters)
    # The transpose is necessary such that the order of dimensions in the final stability table is senseful.
    parameters_paired = np.array(parMesh).T.reshape(-1, numberOfParameters)
    stability = _stabilityListForParameters(dde, parameters_paired, M ,varDelay)
        
    return stability.reshape(tuple([len(par) for par in parameters]))

def mpiStabilityTable(comm: MPI.Intracomm, dde: Callable[..., generalDDE], parameters: tuple, M:int, varDelay: bool = True) -> np.array[np.bool]:
    """Generate a stability Table for the given DDE and parameters using multicore computation.
    The Table will be returned as a Bool-Matrix. 
    The dimensions are in order of the parameters provided.

    Args:
        comm (MPI.Intracomm): the MPI communicator on which the program should be run
        dde (Callable[..., generalDDE]): a callable that maps a set of parameters on a DDE of type general DDE.
        parameters (tuple): A tuple of lists or arrays containing the desired values for each parameter
        M (int): the index of the Meshes for the pseudo differentiation method
        varDelay (bool, optional): Determines whether the delays are varying parameters. 
            Setting it to false will reduce computational time. 
            Setting it to False even if it is not justified, will result in faulty calculation. 
            Defaults to True.

    Returns:
        np.array[np.bool]: The stability table, with the dimensions in order of the parameters in the tuple.
    """
    
    # get the necessary parameter for the mpi environment
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    numberOfParameters = len(parameters)
    # We the parameters such that by taking te i-th entry in each column yields every possible combination of parameters
    parMesh = np.meshgrid(*parameters)
    # Notice the transpose
    global_columns = [coords.T.reshape(-1) for coords in parMesh]
    global_column_length = len(global_columns[0])
    
    # Distribute the parameters on the threads, notice that these operations are performed by every thread.
    # This is due to the fact, that sending the parameters between the threads takes more time than this approach.
    # Get the length of the parameter list for each thread
    small_length = int(global_column_length // size)
    # If the global length is not divisible by the number of threads a remainder might appear
    remainder = int(global_column_length % size)
    # Distribute the remainder on each thread
    small_lengths = [int(small_length + 1) if i < remainder else int(small_length) for i in range(size)]
    # Get the starting indices for each thread
    displacements = [int(np.sum(small_lengths[:i])) for i in range(size)]
    
    local_columns = [np.zeros(small_lengths[rank]) for i in range(numberOfParameters)]
    # simply add the finishing index to the displacements list
    column_idx = [*displacements, global_column_length]
    for i in range(numberOfParameters):
        local_columns[i] = global_columns[i][column_idx[rank] : column_idx[rank + 1]]
    
    # get the list of parameter parings.
    local_parameters_paired = np.column_stack(tuple(local_columns))
    # calculate for each parameter pairing wether or not it is stable or not
    local_stability = _stabilityListForParameters(dde, local_parameters_paired, M, varDelay)
    
    #collect the results and pice them together in an final array
    results = None
    if rank == 0:
        results = np.zeros(global_column_length, dtype=np.bool)
    for i in range(numberOfParameters):
        gatherBufferV = [results, small_lengths, displacements, MPI.BOOL]
        comm.Gatherv(local_stability, gatherBufferV, root=0)
    if rank == 0:
        # reshape the parameters in the desired table format
        return results.reshape(tuple([len(par) for par in parameters]))