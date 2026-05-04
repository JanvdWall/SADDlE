from .utils import generateDiffMatrix, generateGeneralChebyNodes
from .generalDDE import theoreticalDDE
import numpy as np
import scipy.sparse as sparse

def getClenshawWeights(n: int):
    """generates the weights for the Clenshaw-Curtis quadrature.
    This code is a line by line translation of the code proposed by [1].
    
    [1] Jörg Waldvogel. “Fast Construction of the Fejér and Clenshaw–Curtis Quadrature
        Rules”. In: BIT 46.1 (Mar. 2006), pp. 195–202. issn: 0006-3835. doi: 10.1007/s10543-006-0045-4

    Args:
        n (int): the index of the mesh for which the weights should be calculated

    Returns:
        np.ndarray: the n+1 weights
    """
    if n == 1:
        return np.array([1, 1])

    N = np.arange(1, n, 2) 
    l = len(N)
    m = n - l

    w = np.concatenate([2 / N / (N - 2), 1 / N[-1:], np.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    g0 = -np.ones(n)
    g0[l] = g0[l] + n
    g0[m] = g0[m] + n
    g = g0 / (n**2 - 1 + (n % 2))
    w = np.fft.ifft(w + g)

    wr = w.real
    return np.concatenate([wr, wr[:1]])

def _generateMeshes(dde, M) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generates the Chebyshev II meshes for the piecewise method.

    Args:
        dde (_type_): the DDE for which the meshes should be calculated
        M (_type_): the index of the meshes

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: a tuple with first: the delays of the DDE, second the list with meshes.
        The meshes are returned for each delay, meaning that the start and end points of consecutive are equivalent.
    """
    meshes = []
    delays = np.concat((np.array([0]), dde.delays))
    for i in range(1, len(delays)):
        tau1 = delays[i-1]
        tau2 = delays[i]
        meshes.append(generateGeneralChebyNodes(M, tau1, tau2))
    return delays, meshes

def _calculateFirstRow(Am, dde, meshes, M, dim, delays):  
    if dde.hasDistributed:
        # if the dde has distributed delays, quadrature must be evaluated
        # The formula for the elements of the first block-row presented in the bachelor thesis is implemented in the following code.
        Am[0:dim, 0:dim] = dde.noDelayMatrix
        weights =getClenshawWeights(M) # each delay gets the same number of quadrature points
        for k, mesh in enumerate(meshes):
            AmK = np.zeros((dim, (M+1)*dim))
            fac = (delays[k+1] - delays[k])*0.5
            for j in range(M+1):
                AmK[0:dim, j*dim:(j+1)*dim] = fac * weights[j] * dde.distributedMatrices[k](mesh[j])
            AmK[0:dim, M*dim: (M+1)*dim] += dde.delayMatrices[k]
            Am[0:dim, (k)*M*dim : ((k+1)*M + M + 1)*dim] += AmK
               
    else:
        # If no distributed delays are present, the quadrature must not be computed.
        Am[0:dim, 0:dim] = dde.noDelayMatrix
        for i, mesh in enumerate(meshes):
            Am[0:dim, ((i+1)*M)*dim: ((i+1)*M +1)*dim] = dde.delayMatrices[i]
            
def _calculateFirstRowSparse(dde, meshes, M, dim, delays):  
            
    rows, cols, data = [], [], []
    if dde.hasDistributed:
        weights =getClenshawWeights(M) # each delay gets the same number of quadrature points
        
        
        for k, mesh in enumerate(meshes):
            fac = (delays[k+1] - delays[k])*0.5
            for m in range(M+1):
                if k == 0 and m == 0:
                    matrix = sparse.csr_matrix(dde.noDelayMatrix) + fac * weights[m] * sparse.csr_matrix(dde.distributedMatrices[k](mesh[m]))
                    matrix_coo = matrix.tocoo()
                    rows.append(matrix_coo.row)
                    cols.append(matrix_coo.col)
                    data.append(matrix_coo.data)
                elif 1 <= m <= M-1 and 0 <= k <= len(meshes)-1:
                    matrix = fac * weights[m] * sparse.csr_matrix(dde.distributedMatrices[k](mesh[m]))
                    matrix_coo = matrix.tocoo()
                    rows.append(matrix_coo.row)
                    cols.append(matrix_coo.col + ((k)*M + m)*dim)
                    data.append(matrix_coo.data)
                elif m == M and 0 <= k <= len(meshes)-2:
                    matrix = sparse.csr_matrix(dde.delayMatrices[k]) + fac * weights[m] * sparse.csr_matrix(dde.distributedMatrices[k](mesh[m]))
                    + (delays[k+2] - delays[k+1])*0.5 * weights[0] * sparse.csr_matrix(dde.distributedMatrices[k+1](mesh[m]))
                    matrix_coo = matrix.tocoo()
                    rows.append(matrix_coo.row)
                    cols.append(matrix_coo.col + ((k)*M + m)*dim)
                    data.append(matrix_coo.data)
                elif m == M and k == len(meshes)-1:
                    matrix = sparse.csr_matrix(dde.delayMatrices[k]) + fac * weights[m] * sparse.csr_matrix(dde.distributedMatrices[k](mesh[m]))
                    matrix_coo = matrix.tocoo()
                    rows.append(matrix_coo.row)
                    cols.append(matrix_coo.col + ((k)*M + m)*dim)
                    data.append(matrix_coo.data)
    else:
        noDelayMatrix_sparse = sparse.coo_matrix(dde.noDelayMatrix)
        rows.append(noDelayMatrix_sparse.row)
        cols.append(noDelayMatrix_sparse.col)
        data.append(noDelayMatrix_sparse.data)
        
        for i, mesh in enumerate(meshes):
            matrix = sparse.csr_matrix(dde.delayMatrices[i])
            matrix_coo = matrix.tocoo()
            rows.append(matrix_coo.row)
            cols.append(matrix_coo.col + ((i+1)*M)*dim)
            data.append(matrix_coo.data)
    return rows, cols, data
    
def assembleAm(dde: theoreticalDDE, M: int, useSparse:bool = False):
    delays, meshes = _generateMeshes(dde, M)
    dim = dde.dimension
    numberDelays = len(delays)
    size = dim * ((numberDelays - 1) * M + 1)
    Id = sparse.eye(dim) if useSparse else np.identity(dim)
    if useSparse:
        rows, cols, data = [], [], []
        for i, mesh in enumerate(meshes):
            diff = generateDiffMatrix(mesh)
            diff_sparse = sparse.csr_matrix(diff[1:, :])
            AmK = sparse.kron(diff_sparse, sparse.eye(dim), format='coo')
            
            # Calculate offsets for this block
            row_offset = (M * i + 1) * dim
            col_offset = (i * M) * dim

            # Append shifted indices
            rows.append(AmK.row + row_offset)
            cols.append(AmK.col + col_offset)
            data.append(AmK.data)

        firstRows, firstCols, firstData = _calculateFirstRowSparse(dde, meshes, M, dim, delays)
        # Create sparse matrix from accumulated COO data
        all_rows = np.concatenate(rows + firstRows)
        all_cols = np.concatenate(cols + firstCols)
        all_data = np.concatenate(data + firstData)
        Am = sparse.coo_array((all_data, (all_rows, all_cols)), shape=(size, size))

    else:
        Am = np.zeros((dim*((numberDelays-1) * M +1), dim*((numberDelays-1) * M +1)))
    
        #calculate the bottom of the matrix
        for i, mesh in enumerate(meshes):
            # the bottom of the matrix consists of kronecker products of the differentiation matrix with the identity matrix.
            # where the differentiation matrix associated with each sub mesh must be considered
            diff = generateDiffMatrix(mesh)
            AmK = np.kron(diff[1:, :], Id)
            Am[(M*i + 1)*dim : (M*i + 1 + M)*dim, (i*M)*dim : (i*M + M + 1)*dim] = AmK
        
        # generate the first block-row. This is a separate function to be used in other functions regarding the stability tables
        _calculateFirstRow(Am, dde, meshes, M, dim, delays)
        
    return Am