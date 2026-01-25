import numpy as np
from scipy.interpolate import BarycentricInterpolator

def generateChebyNodes(N:int, tau: float) -> np.typing.NDArray:
    """
    Generates the chebyshev nodes on the intervall [-tau, 0] of index N.
    Meaning that N+1 nodes are generated.
    
    :param N: Index of the chebyshev nodes
    :type N: int
    :param tau: max delay
    :type tau: float
    :return: NDArray containing the chebychev noted on [-tau, 0]
    :rtype: NDArray
    """
    return 0.5 * tau * (np.cos(np.pi * np.array(np.arange(N+1, dtype=float) / N)) - 1)

def generateGeneralChebyNodes(N: int, tau1: float, tau2: float):
    return 0.5 * (tau2-tau1) * (np.cos(np.pi * np.array(np.arange(N+1, dtype=float) / N)) - 1)

def generateDiffMatrix(chebyPoints: list[float]) -> np.typing.NDArray:
    """Generate the differentiation matrix for chbyshev nodes.
    It is not necessary to normalize the node to [-1,1] but the nodes must be transformed chebyshev nodes of second order
    This code is a translation of the code proposed by [1]
    
    [1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
    Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

    Args:
        chebyPoints (list[float] | np.typing.NDArray): chebyshev nodes of second order

    Returns:
        NDArray: the differentiation matrix
    """
    
    N = len(chebyPoints) - 1
    if N == 0:
        return np.array([[0]])
    weights = np.ones(N+1)
    weights[0] = 2
    weights[-1] = 2
    weights[1::2] *= -1
    diffPoints = chebyPoints[:, np.newaxis] - chebyPoints[np.newaxis, :] 
    D = weights[:, np.newaxis] * 1/weights[np.newaxis, :] / (diffPoints + np.identity(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D
    