# SADDlE package

## Submodules

## SADDlE.generalDDE module

### *class* SADDlE.generalDDE.generalDDE(dimension, discreteDelays, lowerIntegrationBounds, upperIntegrationBounds, noDelayMatrix, delayMatrices, distributedMatrices)

Bases: `object`

\_summary_

#### \_\_init_\_(dimension, discreteDelays, lowerIntegrationBounds, upperIntegrationBounds, noDelayMatrix, delayMatrices, distributedMatrices)

\_summary_

Args:
: dimension (int): the dimension of the DDE
  discreteDelays (list | np.ndarray | None): list of discrete delays terms
  lowerIntegrationBounds (list): the lower integration bounds r_1, …, r_p
  upperIntegrationBounds (list): the upper integration bounds u_1, …, u_p
  noDelayMatrix (np.ndarray): the matrix for the term without delay
  delayMatrices (list[np.ndarray]): list of matrices multiplied with the discrete delay terms. Must have the same length as discreteDelays
  distributedMatrices (list[callable]): a list of callables that map R to a matrix of the right dimension. These are the matrices that are multiplied inside of the integral
  Must have the same length as the the lower and upper integration bounds.

Raises:
: Exception: if some lengths or dimensions dont match.

#### convertToTheoreticalDDE()

convert a generalDDE into a theoretical one.
This code is heavily inspired by the code provided by [1].

[1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

* **Return type:**
  [`theoreticalDDE`](#SADDlE.generalDDE.theoreticalDDE)

Returns:
: theoreticalDDE: the theoreticalDDE equivalent to the general one this function is called on

### *class* SADDlE.generalDDE.theoreticalDDE(dimension, delays, noDelayMatrix, delayMatrices, distributedMatrices, hasDistributed=True)

Bases: `object`

#### \_\_init_\_(dimension, delays, noDelayMatrix, delayMatrices, distributedMatrices, hasDistributed=True)

## SADDlE.pseudoDiff module

### SADDlE.pseudoDiff.assembleAm(dde, M, useSparse=False)

### SADDlE.pseudoDiff.getClenshawWeights(n)

generates the weights for the Clenshaw-Curtis quadrature.
This code is a line by line translation of the code proposed by [1].

[1] Jörg Waldvogel. “Fast Construction of the Fejér and Clenshaw–Curtis Quadrature
: Rules”. In: BIT 46.1 (Mar. 2006), pp. 195–202. issn: 0006-3835. doi: 10.1007/s10543-006-0045-4

Args:
: n (int): the index of the mesh for which the weights should be calculated

Returns:
: np.ndarray: the n+1 weights

## SADDlE.stabilityCharts module

### SADDlE.stabilityCharts.generateStabilityTable(dde, parameters, M, varDelay=True)

## SADDlE.utils module

### SADDlE.utils.generateChebyNodes(N, tau)

Generates the chebyshev nodes on the intervall [-tau, 0] of index N.
Meaning that N+1 nodes are generated.

* **Parameters:**
  * **N** (*int*) – Index of the chebyshev nodes
  * **tau** (*float*) – max delay
* **Returns:**
  NDArray containing the chebychev noted on [-tau, 0]
* **Return type:**
  NDArray

### SADDlE.utils.generateDiffMatrix(chebyPoints)

Generate the differentiation matrix for chbyshev nodes.
It is not necessary to normalize the node to [-1,1] but the nodes must be transformed chebyshev nodes of second order
This code is a translation of the code proposed by [1]

[1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

* **Return type:**
  `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]

Args:
: chebyPoints (list[float] | np.typing.NDArray): chebyshev nodes of second order

Returns:
: NDArray: the differentiation matrix

### SADDlE.utils.generateGeneralChebyNodes(N, tau1, tau2)

## Module contents

### SADDlE.assembleAm(dde, M, useSparse=False)

### *class* SADDlE.generalDDE(dimension, discreteDelays, lowerIntegrationBounds, upperIntegrationBounds, noDelayMatrix, delayMatrices, distributedMatrices)

Bases: `object`

\_summary_

#### \_\_init_\_(dimension, discreteDelays, lowerIntegrationBounds, upperIntegrationBounds, noDelayMatrix, delayMatrices, distributedMatrices)

\_summary_

Args:
: dimension (int): the dimension of the DDE
  discreteDelays (list | np.ndarray | None): list of discrete delays terms
  lowerIntegrationBounds (list): the lower integration bounds r_1, …, r_p
  upperIntegrationBounds (list): the upper integration bounds u_1, …, u_p
  noDelayMatrix (np.ndarray): the matrix for the term without delay
  delayMatrices (list[np.ndarray]): list of matrices multiplied with the discrete delay terms. Must have the same length as discreteDelays
  distributedMatrices (list[callable]): a list of callables that map R to a matrix of the right dimension. These are the matrices that are multiplied inside of the integral
  Must have the same length as the the lower and upper integration bounds.

Raises:
: Exception: if some lengths or dimensions dont match.

#### convertToTheoreticalDDE()

convert a generalDDE into a theoretical one.
This code is heavily inspired by the code provided by [1].

[1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

* **Return type:**
  [`theoreticalDDE`](#SADDlE.generalDDE.theoreticalDDE)

Returns:
: theoreticalDDE: the theoreticalDDE equivalent to the general one this function is called on

### SADDlE.generateChebyNodes(N, tau)

Generates the chebyshev nodes on the intervall [-tau, 0] of index N.
Meaning that N+1 nodes are generated.

* **Parameters:**
  * **N** (*int*) – Index of the chebyshev nodes
  * **tau** (*float*) – max delay
* **Returns:**
  NDArray containing the chebychev noted on [-tau, 0]
* **Return type:**
  NDArray

### SADDlE.generateDiffMatrix(chebyPoints)

Generate the differentiation matrix for chbyshev nodes.
It is not necessary to normalize the node to [-1,1] but the nodes must be transformed chebyshev nodes of second order
This code is a translation of the code proposed by [1]

[1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

* **Return type:**
  `ndarray`[`tuple`[`Any`, `...`], `dtype`[`TypeVar`(`_ScalarT`, bound= `generic`)]]

Args:
: chebyPoints (list[float] | np.typing.NDArray): chebyshev nodes of second order

Returns:
: NDArray: the differentiation matrix

### SADDlE.generateStabilityTable(dde, parameters, M, varDelay=True)

### SADDlE.getClenshawWeights(n)

generates the weights for the Clenshaw-Curtis quadrature.
This code is a line by line translation of the code proposed by [1].

[1] Jörg Waldvogel. “Fast Construction of the Fejér and Clenshaw–Curtis Quadrature
: Rules”. In: BIT 46.1 (Mar. 2006), pp. 195–202. issn: 0006-3835. doi: 10.1007/s10543-006-0045-4

Args:
: n (int): the index of the mesh for which the weights should be calculated

Returns:
: np.ndarray: the n+1 weights

### *class* SADDlE.theoreticalDDE(dimension, delays, noDelayMatrix, delayMatrices, distributedMatrices, hasDistributed=True)

Bases: `object`

#### \_\_init_\_(dimension, delays, noDelayMatrix, delayMatrices, distributedMatrices, hasDistributed=True)
