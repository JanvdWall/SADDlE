import numpy as np

class generalDDE:
    """_summary_
    """
    def __init__(self, dimension: int, discreteDelays:list, lowerIntegrationBounds: list, upperIntegrationBounds: list, noDelayMatrix:np.ndarray, delayMatrices:list[np.ndarray], distributedMatrices:list[callable]):
        """_summary_

        Args:
            dimension (int): the dimension of the DDE
            discreteDelays (list | np.ndarray | None): list of discrete delays terms
            lowerIntegrationBounds (list): the lower integration bounds r_1, ..., r_p
            upperIntegrationBounds (list): the upper integration bounds u_1, ..., u_p
            noDelayMatrix (np.ndarray): the matrix for the term without delay
            delayMatrices (list[np.ndarray]): list of matrices multiplied with the discrete delay terms. Must have the same length as discreteDelays
            distributedMatrices (list[callable]): a list of callables that map R to a matrix of the right dimension. These are the matrices that are multiplied inside of the integral
            Must have the same length as the the lower and upper integration bounds.

        Raises:
            Exception: if some lengths or dimensions dont match.
        """
        
        # set all fields of the class and set empty arrays if None is provided
        self.discreteDelays = np.array([]) if discreteDelays is None else discreteDelays
        self.delayMatrices = [] if delayMatrices is None else delayMatrices
        self.upperIntegrationBounds = np.array([]) if upperIntegrationBounds is None else np.array(upperIntegrationBounds)
        self.lowerIntegrationBounds = np.array([]) if lowerIntegrationBounds is None else np.array(lowerIntegrationBounds)
        self.noDelayMatrix = np.zeros((dimension,dimension)) if noDelayMatrix is None else noDelayMatrix
        self.distributedMatrices = [] if distributedMatrices is None else distributedMatrices
        self.dimension = dimension
        
        # raise exceptions if lengths dont match.
        if not len(self.lowerIntegrationBounds) == len(self.upperIntegrationBounds) == len(self.distributedMatrices):
            raise Exception("there must be equally many lower and upper integration bounds and distributed Matrices")
        
        if not len(self.discreteDelays) == len(self.delayMatrices):
            raise Exception("there must be equally many discrete delays and discrete Matrices.")
        
        
    def convertToTheoreticalDDE(self) -> theoreticalDDE:
        """convert a generalDDE into a theoretical one.
        This code is heavily inspired by the code provided by [1].
        
        [1] Dimitri Breda, Stefano Maset, and Rossana Vermiglio. Stability of Linear Delay
        Differential Equations a Numerical Approach with MATLAB. Springer-Verlag, 2010.

        Returns:
            theoreticalDDE: the theoreticalDDE equivalent to the general one this function is called on
        """
        countDiscrete = len(self.discreteDelays)
        countDistrebuted = len(self.lowerIntegrationBounds)
        
        if countDistrebuted > 0:
            # remove the integrals with length zero
            idx = np.nonzero(self.upperIntegrationBounds - self.lowerIntegrationBounds)[0]
            length = len(idx)
            cleanLowerBounds = np.zeros(length)
            cleanUpperBounds = np.zeros(length)
            cleanDistMatr = []
            for i in range(len(idx)):
                cleanLowerBounds[i]  = self.lowerIntegrationBounds[idx[i]]
                cleanUpperBounds[i]  = self.upperIntegrationBounds[idx[i]]
                cleanDistMatr.append(self.distributedMatrices[i])
        
        # each integration bound and discrete delay must be a delay i the final product        
        taus = np.unique(np.sort(np.concat((np.array([0]), self.discreteDelays, self.lowerIntegrationBounds, self.upperIntegrationBounds))))
        numberDelays = len(taus) - 1
        
        # the no delay matrix stays the same
        noDelayMatrix = self.noDelayMatrix
        discreteMatr = [noDelayMatrix]
        # for the intermediate steps the noDelayMatrix is included in the list of discrete delay matrices.
        # Later it will return to its special role
        for k in range(1, numberDelays + 1):
            discreteMatr.append(np.zeros(self.noDelayMatrix.shape))
            
        for k in range(countDiscrete):
            # add the appropriate discrete matrices together, multiple zero delays or one delay occurring multiple times is handled this way.
            index = np.searchsorted(taus, self.discreteDelays[k])
            discreteMatr[index] = discreteMatr[index] + self.delayMatrices[k]
        
        # the discrete delay matrices are finalized    
        delayMatr = []
        for matr in discreteMatr[1:]:
            delayMatr.append(matr)
        
        # the same operation must be performed with the distributed delays.
        # these must be initialized as callables
        newDistrMatr = []
        for k in range(numberDelays):
            newDistrMatr.append(lambda theta : np.zeros(self.noDelayMatrix.shape))
            
        hasDistr = False   
        if countDistrebuted > 0:
            # these operations must only be performed if distributed delay terms are present    
            hasDistr = True
            # pair the lower and upper integration bounds, if -lower > -higher these indices must later be flipped
            boundaries = np.column_stack((cleanLowerBounds, cleanUpperBounds))
            # this line only safes the indices of the flipped bounds
            idxFlipped = np.argsort(boundaries)[:, 0]
            # flips the reverse bounds
            boundaries = np.sort(boundaries)[:, ::-1]
            for k  in range(len(cleanDistMatr)):
                if idxFlipped[k] == 0:
                    # if the indices have been flipped, we must evaluate the integral multiplied by -1
                    cleanDistMatr[k] = lambda theta: -cleanDistMatr[k](theta)
            for k in range(len(cleanLowerBounds)):
                # the indices of the delays between two boundaries
                idx = (np.argwhere(np.logical_and(-boundaries[k][0] <= -taus, -taus <-boundaries[k][1])).T)[0]
                for i in idx:
                    # add the matrix to the matrices corresponding to idx
                    # i-1 since in taus the 0 delay is included
                    newDistrMatr[i-1] = lambda theta, oldMatr=newDistrMatr[i-1], newMatr=cleanDistMatr[k]: oldMatr(theta) + newMatr(theta)
                    
        return theoreticalDDE(self.dimension, taus[1:], discreteMatr[0], delayMatr, newDistrMatr, hasDistr)
        
        
class theoreticalDDE:
    def __init__(self, dimension: int, delays: np.ndarray, noDelayMatrix: np.ndarray, delayMatrices: list[np.ndarray], distributedMatrices: list[callable], hasDistributed=True):
        self.delays = delays
        self.noDelayMatrix = noDelayMatrix
        self.delayMatrices = delayMatrices
        self.distributedMatrices = distributedMatrices
        self.hasDistributed = hasDistributed
        self.dimension = dimension
    