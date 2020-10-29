import re
import numpy as np

class Distribution:
    """
        Super-class for distributions, holds estimates and simple learning algorithms such as _updateEstimates which the Agents can access. Which 
        distribution to be used can be specified in the map file: N(mean, var) for normal distribution and E(base, scale) for exponential distribution.
    """

    def __init__(self):
        self._observation = None # Recently made observation from an appropriately distributed random number generator
        self._estimatedMean = 0.0 # Estimated mean based on the observations so far
        self._estimatedSquareMean = 0.0 # Estimated square mean based on the observations so far
        self._observations = 0 # The amount of observations we have made

    def getObservation(self):
        if not self.isObserved():
            self._observation = self._getSample()
            self._updateEstimates()
        return self._observation
    
    def reset(self):
        self._observation = None

    def isObserved(self) -> bool:
        return self._observation is not None

    def _updateEstimates(self):
        self._observations += 1 # Adds 1 to observations since we made another observation.
        self._estimatedMean += 1.0 / self._observations * (self._observation - self._estimatedMean) # Using the amount of observations we can update our estimated mean
        self._estimatedSquareMean += 1.0 / self._observations * (self._observation ** 2 - self._estimatedSquareMean) # Update our estimated square mean in the same way

    def getEstimatedMean(self):
        return self._estimatedMean

    def getEstimatedVariance(self):
        return self._estimatedSquareMean - self._estimatedMean ** 2

    def _getSample(self):
        raise NotImplementedError

    def getMean(self):
        raise NotImplementedError

    def getVariance(self):
        raise NotImplementedError

    @staticmethod
    def getDistribution(s: str):
        
        split = s.split()

        match: re.Match = None
        name = split[0]

        match = re.match(NormalDistribution.REGEX, split[1]) # Check if the requested distribution is normal distribution
        if match:
            return NormalDistribution(name, float(match.group('mean')), np.sqrt(float(match.group('var'))))

        match = re.match(ExponentialDistribution.REGEX, split[1]) # Check if the requested distribution is exponential distribution
        if match:
            return ExponentialDistribution(name, float(match.group('base')), float(match.group('scale')))

        raise ValueError("Found no appropriate distribution.")


class NormalDistribution(Distribution):
    
    REGEX = r"N\((?P<mean>[0-9\.]+),(?P<var>[0-9\.]+)\)"
    """
        The normal distribution class is initialized with a mean and a variance which are then ultimately used to generate a random number within this distribution.
    """

    def __init__(self, name: str, mean: float, std: float):
        super(NormalDistribution, self).__init__()
        self.name = name
        self.mean = mean
        self.std = std
        
    def _getSample(self) -> float:
        # Get a sample from numpys normal distributed random number generator.
        return np.random.normal(
            loc=self.mean,
            scale=self.std
        )

    def getMean(self) -> float:
        return self.mean

    def getVariance(self) -> float:
        return self.std ** 2


class ExponentialDistribution(Distribution):
    
    REGEX = r"E\((?P<base>[0-9\.]+),(?P<scale>[0-9\.]+)\)"
    """
        Exponential distribution is initialized with a base value and a scale value. The base value specifies the minimum value to which the exponential 
        distribution is added to. The scale value is used to generate an exponential distribution where the mean is equal to the scale value.
    """
    def __init__(self, name: str, base: float, scale: float):
        super(ExponentialDistribution, self).__init__()
        self.name = name
        self.base = base
        self.scale = scale
        
    def _getSample(self) -> float:
        # Get a sample from numpys exponentially distributed random number generator.
        return self.base + np.random.exponential(   
            scale=self.scale
        )

    def getBase(self) -> float:
        return self.base

    def getScale(self) -> float:
        return self.scale ** 2

    def getMean(self) -> float:
        return self.base + self.scale
    
    def getVariance(self) -> float:
        return self.scale ** 2
