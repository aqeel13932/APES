from Settings import *
import numpy as np
class Obstacles:
    """Class represent obstacles in the EnvironmentError
    Attributes:
        * ID: Obstacle Unique id
        * See: boolean value determine if agent can see through this obstacle or not
        * Shape: (n,n) numpy array determine the obstacle shape
        * Image: dictionary key to link image with Settings.Images
        * IndiciesOrderName: dictionary key to Settings.ProbabilitiesTable to determine where to assign obstacle first."""
    def __init__(self,ImageKey,See=False,Shape=np.array([1]),PdstName=None):
        """Initialize Obstacles
        Args:
            * ImageKey: The Image key in Settings.py
            * See: True for other agent can see through this Obstacle , False they can't
            * Shape: numpy array for this obstacle shape example np.array([[1,1,1],[0,1,0],[0,1,0]]) will create a (T) shape obstacle
            * PdstName:Probability Distribution Name, name should be linked with 
                Settings.ProbabilitiesTable by using Settings.AddProbabilityDistribution)"""
        self.ID = Settings.GetObstaclesID() # Get unique ID per type
        self.See=See # Determine if Agent can see through the obstacle or not
        self.Shape=Shape # Shape of Obstacle (2d) 0 nothing , 1 exist
        self.Image=ImageKey
        self.IndiciesOrderName = PdstName

    # This functioner return Kronecker product between the shape and the image
    # https://en.wikipedia.org/wiki/Kronecker_product
    def GetImage(self):
        """Description: Generate Image Corresponsing to obstacle shape"""
        return np.kron(self.Shape[...,None],Settings.Images[self.Image])
    