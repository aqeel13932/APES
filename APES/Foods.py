from .Settings import *
class Foods:
    """Class Simulate the Food.
    Attributes:
        * ID: Food Unique id
        * See: boolean value determine if agent can see through this food or not
        * Energy:integer represent how Rewardful this food is, used as factor in rewarding agents.
        * Image: dictionary key to link image with Settings.Images
        * IndiciesOrderName: dictionary key to Settings.ProbabilitiesTable to determine where to assign food first."""
    def __init__(self,ImageKey,Energy=1,See=True,PdstName=None,Range=1):
        """Initialize Food object
        Args:
            * ImageKey: The Image key in Settings.py
            * Energy: integer represent how Rewardful this food is, used as factor in rewarding agents.
            * See: True for other agent can see through this food , False they can't
            * PdstName:(Probability Distribution Name) The name of wanted distribution (name should be linked with Settings.IndiciesOrderTable)
            * Range: define when the agents can eat this food."""
        
        self.Energy=Energy # Determine the reward from eating this food
        self.See=See # Determine if Agent can see through the Food or not
        self.ID = Settings.GetFoodID() # Get unique ID per type
        self.Image=ImageKey
        self.IndiciesOrderName = PdstName
        self.Range = Range
    # This functioner return Kronecker product between the shape and the image
    # https://en.wikipedia.org/wiki/Kronecker_product
    def GetImage(self):
        """get object image
        Return: numpy array image"""
        return Settings.Images[self.Image]
