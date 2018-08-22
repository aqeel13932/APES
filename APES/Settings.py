import numpy as np
from skimage import io,transform
import os
#For Video Genration.
import matplotlib.animation as animation
#import numpy as np
from pylab import *
from datetime import datetime
#Add Thread Management
class Settings:
    """Settings class to handle shared data
    Attributes:
        * Images: dictionary link image name 'Wall' to image picture
        * Agents: Agent ID counter start from 1000
        * Food: Food ID counter start from 2000
        * Obstacles: Obstacles ID counter start from 3000
        * BlockSize: define how many pixles for block images sizes, images like (food,obstacles,blank,unob,agents , etc..), example (50,50)
        * WorldSize: define how many blocks in the world (10,10)-> 100 block
        * FigureSize: define the size of the images outputed.
        * ProbabilitiesTable: Dictionary (Key:name,Value:Probability distribution numpy array).
        * PossibleActions: numpy array contain all possible actions for agent.(Could be modified to represent customized set of actions)"""
    
    #Block Size
    BlockSize=(100,100)
    #Images so all elements, foods reference to those images without replication
    Images={}
    Images[0]=np.tile(1,(BlockSize[0],BlockSize[1],3)) #Empty
    Images[-1] =np.tile(0,(BlockSize[0],BlockSize[1],3)) # black or unobservable

    #Agent ID domain start from
    Agents=1000

    #Food ID domain (types)
    Food=2000

    #Barriers ID domain (types)
    Obstacles=3000

    #World Size
    WorldSize=(11,11)

    #Figure Size
    FigureSize = (10,10)

    #Probability Distributions Table
    ProbabilitiesTable = {}
    
    #List of Possible Actions For Agent
    AllPossibleActions = np.array([['R','L'],['R','R'],['L','N'],['L','S'],['L','E'],['L','W'],\
    ['M','N'],['M','S'],['M','E'],['M','W']])
    #List of Action that we pick from randomly 
    PossibleActions = np.array([[['L','N'],['M','N']],[['L','S'],['M','S']],[['L','W'],['M','W']],[['L','E'],['M','E']],[]])

    @staticmethod
    def AddProbabilityDistribution(Name,IntProbabilityDst):
        """Convert Integerwise distribution array to indicies  
        Args:
            * Name: The name of the distribution
            * IntProbabilityDst: Integer array [0....+inf] of the world size, Used to generate Probability matrix"""
        Settings.ProbabilitiesTable[Name] = (IntProbabilityDst/ IntProbabilityDst.sum()).ravel()
    @staticmethod
    def SetWorldSize(x,y):
        """Update world Size
        Args:
            newsize: the new world size, example : SetWorldSize((20,20))"""
        #print 'Warning: This Value Should not be changed after creating the world'
        Settings.WorldSize=newsize

        Settings.apb = 360.0/(Settings.WorldSize[0]*2 + Settings.WorldSize[1]*2)

    # get unique Agent ID
    @staticmethod
    def GetAgentID():
        """staticmethod to get unique agent ID
        Return: integer for Agents"""
        Settings.Agents+=1
        return Settings.Agents

    #get unique obstacle ID (type)
    @staticmethod
    def GetObstaclesID():
        """staticmethod to get unique obstacle ID
        Return: integer for obstalces"""
        Settings.Obstacles+=1
        return Settings.Obstacles

    #get unique Food ID (type)
    @staticmethod
    def GetFoodID():
        """staticmethod to get unique Food ID
        Return: integer for Foods"""
        Settings.Food+=1
        return Settings.Food

    @staticmethod
    def ImageViewer(image):
        """Convert image from BGR to RGB colors
        Return: RGB image"""
        #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
        return image

    #Store Image with Key
    @staticmethod
    def AddImage(Key,Fname):
        """Add Image to Images dictionary
        Args:
            * Key: Dictionary name for the image
            * Fname: File Directory to the image Example: 'Pics/wall.jpg'
        Exception:
            * IOError: if file doesn't exist
        Example:
            * Settings.AddImage('Wall','Pics/wall.jpg')
        """
        if not os.path.isfile(Fname):
            raise IOError('file:\'{}\' not found'.format(Fname))
        Settings.Images[Key] = Settings.ImageViewer(transform.resize(io.imread(Fname),Settings.BlockSize,mode='constant'))
    @staticmethod
    def SetBlockSize(Newsize):
        """Set new block size in the Settings
        Args :
            * Newsize: integer for block new size. (block can only be square not rectangle).
        Exception:
            * AssertionError: if Newsize is not integer or less or equal 0"""
        assert type(Newsize) is int, "Newsize is not integer"
        assert Newsize>0, "Newsize can't be less or equal zero"
        Settings.BlockSize = (Newsize,Newsize)
        Settings.Images[0]=np.tile(1,(Settings.BlockSize[0],Settings.BlockSize[1],3)) #Empty
        Settings.Images[-1]=np.tile(0,(Settings.BlockSize[0],Settings.BlockSize[1],3)) # black or unobservable
    
    @staticmethod
    def ani_frame(rimages=[],fps=60,dpi = 100,name='demo'):
        """Create Videom from list of images
        Args:
            * rimages: list of images
            * fps: Frame Per Second (image per second)
            * dpi: dot per pixel
            * name: File name looks like  Name_fps:2_DPI:100_YYYY-MM-DD-HH:MM:SS.mp4
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im = ax.imshow(rand(1,1))
        fig.set_size_inches(Settings.FigureSize)
        tight_layout()

        def update_img(n):
            im.set_data(rimages[n])
            return im
        x = datetime.now()
        ani = animation.FuncAnimation(fig,update_img,len(rimages),interval=30)
        writer = animation.writers['ffmpeg'](fps=fps)
        ani.save('{}_fps:{}_DPI:{}_{}-{}-{}-{}:{}:{}.mp4'.format(name,fps,dpi,x.year,x.month,x.day,x.hour,x.minute,x.second),writer=writer,dpi=dpi)
