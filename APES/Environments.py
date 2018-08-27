from .Settings import *
from .World import *
from .Agent import *
from .Obstacles import *
from .Foods import *
from time import time
from pathlib import Path
import os

def CreateEnvironment(preference,ActionMemory=0,max_timesteps=1000,rwrdschem=[-10,1000,-0.1],svision=180):
    Settings.Agents=1000
    Settings.Food=2000
    Settings.Obstacle=3000
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(100)
    Settings.AddImage('Wall',os.path.abspath("APES/Pics/wall.jpg"))#'Pics/wall.jpg')
    Settings.AddImage('Food','APES/Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)

    #print(preference['mesg'])
    if preference['obs']!=(0,0):
        obs[preference['obs']] = 1
    ragnt[preference['sub']] =1
    gagnt[preference['dom']]=1
    food[preference['food']]=1
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    ragnt = Agent(Fname='APES/Pics/red.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt',ActionMemory=ActionMemory)
    gagnt = Agent(Fname='APES/Pics/blue.jpg',Power=10,VisionAngle=180,Range=-1,ControlRange=1,PdstName='gagnt')
        
    #Add Probabilities to Settings
    if preference['obs']!=(0,0):
        Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('food',food)

	#Create World Elements
    if preference['obs']!=(0,0):
        obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')  
    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([gagnt,ragnt])
    #game.AddAgents([ragnt])
    if preference['obs']!=(0,0):
        game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    #print ('Taken:',Start)
    preference['Taken']=Start
    game.GenerateWorld()
    ragnt.Direction=preference['subdir']
    gagnt.Direction=preference['domdir']
    return game
