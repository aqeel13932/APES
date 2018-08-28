from APES import *
import numpy as np
from time import time
import pylab as pl
from IPython import display
Settings.SetBlockSize(100)

# Complex Example.
initalization_time = time()
#Add pictures for items
Settings.AddImage('Wall','APES/Pics/wall.jpg')
Settings.AddImage('Food','APES/Pics/food.jpg')

#Create Probability distribution matrices (PDMs)
obs_pdm = np.zeros(Settings.WorldSize)
agnts_pdm = np.zeros(Settings.WorldSize)
food_pdm = np.zeros(Settings.WorldSize)

# Obstacles can appear from 3rd to 7th row and 5th column
obs_pdm[3:8,5] = 1 
agnts_pdm[2,[0,10]] = 1
food_pdm[:,4:7] = 1

#Add PDMs to Settings
Settings.AddProbabilityDistribution('Obs_pdm',obs_pdm) 
Settings.AddProbabilityDistribution('agnts_pdm',agnts_pdm)
Settings.AddProbabilityDistribution('food_pdm',food_pdm)

#Create World Elements
#Create vertical obastacle with length 4
obshape = np.array([[1],[1],[1],[1]]) 
obs = Obstacles('Wall',Shape=obshape,PdstName='Obs_pdm')

#Create two agents
ragnt = Agent(Fname='APES/Pics/red.jpg',PdstName='agnts_pdm')
bagnt = Agent(Fname='APES/Pics/blue.jpg',PdstName='agnts_pdm')
food = Foods('Food',PdstName='food_pdm')

#Reward food by 10, time step by -0.1
game = World(RewardsScheme=[0,10,-0.1])

#Adding Agents in Order of Following the action
game.AddAgents([ragnt,bagnt])
game.AddObstacles([obs])
game.AddFoods([food])
initalization_time = initalization_time-time()

#Execute at the beginning of every episode
world_generating = time()
game.GenerateWorld()
world_generating= time()-world_generating

#Execute every time step
counter=0
steps_time=0
while not game.Terminated[0]:
    # Agents taking action
    bagnt.RandomAction()
    ragnt.RandomAction()
    counter+=1

    step_time = time()
    game.Step()
    steps_time += time()-step_time
avg = steps_time/counter
print('initialization time:{}\nGenerating world:{}\nAvg step time:{}\nAvg steps per second:{}'.format(initalization_time,world_generating,avg,1/avg))
