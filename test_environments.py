filesignature = 189
batch_size = 32
exploration= 0.0
tau=0.001
activation='tanh'
advantage='max'
seed=4917
totalsteps=5000 
details="A*,180,FR,E186"
hidden_size=100
layers=1
batch_norm=False
replay_size=100000
train_repeat=1
gamma=0.99
max_timesteps=1000
optimizer='adam'
vanish=0.75
rwrdschem=[-10,1000,-0.1]
svision=180

import numpy as np
np.random.seed()
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from PD_Map import DPMP
from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
from time import time
from copy import deepcopy
from buffer import Buffer
import os


def Environment1():
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    print('Env:1,Should not go')
    obs[5,5] = 1
    ragnt[10,0] =1
    gagnt[10,10]=1
    food[10,5]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt,gagnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def Environment2():
    print('Env:2,Should go')
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    obs[5,5] = 1
    ragnt[10,0] =1
    gagnt[4,4]=1
    food[10,6]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt,gagnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def Environment3():
    print('Env:3,Should not go')
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    obs[2,5] = 1
    ragnt[10,0] =1
    gagnt[1,9]=1
    food[10,5]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt,gagnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def Environment4():
    print('Env:4,Should go')
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    obs[5,5] = 1
    ragnt[10,0] =1
    gagnt[4,4]=1
    food[10,6]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt,gagnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game


def Environment5():
    print('Env:5,Should go')
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    obs[5,5] = 1
    ragnt[10,0] =1
    gagnt[4,6]=1
    food[10,6]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt,gagnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game
