msgs=[',should go',',should not go']
msg='Env:{}'
preferences={
    1:{'sub':(1,1),'dom':(1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    2:{'sub':(1,1),'dom':(7,4),'food':(2,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
    3:{'sub':(1,1),'dom':(10,0),'food':(2,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
    4:{'sub':(1,1),'dom':(9,9),'food':(2,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    5:{'sub':(1,1),'dom':(1,9),'food':(2,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' explore'},
    6:{'sub':(1,1),'dom':(1,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' analyze the results'},
    7:{'sub':(1,1),'dom':(3,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+' surprise me'},
    8:{'sub':(1,1),'dom':(9,7),'food':(2,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    9:{'sub':(1,1),'dom':(1,9),'food':(2,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    10:{'sub':(1,1),'dom':(1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]},
    11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+' race'},
    12:{'sub':(1,1),'dom':(1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+' SM'}
	            }
'''
preferences={
    1:{'sub':(1,1),'dom':(1,9),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    2:{'sub':(1,1),'dom':(7,4),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
    3:{'sub':(1,1),'dom':(10,0),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
    4:{'sub':(1,1),'dom':(9,9),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    5:{'sub':(1,1),'dom':(1,9),'food':(2,6),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' explore'},
    6:{'sub':(1,1),'dom':(1,9),'food':(6,5),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' analyze the results'},
    #7:{'sub':(1,1),'dom':(1,9),'food':(6,5),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' Replicated'},
    7:{'sub':(1,1),'dom':(3,1),'food':(6,5),'obs':(2,5),'subdir':'W','domdir':'W','mesg':msg+' surprise me'},
    8:{'sub':(1,1),'dom':(9,7),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
    9:{'sub':(1,1),'dom':(1,9),'food':(2,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    10:{'sub':(1,1),'dom':(1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]},
    11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(2,5),'subdir':'E','domdir':'E','mesg':msg+' race'},
    12:{'sub':(1,1),'dom':(1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+' SM'}
    #,14:{'sub':(1,1),'dom':(1,9),'food':(2,4),'obs':(2,5),'subdir':'E','domdir':'W','mesg':msg+'Not Used'}
            }
'''
import numpy as np
np.random.seed(4917)
from keras.models import Model,load_model
from output.Test_Functions import log_progress
import matplotlib.pyplot as plt
import skvideo.io
from time import time
from Settings import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from Environments import CreateEnvironment

def AddTextToImage(img,action,AgentView=0):
    img = np.array(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    #img = Image.fromarray(game.BuildImage())
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 12)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if AgentView:
        draw.text((0, 0),"Action:{}".format(action),(255,0,0),font=font)
    else:
        draw.text((0, 0),"Action:{}".format(action),(0,0,0),font=font)
    return img

#### Load the model
train_m=target_m = 434
model = load_model('output/{}/MOD/model.h5'.format(target_m))
for env in range(1,13):
    counter = env+(env-1)*2
    preferences[env]['mesg']=preferences[env]['mesg'].format(env)
    game = CreateEnvironment(preferences[env])
    #DAgent,AIAgent = [game.agents[x] for x in game.agents]

    AIAgent = [game.agents[x] for x in game.agents][0]

    print('Testing Target Model{}'.format(env))
    TestingCounter=0
    TestingCounter+=1
    writer = skvideo.io.FFmpegWriter("Final_Results/VID_ENV_{}_Test_{}.avi".format(env,TestingCounter))
    writer2 = skvideo.io.FFmpegWriter("Final_Results/VID_ENV_{}_TestAG_{}.avi".format(env,TestingCounter))
    #game.GenerateWorld()
    img = game.BuildImage()
    game.Step()
    plt.imsave('Final_Results/VID_ENV_{}_Test_{}.png'.format(env,TestingCounter),img)
    Start = time()
    episode_reward=0
    observation = AIAgent.Flateoutput()

    writer.writeFrame(AddTextToImage(game.BuildImage(),AIAgent.NextAction,0))
    writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),AIAgent.NextAction,1))
    for t in range(1000):
        s =np.array([observation])
        q = model.predict(s, batch_size=1)
        #if t==1:
        #print('Input type:',AIAgent.Flateoutput())
        #print('Q value type:',q)
        if np.random.random()<0.05:
            action=AIAgent.RandomAction()
        else:
            #action = np.argmax(q[0,:-1])
            action = np.argmax(q[0])
        #print(Settings.PossibleActions[action],action)
        AIAgent.NextAction = Settings.PossibleActions[action]
        #print(AIAgent.NextAction)
        #if env not in [4,5]:
        #DAgent.DetectAndAstar()
        #print(DAgent.NextAction)
        game.Step()
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        #print(reward)
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += reward
        writer.writeFrame(AddTextToImage(game.BuildImage(),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),0))
        writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),1))
        if done:
            break

    writer.close()
    writer2.close()
    Start = time()-Start
    print(t)

