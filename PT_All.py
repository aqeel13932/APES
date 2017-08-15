import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=[],type=int)
parser.add_argument('--naction',type=int,default=0)
args = parser.parse_args()
msgs=[',should go',',should not go']
msg='Env:{}'
#'''
#Specific Cases
preferences={
    1:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    2:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(6,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    3:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(5,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    4:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(4,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    5:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    6:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(6,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    7:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(5,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    8:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(4,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]}

}
#'''
'''
preferences={
1:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
3:{'sub':(2,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
4:{'sub':(2,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
5:{'sub':(2,1),'dom':(2,9),'food':(3,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' explore'},
6:{'sub':(2,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' analyze the results'},
7:{'sub':(2,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+' surprise me'},
8:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
9:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
10:{'sub':(2,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]},
11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+' race'},
12:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+' SM'},
13:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+' SM'}
        }
'''
import numpy as np
np.random.seed(4917)
from keras.models import load_model
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
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 24)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if AgentView:
        draw.text((0, 0),"Action:{}".format(action),(255,0,0),font=font)
    else:
        draw.text((0, 0),"Action:{}".format(action),(0,0,0),font=font)
    return img


def WriteInfo(model_id,mod_type,env,steps,reward,time,dom):#,rwsc,rwprob,aiproba,eptype,trqavg,tsqavg):
    with open('Final_Results/FinalResults.csv','a') as outp:
        print('{},{},{},{},{},{},{}\n'.format(model_id,mod_type,env,steps,reward,time,dom))
        outp.write('{},{},{},{},{},{},{}\n'.format(model_id,mod_type,env,steps,reward,time,dom))
#### Load the model
for i in args.models:
    if not os.path.exists('Final_Results/{}'.format(i)):
        os.makedirs('Final_Results/{}'.format(i))
    train = load_model('output/{}/MOD/model.h5'.format(i))
    target = load_model('output/{}/MOD/target_model.h5'.format(i))
    #models={'train':train,'target':target}
    models={'target':target}
    for mod in models:
        np.random.seed(1337)
        model = models[mod]
        #domagnt=[True,False]
        domagnt=[False]
        for dom in domagnt:
            #for env in range(1,13):
            for env in preferences:
                counter = env+(env-1)*2
                preferences[env]['mesg']=preferences[env]['mesg'].format(env)
                game = CreateEnvironment(preferences[env],args.naction)
                DAgent,AIAgent = [game.agents[x] for x in game.agents]
                AIAgent = game.agents[1001]
                #print('D:{},AI:{}'.format(DAgent.ID,AIAgent.ID))
                #AIAgent = [game.agents[x] for x in game.agents][0]
                TestingCounter=0
                TestingCounter+=1
                writer = skvideo.io.FFmpegWriter("Final_Results/{}/ENV:{},mod:{},dom:{}.avi".format(i,env,mod,dom))
                writer2 = skvideo.io.FFmpegWriter("Final_Results/{}/ENV:{},mod:{},dom:{}_AG.avi".format(i,env,mod,dom))
                #game.GenerateWorld()
                img = game.BuildImage()
                game.Step()
                #plt.imsave('Final_Results/{}/ENV_{}.png'.format(i,env,TestingCounter),img)
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
                    #if np.random.random()<0.05:
                    #    action=AIAgent.RandomAction()
                    #else:
                        #action = np.argmax(q[0,:-1])
                    action = np.argmax(q[0])
                    #print(Settings.PossibleActions[action],action)
                    AIAgent.NextAction = Settings.PossibleActions[action]
                    #print(AIAgent.NextAction)
                    #if env not in [4,5]:
                    if dom:
                        DAgent.DetectAndAstar()
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
                WriteInfo(i,mod,env,t,episode_reward,Start,dom)

