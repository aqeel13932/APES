import numpy as np
from .Settings import *
from .FOV_Calculator import FOV_SC
from collections import OrderedDict
from math import floor
#np.set_printoptions(linewidth=150,precision=0)
class World:
    """This class represent the Grid world and organize the policy between world elements.
    Attributes:
        * agents: Ordered Dictionary (Key: Agent ID, Value: Agent Object)
        * foods: Dictionary (Key:Food ID, Value: Food object)
        * obstacles: Dictionary (Key: obstacle ID, Value: Obstacle object)
        * world: numpy array (n,n) represent the World
        * VBM: Vision border map. It's binary numpy array (True:can't see through,False: Can see through)
        * options: dictionary link letters with action example Key:'W'-> Value:_WestPosition
        * RotationAction: dictionary (Key:(Rotation direction,Current Direction),Value:New Direction)
        * ANNIG: more info in __init__
        * RF: Reward function, Variable to store the function responsible for distributing the rewrar between agents.
        * RewardScheme: array determine the default reward (more info in __init__)
        * AES: constant number of steps where the game continue even if the food had been eaten.
        * AfterEndSteps: one element array represent number of steps where the game continue even if the food had been eaten (by ref value change each step.)
        * Terminated: one element array represent the status of the game.
        * StepsLimit: maximum number of steps for the game to terminate.
        * StepsCounter: the number of steps up to now.
        * Limited: identify if this game is limited by steps number or not.
    TODO:
        * Improve function _GetVisionBorders by removing all if statements in that function.
            and replace it with dictinary someway.
    """
    def __init__(self,ANNIG=None,RewardsScheme=[-10,10,-1],RewardFunction=None,AES=0,StepsLimit=None):
        """Description: Initialize the world depending on the settings from Settings.py file
        IMPORTANT: AGENTS ARE ORDERED DICTIONARY, SO IF ORDER IMPORTANT ORDER THEM IN INPUT LIST (MORE POWERFUL BEFORE THE WEAK OR OPPIST DEPENDING ON YOUR NEED)
        Args:
            * ANNIG:(Agent Neural Network Input Gnerator) Function to generate the input for the neural network, info about the parameters from _FindAgentOutput function.
                    EXAMPLE FUNCTION (Default at version 0.1): Check _FindAgentOutput
            * RewardsScheme: [Punishment Factor (agents meet) ,Reward Factor(Food in Range), Punishement per Step]
            * RewardFunction: Function responsible to distribute reward and punishments on agents, info about the parameters from _EstimateRewards
                    EXAMPLE FUNCTION (Default at version 0.1): Check _EstimateRewards
            * AES: (After End Steps) number of steps to keep the game going while no food exist.
            * StepsLimit: maximum number of steps for the game to terminate,Default is none (unlimited)."""
        self.agents = OrderedDict()
        self.foods={}
        self.obstacles = {}
        self.world = np.zeros(Settings.WorldSize,dtype=int)
        # Vision Borders Map (Places where agent can't see behind)
        self.VBM = np.zeros(Settings.WorldSize,dtype=np.bool)
        self.options = {'W' : self._WestPosition,'E' : self._EastPosition,'N':self._NorthPosition, 'S': self._SouthPosition,'L':self.Look,'R':self.Rotate,'M':self.Move}
        self.RotationAction ={('R','W') : 'N',('R','E') : 'S',('R','N'):'E', ('R','S'):'W',\
        ('L','W') : 'S',('L','E') :'N',('L','N'):'W', ('L','S'): 'E'}

        if ANNIG:
            self.ANNIG = ANNIG
        else:
            self.ANNIG = self._FindAgentOutput
        if RewardFunction:
            self.RF = RewardFunction
        else:
            self.RF = self._EstimateRewards
        
        self.RewardsScheme = RewardsScheme
        self.AES = AES
        self.AfterEndSteps = [AES]
        self.Terminated = [False]
        self.Limited=False
        if StepsLimit:
            self.StepsLimit= StepsLimit
            self.Limited=True
        self.StepCounter=0
        #Same direction
        self.ego_map = dict.fromkeys([('N','N'),('W','W'),('S','S'),('E','E')],np.array([1,0,0,0]))
        #Oppesite direction
        self.ego_map.update(dict.fromkeys([('N','S'),('S','N'),('E','W'),('W','E')],np.array([0,1,0,0])))
        #Looking left
        self.ego_map.update(dict.fromkeys([('N','W'),('S','E'),('E','N'),('W','S')],np.array([0,0,1,0])))
        #Looking right
        self.ego_map.update(dict.fromkeys([('N','E'),('S','W'),('E','S'),('W','N')],np.array([0,0,0,1])))

    def Ego_centric_array(self,direction,array):
        """ Rotate the array to match an ego-centric agent.
        direction: the agent current direction.
        array: the agent map of the world.
        """
        if direction=='E':
            return np.rot90(array)
        elif direction=='S':
            return np.rot90(array,2)
        elif direction=='W':
            return np.rot90(array,3)
        else:
            return array
    def AddAgents(self,agents):
        """Add List of Agents to the world
        Args:
            * agents: [agent1,agent2,...]
        Exceptions:
            * AssertionError: if agents wasn't a list"""
        assert type(agents) is list, "agents are not list"
        for a in agents:
            self.agents[a.ID]= a

    def AddFoods(self,foods):
        """Add List of foods to the world
        foods: [food1,food2,...]"""
        assert type(foods) is list, "foods are not list"
        for a in foods:
            self.foods[a.ID]= a

    def AddObstacles(self,obstacles):
        """Add List of obstacles to the world
        obstacles: [obstacle1,obstacle2,...]"""
        assert type(obstacles) is list, "obstacles are not list"
        for a in obstacles:
            self.obstacles[a.ID]= a

    #Randomly get and an index of empty place
    def GetRandomPlace(self):
        """Get occupied random place from world map
        Return: row,column"""
        #Another implementation of the method
        zeros = np.where(self.world==0)
        #print zeros
        if len(zeros[0])<1:
            raise Exception('World Map is Full')
        coords =  np.random.randint(0,len(zeros[0]))
        #print coords,len(zeros[0]),zeros[0][coords],zeros[1][coords]
        r = zeros[0][coords]
        c=zeros[1][coords]
        return r,c

    #Get domain around center in case of odd number
    @staticmethod
    def GetOddDomain(number,center):
        """Get [start,end] domain for odd number , example (number-> 3,center -> 0) -> [-1,2] which is 3 elements in total (-1,0,1) around the center 0
        Args :
            * number: how many elements wanted.
            * center: the value of the center
        Return: start,end of type int"""
        number = floor(number/2)
        return center-number,center+number+1

    # Place Obstacle no matter of it's shape on world map
    def PlaceObstacle(self,center,key):
        """Put an obstacle on the world map
        Args:
            * center: (row,column)
            * key: obstacle ID"""
        osh,oid = self.obstacles[key].Shape.shape,self.obstacles[key].ID
        rmin,rmax,cmin,cmax = 0,0,0,0
        # Row obstacles has shape (x,)-> become (1,x)
        # Columns obstacles has shape (x,1) So we change the R
        modified = False
        if len(osh)==1:
            osh = (1,osh[0])
            modified=True

        #Fix the domain for rows
        if osh[0]%2==0:
            rmin,rmax = self.GetOddDomain(osh[0]-1,center[0])
            rmax+=1
        else:
            rmin,rmax = self.GetOddDomain(osh[0],center[0])

        #Regulize domains
        rmin = 0 if rmin<0 else rmin
        rmax = self.world.shape[0] if rmax>=self.world.shape[0] else rmax

        #Fix the domain for columns
        if osh[1]%2==0:
            cmin,cmax = self.GetOddDomain(osh[1]-1,center[1])
            cmax+=1
        else:
            cmin,cmax = self.GetOddDomain(osh[1],center[1])

        #Regulaize domains
        cmin = 0 if cmin<0 else cmin
        cmax = self.world.shape[1] if cmax>=self.world.shape[1] else cmax
        rmin,cmin,rmax,cmax = int(rmin),int(cmin),int(rmax),int(cmax)
        updatearea = self.world[rmin:rmax,cmin:cmax]

        # Deal with obstacles
        if modified:
            ToChange =  self.obstacles[key].Shape[0:updatearea.shape[1]]
        else:
            ToChange =  self.obstacles[key].Shape[0:updatearea.shape[0],0:updatearea.shape[1]]
        # Update the Vision Border Map(VBM)
        ## the area that would be updated * the area that should be changed -> new look
        newlook= np.logical_and((updatearea==0),ToChange)
        ## Take into account the type of obstacle (water or solid)
        newlook = newlook * (not self.obstacles[key].See)
        ## Combine the the current state of the important area with the new look. 
        self.VBM[rmin:rmax,cmin:cmax] =np.logical_or(self.VBM[rmin:rmax,cmin:cmax],newlook )
        #Assigen Obstacle ID to valid places on the world map
        updatearea  += np.logical_and((updatearea==0),ToChange)*key

        #print rmin,rmax,cmin,cmax
        #print 'Center:{},OShape:{},OID:{},Result Shape:{}'.format((center[0],center[1]),osh,oid,updatearea.shape)
        #print ToChange
    
    def PlaceAgents(self):
        """Assigen Places to Agents"""
        for key in self.agents.keys():
            c = self.GetPlace(Name=self.agents[key].IndiciesOrderName)
            self.world[c] = self.agents[key].ID
            self.VBM[c] = not self.agents[key].See
        #print '{}: Placed'.format(len(self.agents.keys()))
        
    def PlaceFoods(self):
        """Assigen places to Foods"""
        for key in self.foods.keys():
            c = self.GetPlace(Name=self.foods[key].IndiciesOrderName)
            self.world[c] = self.foods[key].ID
            self.VBM[c] = not self.foods[key].See
        #print '{}: Foods Placed'.format(len(self.foods.keys()))
    
    def GetPlace(self,Name=None):
        """Get valid place for obstacle (Totally random or depending or probability distribution Provided to the obstacle it self)
        Args:
            * key: Obstacle ID"""
        if Name:

            #Get The Probability distribution to use 
            tmp = Settings.ProbabilitiesTable[Name]

            #Get Total possible elements
            count = np.sum(tmp>0)

            #Get Indicies of Random Choices the corrospond to the probabilities.
            choice = np.unravel_index(np.random.choice(Settings.WorldSize[0]*Settings.WorldSize[0],replace=False,size=count,p=tmp),Settings.WorldSize)

            #Try all possible choices.
            for i in range(choice[0].shape[0]):
                if self.world[choice[0][i],choice[1][i]]==0:
                    return choice[0][i],choice[1][i]

        return self.GetRandomPlace()

    def PlaceObstacles(self):
        """Places  to obstacles""" 
        for key in self.obstacles.keys():
            
            self.PlaceObstacle(self.GetPlace(Name=self.obstacles[key].IndiciesOrderName),key)
        #print '{}: Obstacles Placed'.format(len(self.obstacles.keys()))

    def GenerateWorld(self):
        """Description: Generate the world depending on the inserted data"""
        self.StepCounter=0
        self.AfterEndSteps = [self.AES]
        self.Terminated = [False]
        self.world.fill(0)
        self.VBM.fill(0)
        #map(lambda x: self.agents[x].FullReset(),self.agents.keys())
        for x in self.agents.keys():
            self.agents[x].FullReset()

        self.PlaceAgents()
        self.PlaceFoods()
        self.PlaceObstacles()

    #Get The Image for single Element
    def GetElementImage(self,ID):
        """Get element image from element id
        Args:
            * ID: Element ID
        Return: Image in numpy array"""
        
        if ID <=0:
            return Settings.Images[ID]
        #Agent ID domain start from 1000
        if ID<=2000:
            return self.agents[ID].GetImage()
        #Food ID domain (types)2000
        if ID<=3000:
            return Settings.Images[self.foods[ID].Image]
        #Barriers ID domain (types) 3000
        if ID>3000:
            return Settings.Images[self.obstacles[ID].Image]

    #This is the Slowest Part Enhancement could be applied
    # Simply linking values with images in hash table would enhance the performance
    def BuildImage(self):
        """Build map image
        Return: ((WorldSize * BlockSize)*3) Image"""
        totalrows=[]
        for row in range(self.world.shape[0]):
            cells=[]
            for column in range(self.world.shape[1]):
                cells.append(self.GetElementImage(self.world[row,column]))
            totalrows.append(np.concatenate(cells,axis=1))
        return np.concatenate(totalrows,axis=0)

    def AgentViewPoint(self,ID):
        """Build image from agent prespictive
        Args:
            * ID: agent ID
            * EgoCentric: True EgoCentric View , False: Normal map view
        Return: 
            * ((WorldSize * BlockSize)*3) Image"""
        array = self._GetAgentMap(ID)
        totalrows=[]
        for row in range(array.shape[0]):
            cells=[]
            for column in range(array.shape[1]):
                cells.append(self.GetElementImage(array[row,column]))
            totalrows.append(np.concatenate(cells,axis=1))
        
        if self.agents[ID].EgoCentric:
            return self.Ego_centric_array(self.agents[ID].Direction,np.concatenate(totalrows,axis=0))
        return np.concatenate(totalrows,axis=0)
    
    def _GetAgentMap(self,ID):
        """Return map from Agent Prospective.
        Args:
            * ID: agent ID
            * EgoCentric: True EgoCentric View , False: Normal map view
        Return: 
            * Array [worldsize/(2*worldsize -1)]"""
        agnt = self.agents[ID]
        if agnt.EgoCentric:
            array =agnt.FullEgoCentric
        else:
            array = agnt.FullEgoCentric[agnt.borders[0]:agnt.borders[1],agnt.borders[2]:agnt.borders[3]]
        return array

    #### Main Function Consume All Agents NextActions #######
    def Step(self):
        """Execute all agents actions, calculate there vision, and ditribute rewards"""
        self.StepCounter+=1
        for x in self.agents.keys():
            self.agents[x].Reset()

        for x in self.agents.keys():
            self._ConsumAgentActions(x)
        #Vision Should be done after all the agents done there moves.
        for x in self.agents.keys():
            self._GetTotalVision(x)

        for x in self.agents.keys():
            self._AgentNNInput(x)

        self.RF(self.agents,self.foods,self.RewardsScheme,self.world,self.AfterEndSteps,self.Terminated)
        if self.Limited:
            self.Terminated[0]= (self.Limited and (self.StepsLimit<self.StepCounter)) or self.Terminated[0]

    def _ConsumAgentActions(self,ID):
        """ Consume and clear agent actions queue
        Args:
            * ID: Agent ID"""
        for x in self.agents[ID].NextAction:
            self.DoAction(ID,x)
        #map(lambda x: self.DoAction(ID,x),self.agents[ID].NextAction)
    
    ########## Agent Actions ###########
    # General Agent Action # 
    def DoAction(self,ID,Action):
        """Main action function that bind to all actions
        ID : the agent ID
        Action : (X,Y) where if 
                    X: L (Look) or M (Move) -> Y: ('N','E','W','S')
                       R (Rotate)-> Y:(R,L)"""
        self.options[Action[0]](ID,Action[1])

    # Look Actions #
    def Look(self,ID,Direction):
        """Change the agent Direction
        Args:
            * ID : the agent ID
            * Direction : 'W', 'E', 'N','S'"""
        self.agents[ID].Direction = Direction
    
    # Rotat Actions #
    def Rotate(self,ID,Direction):
        """Rotate in specific direction
        Args:
            * ID : the agent ID
            * Direction : 'R','L'"""
        self.Look(ID,self.RotationAction[(Direction,self.agents[ID].Direction)])

    # Move Actions #
    def Move(self,ID,Direction):
        """Move agent with ID in the specificed direction
        ID : the agent ID
        Direction : 'W', 'E', 'N','S'"""
        p = np.where(self.world==ID)
        p = (p[0][0],p[1][0])
        # using dictionary to fasten the operation
        dest = self.options[Direction](p)
        #print p , dest
        #Make sure destination within boundaries
        if dest[0]<0 or dest[1]<0 or dest[0]>self.world.shape[0]-1 or dest[1]>self.world.shape[1]-1:
            return

        #####  Notes ####
        #Description: Check Destination possibilities
        #Empty: We just move,
        #Agent: Should be discussed
        #Food: Eaten but what happen after it ?
        
        #Empty
        if self.world[dest]==0:
            #Move
            self.world[dest]=ID
            self.world[p] = 0
            self.VBM[dest] = not  self.agents[ID].See
            self.VBM[p] = False
            return

        #Agent
        if 0<self.world[dest]<=2000:
            #print 'Obs its an other agent'
            return
        
        #Food
        if self.world[dest]<=3000:
            self.agents[ID].IAteFoodID = self.world[dest]
            self.world[dest]=ID
            self.world[p]=0
            self.VBM[dest] = not  self.agents[ID].See
            self.VBM[p] = False
            return

        '''
        # Barriers
        #if self.world[dest]>3000:
        #    return
        '''

    def _NorthPosition(self,position):
        """Calculate north position depending on the current position
        Args :
            * position: current position
        Return:
            new position (Row,Column)"""
        return (position[0]-1,position[1])

    def _SouthPosition(self,position):
        """Calculate South position depending on the current position
        Args :
            * position: current position
        Return:
            new position (Row,Column)"""
        return (position[0]+1,position[1])

    def _WestPosition(self,position):
        """Calculate East position depending on the current position
        Args :
            * position: current position
        Return:
            new position (Row,Column)"""
        return (position[0],position[1]-1)
         
    def _EastPosition(self,position):
        """Calculate West position depending on the current position
        Args :
            * position: current position
        Return:
            new position (Row,Column)"""
        return (position[0],position[1]+1)
    ######### AGENTS VISION CALCULATION ######
    def _GetTotalVision(self,ID):
        """Get the image from agent ID prospective.
        Args:
            * ID: agent ID
        Return: 
            Worldsize map with the visibility of the agent."""
        agntcoords = self._GetElementCoords(ID,self.world)
        agnt= self.agents[ID]
        
        #Find borders of world in Egocentric vision For Current Agent Location
        b0 = agnt.CenterPoint[0]-agntcoords[0]
        b1 = agnt.CenterPoint[0]+(self.world.shape[0]-agntcoords[0])
        b2 = agnt.CenterPoint[1]-agntcoords[1]
        b3 = agnt.CenterPoint[1]+(self.world.shape[1]-agntcoords[1])
        borders = [b0,b1,b2,b3]
        agnt.borders = borders

        Rmin,Rmax,Cmin,Cmax = self._GetVisionBorders(agntcoords,agnt.Range,self.world.shape)

        tmp = FOV_SC(self.VBM[Rmin:Rmax,Cmin:Cmax])
        NAgCoord = self._GetElementCoords(ID,self.world[Rmin:Rmax,Cmin:Cmax])
        tmp.do_fov(NAgCoord[1],NAgCoord[0])

        #### MERGING LAYERS TO CREATE AGENT VISION ########
        #Prepare The values in this array
        tmpworld = np.array(self.world,copy=True)
        tmpworld[tmpworld==0]= -10
        #Find What is Visible By Light
        tmpworld[Rmin:Rmax,Cmin:Cmax]*= tmp.light
        #Find what is visible by face direction
        tmpworld *=agnt.VisionFields[agnt.Direction][borders[0]:borders[1],borders[2]:borders[3]]
        
        # what ever equal 0 is unobserved by agent (unobserved code is -1)
        tmpworld[tmpworld==0] = -1

        #Late Modification
        tmpworld2 = np.zeros(self.world.shape)
        tmpworld2.fill(-1)
        tmpworld2[Rmin:Rmax,Cmin:Cmax] = tmpworld[Rmin:Rmax,Cmin:Cmax]

        # what ever equal to -10 is empty space (empty space code is 0)
        tmpworld2[tmpworld2==-10]=0                   
        agnt.FullEgoCentric.fill(-1)
        agnt.FullEgoCentric[borders[0]:borders[1],borders[2]:borders[3]] = tmpworld2
        ######## REMOVE ME IF YOU WANT VISION TO COME BACK TO NORMAL######
        #if agnt.ID==1001:
        #    agnt.FullEgoCentric[borders[0]:borders[1],borders[2]:borders[3]] = np.array(self.world,copy=True)
    @staticmethod
    def _GetElementCoords(ID,array):
        """Get Element coordinates in specific array"""
        coords = np.where(array==ID)
        return coords[0][0],coords[1][0]
    
    @staticmethod
    def _GetVisionBorders(center,Range,shape):
        """Get the borders of vision or [limits] giving:
        Args:
            center: specificy the center coordinates example (0,0)
            range: integer specifiy how far from the center is the limit. (-1 infinit)
            shape: the array diminsions where the provided center belong too. Example: (10,10)
        Return:
            Rmin: minimum Row for agent vision
            Rmax: maximum Row for agent vision
            Cmin: minimum column for agent vision
            Cmax: maximum column for agent vision
        * Performance Could be improved."""
        Rmin = Rmax = center[0]
        Cmin = Cmax = center[1]
        #Full Range (Then Only field of vision affect the view.)
        if Range <=-1:
            Rmin = Cmin = 0
            Rmax = shape[0]
            Cmax = shape[1]
        #Specific Range
        else:
            Rmin -= Range
            Rmax += Range+1
            Cmin -= Range
            Cmax += Range+1

            ##Regulaize Parameters###
            Rmin = 0 if Rmin<0 else Rmin
            Rmax = Rmax if Rmax<=shape[0] else shape[0]
            Cmin = 0 if Cmin<0 else Cmin
            Cmax = Cmax if Cmax<=shape[1] else shape[1]
        return  Rmin,Rmax,Cmin,Cmax
            
    ######## Get Agent Reward #####
    def _EstimateRewards(self,agents,foods,rwrdschem,world,AES,Terminated):
        """Calculate All agents rewards
        Args:
            * agents: dictionary of agents contain all agents by ID
            * foods: dictionary of all foods
            * rwrdschem: Reward Schema (More info in World __init__)
            * world: World Map
            * AES: one element array
        TODO:
            * copy this function to class or __init__ documentation as example of how to build customer reward function
            * Assign Reward To Agents
            * Impelent the Food Reward Part depending on the decision of who take the food reward if two 
              agent exist in food range in same time
            * Change All Ranges to .ControlRange not (-1) it's -1 only for testing purpuse
            * Change Punish per step to not punish when agent do nothing"""
        # Check Agents in Foods Range
        def ResetagentReward(ID):
            #Punish for step 
            agents[ID].CurrentReward= rwrdschem[2] # -1 # rwrdschem[2] if len(agents[ID].NextAction)>0 else 0
            
        for x in agents:
            ResetagentReward(x)

        AvailableFoods = world[(world>2000)&(world<=3000)]
        if len(AvailableFoods)==0:
            AES[0]-=1
            Terminated[0]= True if AES[0]<=0 else Terminated[0]
        #If Food Could be eaten without being in agent vision activate this
        #for ID in AvailableFoods:
        #    foodcenter = World._GetElementCoords(ID,world)
        #    fborder = World._GetVisionBorders(foodcenter,foods[ID].Range,world.shape)
        #    crff = world[fborder[0]:fborder[1],fborder[2]:fborder[3]]
            #Find location of all elements between 0 and Food ID (2000 as default)
        #    agnts = crff[(crff>1000)&(crff<=2000)]
            
        #    for aID in agnts:
        #        agents[aID].CurrentReward+= foods[ID].Energy* rwrdschem[1]
        #        world[world==ID]=0
            
        for ID in agents.keys():
            if agents[ID].IAteFoodID >-1:
                agents[ID].CurrentReward+= foods[agents[ID].IAteFoodID].Energy* rwrdschem[1]
            agntcenter = World._GetElementCoords(ID,agents[ID].FullEgoCentric)
            aborder = World._GetVisionBorders(agntcenter,agents[ID].ControlRange,agents[ID].FullEgoCentric.shape)
            #print 'Control Range For Agent ID:',ID
            #Critical Area For Agent
            crfa = agents[ID].FullEgoCentric[aborder[0]:aborder[1],aborder[2]:aborder[3]]
            # List of Agents in Control Rane + In Vision Range
            
            for EnemyID in crfa[(crfa>1000)&(crfa<=2000)&(crfa!=ID)]:
                #If I have more power I punish
                if agents[ID].Power>agents[EnemyID].Power:
                    agents[EnemyID].CurrentReward+= rwrdschem[0]*agents[ID].Power

            #Activate this if Food reward when in vision   
    ######## Get NN INPUT #########
    def _AgentNNInput(self,ID):
        """Prepare variables for ANNIG variable to do it's job without the need to 'self'
        Args:
            * ID: agent ID
            * EgoCentric: True:EgoCentric or False:Normal"""
        array = self._GetAgentMap(ID)
        self.agents[ID].NNFeed= self.ANNIG(ID,array,self.agents)    

    def _FindAgentOutput(self,ID,array,agents):
        
        """ Generate the desired output for agent.NNFeed which will be provided later to neural network
        Args:
            * ID: agent ID
            * array: the world in the agent prospective
            * agents: Dictionary (Key: agent ID, Value: Agent Object)"""

        def _agentdirection(direction):
            """ Get a vector of current agent direction
            Args:
                * direction: direction ('N'or 'S' or 'W' or 'E')
            return:
                array (1,4) example [1,0,0,0]
            """
            return np.array([direction=='N',direction=='S',direction=='E',direction=='W'])
        #Used Ordered here so the output keys will be always in the same manner in case the values
        # feed to some network they will always be in same order.
        ls = OrderedDict()
        if agents[ID].EgoCentric:
            array = self.Ego_centric_array(agents[ID].Direction,array)
        #observed (True)/unobeserved(False) layer
        ls['observed']= (array!=-1)

        #My Place
        ls['mypos']= (array==ID)
        # in case of ego centric my orientation is not needed (alwasy looking forward).
        if agents[ID].EgoCentric:
            ls['myori']=[] 
        else:
            ls['myori']= _agentdirection( agents[ID].Direction)
        ls['obstacles'] = (array>3000)
        ls['food'] = np.logical_and(array>2000,array<3001)
        #Get list of only observed agents.
        observedagents = array[(array>1000)&(array<2000)]
        for oID in agents.keys():
            if oID == ID:
                continue
            if oID in observedagents:
                ls['agentpos{}'.format(oID)]= (array==oID)
                if agents[ID].EgoCentric:
                    ls['agentori{}'.format(oID)]= self.ego_map[(agents[ID].Direction,agents[oID].Direction)]
                else:
                    ls['agentori{}'.format(oID)]= _agentdirection(agents[oID].Direction)
                
            else:
                ls['agentpos{}'.format(oID)]= np.zeros(array.shape,dtype=bool)# (array==oID)
                ls['agentori{}'.format(oID)]=np.array([0,0,0,0],dtype=bool)
        #ls['zombiagent']=np.zeros(125,dtype=bool)
        return ls
