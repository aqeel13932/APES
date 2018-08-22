import numpy as np
from time import time
np.set_printoptions(linewidth=200)
#Function to generate the (Flate position) out of index(r,c) position.
def nl(indx,shape):
    return np.ravel_multi_index(indx,shape)

def Get_Surrondings(location,shape,wm):
    def ValidateLocation(location):
        if wm[location]<3000:
            return True
        else:
            return False
    #Up,Down,Left,Right,Same location.
    AllPossibilities = 5.0
    #Possible Sides
    ActualPossibilities=0
    #output positions
    op=[]
    #output values
    pv=[]
    #Is Left Possible
    if location[0]>=1:
        #Is Left Empty
        if ValidateLocation((location[0]-1,location[1])):
            op.append((location[0]-1,location[1]))
            pv.append(1/AllPossibilities)
            ActualPossibilities+=1
    #Is Right Possible
    if location[0]<(shape[0]-1):
        #Is Right Empty
        if ValidateLocation((location[0]+1,location[1])):
            op.append((location[0]+1,location[1]))
            pv.append(1/AllPossibilities)
            ActualPossibilities+=1
    #Is Up Possible
    if location[1]>=1:
        #Up is Empty
        if ValidateLocation((location[0],location[1]-1)):
            op.append((location[0],location[1]-1))
            pv.append(1/AllPossibilities)
            ActualPossibilities+=1
    #Is Down Possible
    if location[1]<(shape[1]-1):
        #Is Down Empty
        if ValidateLocation((location[0],location[1]+1)):
            op.append((location[0],location[1]+1))
            pv.append(1/AllPossibilities)
            ActualPossibilities+=1
    #Stay at the same place (All Possibilities - Available possibilities)/All possibilities
    op.append(location)
    op= [nl(x,shape) for x in op]
    pv.append((AllPossibilities-ActualPossibilities)/AllPossibilities)
    return op,pv

def DPMP(wm,strtloc,endloc,wantedsteps):
    """Distribution Probability Map
    Args:
        * wm : world map
        * strtloc: start Location (start point for the agent)
        * endloc: The location where the game ends (food location)
        * wantedsteps: How many steps
    Return:
        * The probability of being at endlocation starting from start location after Wanted steps"""
    #We substract 1 because when we create the map it's already 1 step
    strtloc = nl(strtloc,wm.shape)
    endloc = nl(endloc,wm.shape)
    wantedsteps -=1
    G = np.zeros((wm.size,wm.size))
    for loc,val in np.ndenumerate(wm):
        if val<3000:
            strtpos = nl(loc,wm.shape)
            if strtpos==endloc:
                G[strtpos,endloc]=1
                continue
            op,pv = Get_Surrondings(loc,wm.shape,wm)
            for i in range(len(op)):
                G[strtpos,op[i]]=pv[i]
    output = G
    imp = G[strtloc]
    if wantedsteps==0:
        return G[strtloc,endloc]
    curr= G[strtloc]
    for j in range(wantedsteps):
        imp=output[strtloc]
        output = output.dot(G)
        curr = output[strtloc]
    return (curr-imp)[endloc]

if __name__ == "__main__":
    x= np.zeros((4,4))
    x[2,2]=3000
    print(DPMP(x,(0,0),(3,3),10))
    print(DPMP(x,(0,0),(3,3),10))

