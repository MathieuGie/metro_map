from typing import List, Dict
import numpy as np

def dijkstra(V, E, source, end):


    visited=[]
    #Source and end can not belong to V
    distance={source:[0,None], end:[np.infty, None]}
    
    for u in V:
        if u != source and u!=end:
            distance[u] = [np.infty, None]

    
    unvisited=[u for u in V]
    study=source

    while unvisited!=[]:

        #Find current distance from source
        dis=distance[study][0]

        #Find the neighbours of study
        neighbours={}

        for key in E:

            if study==key[0]:
                neighbours[key[1]]=E[key]

            elif study==key[1]:
                neighbours[key[0]]=E[key]

        #Update the distance from source
        for key in neighbours:

            if dis+neighbours[key]<distance[key][0]:
                distance[key]=[dis+neighbours[key], study]

        distance=dict(sorted(distance.items(), key=lambda item: item[1][0]))

        #Choose what to study next and update visited and unvisited:
        #print("remove", study, unvisited)
        unvisited.remove(study)
        visited.append(study)

        for key in list(distance.keys()):
            #print(key)
            if key in unvisited:
                study=key
                break

    return distance[end][0], distance

#V=[0,1,2,3,4,5,6,7,8,9,10]
#E={(0,1):1, (1,2):2, (0,3):6, (3,4):4, (4,5):1, (3,6):3, (2,7):3, (5,8):10, (7,8):20, (7,9):22, (8,9):1, (2,10):100, (9,10):1}
#print(dijkstra(V, E, 0,1))