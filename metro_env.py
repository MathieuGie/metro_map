import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import time

#from city import euclidean
from dijkstra import dijkstra

def euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

class Station():

    def __init__ (self, i, j):

        self.location = (i,j)
        self.line=None

        #Initialise the sequence of stations
        self.previous=None 
        self.next=None

        #Initialise the stations on oher lines to which it is connected:
        self.connected=set() #station: length

class Line:

    def __init__(self, number:int, starting:Station, ending:Station):

        self.number = number

        self.starting = starting
        self.ending = ending


class Stations_network:

    def __init__(self, size, initial, max_connected:int, speed_walk, speed_metro, speed_change, r_walking, k_walking, waiting_for_train, waiting_when_stopping):

        self.size=size
        initial_station = Station(initial[0], initial[1])

        self.all_stations=[initial_station]
        self.incomplete = [initial_station]
        self.complete = set() #For complete stations (so we don't build again at same place more)

        self.max_connected=max_connected

        self.lines = {}
        self.lines[1]=Line(1, initial_station, initial_station)
        initial_station.line = 1

        self.V=set()
        self.E={}

        self.speed_metro = speed_metro
        self.speed_change = speed_change
        self.speed_walk = speed_walk

        self.waiting_for_train = waiting_for_train
        self.waiting_when_stopping = waiting_when_stopping

        self.r_walking = r_walking
        self.k_walking = k_walking

    ################################################ 1.
    def make_change_station(self, station:Station):

        new = None

        if station not in self.complete and len(station.connected)<self.max_connected:

            new = Station(station.location[0], station.location[1])

            new.connected = copy.deepcopy(station.connected)
            new.connected.add(station)

            for co in set(station.connected):
                co.connected.add(new)

            station.connected.add(new)

            self.all_stations.append(new)
        
        else:
            self.complete.add(station)

        return new
    
    ################################################ 2.
    def make_new_station(self, station, i, j):

        new = Station(i,j)
        L = self.display(frame=False)

        #Previous
        if station.previous is None and new.location not in L[station.line]:
            station.previous=new
            new.next = station
            new.line = station.line

            if self.lines[station.line].starting == station:
                self.lines[station.line].starting = new

            self.all_stations.append(new)

            #print("adding previous", station.location, station.line, new.location, new.line)

        #Next
        elif station.next is None and new.location not in L[station.line]:
            station.next=new
            new.previous=station
            new.line = station.line

            if self.lines[station.line].ending == station:
                self.lines[station.line].ending = new

            self.all_stations.append(new)

            #print("adding next", station.location, station.line, new.location, new.line)

        #Co with available spot (previous or next)
        else:

            found=0
            if station.connected!=set():
                for co in set(station.connected):

                    if found==0 and co.previous is None and new.location not in L[co.line]:
                        found=1
                        co.previous=new
                        new.next = co
                        new.line = co.line

                        if self.lines[station.line].starting == co:
                            self.lines[co.line].starting = new

                        #print("adding previous to co", station.location,station.line,  new.location, new.line)

                        self.all_stations.append(new)


                    if found==0 and co.next is None and new.location not in L[co.line]:
                        found=1
                        co.next=new
                        new.previous=co
                        new.line = co.line

                        if self.lines[station.line].ending == co:
                            self.lines[co.line].ending = new

                        #print("adding next to co", station.location, station.line, new.location, new.line)

                        self.all_stations.append(new)

                    if found==1:
                        break

            if found==0:

                new_co = self.make_change_station(station)
                if new_co is not None: #If none, cannot, just whatever
                    new_co.previous = new
                    new.next = new_co

                    #print("add new co to", station.location, "with previous", new.location, "n_lines until now:", len(list(self.lines.keys())))

                    n_lines = len(list(self.lines.keys()))
                    
                    self.lines[n_lines+1] = Line(n_lines+1, new, new_co)
                    print("new line", new_co.location, new.location)

                    new_co.line = n_lines+1
                    new.line = n_lines+1

                    #print("new", new_co.location, new_co.line)
                    #print("new", new.location, new.line)

                    self.all_stations.append(new)


    ################################################ 3.
    def get_fastest(self, a, b):

        #First see if a or b is not connected to the graph (so no diksjtra)
        neighbours = [{},{}]
        done=[]
        i=0
        for p in [a, b]:
            for station in self.all_stations:

                if euclidean(p, station.location)<=self.r_walking:

                    if p not in done:
                        done.append(p)
        
                    if len(list(neighbours[i].keys()))<self.k_walking:
                        neighbours[i][(p, (station.location, station.line))]=euclidean(p, station.location)/self.speed_walk + self.waiting_for_train
                        neighbours[i]={k: v for k, v in sorted(neighbours[i].items(), key=lambda item: item[1])}

                    elif euclidean(p, station.location)<list(neighbours[i].values())[-1]:
                        del neighbours[i][list(neighbours[i].keys())[-1]]
                        neighbours[i][(p, (station.location, station.line))]=euclidean(p, station.location)/self.speed_walk + self.waiting_for_train
                        neighbours[i]={k: v for k, v in sorted(neighbours[i].items(), key=lambda item: item[1])}

            i+=1

        if len(done)!=2:

            walking_time=euclidean(a, b)/self.speed_walk
            return (walking_time, np.infty, None)
        
        else:

            self.V = set()

            for s in self.all_stations:
                self.V.add((s.location, s.line))

            self.V.add(a)
            self.V.add(b)

            self.E={}

            for line in self.lines:
                
                station = self.lines[line].starting
  
                while station is not None:

                    if station.next is not None and station.location!=station.next.location:
                        if ((station.location, station.line), (station.next.location, station.next.line)) not in list(self.E.keys()) and ((station.next.location, station.next.line), (station.location, station.line)) not in list(self.E.keys()):
                            self.E[((station.location, station.line), (station.next.location, station.next.line))]=euclidean(station.location, station.next.location)/self.speed_metro + self.waiting_when_stopping

                    if station.connected!=set():
                        for co in station.connected:

                            if co==station:
                                print("problem", co.location, co.line)
                            
                            if ((station.location, station.line), (co.location, co.line)) not in list(self.E.keys()) and ((co.location, co.line), (station.location, station.line)) not in list(self.E.keys()):
                                self.E[((station.location, station.line), (co.location, co.line))]=2/self.speed_change

                    station = station.next

            for i in range(2):
                for k in neighbours[i]:
                    self.E[k]=neighbours[i][k]


            if 0==1:

                print(self.display(False))
                time.sleep(2)
                # Create a graph object
                G = nx.Graph()

                # Add vertices and edges to the graph
                G.add_nodes_from(self.V)
                G.add_edges_from(self.E)

                color_map = {1: 'red', 2: 'orange', 3: 'yellow', 4:"lime", 5:"green", 6:"blue", 7:"cyan", 8:"magenta", 9:"purple", 10:"gray", 11:"black"}
                default_color = 'white'

                node_color = [
                    color_map.get(node[1], default_color)
                    if isinstance(node, tuple) and isinstance(node[0], tuple) else default_color
                    for node in G.nodes()
]
                # Draw the graph
                plt.figure(figsize=(8, 6))
                nx.draw(G, with_labels=True, node_color=node_color, node_size=500, font_size=8, font_weight='bold')
                plt.title("Graph Visualization")

                # Save the graph image to a file
                graph_image_path = '/Users/mathieugierski/Library/CloudStorage/OneDrive-Personnel/metro/metro_map/graph_visualization.png'  # Replace with your desired file path
                plt.savefig(graph_image_path)

            #Run Dijkstra on this graph and compare:
            metro_time, summary_metro=dijkstra(self.V,self.E,a,b)
            walking_time=euclidean(a, b)/self.speed_walk

            return (walking_time, metro_time, summary_metro)
        

    ################################################ 4.
    def display(self, frame=True):

        L = {}
        
        #### MAKE IT CLEANER!!
        if frame==True:
            mid = int(self.size/2)
        else:
            mid=0

        for line in self.lines:
            way=[]
            station = self.lines[line].starting

            while station is not None:
                way.append((station.location[0]+mid, station.location[1]+mid))
                station=station.next

            L[line] = way

        return L

