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

    def __init__(self, size, initial, max_connected:int, speed_walk, speed_metro, speed_change, r_walking, k_walking, waiting_for_train, waiting_when_stopping, make_connection_distance):

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
        self.make_connection_distance = make_connection_distance

        self.set_V_E()

    ################################################ 1.
    def make_change_station(self, station:Station):
        #adding a new line from a given station

        new = None
        #print("making a new", station.location, station.line)

        if station not in self.complete and len(station.connected)<self.max_connected:

            new = Station(station.location[0], station.location[1])

            new.connected = set()

            #They will get all the existing connections, if they are not at max capacity of connection
            new.connected.add(station)

            for co in set(station.connected):
                if co not in self.complete and new not in self.complete:

                    co.connected.add(new)
                    new.connected.add(co)

                    if len(co.connected)>=self.max_connected:
                        self.complete.add(co)

                    if len(new.connected)>=self.max_connected:
                        self.complete.add(new)
                    

            station.connected.add(new)

            if len(station.connected)>=self.max_connected:
                self.complete.add(station)

            self.all_stations.append(new)
        
        else:
            self.complete.add(station)

        return new
    
    ################################################ 2.
    def make_new_station(self, station, i, j, returning=False):

        already_exists=0
        for s in self.complete:
            if (i,j)==s.location:

                print("already one complete there")
                #Cannot put a station where there is a complete change already
                return None
            
        for s in self.all_stations:
            if (i,j)==s.location:

                if s.previous is not None:
                    if s.previous.location == station.location:
                        already_exists=1
                        print("already the same link found")
                        break

                if s.next is not None:
                    if s.next.location == station.location:
                        already_exists=1
                        print("already the same link found")
                        break

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

                    #print("hello", co.location, new.location)

                    if found==0 and co.previous is None and new.location not in L[co.line]:
                        found=1
                        co.previous=new
                        new.next = co
                        new.line = co.line

                        if self.lines[co.line].starting == co:
                            self.lines[co.line].starting = new

                        #print("adding previous to co", station.location,station.line,  new.location, new.line)

                        self.all_stations.append(new)


                    if found==0 and co.next is None and new.location not in L[co.line]:
                        found=1
                        co.next=new
                        new.previous=co
                        new.line = co.line

                        if self.lines[co.line].ending == co:
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
                    #print("new line", new_co.location, new.location)

                    new_co.line = n_lines+1
                    new.line = n_lines+1

                    #print("new", new_co.location, new_co.line)
                    #print("new", new.location, new.line)

                    self.all_stations.append(new)

                #Need to add this for the returning otherwise returns stupid thing
                elif returning:
                    print("did everything")
                    return None

        if returning:

            if already_exists==1:
                return (new, 0)
            else:
                #print("new", new.location, new.line)
                return new


    ################################################ 3.
            
    def set_V_E(self):

        self.V = set()
        self.E={}

        for s in self.all_stations:
            self.V.add((s.location, s.line))

        #Adding all edges of metro (between next and previous and connections)
        for line in self.lines:
            
            station = self.lines[line].starting

            while station is not None:

                if station.next is not None and station.location!=station.next.location:
                    if ((station.location, station.line), (station.next.location, station.next.line)) not in list(self.E.keys()) and ((station.next.location, station.next.line), (station.location, station.line)) not in list(self.E.keys()):
                        self.E[((station.location, station.line), (station.next.location, station.next.line))]=euclidean(station.location, station.next.location)/self.speed_metro + self.waiting_when_stopping

                if station.connected!=set():

                    if len(station.connected)>self.max_connected:
                        print("problem, station has too many connections")

                    for co in station.connected:

                        if co==station:
                            print("problem, station connected to itself", co.location, co.line)

                        
                        if ((station.location, station.line), (co.location, co.line)) not in list(self.E.keys()) and ((co.location, co.line), (station.location, station.line)) not in list(self.E.keys()):
                            self.E[((station.location, station.line), (co.location, co.line))]=(2+euclidean(station.location, co.location))/self.speed_change + self.waiting_for_train

                station = station.next

    def add_points(self, a, b):

        #Add points a anc b to V and E
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

            return None
  
        else:

            V = copy.deepcopy(self.V)
            E = copy.deepcopy(self.E)

            V.add(a)
            V.add(b)

            for i in range(2):
                for k in neighbours[i]:
                    E[k]=neighbours[i][k]

            return V, E


    """
    def get_fastest(self, a, b, display=False):

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
                #print("V", (s.location, s.line))
                #if s.line is None:
                    #print("ALERT", s.location, s.previous, s.next)

            self.V.add(a)
            self.V.add(b)

            self.E={}

            #Adding all edges of metro (between next and previous and connections)
            for line in self.lines:
                
                station = self.lines[line].starting
  
                while station is not None:

                    if station.next is not None and station.location!=station.next.location:
                        if ((station.location, station.line), (station.next.location, station.next.line)) not in list(self.E.keys()) and ((station.next.location, station.next.line), (station.location, station.line)) not in list(self.E.keys()):
                            self.E[((station.location, station.line), (station.next.location, station.next.line))]=euclidean(station.location, station.next.location)/self.speed_metro + self.waiting_when_stopping

                    if station.connected!=set():

                        if len(station.connected)>self.max_connected:
                            print("problem, station has too many connections")

                        for co in station.connected:

                            if co==station:
                                print("problem, station connected to itself", co.location, co.line)

                            
                            if ((station.location, station.line), (co.location, co.line)) not in list(self.E.keys()) and ((co.location, co.line), (station.location, station.line)) not in list(self.E.keys()):
                                self.E[((station.location, station.line), (co.location, co.line))]=(2+euclidean(station.location, co.location))/self.speed_change + self.waiting_for_train

                    station = station.next

            #Adding connections to the 2 given points and stations
            for i in range(2):
                for k in neighbours[i]:
                    self.E[k]=neighbours[i][k]


            if display is True:

                print(self.display(False))
                #time.sleep(5)
                # Create a graph object
                G = nx.Graph()

                # Add vertices and edges to the graph
                G.add_nodes_from(self.V)
                G.add_edges_from(self.E)

                color_map = {1:"red", 2:"darkorange", 3:"gold", 4:"yellow", 5:"lime", 6:"green", 7:"cyan",8:"dodgerblue", 
                             9:"blue",10:"purple",11:"blueviolet", 12:"magenta", 13:"pink",14:"crimson", 15:"maroon"}
                default_color = 'gray'

                node_color = [
                    color_map.get(node[1], default_color)
                    if isinstance(node, tuple) and isinstance(node[0], tuple) else default_color
                    for node in G.nodes()
]
                # Draw the graph
                plt.figure(figsize=(8, 6))
                nx.draw(G, with_labels=True, node_color=node_color, node_size=500, font_size=8, font_weight='bold')
                plt.title("Graph Visualization")

                metro_time, summary_metro=dijkstra(self.V,self.E,a,b)
                walking_time=euclidean(a, b)/self.speed_walk
                plt.text(0.5, 0.95, "walking: "+str(walking_time)+" , "+"metro: "+str(metro_time), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

                # Save the graph image to a file
                graph_image_path = '/Users/mathieugierski/Nextcloud/Macbook M3/metro/metro_map/graph_visualization.png'  # Replace with your desired file path
                plt.savefig(graph_image_path)

                plt.close()
                return 0

            #Run Dijkstra on this graph and compare:
            metro_time, summary_metro=dijkstra(self.V,self.E,a,b)
            walking_time=euclidean(a, b)/self.speed_walk

            #print("summary", walking_time, metro_time)

            return (walking_time, metro_time, summary_metro)

    """

    def get_fastest(self, a, b, display=False, run_number=None):

        connections = self.add_points(a, b)

        if connections is None:
            return (euclidean(a, b)/self.speed_walk, np.infty, None)
        
        else:

            if display is True:

                print(self.display(False))
                #time.sleep(5)
                # Create a graph object
                G = nx.Graph()

                # Add vertices and edges to the graph
                G.add_nodes_from(self.V)
                G.add_edges_from(self.E)

                color_map = {1:"red", 2:"darkorange", 3:"gold", 4:"yellow", 5:"lime", 6:"green", 7:"cyan",8:"dodgerblue", 
                             9:"blue",10:"purple",11:"blueviolet", 12:"magenta", 13:"pink",14:"crimson", 15:"maroon"}
                default_color = 'gray'

                node_color = [
                    color_map.get(node[1], default_color)
                    if isinstance(node, tuple) and isinstance(node[0], tuple) else default_color
                    for node in G.nodes()
]
                # Draw the graph
                plt.figure(figsize=(8, 6))
                nx.draw(G, with_labels=True, node_color=node_color, node_size=500, font_size=8, font_weight='bold')
                plt.title("Graph Visualization")

                metro_time, summary_metro=dijkstra(connections[0],connections[1],a,b)
                walking_time=euclidean(a, b)/self.speed_walk
                plt.text(0.5, 0.95, "walking: "+str(walking_time)+" , "+"metro: "+str(metro_time), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

                # Save the graph image to a file
                graph_image_path = '/Users/mathieugierski/Nextcloud/Macbook M3/metro/metro_map/graph_visualization_'+str(run_number)+'.png'  # Replace with your desired file path
                plt.savefig(graph_image_path)

                plt.close()
                return 0

            #Run Dijkstra on this graph and compare:
            metro_time, summary_metro=dijkstra(connections[0],connections[1],a,b)
            walking_time=euclidean(a, b)/self.speed_walk

            #print("summary", walking_time, metro_time)

            return (walking_time, metro_time, summary_metro)



    def get_dis_closest_station(self, point):

        max_dis = np.inf
        for station in self.all_stations:
            if euclidean(point, station.location)<=max_dis:

                max_dis = euclidean(point, station.location)

        return max_dis



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
    
    ################################################ 5.
    def station_already(self, location):

        found = 0
        found_complete = 0

        for station in self.all_stations:
            if euclidean(station.location, location)<=self.r_walking:
                found=1

                if station in self.complete:
                    found_complete=1
            
        return found, found_complete
    
    ################################################ 6.
    def make_connection_close(self, station:Station):
        #Take all closest stations and add as connected
        #Here we make a connection if stations are close enough but they are not necessarily exactly at the same spot on the map

        closest={}
        if len(station.connected)<self.max_connected:

            for s in self.all_stations:
                if euclidean(s.location, station.location)<=self.make_connection_distance and s.line!=station.line:
                    if len(s.connected)<self.max_connected and station not in s.connected and s not in station.connected and s!=station:
                
                        if len(list(closest.keys()))<self.max_connected:
                            closest[s]=euclidean(s.location, station.location)
                            closest = {k: v for k, v in sorted(closest.items(), key=lambda item: item[1])}
                        
                        elif euclidean(s.location, station.location)<list(closest.values())[-1]:
                            del closest[list(closest.keys())[-1]]
                            closest[s]=euclidean(s.location, station.location)
                            closest = {k: v for k, v in sorted(closest.items(), key=lambda item: item[1])}

            #Station can be connected to a and b but that does not imply that a and b are directly connected (though dijkstra can indirectly connect a and b through station)
            for s in closest:

                s.connected.add(station)
                station.connected.add(s)

                if len(s.connected)>=self.max_connected:
                    self.complete.add(s)

                if len(station.connected)>=self.max_connected:
                    self.complete.add(station)



    def is_on_shortest_path(self, a, b, v):

        connections = self.add_points(a, b)

        if connections is None:
            return None
        
        _, distance = dijkstra(connections[0], connections[1], a, b)
        if v not in distance:
            return None

        current = b
        path_nodes = [b]
        while current != a:
            current = distance[current][1]  # Move to the predecessor
            if current is None:
                return "not on path"  # If there's no predecessor, v isn't on the path
            path_nodes.append(current)

        if v in path_nodes:
            if path_nodes[1] == v and path_nodes[-2] == v:
                return 0
            elif path_nodes[-2] == v:
                return 1
            elif path_nodes[1] == v:
                return 2
            else:
                return 3
        else:
            return None




