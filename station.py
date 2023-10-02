import numpy as np
import copy

#Why cannot import ? 
def euclidean(a,b):
    result=0
    for i in range(2):
        result+=(a[i]-b[i])**2
    return np.sqrt(result)

#from city import euclidean
from typing import List
from dijstra import dijstra


class Point:

    def __init__(self, i, j):
        self.location=(i,j)
        self.neighbours={} #for the neighbouring stations
        #coordinator builds it. 


class Station(Point):

    def __init__ (self, i, j):
        super().__init__(i, j)

        self.line=None

        #Initialise the sequence of stations
        self.previous=None 
        self.next=None

        #Initialise the stations on oher lines to which it is connected:
        self.connected={} #station: length

class Line:

    def __init__(self):
        self.stations=[]


class Stations_network:

    def __init__(self, initial_set: List[Point]):

        self.n_stations=len(initial_set)
        self.all_stations={}
        self.reverse_all_stations={}

        i=0
        for p in initial_set:
            self.all_stations[i]=p
            self.reverse_all_stations[p]=i
            i+=1

        self.V=[]
        self.E=[]

        self.display_lines = {}

    def set_neighbours(self, r:float, k:int):
        #For any station, find at most k neighbours within radius r. Neighbours are not necessarilly connected by a line
        #Change this to be able to set the lines => just use the previous and next
        for station_ind in range(len(self.all_stations)):
            neighb = {}
            if self.all_stations[station_ind].previous is not None:
                neighb[self.all_stations[station_ind].previous]=euclidean(self.all_stations[station_ind].location, self.all_stations[station_ind].previous.location)

            if self.all_stations[station_ind].next is not None:
                neighb[self.all_stations[station_ind].next]=euclidean(self.all_stations[station_ind].location, self.all_stations[station_ind].next.location)

            self.all_stations[station_ind].neighbours=neighb

    def set_neighbours2(self, r:float, k:int):
        #For any station, find at most k neighbours within radius r. Neighbours are not necessarilly connected by a line
        #Change this to be able to set the lines==> Graph will be made from the lines
        for station_ind in range(len(self.all_stations)):

            neighbours={}

            for other_ind in range(len(self.all_stations)):
                if station_ind!=other_ind and euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)<=r:

                    if len(list(neighbours.keys()))<k:
                        neighbours[self.all_stations[other_ind]]=euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    elif euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)<neighbours[list(neighbours.keys())[-1]]:
                        del neighbours[list(neighbours.keys())[-1]]
                        neighbours[self.all_stations[other_ind]]=euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    
            self.all_stations[station_ind].neighbours=neighbours

    def build_graph(self, speed_metro, speed_change):
        #Makes the graph of stations

        V=[]
        E={}

        for station_ind in self.all_stations:

            station = self.all_stations[station_ind]

            V.append(station)


            if station.previous is not None:
                if (station, station.previous) not in list(E.keys()) or (station.previous, station) not in list(E.keys()):
                    E[(station, station.previous)]=euclidean(station.location, station.previous.location)/speed_metro
                    #print("adding1", (station, station.previous))
                    #print("add1", station.previous in list(self.all_stations.values()))

            if station.next is not None:
                if (station, station.next) not in list(E.keys()) or (station.next, station) not in list(E.keys()):
                    E[(station, station.next)]=euclidean(station.location, station.next.location)/speed_metro
                    #print("adding2", (station, station.next))
                    #print("add2", station.next in list(self.all_stations.values()))

            if station.connected!={}:
                #print(station.connected)
                for other in list(station.connected.keys()):
                    if (station, other) not in list(E.keys()) or (other, station) not in list(E.keys()):
                        E[(station, other)]=2/speed_change
                        #print("adding3", (station, other))
                        #print("add3", other in list(self.all_stations.values()))


            self.V=V
            self.E=E
            #print("E", E)

            

    def get_fastest(self, a: Point, b:Point, speed_metro, speed_change, speed_walk, r_walking, k_walking):

        # This function does everything: building the stations graph, getting the k-neighbours in r-radius for initial and final points
        # And of course computes the fastest distance between the 2.

        #First set the neighbouring stations from points a and from points b:
        all_neighbours=[]
        for p in [a, b]:

            neighbours={}

            for station_ind in self.all_stations:
                if euclidean(p.location, self.all_stations[station_ind].location)<=r_walking:

                    if len(list(neighbours.keys()))<k_walking:
                        neighbours[station_ind]=euclidean(self.all_stations[station_ind].location, p.location)/speed_walk
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    elif euclidean(p.location, self.all_stations[station_ind].location)<list(neighbours.keys())[-1]:
                        del neighbours[list(neighbours.keys())[-1]]
                        neighbours[station_ind]=euclidean(self.all_stations[station_ind].location, p.location)/speed_walk
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    
            all_neighbours.append(neighbours)

        #print(all_neighbours)
        #print("heelloooooo")
        #Now build the stations graph
        self.build_graph(speed_metro, speed_change)

        V=self.V
        E=self.E
        #NNNNNNEEEDDDD TO SET NEIGBOURS 
        #Add the points into the stations graph

        V.append(a)
        V.append(b)

        points=[a,b]

        for point in range(2):
            for neighb in all_neighbours[point]:
                #BE CAREFUL: neighbours is with indices
                E[(points[point],self.all_stations[neighb])]=all_neighbours[point][neighb]/speed_walk

        #Run Dijstra on this graph and compare:

        metro_time, summary_metro=dijstra(V,E,a,b)
        walking_time=euclidean(a.location, b.location)/speed_walk

        return (walking_time, metro_time, summary_metro)

    def make_new_station(self, i, j):

        self.n_stations+=1
        self.all_stations[self.n_stations-1]=Station(i,j)
        print("new1", self.n_stations)

    def make_change_station(self, index:int):

        self.n_stations+=1
        self.all_stations[self.n_stations-1]=copy.deepcopy(self.all_stations[index])
        print("new2", self.n_stations)

        self.all_stations[self.n_stations-1].previous=None
        self.all_stations[self.n_stations-1].next=None

        self.all_stations[index].connected[self.all_stations[self.n_stations-1]]=2

        #Add to all connections
        #Need this step of reconstituting the connected dictionary to help with the hashing
        for connection in self.all_stations[index].connected:
            connection.connected = {}
            connection.connected[self.all_stations[index]] = 2
            for other in self.all_stations[index].connected:
                if other!=connection:
                    connection.connected[other]=2

            #print("if add", connection, connection.connected, connection in list(self.all_stations.values()))

    def display(self, size):

        #assembles stations by lines
        visited = []
        unvisited = []
        for station_ind in self.all_stations:
            unvisited.append(self.all_stations[station_ind])

        lines={}
        n_lines=1

        while len(visited)<self.n_stations:

            new_line=[]
            
            index = np.random.randint(len(unvisited))
            station = unvisited[index]

            #print("unvisited", unvisited)
            #print("station", index, len(visited), self.n_stations)
            #print("intermediate", lines)

            if station.previous is not None:

                while station.previous is not None:
                    station = station.previous
                    #print("previous loop", station.location)

                ok =1
                while ok==1:

                    #print("next loop", station.location, station.connected)

                    i = station.location[0] + size/2
                    j = station.location[1] + size/2

                    new_line.append((i,j))

                    new = []
                    seen = 0
                    for sta in unvisited:
                        if seen==0:
                            if sta.location!=station.location or sta.next!=station.next or sta.previous!=station.previous:
                                new.append(sta)
                                seen=1
                        else:
                            new.append(sta)

                    unvisited=new

                    visited.append(station)

                    if station.next is not None:
                        station = station.next

                    else: #No more station
                        ok=0



            if station.previous is None and station.next is None:

                #print("appart", station.location)

                i = station.location[0] + size/2
                j = station.location[1] + size/2

                new_line.append((i,j))

                new = []
                seen = 0
                for sta in unvisited:
                    if seen==0:
                        if sta.location!=station.location or sta.next!=station.next or sta.previous!=station.previous:
                            new.append(sta)
                            seen=1
                    else:
                        new.append(sta)

                unvisited=new
                visited.append(station)

            found=0
            for line_ind in lines:
                if new_line == lines[line_ind]:
                    found=1
            if found==0 and new_line!=[]:
                lines[n_lines]=new_line
                n_lines+=1


        self.display_lines = lines
