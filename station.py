import numpy as np

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

        self.lines=[]
        self.neighbours={}

        #Initialise the sequence of stations
        self.previous=None
        self.next=None

        #Initialise the stations on oher lines to which it is connected:
        self.connected={} #station: line_number

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

        for station_ind in range(len(self.all_stations)):

            neighbours={}

            for other_ind in range(len(self.all_stations)):
                if station_ind!=other_ind and euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)<=r:

                    if len(list(neighbours.keys()))<k:
                        neighbours[self.all_stations[other_ind]]=euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    elif euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)<list(neighbours.keys())[-1]:
                        del neighbours[list(neighbours.keys())[-1]]
                        neighbours[self.all_stations[other_ind]]=euclidean(self.all_stations[station_ind].location, self.all_stations[other_ind].location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    
            self.all_stations[station_ind].neighbours=neighbours

    def build_graph(self, speed_metro, speed_change):
        #Makes the graph of stations

        V=[]
        E={}

        for station in self.all_stations:

            V.append(station)

            if (station, station.previous) not in list(E.keys()) or (station.previous, station) not in list(E.keys()):
                E[(station, station.previous)]=speed_metro*euclidean(station.location, station.previous.location)

            if (station, station.next) not in list(E.keys()) or (station.next, station) not in list(E.keys()):
                E[(station, station.next)]=speed_metro*euclidean(station.location, station.next.location)

            if station.connected!={}:
                for other in list(station.connected.keys()):

                    if (station, other) not in list(E.keys()) or (other, station) not in list(E.keys()):
                        E[(station, other)]=speed_change*2

            self.V=V
            self.E=E

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
                        neighbours[station_ind]=euclidean(self.all_stations[station_ind].location, p.location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    elif euclidean(self.location, self.all_stations[station_ind].location)<list(neighbours.keys())[-1]:
                        del neighbours[list(neighbours.keys())[-1]]
                        neighbours[station_ind]=euclidean(self.all_stations[station_ind].location, p.location)
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    
            all_neighbours.append(neighbours)

        #Now build the stations graph
        self.build_graph(speed_metro, speed_change)

        V=self.V
        E=self.E

        #Add the points into the stations graph

        V.append(a)
        V.append(b)

        points=[a,b]

        for point in range(2):
            for neighb in all_neighbours[point]:
                E[(points[point],neighb)]=all_neighbours[point][neighb]*speed_walk

        #Run Dijstra on this graph and compare:

        metro_time, summary_metro=dijstra(V,E,a,b)
        walking_time=euclidean(a.location, b.location)*speed_walk

        return (walking_time, metro_time, summary_metro)

    def make_new_station(self, i, j):

        self.n_stations+=1
        self.all_stations[self.n_stations-1]=Station(i,j)
        self.reverse_all_stations[Station(i,j)]=self.n_stations-1

    def make_change_station(self, index:int):

        self.n_stations+=1
        self.all_stations[self.n_stations-1]=self.all_stations[index]

        self.all_stations[self.n_stations-1].connected[self.all_stations[index]]=2
        self.all_stations[index].connected[self.all_stations[self.n_stations-1]]=2

    def display(self):

        visited = []
        lines={}
        n_lines=0

        while len(visited)<self.n_stations:

            n_lines+=1
            station = min(list(self.all_stations.keys()))

            while station.previous is not None:
                station = station.previous

            lines[n_lines] = []

            while station.next is not None:

                lines[n_lines].append(station.location)
                station = station.next

        self.display_lines = lines

            
                


