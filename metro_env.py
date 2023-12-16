import numpy as np
import copy

#from city import euclidean
from dijstra import dijstra

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

    def __init__(self, initial, max_connected:int, speed_walk, speed_metro, speed_change, r_walking, k_walking):

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

        self.r_walking = r_walking
        self.k_walking = k_walking

    ################################################ 1.
    def make_change_station(self, station:Station):

        new = None

        if station not in self.complete and len(station.connected)<self.max_connected:

            new = Station(station.location[0], station.location[1])

            new.connected = station.connected
            new.connected.add(station)

            for co in set(station.connected):
                co.connected.add(new)

            station.connected.add(new)

            self.all_stations.append(new)

        return new
    
    ################################################ 2.
    def make_new_station(self, station, i, j):

        new = Station(i,j)

        #Previous
        if station.previous is None:
            station.previous=new
            new.next = station
            new.line = station.line

            if self.lines[station.line].starting == station:
                self.lines[station.line].starting = new

            self.all_stations.append(new)

            #print("adding previous", station.location, new.location)

        #Next
        elif station.next is None:
            station.next=new
            new.previous=station
            new.line = station.line

            if self.lines[station.line].ending == station:
                self.lines[station.line].ending = new

            self.all_stations.append(new)

            #print("adding next", station.location, new.location)

        #Co with available spot (previous or next)
        else:

            found=0
            if station.connected!=set():
                for co in set(station.connected):

                    if found==0 and co.previous is None:
                        found=1
                        co.previous=new
                        new.next = co
                        new.line = co.line

                        if self.lines[station.line].starting == co:
                            self.lines[co.line].starting = new

                        #print("adding previous to co", station.location, new.location)

                        self.all_stations.append(new)


                    if found==0 and co.next is None:
                        found=1
                        co.next=new
                        new.previous=co
                        new.line = co.line

                        if self.lines[station.line].ending == co:
                            self.lines[co.line].ending = new

                        #print("adding next to co", station.location, new.location)

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

                    new_co.line = n_lines+1
                    new.line = n_lines+1

                    #print("new", new_co.previous.location, self.lines[n_lines+1].starting.location)
                    #print("new", new.next.location, self.lines[n_lines+1].ending.location)

                    self.all_stations.append(new)

                
                else:
                    self.complete.add(station) #ONE DAY USE THIS TO LIMIT n of lines


        #print(new.location)
        #print("LINES", self.display(200))
        #print("\n")


    ################################################ 3.
    def get_fastest(self, a, b):

        #First see if a or b is not connected to the graph (so no diksjtra)
        neighbours = {}
        done=[]
        for p in [a, b]:
            for station in self.all_stations:

                if euclidean(p, station.location)<=self.r_walking:

                    if p not in done:
                        done.append(p)
        

                    if len(neighbours)<self.k_walking:
                        neighbours[(p, station)]=euclidean(p, station.location)/self.speed_walk
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

                    elif euclidean(p, station.location)<list(neighbours.values())[-1]:
                        del neighbours[list(neighbours.keys())[-1]]
                        neighbours[(p, station)]=euclidean(p, station.location)/self.speed_walk
                        neighbours={k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1])}

        if len(done)!=2:

            walking_time=euclidean(a, b)/self.speed_walk

            return (walking_time, np.infty, None)
        
        else:

            self.V = self.all_stations
            self.V.add(a)
            self.V.add(b)

            self.E={}

            for line in self.lines:
                
                station = self.lines[line].starting
  
                while station is not None:

                    if (station, station.next) not in list(self.E.keys()) and (station.next, station) not in list(self.E.keys()):
                        self.E[(station, station.next)]=euclidean(station.location, station.next.location)/self.speed_metro

                    if station.connected!=set():
                        for co in station.connected:
                            if (station, co) not in list(self.E.keys()) and (co, station) not in list(self.E.keys()):
                                self.E[(station, co)]=5/self.speed_change

                    station = station.next

            for k in neighbours:
                self.E[k]=neighbours[k]

            #Run Dijstra on this graph and compare:
            metro_time, summary_metro=dijstra(self.V,self.E,a,b)
            walking_time=euclidean(a.location, b.location)/self.speed_walk

            return (walking_time, metro_time, summary_metro)
        

    ################################################ 4.
    def display(self, size):

        L = {}
        mid = int(size/2)

        for line in self.lines:
            way=[]
            station = self.lines[line].starting

            while station is not None:
                way.append((station.location[0]+mid, station.location[1]+mid))
                station=station.next

            L[line] = way

        return L

