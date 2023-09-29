from city import City, Metropolis
from station import Point, Station, Stations_network
from learner import Learner
from nn import nn

import matplotlib.pyplot as plt

from dijstra import dijstra
import torch 
import numpy as np


class Coordinator:

    def __init__(self, name:str, size:int, starting_station: Station, metro_params, city_params, gamma, tau):

        self.time = 0

        #City params

        central_city = City(name, self.time, [[0,0]])

        self.size=size
        self.metropolis=Metropolis(central_city, [], self.time, size)

        self.p_center = city_params["p_center"]
        self.p_other = city_params["p_other"]
        self.p_new = city_params["p_new"]
        self.p_growth = city_params["p_growth"]

        for _ in range(15):
            self.metropolis.new_round(self.p_center, self.p_other, self.p_new)

        #Metro params

        self.stations_network=Stations_network([starting_station])
        self.learner=Learner(metro_params["k_stations"], gamma)
         
        self.speed_metro=metro_params["speed_metro"]
        self.r_stations=metro_params["r_stations"]
        self.k_stations=metro_params["k_stations"]
        self.speed_change=metro_params["speed_change"]
        self.speed_walk=metro_params["speed_walk"]
        self.r_walking=metro_params["r_walking"]
        self.k_walking= metro_params["k_walking"]

        self.info=torch.zeros((self.stations_network.n_stations, 13+4*self.k_stations)) #Will be a tensor stacking all info of each one of the stations

        self.nn=nn(self.k_stations)
        self.nn_target=nn(self.k_stations)

        self.tau = tau

        self.optimiser =torch.optim.Adam(self.learner.prediction_nn.parameters(), lr=0.001)


    def get_stations_info(self, n_trips:int):

        #First get geographical features (location, in main city or not...)

        self.info=torch.zeros((self.stations_network.n_stations, 13+4*self.k_stations))
        self.stations_network.set_neighbours(self.r_stations, self.k_stations) #set neighbours for each station

        #print(self.info.shape)
        for i in self.stations_network.all_stations:

            station = self.stations_network.all_stations[i]

            self.info[i, 0]=station.location[0]
            self.info[i, 1]=station.location[1]

            if station.location in self.metropolis.center.area:
                self.info[i, 2]=1

            if station.location[0]+15<self.size/2:
                self.info[i, 3]=self.metropolis.frame[(station.location[0]+15, station.location[1])]

            if station.location[1]+15<self.size/2:
                self.info[i, 4]=self.metropolis.frame[(station.location[0], station.location[1]+15)]

            if station.location[0]-15>-self.size/2:
                self.info[i, 5]=self.metropolis.frame[(station.location[0]-15, station.location[1])]

            if station.location[1]-15>-self.size/2:
                self.info[i, 6]=self.metropolis.frame[(station.location[0], station.location[1]-15)]

            n=0
            for neighb in station.neighbours:
                self.info[i, 7+n*4]=station.location[0]-neighb.location[0]
                self.info[i, 7+n*4+1]=station.location[1]-neighb.location[1]

                if neighb.line == station.line:
                    self.info[i, 7+n*4+2]=1

                if station.connected!={}:
                    self.info[i, 7+n*4+3]=1

                n+=1

            i+=1

        #Then, simulate trips to get information about the flow

        for i in range(n_trips):

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()
            initial=Point(i_initial, j_initial)
            final=Point(i_final, j_final)

            walking_time, metro_time, summary_metro = self.stations_network.get_fastest(initial, final, self.speed_metro, self.speed_change, self.speed_walk, self.r_walking, self.k_walking)
            point=final

            if metro_time != np.infty:
                while point!=initial:

                    new=summary_metro[point][1]

                    if point==final: #last station
                        if walking_time < metro_time:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations+2]+=1
                        else:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations+3]+=1

                    if new==initial: #first station
                        if walking_time < metro_time:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations]+=1
                        else:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations+1]+=1

                    else: #Just a station through
                        if walking_time < metro_time:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations+4]+=1
                        else:
                            self.info[self.stations_network.reverse_all_stations[new],7+4*self.k_stations+5]+=1


        self.info[:,-6:]/=n_trips

    def change_network(self, index:int , action:int):

        r = None
        i,j=self.stations_network.all_stations[index].location
        new_location=None

        if action<8:
            scale=10
        else:
            scale=25

        if action==0:
            r=0
        elif (action-1)%7==0 and i-scale>-self.size/2 and j-scale>-self.size/2:
            new_location = (i-scale,j-scale)
        elif (action-1)%7==1 and i-scale*np.sqrt(2)>-self.size/2:
            new_location = (i-scale*np.sqrt(2),j)
        elif (action-1)%7==2 and i-scale>-self.size/2 and j+scale<self.size/2:
            new_location = (i-scale,j+scale)
        elif (action-1)%7==3 and j+scale*np.sqrt(2)<self.size/2:
            new_location = (i,j+scale*np.sqrt(2))
        elif (action-1)%7==4 and i+scale<self.size/2 and j+scale<self.size/2:
            new_location = (i+scale,j+scale)
        elif (action-1)%7==5 and i+scale*np.sqrt(2)<self.size/2:
            new_location = (i+scale*np.sqrt(2),j)
        elif (action-1)%7==6 and i+scale<self.size/2 and j-scale>-self.size/2:
            new_location = (i+scale,j-scale)

        else:
            r=-1

        if new_location is not None:
            self.stations_network.make_new_station(int(new_location[0]), int(new_location[1]))

        if r is None:
            self.stations_network.build_graph(self.speed_metro, self.speed_change)
            self.get_reward(self.stations_network.all_stations[index], 100)
        
        if r is None:
            r=0


        #Also need to settle lines:

        if self.stations_network.all_stations[index].next is None:
            self.stations_network.all_stations[index].next = self.stations_network.all_stations[self.stations_network.n_stations-1]

        elif self.stations_network.all_stations[index].previous is None:
            self.stations_network.all_stations[index].previous = self.stations_network.all_stations[self.stations_network.n_stations-1]

        else:
            self.stations_network.make_change_station(index)

            #Now settle the previous/ next with next interchange
            self.stations_network.all_stations[self.stations_network.n_stations-1].next = self.stations_network.all_stations[self.stations_network.n_stations-2]
            self.stations_network.all_stations[self.stations_network.n_stations-2].previous = self.stations_network.all_stations[self.stations_network.n_stations-1]

        return r

    def change_metropolis(self):

        #print(self.stations_network.all_stations)
        self.metropolis.new_round(self.p_center, self.p_other, self.p_new)

        for index in self.stations_network.all_stations:
            #print("NEW STATION:", self.stations_network.all_stations[index].location)
            self.metropolis.grow_station(self.stations_network.all_stations[index], self.p_growth)

    def get_reward(self, station: Station, n_trips:int): 

        reward=0

        for i in range(n_trips):

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()
            initial=Point(i_initial, j_initial)
            final=Point(i_final, j_final)

            walking_time, metro_time, summary_metro = self.stations_network.get_fastest(initial, final, self.speed_metro, self.speed_change, self.speed_walk, self.r_walking, self.k_walking)
            point=final

            if summary_metro[point][0] != np.infty:
                while point!=initial:

                    new=summary_metro[point][1]

                    if point==final and new==station: #last station
                        reward+=1

                    if new==initial and point==station: #first station
                        reward+=1

                    elif new==station: #Just a station through
                        reward+=0.8

                    point = new
                

        return reward/n_trips

    def backprop(self, L):

        #Backprop on nn
        self.optimiser.zero_grad()
        L.backward()
        self.optimiser.step()

        #Update nn_target
        for target_param, param in zip(self.learner.target_nn.parameters(), self.learner.prediction_nn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def feed_play(self, state: torch.tensor, n_selected:int, n_allowed_per_play: int, epsilon: float):

        self.time+=1
        L = None

        actions_left=1 #this will decrease the more actions we deecide to do.

        #Select the desired station
        done=[-1]
        for i in range(n_selected):

            ind=i
            if i>=state.shape[0]:
                break
            elif n_selected>state.shape[0]:

                ind=-1
                while ind in done:
                    ind = np.random.randint(0, state.shape[0])

            done.append(ind)

            #Play action:
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            row = state[i,:]
            row = row.reshape(1,-1)

            vec = torch.cat((remaining, row), axis=1)

            self.learner.predict(vec, epsilon)
            action = self.learner.action

            if action != 0:
                actions_left-=1/n_allowed_per_play

            r=self.change_network(i, action)
            self.change_metropolis()

            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            vec = torch.cat((remaining, row), axis=1)

            self.learner.target(vec, r)

            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()

        return L

    def step(self, n_iter:int):

        for _ in range(2):

            self.metropolis.new_round(self.p_center, self.p_other, self.p_new)

        for i in range(n_iter):

            print("iteration_coord:", i)

            self.get_stations_info(50)

            L = self.feed_play(self.info, 5, 2, 0.1)
            #This one also modifies the state of stations network and city

            self.backprop(L)


    def display(self):

        density = self.metropolis.frame
        plt.imshow(density, cmap='viridis', origin='lower')

        self.stations_network.display(self.size)
        points_dict = self.stations_network.display_lines

        print(points_dict)

        x_coords = []
        y_coords = []

        for point in points_dict:
            x_coords.append(points_dict[point][0])
            y_coords.append(points_dict[point][1])
        
        for key, points in points_dict.items():
            #x_coords, y_coords = zip(*points)  # Unzip the list of tuples into two lists
            plt.scatter(x_coords, y_coords)  # Scatter plot for points

        plt.colorbar(label='Density')
        plt.title('Density and Points')
        plt.xlabel('x')
        plt.ylabel('y')

        # Add legend to describe each group
        plt.legend()

        # Show the plot
        plt.show()


###############################


metro_params={
    "speed_metro" : 5,
    "speed_change" : 2,
    "speed_walk" : 1,

    "r_stations" : 20,
    "k_stations" : 4,

    "r_walking" : 30,
    "k_walking" : 2,
}

city_params={
    "p_center" : 0.3,
    "p_other" : 0.1,
    "p_new" : 0.7,
    "p_growth" : 0.3,
}

coord = Coordinator("dummy", 500, Station(10, 5), metro_params, city_params, 0.3, 0.1)

coord.step(4)
coord.display()
