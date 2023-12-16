from city import City, Metropolis
from station import Point, Station, Stations_network
from learner import Learner
from nn import nn

import matplotlib.pyplot as plt

from collections import deque
from dijstra import dijstra
import torch 
import numpy as np
import random

def euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        # Adds an experience to the buffer. Oldest experiences are automatically dropped if capacity is reached.
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly samples a batch of experiences from the buffer.
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class Coordinator:

    def __init__(self, name:str, size:int, starting_station: Station, metro_params, city_params, gamma, tau, epsilon):

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

        self.starting_station = starting_station
        self.stations_network=Stations_network([starting_station])

        self.speed_metro=metro_params["speed_metro"]
        self.r_stations=metro_params["r_stations"]
        self.k_stations=metro_params["k_stations"]
        self.speed_change=metro_params["speed_change"]
        self.speed_walk=metro_params["speed_walk"]
        self.r_walking=metro_params["r_walking"]
        self.k_walking= metro_params["k_walking"]

        #Learner
        self.learner=Learner(self.k_stations, gamma)
        self.epsilon=epsilon

        self.info=torch.zeros((self.stations_network.n_stations, 33+3*self.k_stations)) #Will be a tensor stacking all info of each one of the stations

        self.nn=nn(self.k_stations)
        self.nn_target=nn(self.k_stations)

        self.tau = tau
        self.optimiser =torch.optim.Adam(self.learner.prediction_nn.parameters(), lr=0.001)

        self.average_reward = 0

        self.buffer = ReplayBuffer(1000)


    def get_stations_info(self, n_trips:int):

        #First get geographical features (location, in main city or not...)

        self.info=torch.zeros((self.stations_network.n_stations, 33+3*self.k_stations))
        self.stations_network.set_neighbours() #set neighbours for each station

        #print(self.info.shape)
        for i in self.stations_network.all_stations:

            station = self.stations_network.all_stations[i]

            self.info[i, 0]=station.location[0]/(self.size)+0.5
            self.info[i, 1]=station.location[1]/(self.size)+0.5


            if station.location in self.metropolis.center.area:
                self.info[i, 2]=1

            neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
            scales = [15, 25, 50]
            j=0

            for neighb in neighbours:
                for n in scales:
                    
                    if station.location[0]+n*neighb[0]<self.size/2 and station.location[0]+n*neighb[0]>-self.size/2 and station.location[1]+n*neighb[1]<self.size/2 and station.location[1]+n*neighb[1]>-self.size/2:
                        #print((station.location[0]+int(n*neighb[0]), station.location[1]+int(n*neighb[1])))
                        if (station.location[0]+int(n*neighb[0]), station.location[1]+int(n*neighb[1])) in list(self.metropolis.density.keys()):
                            #print("yes")
                            self.info[i, 3+j]=self.metropolis.density[(station.location[0]+int(n*neighb[0]), station.location[1]+int(n*neighb[1]))][0]
                    j+=1

            n=0
            #print("before")
            #print(self.metropolis.density)
            #print(self.metropolis.frame[station.location[0]-8:station.location[0]+8, station.location[1]-8:station.location[1]+8])
            #print(self.info[i,:])
            if station.previous is not None:
                self.info[i, 27+n*3]=1/50*(station.location[0]-station.previous.location[0])
            
            if station.next is not None:
                self.info[i, 27+n*3+1]=1/50*(station.location[1]-station.next.location[1])

            if station.previous is not None or station.next is not None:
                self.info[i, 27+n*3+2]=1

            n=1 #Because already seen for the station itself

            if station.connected!={}:
                for co in station.connected:


                    if co.previous is not None:
                        self.info[i, 27+n*3]=1/50*(co.location[0]-co.previous.location[0])
                    
                    if co.next is not None:
                        self.info[i, 27+n*3+1]=1/50*(co.location[1]-co.next.location[1])

                    if co.previous is not None or co.next is not None:
                        self.info[i, 27+n*3+2]=1

                    n+=1

            i+=1

        #Then, simulate trips to get information about the flow

        for i in range(n_trips):

            #redefine the reverse every time
            reverse_stations = {v:k for k,v in self.stations_network.all_stations.items()}

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()
            while euclidean((i_initial, j_initial),(i_final, j_final))<20:
                i_initial, j_initial = self.metropolis.pick_point()
                i_final, j_final = self.metropolis.pick_point()
            initial=Point(i_initial, j_initial)
            final=Point(i_final, j_final)

            walking_time, metro_time, summary_metro = self.stations_network.get_fastest(initial, final, self.speed_metro, self.speed_change, self.speed_walk, self.r_walking, self.k_walking)
            point=final

            #print(metro_time, summary_metro)

            if metro_time != np.infty:
                while point!=initial:

                    #print(initial.location, final.location, point.location, type(point))
                    new=summary_metro[point][1]

                    next=0
                    if point==final: #last station
                        next = 1

                    if next==1 and (point!=initial and point!=final):
                        next=0
                        if walking_time < metro_time:
                            self.info[reverse_stations[point],27+3*self.k_stations+2]+=1
                        else:
                            self.info[reverse_stations[point],27+3*self.k_stations+3]+=1

                    if new==initial and (point!=initial and point!=final): #first station
                        if walking_time < metro_time:
                            self.info[reverse_stations[point],27+3*self.k_stations]+=1
                        else:
                            self.info[reverse_stations[point],27+3*self.k_stations+1]+=1

                    elif (point!=initial and point!=final): #Just a station through, can also be the final due to next
                        if walking_time < metro_time:
                            self.info[reverse_stations[point],27+3*self.k_stations+4]+=1
                        else:
                            self.info[reverse_stations[point],27+3*self.k_stations+5]+=1

                    point=new
                    #print("new:", new.location)


        self.info[:,-6:]/=n_trips

    def change_network(self, index:int , action:int):

        r = None
        i,j=self.stations_network.all_stations[index].location
        new_location = None

        if action==0:
            r=0.01

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [10, 25, 50]
        act = 0


        for scale in scales:
            for neighb in neighbours:
                act+=1

                if action == act:
                    if i+scale*neighb[0]<self.size/2 and i+scale*neighb[0]>-self.size/2 and j+scale*neighb[1]<self.size/2 and j+scale*neighb[1]>-self.size/2:
                        new_location = (i+scale*neighb[0],j+scale*neighb[1])


        int_new_location = None
        if new_location is not None:
            int_new_location = (int(new_location[0]), int(new_location[1]))


        #Check that the new station is not already linked to the station itself or a connected version:
        already_found=0
        if self.stations_network.all_stations[index].previous is not None:
            i_pos = int(self.stations_network.all_stations[index].previous.location[0])
            j_pos = int(self.stations_network.all_stations[index].previous.location[1])
            if (i_pos,j_pos) == int_new_location:
                already_found=1

        if self.stations_network.all_stations[index].next is not None:
            i_pos = int(self.stations_network.all_stations[index].next.location[0])
            j_pos = int(self.stations_network.all_stations[index].next.location[1])
            if (i_pos,j_pos) == int_new_location:
                already_found=1

        if self.stations_network.all_stations[index].connected!={}:
            for co in self.stations_network.all_stations[index].connected:

                if co.previous is not None:
                    i_pos = int(co.previous.location[0])
                    j_pos = int(co.previous.location[1])
                    if co.previous.location == int_new_location:
                        already_found=1

                if co.next is not None:
                    i_pos = int(co.next.location[0])
                    j_pos = int(co.next.location[1])
                    if (i_pos, j_pos) == int_new_location:
                        already_found=1

        
        #IF ADD A NEW STATION:
        if new_location is not None and already_found==0: #Only add a new station if this new station not in the neeighbourhood already
            self.stations_network.make_new_station(int(new_location[0]), int(new_location[1]))

            #Also need to settle lines:
            if self.stations_network.all_stations[index].next is None:
                #print(new_location, "next of",self.stations_network.all_stations[index].location )
                self.stations_network.all_stations[index].next = self.stations_network.all_stations[self.stations_network.n_stations-1]
                self.stations_network.all_stations[self.stations_network.n_stations-1].previous = self.stations_network.all_stations[index]

            elif self.stations_network.all_stations[index].previous is None:
                #print(new_location, "previous of",self.stations_network.all_stations[index].location )
                self.stations_network.all_stations[index].previous = self.stations_network.all_stations[self.stations_network.n_stations-1]
                self.stations_network.all_stations[self.stations_network.n_stations-1].next = self.stations_network.all_stations[index]

            else:
                #If it is not a change station:
                if self.stations_network.all_stations[index].connected == {}:
                    self.stations_network.make_change_station(index)
                    #settle the previous/ next with next interchange
                    #print(self.stations_network.all_stations[index].location, "became connection with next being", new_location)
                    #########################
                    self.stations_network.all_stations[self.stations_network.n_stations-1].next = self.stations_network.all_stations[self.stations_network.n_stations-2]
                    self.stations_network.all_stations[self.stations_network.n_stations-2].previous = self.stations_network.all_stations[self.stations_network.n_stations-1]

                else:
                    found=0
                    for change in self.stations_network.all_stations[index].connected:
                        if change.previous is None and found==0:
                            found=1
                            change.previous = self.stations_network.all_stations[self.stations_network.n_stations-1]
                            self.stations_network.all_stations[self.stations_network.n_stations-1].next = change
                            #print(change.location, "found other with previous being", new_location)

                        elif change.next is None and found==0:
                            found=1
                            change.next = self.stations_network.all_stations[self.stations_network.n_stations-1]
                            self.stations_network.all_stations[self.stations_network.n_stations-1].previous = change
                            #print(change.location, "found other with next being", new_location)

                    if found==0 and len(list(self.stations_network.all_stations[index].connected))<self.k_stations:#Only add a new change station if less than k_stations connected togetther
                        self.stations_network.make_change_station(index)
                        #print("adding new connection of", self.stations_network.all_stations[self.stations_network.n_stations-1].location, "which has next:", new_location)
                        #settle the previous/ next with next interchange
                        self.stations_network.all_stations[self.stations_network.n_stations-1].next = self.stations_network.all_stations[self.stations_network.n_stations-2]
                        self.stations_network.all_stations[self.stations_network.n_stations-2].previous = self.stations_network.all_stations[self.stations_network.n_stations-1]


        if r is None and already_found==0:
            self.stations_network.build_graph(self.speed_metro, self.speed_change)
            r=self.get_reward(self.stations_network.all_stations[index], 2000)

        elif r is None and already_found==1:
            r=-0.01
        
        #print("r:", r)
        return r

    def change_metropolis(self):

        #self.metropolis.new_round(self.p_center, self.p_other, self.p_new)

        for index in self.stations_network.all_stations:
            
            if random.uniform(0,1)<0.7: #Don't grow stations everytime
                self.metropolis.grow_station(self.stations_network.all_stations[index], self.p_growth)

    def get_reward(self, station: Station, n_trips:int): 

        reward=0

        for _ in range(n_trips):

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()

            while euclidean((i_initial, j_initial),(i_final, j_final))<20:
                i_initial, j_initial = self.metropolis.pick_point()

            initial=Point(i_initial, j_initial)
            final=Point(i_final, j_final)

            walking_time, metro_time, summary_metro = self.stations_network.get_fastest(initial, final, self.speed_metro, self.speed_change, self.speed_walk, self.r_walking, self.k_walking)
            #point=final

            if 5*metro_time<walking_time:
                reward+=20
            elif 2*metro_time<walking_time:
                reward+=8
            elif metro_time<walking_time:
                reward+=3
            elif 0.75*metro_time<walking_time:
                reward+=1
                
                
        return reward/n_trips

    def backprop(self, L, time):

        #Backprop on nn
        self.optimiser.zero_grad()
        L.backward()
        self.optimiser.step()

        #Update nn_target

        if time%80==0:
            print("UPDATING TARGET")
            for target_param, param in zip(self.learner.target_nn.parameters(), self.learner.prediction_nn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def feed_play(self, n_selected:int, n_allowed_per_play: int, epsilon: float):

        self.time+=1
        L = None

        actions_left=1 #this will decrease the more actions we deecide to do.

        #Select the desired station
        done=set()
        done.add(-1) #Add a starting fake station

        for i in range(n_selected):

            #Make state s: (needed at every new step because changes are made)
            self.get_stations_info(1000)
            state = self.info

            if n_selected<=state.shape[0]: ###SMALLER
                ind=-1
                while ind in done:
                    ind = np.random.randint(0, state.shape[0])

            elif i==state.shape[0]: 
                break

            else:
                ind=i

            done.add(ind)

            #Play action:
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            row = state[i,:]
            row = row.reshape(1,-1)

            vec = torch.cat((remaining, row), axis=1)

            self.learner.predict(vec, epsilon)
            action = self.learner.action

            #When not allowed to play the action
            if action != 0 and actions_left<=0:
                actions_left=0
                action = 0
                r = 0

            #When action is playable
            else:
                if action != 0:
                    actions_left-=1/n_allowed_per_play

                r=self.change_network(i, action)

                self.change_metropolis()

            STATE = vec
            print(STATE)
            ACTION = action
            REWARD = r
            self.average_reward+=r

            ### REMAINING ###
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            #Make state s':
            self.get_stations_info(500)
            new_state = self.info

            new_row = new_state[i,:]
            new_row = new_row.reshape(1,-1)

            vec = torch.cat((remaining, new_row), axis=1)

            NSTATE = vec
            self.buffer.push((STATE, ACTION, REWARD, NSTATE))

            self.learner.target(vec, r)

            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()

            ###Now sample:

            samples = self.buffer.sample(64)

            for sample in samples:

                self.learner.predict_for_replay(sample[0], sample[1])
                self.learner.target(sample[3], sample[2])
                L+=self.learner.get_loss()

        
        self.metropolis.new_round(self.p_center, self.p_other, self.p_new)
        return L/(n_selected+64) #Need to divide

    def step(self, n_iter:int, time_target:int):


        for i in range(n_iter):

            print("iteration_coord:", i)

            #self.get_stations_info(50)

            L = self.feed_play(10, 7, self.epsilon)
            #This one also modifies the state of stations network and city
            self.epsilon*=0.999

            self.backprop(L, n_iter*time_target+self.time)


    def display(self):

        density = self.metropolis.frame
        plt.imshow(density, cmap='viridis', origin='lower')

        self.stations_network.display(self.size)
        LINES = self.stations_network.display_lines
        print(LINES)

        # Predefined list of colors.
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for idx, (key, line) in enumerate(LINES.items()):
            color = colors[idx % len(colors)]  # Cycle through colors if there are more lines than colors.
            for i in range(len(line)-1):
                x_values = [line[i][0], line[i+1][0]]
                y_values = [line[i][1], line[i+1][1]]
                plt.plot(x_values, y_values, color, label=f"Line {key}" if i == 0 else "", marker="o")

        plt.colorbar(label='Density')
        plt.title('Density and Points')
        #plt.xlabel('x')
        #plt.ylabel('y')

        # Add legend to describe each group
        plt.legend()

        # Show the plot
        plt.show()

    
    def reset(self, n_iter:int):

        self.average_reward /= n_iter
        out = self.average_reward

        self.average_reward=0
        self.time=0

        central_city = City("hello", self.time, [[0,0]])

        self.metropolis=Metropolis(central_city, [], self.time, self.size)

        for _ in range(15):
            self.metropolis.new_round(self.p_center, self.p_other, self.p_new)

        self.starting_station.previous=None
        self.starting_station.next=None
        self.starting_station.connected={}
        self.stations_network=Stations_network([self.starting_station])

        self.info=torch.zeros((self.stations_network.n_stations, 17+3*self.k_stations))

        return out



###############################


metro_params={
    "speed_metro" : 8,
    "speed_change" : 2,
    "speed_walk" : 1,

    "r_stations" : 50, #Useless
    "k_stations" : 3, #A change station has at most 3 connections

    "r_walking" : 15,
    "k_walking" : 2,
}

city_params={
    "p_center" : 0.2,
    "p_other" : 0.1,
    "p_new" : 0.1,
    "p_growth" : 0.5,
}

coord = Coordinator("dummy", 200, Station(0, 0), metro_params, city_params, 0.99, 0.4, 0.98)


all = []
time_target=0

for i in range(0):

    print("reset", i)
    coord.step(6, time_target)
    time_target+=1

    r = coord.reset(6)
    print("r:", r)
    print("epsilon:", coord.epsilon)
    #print(coord.stations_network.all_stations)
    all.append(r)

    # Generate your plot with the current state of your_list
    plt.figure()
    plt.plot(all)
    plt.title('Rewards')
    
    # Save the figure to the same file location every time
    plt.savefig('reward.png')
    
    # Close the figure to free memory
    plt.close()

#Last one to plot:
coord.step(10, time_target)


print("all_rewards:", all)


for station_ind in coord.stations_network.all_stations:
    print(coord.stations_network.all_stations[station_ind].location)
    if coord.stations_network.all_stations[station_ind].previous is not None:
        print("previous",coord.stations_network.all_stations[station_ind].previous.location)
    if coord.stations_network.all_stations[station_ind].next is not None:
        print("next",coord.stations_network.all_stations[station_ind].next.location)

"""
print("now comparing")
for ind in range(coord.stations_network.n_stations):
    for other_ind in range(coord.stations_network.n_stations):
        if ind>other_ind:
            print(coord.stations_network.all_stations[ind]==coord.stations_network.all_stations[other_ind])
"""

coord.display()