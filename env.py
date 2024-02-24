from city_env import Metropolis, City
from metro_env import Station, Line, Stations_network

import numpy as np
import torch 
from collections import deque
import random
import copy
import time
import matplotlib.pyplot as plt

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



class Environment():

    def __init__ (self, size, n_simulations, city_center, city_facts, first_station, metro_facts):

        self.size = size
        self.n_simulations = n_simulations

        self.city_center = city_center
        self.p_center = city_facts["p_center"]
        self.p_new = city_facts["p_new"]
        self.p_station = city_facts["p_station"]
        self.at_most_new = city_facts["at_most_new"]

        self.first_station = first_station
        self.max_connected = metro_facts["max_connected"]
        self.speed_walk = metro_facts["speed_walk"]
        self.speed_metro = metro_facts["speed_metro"]
        self.speed_change = metro_facts["speed_change"]

        self.waiting_for_train = metro_facts["waiting_for_train"]
        self.waiting_when_stopping = metro_facts["waiting_when_stopping"]

        self.r_walking = metro_facts["r_walking"]
        self.k_walking = metro_facts["k_walking"]
        self.make_connection_distance = metro_facts["make_connection_distance"]

        self.p_selecting_station = metro_facts["p_selecting_station"]

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking, self.waiting_for_train, self.waiting_when_stopping, self.make_connection_distance)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

        self.info = None

        self.before_reward = 0

        #Get the max number of pixels that can be served by a station
        self.max_station_area = self.get_max_station_area()

        self.initial_points=[]
        self.final_points=[]
    
    ################################################ 1.
    def make_state(self, selected, n_iter, final, state_of_final):

        #T=[]
        #start_time = time.time()

        #T.append((time.time()-start_time, "beginning"))
        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [6, 12]

        #neighbours = [(1,0), (0,1), (-1,0), (0, -1)]
        #scales = [6]

        self.info=torch.zeros((1,211))
        J=0

        #0. If final or not:
        #tate_of_final is used to see whether we are at state t or targeting state t+1 for the final state.
        if final is True:
            self.info[0, J] = 1
            if state_of_final==1:
                self.info[0, J+1] = 1

        J+=2
        J_before = J

        #1. Location:
        
        lines_to_check = [selected.line]
        self.info[0, J]=selected.location[0]/(self.size)+0.5
        self.info[0, J+1]=selected.location[1]/(self.size)+0.5

        if selected.previous is None:
            self.info[0, J+2] = 1  
        else:
            self.info[0, J+3] = selected.previous.location[0]/(self.size)+0.5
            self.info[0, J+4] = selected.previous.location[1]/(self.size)+0.5

        if selected.next is None:
            self.info[0, J+5] = 1  
        else:
            self.info[0, J+6] = selected.next.location[0]/(self.size)+0.5
            self.info[0, J+7] = selected.next.location[1]/(self.size)+0.5

        if selected.connected=={}:
            self.info[0, J+8] = 1
        else:

            if selected in self.metro.complete:
                self.info[0, J+9] = 1

            J=J_before+10
            for co in set(selected.connected):

                lines_to_check.append(co.line)

                if co.previous is None:
                    self.info[0, J] = 1  
                else:
                    self.info[0, J+1] = co.previous.location[0]/(self.size)+0.5
                    self.info[0, J+2] = co.previous.location[1]/(self.size)+0.5

                if co.next is None:
                    self.info[0, J+3] = 1  
                else:
                    self.info[0, J+4] = co.next.location[0]/(self.size)+0.5
                    self.info[0, J+5] = co.next.location[1]/(self.size)+0.5

                J+=6

        
        J=J_before + 10+2*6

        #2. n_iter
        #size of metro:
        self.info[0, J] = len(self.metro.all_stations)/n_iter
        J+=1
        J_before=J

        #3. cities
        #Info about best cities:
        best_dens, max_dens, max_area = self.metropolis.get_best_city_centers(6)

        for city in best_dens:

            self.info[0, J] = city[0]/(self.size)+0.5
            self.info[0, J+1] = city[1]/(self.size)+0.5

            if J!=J_before:
                self.info[0, J+2] = best_dens[city][0]/max_dens
                self.info[0, J+3] = best_dens[city][1]/max_area
                J+=4
            else:
                J+=2

        J = 23 + 5*4+2
        J_before = J

        #4. Info about locations around
        L = self.metro.display(frame=False)
        #print(L, lines_to_check)

        for n in scales:
            for direction in neighbours:
                new_loc = (selected.location[0] + int(n*direction[0]), selected.location[1] + int(n*direction[1]))
                i=0

                #If on a line already
                for l in lines_to_check:
                    if new_loc in L[l]:
                        self.info[0, J+i] = 1         
                    i+=1

                #Get the density and occupation
                    
                if new_loc[0]<self.size/2 and new_loc[0]>-self.size/2 and new_loc[1]<self.size/2 and new_loc[1]>-self.size/2:
                    dens, area = self.get_dense_around(new_loc)
                    dens_occupied  = self.get_share_already_served(new_loc)

                    self.info[0, J+i+1]=area/self.max_station_area
                    self.info[0, J+i+2]=dens/area
                    self.info[0, J+i+3]=dens_occupied/dens

                    closest_station, closest_dis = self.get_nearest_station(new_loc)
                    self.info[0, J+i+4] = closest_dis/np.max(scales)

                    if closest_dis==0:
                        for s in self.metro.all_stations:
                            if s.location == new_loc:

                                #Find if can connect to one complete station already
                                if s in self.metro.complete:
                                    self.info[0, J+i+5] = 1

                                #Find the same link already
                                if s.previous is not None:
                                    if s.previous.location == selected.location:

                                        self.info[0, J+i+6] = 1 #Found the same link already

                                if s.next is not None:
                                    if s.next.location == selected.location:

                                        self.info[0, J+i+6] = 1 #Found the same link already

                    #print("DISTANCES", closest_dis/np.max(scales), closest_station.location, new_loc)
                    self.info[0, J+i+7] = 1 #In boundary
                
                J+=10

        #T.append((time.time()-start_time, "locations around"))

        J = J_before+10*16 #16 before (with the 16 actions)

        #5. Check current area:
        dens, area = self.get_dense_around(selected.location)
        dens_occupied  = self.get_share_already_served(selected.location)

        self.info[0, J+1]=area/self.max_station_area
        self.info[0, J+2]=dens/area
        self.info[0, J+3]=dens_occupied/dens

        return self.info

    
    ################################################ 2.
    def change_metro(self, station, action):

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [6, 12]

        #neighbours = [(1,0), (0,1), (-1,0), (0, -1)]
        #scales = [6]

        n = scales[(action)//8]
        direction = neighbours[(action)%8]

        #New location added
        new_loc = (station.location[0] + int(n*direction[0]), station.location[1] + int(n*direction[1]))

        if new_loc[0]<self.size/2 and new_loc[0]>-self.size/2 and new_loc[1]<self.size/2 and new_loc[1]>-self.size/2:
            new = self.metro.make_new_station(station, new_loc[0], new_loc[1], returning=True)

            if new is not None and isinstance(new, tuple) is False:
                self.metro.make_connection_close(new)
                return new

            elif isinstance(new, tuple) is True:
                self.metro.make_connection_close(new[0])
                return 0
            
            else:
               return new 
        
        else:
            print("outside boundaries")
            return None

    
    ################################################ 3.
    def change_metropolis(self):

        self.metropolis.step(self.at_most_new)

        for station in self.metro.all_stations:
            if random.uniform(0,1)<self.p_station: #Don't grow stations everytime

                self.metropolis.step_station(station.location)


    ################################################ 4.
                
    def get_reward(self):

        reward=0
        #_, max_dens, _ = self.metropolis.get_best_city_centers(5)

        #Compute only once the points as city does not change (for now)
        if self.initial_points==[]:

            for _ in range(self.n_simulations):

                i_initial, j_initial = self.metropolis.pick_point()
                i_final, j_final = self.metropolis.pick_point()

                done=1
                coef = 2
                while euclidean((i_initial, j_initial),(i_final, j_final))<coef*self.r_walking:
                    i_initial, j_initial = self.metropolis.pick_point()
                    done+=1

                    if done>=3:
                        coef = 1.5


                initial = (i_initial, j_initial) #No +mid here
                final = (i_final, j_final)

                self.initial_points.append(initial)
                self.final_points.append(final)

        
        for i in range(len(self.initial_points)):

            walking_time, metro_time, _ = self.metro.get_fastest(self.initial_points[i], self.final_points[i])

            if metro_time==np.inf:

                dis_initial = self.metro.get_dis_closest_station(self.initial_points[i])
                dis_final = self.metro.get_dis_closest_station(self.final_points[i])
                l_dis = [dis_initial, dis_final]

                for dis in l_dis:

                    if dis>2.5*self.r_walking:
                        reward+=0.05/2
                    elif dis>1.6*self.r_walking:
                        reward+=0.1/2
                    elif dis>self.r_walking:
                        reward+=0.25/2
                    else:
                        reward+=0.4/2

            else:
                #x = (metro_time-walking_time)/walking_time
                #if x<=1/4:
                    #reward+= -(4/9)*x+1/9
                
                if metro_time*4<walking_time:
                    reward+=1.2
                elif metro_time*3<walking_time:
                    reward+=1.1
                elif metro_time*2.25<walking_time:
                    reward+=0.95
                elif metro_time*1.8<walking_time:
                    reward+=0.9
                elif metro_time*1.5<walking_time:
                    reward+=0.85
                elif metro_time<walking_time:
                    reward+=0.66
                else:
                    reward+=0.5

        """
        # Plotting
        # Plot initial points
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
        initial_x, initial_y = zip(*initial_points)  # Unpack points
        plt.hexbin(initial_x, initial_y, gridsize=30, cmap='Blues')  # Adjust gridsize as needed
        plt.colorbar()  # Show color scale
        plt.title('Density of Initial Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()  # Invert y-axis to align with the frame plot

        # Plot density of final points
        plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
        final_x, final_y = zip(*final_points)  # Unpack points
        plt.hexbin(final_x, final_y, gridsize=30, cmap='Reds')  # Adjust gridsize as needed
        plt.colorbar()  # Show color scale
        plt.title('Density of Final Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()  # Invert y-axis to align with the frame plot

        frame = self.metropolis.display()
        # Display the frame as a 2D matrix
        plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd subplot
        plt.imshow(frame, cmap='viridis', interpolation='nearest') # Adjust cmap as needed
        plt.title('Frame Matrix')
        plt.colorbar() # Optional: add a color bar to indicate scale

        plt.tight_layout()
        plt.show()
        """

        return reward/self.n_simulations

    
    ################################################ 5.
    def reset(self):

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking, self.waiting_for_train, self.waiting_when_stopping, self.make_connection_distance)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

    ################################################ 6.
    def select_station(self):

        if np.random.uniform(0,1)<self.p_selecting_station:

            line = np.random.randint(1, len(list(self.metro.lines.keys()))+1)
            station_ind = np.random.randint(0, 2)

            if station_ind==0:
                station = self.metro.lines[line].starting

            else:
                station = self.metro.lines[line].ending

            return station

        else:
            return self.metro.all_stations[np.random.randint(0, len(self.metro.all_stations))]
        


    ################################################ 7.

    def get_dense_around(self, location):

        all_possible_pixels_array = np.array(list(self.metropolis.all_possible_pixels))
        distances = np.linalg.norm(all_possible_pixels_array - np.array(location), axis=1)
        within_r_walking_mask = distances <= self.metro.r_walking
        pixels_within_r_walking = all_possible_pixels_array[within_r_walking_mask]
        
        density = 0
        total_seen = 0

        for pixel in map(tuple, pixels_within_r_walking):
            if pixel in self.metropolis.area:
                density += self.metropolis.density[pixel][0]
                total_seen += 1
        
        density = max(density, 0.001)
        total_seen = max(total_seen, 0.001)
        
        return density, total_seen

    def get_nearest_station(self, location):

        closest = None
        dis_closest = np.inf

        for station in self.metro.all_stations:
            distance = euclidean(station.location, location)
            if distance < dis_closest:
                closest = station
                dis_closest = distance 
        
        return closest, dis_closest

    def get_share_already_served(self, location):

        density_served = 0
        station_locations = np.array([station.location for station in self.metro.all_stations])

        for pixel in self.metropolis.area:
            if euclidean(pixel, location) <= self.metro.r_walking:
                distances = np.linalg.norm(station_locations - np.array(pixel), axis=1)
                
                if np.any(distances <= self.metro.r_walking):
                    density_served += self.metropolis.density[pixel][0]

        return density_served


    def get_max_station_area(self):

        radius_grid_cells = int(self.r_walking)
        return (radius_grid_cells * 2 + 1) ** 2

    """
    ################################################ 7.
    def get_dense_around(self, location):

        density=0
        total_seen = 0
        for pixel in self.metropolis.all_possible_pixels:

            if euclidean(pixel, location)<=self.metro.r_walking:

                if pixel in self.metropolis.area:
                    density+=self.metropolis.density[pixel][0]

                total_seen+=1

        if density==0:
            density=0.001

        if total_seen==0:
            total_seen=0.001

        return density, total_seen
    
    def get_nearest_station(self, location):

        closest = None
        dis_closest = np.inf
        for station in self.metro.all_stations:
            if euclidean(station.location, location)<dis_closest:
                closest = station
                dis_closest = euclidean(station.location, location)

        return closest, dis_closest
    
    def get_share_already_served(self, location):

        density_served = 0
        for pixel in self.metropolis.area:
            if euclidean(pixel, location)<=self.metro.r_walking:

                for station in self.metro.all_stations:
                    if euclidean(station.location, pixel)<=self.metro.r_walking:
                        density_served+= self.metropolis.density[pixel][0]
                        break

        return density_served
    

    #Get max possible area 
    def get_max_station_area(self):

        center = (int(self.size/2), int(self.size/2))
        area=0

        for i in range(self.size):
            for j in range(self.size):
                if euclidean((i,j), center)<self.r_walking:
                    area+=1

        return area

    """