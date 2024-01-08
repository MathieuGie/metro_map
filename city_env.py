import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from typing import List
import copy

class City:

    def __init__(self, number: int, center):

        self.number = number
        self.center=center

        self.area=set()
        self.area.add(center)

        self.border = copy.deepcopy(self.area) #To only consider border points (initialise at center)


class Metropolis:

    def __init__(self, central_city, size: int, p_center, p_new, p_station):

        self.central_city=City(0, central_city)
        self.size=size
        self.all_possible_pixels = {(a, b) for a in range(-self.size/2+1, self.size/2) for b in range(-self.size/2+1, self.size/2)}

        self.p_center = p_center #Evolution rate 
        self.p_new = p_new #Add a new city
        self.p_station = p_station


        self.all_cities=[self.central_city]
        self.area=copy.deepcopy(self.central_city.area)
        
        self.density = {}
        self.density[self.central_city.center]=(0.5,central_city)

    ################################################ 1.
    def step(self, n_new_cities:int):

        #index is the city index we care about
        neighbors=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]


        for city in self.all_cities:

            #Compute p of growing 
            if city == self.central_city:
                p=self.p_center
            else:
                p=np.clip(self.p_center - np.sqrt(city.center[0]**2+city.center[1]**2)*0.0025, 0.05, self.p_center)

            #print("p", p, len(city.area), city.center)

            #Harmonise first
            for pixel in city.area:

                p_tilde = 0
                dens = self.density[pixel][0]
                n = 0

                for new in neighbors:
                    possible = (new[0] + pixel[0], new[1] + pixel[1])

                    if possible in self.area:
                        p_tilde += self.density[possible][0]
                        n += 1

                if n != 0:
                    p_tilde /= n

                if np.random.uniform(0, 1)<p_tilde:
                    self.density[pixel] = (0.7*p_tilde+0.3*dens+np.random.uniform(-0.01, 0.08), city)

            #Add neighbours
            new_pixels = set()
            remove_from_border=set()

            for pixel in city.border:
                still_bordering=0

                for new in neighbors:
                    possible=(new[0]+pixel[0], new[1]+pixel[1])

                    if (possible not in self.area) and (possible not in new_pixels):
                        still_bordering=1 #Can still add pixels

                        if random.uniform(0,1)< p : 
                            if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:
                                
                                new_pixels.add(possible)
                                self.density[possible] = (p, city)
                                
                if still_bordering==0:
                    remove_from_border.add(pixel)

            city.area.update(new_pixels)
            city.border-=remove_from_border
            city.border.update(new_pixels)
            self.area.update(new_pixels)

        for _ in range(n_new_cities): #Add new cities
            if np.random.uniform(0,1)<self.p_new:

                ok=False
                while ok==False:
                    i=np.random.randint(-self.size/2,self.size/2)
                    j=np.random.randint(-self.size/2,self.size/2)
                    new_center=(i,j)
                    if new_center not in self.area:

                        ok=True
                        new = City(len(self.all_cities), new_center)

                        self.all_cities.append(new)
                        self.area.add(new_center)
                        self.density[new_center] = (np.clip(self.p_center - np.sqrt(city.center[0]**2+city.center[1]**2)*0.0025, 0.05, self.p_center), new)

    ################################################ 2.
    def step_station(self, location):

        neighbors=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
        new_pixels = set()

        if location in self.area:

            _, city_station = self.density[location]

            #Grow the neighbours
            for neighb in neighbors:
                possible=(location[0]+neighb[0], location[1]+neighb[1])

                if possible in self.area: 

                    dens, city = self.density[possible]
                    dens = np.clip(dens+0.5, 0.15, 10)
                    self.density[possible] = (dens, city)

                elif (possible not in self.area) and (possible not in new_pixels):
                    if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:

                        city_station.area.add(possible)
                        city_station.border.add(possible)
                        new_pixels.add(possible)
                        self.density[possible] = (0.15, city_station)

        else:
            #Create new city around station probabilistically
            new = City(len(self.all_cities), location)

            self.all_cities.append(new)
            self.area.add(location)
            self.density[location] = (np.clip(self.p_center - np.sqrt(location[0]**2+location[1]**2)*0.0025, 0.05, self.p_center), new)

        self.area.update(new_pixels)

    ################################################ 3.
    def pick_point(self):

        grid_points = list(self.density.keys())
        densities = [self.density[point][0] for point in grid_points]

        chosen_point = random.choices(grid_points, weights=densities, k=1)[0]

        return chosen_point
    
    ################################################ 4.
    def display(self):

        frame = np.zeros((self.size, self.size))
        mid=int(self.size/2)

        for pix in self.density:
            frame[pix[1]+mid, pix[0]+mid] = self.density[pix][0]
        

        return frame
    
    ################################################ 5.
    def get_best_city_centers(self, n:int):

        total_dens = {}
        for city in self.all_cities:

            total=0
            size=len(city.area)

            for pixel in city.area:
                total+=self.density[pixel][0]

            total_dens[(city.center)]=(total, size)

        sorted_dict = dict(sorted(total_dens.items(), key=lambda item: item[1][0], reverse=True)[:n])
        max_dens = max(total_dens.values(), key=lambda x: x[0])[0]
        max_area = max(total_dens.values(), key=lambda x: x[1])[1]

        return sorted_dict, max_dens, max_area
    



            

            

            

