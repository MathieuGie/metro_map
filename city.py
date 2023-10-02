import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from typing import List

from station import Station

center=([0], [0])

def gaussian_filter(size, sigma): #usually put sigma to 3
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

def euclidean(a,b):
    result=0
    for i in range(2):
        result+=(a[i]-b[i])**2
    return np.sqrt(result)

class City:

    def __init__(self, name: str, time: int, center: List[List[int]]):
        self.name=name
        self.time=time

        self.center=center #fixed center of the city
        self.area=center #will grow over time

class Metropolis:

    def __init__(self, central_city: City, other_cities: List[City], time: int, size: int):

        self.center=central_city
        self.others=other_cities

        all=[central_city]
        for town in other_cities:
            all.append(town)
        self.cities=all

        self.time=time
        self.size=size

        self.area=[] #area of the whole metropolis (all the cities counted)
        self.density = {}

        for city in self.cities:
            for pixel in city.area:
                self.area.append(pixel)

                if city == central_city:
                    self.density[(pixel[0], pixel[1])] = (0.5, city)
                else:
                    self.density[(pixel[0], pixel[1])] = (0.3, city)

        
        self.frame=np.zeros((size,size)) #initialise the frame to empty map
        

    def update_frame(self):

        frame=np.zeros((self.size,self.size))
        mid=int(self.size/2)

        for pixel in self.density:
            #print(pixel)
            frame[mid+pixel[0], mid+pixel[1]]=self.density[pixel][0]

        self.frame=frame

    def grow(self, p_center, p_others, index):
        #grows one city in the metropolis, not overlapping on other cities

        #index is the city index we care about

        neighbors=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
        new_pixels=[]

        #Determine if you are in the main city or not
        if index==0:
            p=p_center
        else:
            center=self.cities[index].center[0]
            d=np.sqrt(center[0]**2+center[1]**2)
            #p=((1-p_others)/(np.sqrt(2)*250))*d+p_others
            p = p_others + p_others/(d*0.03)

        #Probabilistically grow the desired city
        for pixel in self.cities[index].area:
            for new in neighbors:

                possible=[new[0]+pixel[0], new[1]+pixel[1]]

                if random.uniform(0,1)<p: #< instead of >
                    if (possible not in self.area) and (possible not in new_pixels):
                        if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:
                            
                            new_pixels.append(possible)

                            

        #Then harmonise all the pixels of thee city:
        for pixel in self.cities[index].area:
            for new in neighbors:
                
                possible=[new[0]+pixel[0], new[1]+pixel[1]]
                possible_tuple = (new[0]+pixel[0], new[1]+pixel[1])
                p_tilde=0
                n=0

                if possible in self.area:
                    p_tilde+=self.density[possible_tuple][0]
                    n+=1

            if n!=0:
                p_tilde/=n

            p = self.density[(pixel[0], pixel[1])][0]
            city = self.density[(pixel[0], pixel[1])][1]

            if random.uniform(0,1)<p_tilde:
                p += 0.01

            if p>1:
                p = 1

            self.density[(pixel[0], pixel[1])] = (p, city)

            
        for integrated in new_pixels:

            self.cities[index].area.append(integrated)
            self.area.append(integrated)

            if index==0:
                self.density[(integrated[0], integrated[1])] = (0.5, self.cities[index])
            else:
                dens = (-0.5/360)*euclidean(integrated, [0,0])+0.5
                self.density[(integrated[0], integrated[1])] = (dens, self.cities[index])


        self.time+=1

    def pick_point(self):

        frame=self.frame
        rows=np.cumsum(np.array([np.sum(frame, axis=1)/np.sum(frame)]), axis=1)[0]

        #print("rows:", rows)

        row=np.random.random()

        for i in range(len(rows)):
            if i==0 and row<rows[i]:
                I=0
                break
            elif i==len(rows)-1:
                I=len(rows)-1
            elif rows[i]<=row and rows[i+1]>row:
                I=i+1
                break

        columns=np.cumsum(np.array([(frame[I]/np.sum(frame[I]))]), axis=1)[0]

        #print("columns", columns)

        col=np.random.random()

        for j in range(len(columns)):
            if j==0 and col<columns[j]:
                J=0
                break
            elif j==len(columns)-1:
                J=len(columns)-1
            elif columns[j]<=col and columns[j+1]>col:
                J=j+1
                break

        mid = self.size/2
        return (I-mid,J-mid)


    def new_round(self, p_center, p_others, p_new): 
        #new_round allows to compute the extensions of all cities
        
        #p is the probability of accepting a neighboring pixel
        #p_new is the probability of having a new city being born

        mid=self.size/2

        for city_index in range(len(self.cities)):
            self.grow(p_center, p_others, city_index)

        for _ in range(5): #Attempt to create at most 5 cities per round
            proba=random.uniform(0,1)
            if proba<p_new:

                ok=False
                while ok==False:
                    i=np.random.randint(-mid,mid)
                    j=np.random.randint(-mid,mid)
                    new_center=[i,j]
                    if new_center not in self.area:
                        ok=True
                        self.cities.append(City("hello"+str(self.time), self.time, [new_center]))
                        dens = (-0.5/360)*euclidean(new_center, [0,0])+0.5
                        self.density[(new_center[0], new_center[1])] = (dens, self.cities[-1])

        self.update_frame()

    def grow_station(self, station:Station, p_growth:float):

        neighbors=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]

        if station.location in self.area:

            p_station, city_station = self.density[(station.location[0], station.location[1])]

            for neighb in neighbors:

                possible=[station.location[0]+neighb[0], station.location[1]+neighb[1]]
                possible_tuple = (station.location[0]+neighb[0], station.location[1]+neighb[1])

                if (possible in self.area) and (random.uniform(0,1)<p_growth): 
                    p, city = self.density[possible_tuple]
                    p+=0.01

                    if p>1:
                        p=1

                    self.density[possible_tuple] = (p,city)

                if (possible not in self.area) and (random.uniform(0,1)<p_growth):
                    if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:

                        self.area.append(possible)

                        if city_station == self.cities[0]:
                            self.density[possible_tuple]=(0.5, city_station)
                        else:
                            self.density[possible_tuple]=(p_station, city_station)

        elif random.uniform(0,1)<p_growth:

            #Create new city around station
            new_center = station.location
            self.cities.append(City("hello"+str(self.time), self.time, [new_center]))
            dens = (-0.5/360)*euclidean(new_center, [0,0])+0.5
            self.density[(new_center[0], new_center[1])] = (dens, self.cities[-1])
            


        self.update_frame()

            
"""
city=City("hello", 0, [[0,0]])
metropolis=Metropolis(city, [],0, 500)

print(np.sqrt(250**2+250**2))
for i in range(25):
    print("iteration:", i)
    metropolis.new_round(0.4,0.1,0.7)

print(metropolis.density[(0,0)])


fig, ax = plt.subplots()
ax.imshow(metropolis.frame)

plt.show()
"""