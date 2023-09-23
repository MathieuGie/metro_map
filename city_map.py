import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from typing import List

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

        self.area=center #will grow over time
        self.center=center

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
        for city in self.cities:
            for pixel in city.area:
                self.area.append(pixel)
        
        self.frame=np.zeros((size,size)) #initialise the frame to empty map
        
    def display_on_frame(self):

        frame=np.zeros((self.size,self.size))
        mid=int(self.size/2)

        for pixel in self.area:
            frame[mid+pixel[0], mid+pixel[1]]=1

        self.frame=frame

    def grow(self, p_center, p_others, index):
        #grows one city in the metropolis, not overlapping on other cities

        #index is the city index we care about

        neighbors=[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
        new_pixels=[]

        if index==0:
            p=p_center
        else:
            center=self.cities[index].center[0]
            d=np.sqrt(center[0]**2+center[1]**2)
            p=((0.999-p_others)/(np.sqrt(2)*250))*d+p_others

        for pixel in self.cities[index].area:
            for new in neighbors:

                possible=[new[0]+pixel[0], new[1]+pixel[1]]

                if random.uniform(0,1)>p:
                    if (possible not in self.area) and (possible not in new_pixels):
                        if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:
                            new_pixels.append(possible)

        for integrated in new_pixels:
            self.cities[index].area.append(integrated)
            self.area.append(integrated)

        self.time+=1

    def new_round(self, p_center, p_others, p_new): 
        #new_round allows to compute the extensions of all cities
        
        #p is the probability of accepting a neighboring pixel
        #p_new is the probability of having a new city being born

        mid=self.size/2

        for city_index in range(len(self.cities)):
            self.grow(p_center, p_others, city_index)

        proba=random.uniform(0,1)
        if proba>p_new:

            ok=False
            while ok==False:
                i=np.random.randint(-mid,mid)
                j=np.random.randint(-mid,mid)
                new_center=[i,j]
                if new_center not in self.area:
                    ok=True
                    self.cities.append(City("hello"+str(self.time), self.time, [new_center]))

            

city1=City("hello", 0, [[0,0]])
metropolis=Metropolis(city1, [],0, 500)

for i in range(200):
    metropolis.new_round(0.97,0.975,0.3)
    print(i)

directions=[i*30*3.1415/180 for i in range(12)] #in rad

"""
class Metro:

    def __init__(self, lines, metropolis):
    #in lines, set of points linked together linearly.
    #need the metropolis that goes with the metro system.
        self.lines=lines
        metropolis.display_on_frame()
        self.map=metropolis.frame #give the matrix of popoulation density
        self.size=metropolis.size

    def new_line(self):
        station=np.random.rand(1,2)[0]
        station*=500
        station[0]=int(station[0])
        station[1]=int(station[1])
        self.lines.append([station])

    def new_station(self, gamma, distance, line_number):
        
        station=self.lines[line_number][-1]
        filter=gaussian_filter(20, 3)
        possible=[]

        for theta in directions:

            poss=0
            I,J=int(station[0]), int(station[1])

            for index in range(10):

                I+=int(np.sin(theta)*distance)
                J+=int(np.cos(theta)*distance)

                if I+10>self.size or I-10>self.size or J+10>self.size or J-10>self.size:
                    square=np.zeros((20,20))
                    
                elif I+10<0 or I-10<0 or J+10<0 or J-10<0:
                    square=np.zeros((20,20))

                else:
                    square=self.map[I-10:I+10, J-10:J+10]

                for line in range(len(self.lines)):
                    for other_station in self.lines[line]:
                        if euclidean([I,J], other_station)<distance*0.8 and euclidean([I,J], [self.size/2, self.size/2])>100:
                            square=np.zeros((20,20)) 

                poss+=np.sum(np.multiply(filter, square))*gamma**index
            
            possible.append(poss)

        I,J=int(station[0]), int(station[1])
        theta_star=directions[np.argmax(possible)]
        self.lines[line_number].append([I+np.sin(theta_star)*distance, J+np.cos(theta_star)*distance])

"""
#Need to put display on frame after so that we treat all the data as being centered in (0,0) before
metropolis.display_on_frame()

"""
metro=Metro([], metropolis)

for i in range(12):
    metro.new_line()
    for j in range(30):
        metro.new_station(0.6, 20, i)

fig, ax = plt.subplots()
ax.imshow(metropolis.frame)

for i in range(8):
    print(metro.lines[i])
    c=np.random.rand(3,)
    print("points ", (np.array(metro.lines[i])[:, 0].shape, np.array(metro.lines[i])[:, 1].shape))
    #plt.scatter(np.array(metro.lines[i])[:, 0], np.array(metro.lines[i])[:, 1], c)
    plt.plot(np.array(metro.lines[i])[:, 0], np.array(metro.lines[i])[:, 1], c)
plt.show()
"""