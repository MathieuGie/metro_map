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

#all = [0.4674999999999991, 0.5624999999999989, 0.3112499999999994, 0.6599999999999989, 0.35749999999999954, 0.4462499999999992, 0.3687499999999997, 0.2974999999999998, 0.5274999999999991, 0.24250000000000005, 0.42874999999999924, 0.45874999999999927, 0.3524999999999993, 0.4099999999999992, 0.22000000000000006, 0.3699999999999996, 0.22625000000000006, 0.4612499999999994, 0.28624999999999984, 0.3012499999999997, 0.32374999999999965, 0.30749999999999983, 0.31874999999999987, 0.25375, 0.3299999999999996, 0.16750000000000007, 0.5299999999999991, 0.2537499999999999, 0.35624999999999934, 0.3424999999999996, 0.38374999999999954, 0.43249999999999955, 0.34124999999999983, 0.43374999999999925, 0.4299999999999992, 0.32749999999999957, 0.23000000000000007, 0.3599999999999999, 0.2624999999999998, 0.31624999999999964, 0.31624999999999953, 0.30624999999999986, 0.35249999999999954, 0.43874999999999925, 0.22375000000000006, 0.32749999999999946, 0.3899999999999996, 0.4712499999999994, 0.4012499999999996, 0.4174999999999993, 0.27624999999999983, 0.5012499999999992, 0.4924999999999991, 0.28499999999999953, 0.34999999999999937, 0.48249999999999915, 0.4362499999999994, 0.24375000000000005, 0.544999999999999, 0.3774999999999996, 0.4949999999999992, 0.2637499999999997, 0.3687499999999994, 0.39249999999999935, 0.3962499999999994, 0.609999999999999, 0.3799999999999994, 0.42249999999999927, 0.4499999999999991, 0.4924999999999994, 0.3637499999999992, 0.4224999999999992, 0.5337499999999992, 0.3762499999999992, 0.3649999999999994, 0.4287499999999994, 0.3199999999999997, 0.4512499999999993, 0.3449999999999996, 0.36874999999999963, 0.19500000000000006, 0.4199999999999994, 0.3049999999999996, 0.39999999999999936, 0.6337499999999989, 0.5724999999999988, 0.29374999999999946, 0.24875000000000005, 0.2987499999999995, 0.5387499999999994, 0.4812499999999992, 0.40874999999999934, 0.34124999999999933, 0.2874999999999997, 0.35624999999999946, 0.32499999999999957, 0.24250000000000008, 0.17000000000000004, 0.43749999999999944, 0.5824999999999989, 0.49749999999999905, 0.42249999999999943, 0.531249999999999, 0.25125000000000003, 0.3887499999999994, 0.5562499999999989, 0.44374999999999937, 0.4562499999999993, 0.3412499999999996, 0.4262499999999993, 0.4012499999999996, 0.39749999999999924, 0.39749999999999924, 0.31499999999999967, 0.4949999999999993, 0.3824999999999992, 0.29749999999999976, 0.27249999999999985, 0.4512499999999993, 0.3274999999999997, 0.4187499999999989, 0.28374999999999984, 0.2862499999999996, 0.36749999999999927, 0.2687499999999998, 0.44249999999999934, 0.6249999999999989, 0.4437499999999994, 0.46374999999999966, 0.509999999999999, 0.3499999999999994, 0.21500000000000002, 0.5424999999999992, 0.3112499999999997, 0.4212499999999994, 0.24625000000000002, 0.4462499999999992, 0.3849999999999994, 0.3299999999999996, 0.2862499999999999, 0.4199999999999995, 0.4424999999999993, 0.5499999999999989, 0.3024999999999996, 0.4312499999999995, 0.612499999999999, 0.623749999999999, 0.42624999999999935, 0.5499999999999992, 0.40124999999999933, 0.26249999999999984, 0.6949999999999992, 0.4937499999999991, 0.39749999999999924, 0.39374999999999943, 0.35999999999999943, 0.43249999999999933, 0.511249999999999, 0.3499999999999994, 0.4462499999999993, 0.3199999999999996, 0.5712499999999989, 0.2824999999999996, 0.5449999999999993, 0.28999999999999965, 0.4487499999999994, 0.5124999999999992, 0.28124999999999967, 0.3574999999999996, 0.37999999999999917, 0.3224999999999996, 0.4799999999999992, 0.47499999999999937, 0.29749999999999965, 0.3987499999999992, 0.449999999999999, 0.6362499999999991, 0.2924999999999996, 0.5074999999999992, 0.577499999999999, 0.3362499999999996, 0.2612499999999998, 0.2899999999999996, 0.3274999999999998, 0.43124999999999925, 0.26749999999999985, 0.2774999999999997, 0.4474999999999992, 0.3299999999999997, 0.636249999999999, 0.34249999999999964, 0.4374999999999994, 0.3012499999999999, 0.36624999999999946, 0.487499999999999, 0.23625000000000002, 0.3574999999999995, 0.3137499999999995, 0.25249999999999995, 0.4087499999999991]
#plt.plot(all)
#plt.show()