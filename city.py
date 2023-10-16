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

city=Metropolis(City("hey", 0, [[0,0]]), [], 0, 200)
for _ in range(20):
    city.new_round(0.3,0.1,0.1)

city.update_frame()
rows=np.cumsum(np.array([np.sum(city.frame, axis=1)/np.sum(city.frame)]), axis=1)[0]
#print(rows)
columns=np.cumsum(np.array([(city.frame[100]/np.sum(city.frame[100]))]), axis=1)[0]
print(columns)

for _ in range(10):
    i,j = city.pick_point()
    print((i+100, j+100))

fig, ax = plt.subplots()
ax.imshow(city.frame)

plt.show()
"""
#all = [0.5173999999999995, 0.29946666666666677, 0.612533333333333, 0.1310666666666667, 0.18776666666666667, 0.3677666666666666, 0.08300000000000002, 0.24106666666666668, 0.1872666666666667, 0.33160000000000006, 0.24640000000000004, 0.29906666666666665, 0.45616666666666644, 0.3527666666666665, 0.09773333333333334, 0.1267, 0.2962333333333334, 0.542733333333333, 0.0598, 0.2331, 0.48799999999999977, 0.28440000000000004, 0.613033333333333, 0.3751333333333331, 0.17803333333333338, 0.3534333333333332, 0.18266666666666673, 0.9034666666666662, 0.29006666666666664, 0.29683333333333334, 0.24680000000000005, 0.4983333333333331, 0.21060000000000004, 0.27029999999999993, 0.08300000000000003, 0.3076666666666667, 0.24206666666666674, 0.4113666666666665, 0.14080000000000004, 0.24806666666666674, 1.0335999999999996, 0.13566666666666666, 0.11076666666666668, 0.20866666666666675, 0.07033333333333334, 0.2884666666666667, 0.3005333333333334, 0.1883666666666667, 0.2519333333333334, 0.1691, 0.42206666666666653, 0.3692666666666666, 0.3637999999999999, 0.6206666666666665, 0.31880000000000003, 0.20933333333333334, 0.1879666666666667, 0.6040666666666664, 0.3465666666666665, 0.15240000000000006, 0.18843333333333334, 0.2108, 0.24296666666666666, 0.1806, 0.1877333333333334, 0.40153333333333324, 0.2695333333333334, 0.13003333333333336, 0.12073333333333336, 0.2564666666666667, 0.7901333333333329, 0.19710000000000003, 0.14610000000000004, 0.2929333333333333, 0.1628666666666667, 0.13520000000000001, 0.1507, 0.2618666666666667, 0.11123333333333334, 0.13670000000000002, 0.17443333333333333, 0.4899333333333333, 0.4772999999999998, 0.08756666666666668, 0.2957000000000001, 0.3042333333333334, 0.33753333333333324, 0.21580000000000002, 0.2820666666666667, 0.43696666666666656, 0.17770000000000005, 0.13313333333333333, 0.17463333333333333, 0.15933333333333335, 0.3082333333333334, 0.39789999999999986, 0.2406666666666667, 0.13136666666666666, 0.3063, 0.40986666666666655, 0.0822, 0.1078, 0.29353333333333337, 0.10683333333333334, 0.3079, 0.2615, 0.13013333333333335, 0.21803333333333338, 0.23153333333333337, 0.7472666666666664, 0.0954, 0.17423333333333338, 0.29036666666666666, 0.3402999999999999, 0.30656666666666665, 0.1755333333333334, 0.22586666666666666, 0.11563333333333335, 0.10170000000000003, 0.2546666666666667, 0.17443333333333336, 0.17543333333333336, 0.12250000000000005, 0.13623333333333335, 0.12203333333333333, 0.22290000000000001, 0.1302, 0.2196666666666667, 0.1648666666666667, 0.18076666666666666, 0.2213666666666667, 0.3195666666666667, 0.18183333333333332, 0.23183333333333334, 0.15650000000000003, 0.1893666666666667, 0.1962666666666667, 0.34136666666666665, 0.2538333333333334, 0.1800666666666667, 0.09836666666666667, 0.5797666666666663, 0.1912666666666667, 0.9572999999999995, 0.20656666666666668, 0.23580000000000004, 0.5208666666666663, 0.4161999999999999, 0.1560666666666667, 0.37393333333333323, 0.3067666666666667, 0.4657333333333333, 0.3840999999999999, 0.15803333333333336, 0.19683333333333333, 0.24820000000000006, 0.32606666666666667, 0.33779999999999993, 0.12073333333333336, 0.16893333333333335, 0.26066666666666666, 0.4240333333333332, 0.34779999999999983, 0.3245, 0.2137, 0.6985999999999996, 0.4200666666666664, 0.1878, 0.19450000000000003, 0.42689999999999967, 0.10453333333333337, 0.21769999999999998, 0.4236666666666666, 0.4352, 0.2571666666666667, 0.16993333333333335, 0.14880000000000002, 0.2012666666666667, 0.2513, 0.19116666666666673, 0.2625000000000001, 0.18163333333333334, 0.5101666666666663, 0.13783333333333334, 0.3806666666666665, 0.2430333333333334, 0.15793333333333334, 0.4538999999999999, 0.31803333333333345, 0.24953333333333338, 0.5901999999999997, 0.11803333333333337, 0.34926666666666656, 0.5538666666666664, 0.14233333333333337, 0.4404333333333332, 0.12443333333333335, 0.17220000000000002, 0.2227333333333333, 0.23136666666666672]
#plt.plot(all)
#plt.show()