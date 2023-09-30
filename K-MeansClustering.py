#### K-Means Clustering on iris dataset

#Steps

#### Expectation steps

# Step 1: assign centroids ==> randomly select the points from the dataset
# Step 2: calculate the distance between the centroid and each data point and assign the data points to the centroids

#### Maximisation steps

# Step 3: calculate the mean of the clusters and shift the centroid to the mean
# Step 4: repeat Step 2 and Step 3


#### Data prep

from sklearn import datasets
import numpy as np
from collections import defaultdict



def distance_calc(dataa, centroiid):
   # dist =  np.dot(dataa, centroiid)/np.linalg.norm(dataa)* np.linalg.norm(centroiid)
    dist = np.linalg.norm(dataa-centroiid)
    return dist


def min_disfn(datap, centroidp):
    min_dis = float('inf')
    for j, cd in enumerate(centroidp):
     distance = distance_calc(datap, cd)
     if distance < min_dis:
        min_dis = distance    
        centroid_point = j
    return centroid_point


def mean_calc(pointsincentroid):
    return np.mean(pointsincentroid, axis = 0)


def clustering_algo(data, k, epochs = 10):
  
  # select centroids

   centroid_indices = np.random.randint(low=0, high = np.shape(data)[0], size = k) 
   centroid_datapoints = data[centroid_indices]
   
   l=np.shape(data)[0]
   centroidsfordatapoints = [0]*l
   dataindextocentroids = defaultdict(list)

   for epoch in range(0,epochs):

    for i in range(0,len(data)):
        centroidsare = min_disfn(data[i], centroid_datapoints)
        centroidsfordatapoints[i] = centroidsare
        dataindextocentroids[centroidsare].append(i) 

    # calculate new centroids

    for cent, data_indx in dataindextocentroids.items():
        centroid_datapoints[cent] = mean_calc(data[data_indx])
        
        
   return centroidsfordatapoints
        
iris = datasets.load_iris()
X = iris.data

inp_data = X
print("shape is", np.shape(data))
num_clusters = 3

clusters = clustering_algo(data = inp_data, k = num_clusters)
print("clsuters are:", clusters)
