'''
CSE 547 - HW2, Q2b
Kiana Mittelstaedt
This runs in Google Colab, but it is slow because I used the
Cartesian function. You may need to increase your memory in Colab.
Please load the "data.txt" "c1.txt" and "c2.txt" files into your
Google Colab Files tab.
'''

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import itertools
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# load the datasets
data = sc.textFile("data.txt") # this is a pyspark.rdd.RDD, each row has 58 entries 
c1 = sc.textFile("c1.txt") # this is a pyspark.rdd.RDD, this has 10 rows, each with 58 entries 
c2 = sc.textFile("c2.txt") # this is a pyspark.rdd.RDD, this has 10 rows, each with 58 entries 

# assign values to all points and centroids
data_string_list = data.map(lambda point: (point.split(' ') ))
data_ints_list = data_string_list.map(lambda line: [float(x) for x in line])

c1_string_list = c1.map(lambda point: (point.split(' ') ))
c1_ints_list = c1_string_list.map(lambda line: [float(x) for x in line])

c2_string_list = c2.map(lambda point: (point.split(' ') ))
c2_ints_list = c2_string_list.map(lambda line: [float(x) for x in line])

# points are in form [(0,[58-floats]),(1,[58-floats]),...]
temp_points = data_ints_list.zipWithIndex()
points = temp_points.map(lambda i : (i[1],np.array(i[0])))

# centroids are in the same form 
temp_cent1 = c1_ints_list.zipWithIndex()
cent1 = temp_cent1.map(lambda i : (i[1], np.array(i[0])))

temp_cent2 = c2_ints_list.zipWithIndex()
cent2 = temp_cent2.map(lambda i : (i[1], np.array(i[0])))

#print('Number of points: ' + str(points.count()) + '\n' + 
#      'Number of centroids in c1: ' + str(cent1.count()) + '\n' + 
#      'Number of centroids in c2: ' + str(cent2.count()))

# initialize 
centroids = cent1
max_iter = 20
cost_vec = []

for i in range(max_iter+1):
  # creates all point, centroid pairs: the first row is 
  # [((point 0, [coordinates of point 0]),(centroid 0, [coordinates of centroid 0]))]
  pairs = points.cartesian(centroids)

  # compute the distance between the point and the centroid 
  # want to save (point ID,coordinates of point,centroid ID,summed distance)
  pairs_distance = pairs.map(lambda x: (x[0][0],(x[0][1], x[1][0],np.sum(np.abs(x[0][1]-x[1][1])))))

  # we have 10 point-centroid combinations for each point
  # want to keep only the centroid with the smallest distance from each point
  # we have 4601 items in the RDD - each point ID appears only once 
  closest_points = pairs_distance.reduceByKey(lambda x,y: x if x[-1]<y[-1] else y)

  # computes the cost by summing all of the distances of every point from its nearest centroid 
  cost = closest_points.map(lambda x : x[1][-1]).reduce(lambda x,y: x+y)
  cost_vec.append(cost)
  
  # now we need to create new centroids 
  # [(centroid ID, some point that is close to it),...]
  all_points_centroids = closest_points.map(lambda x: (x[1][1],[x[1][0]]))

  # this is [(centroid ID, array(where each row is a point value))]
  concat_arrays = all_points_centroids.reduceByKey(lambda x,y: np.concatenate((x,y)))

  # takes the median column-wise and updates the np.array
  centroids = concat_arrays.map(lambda x: (x[0], np.median(x[1],axis=0)))

  # help clear some memory
  pairs.unpersist()
  pairs_distance.unpersist()
  closest_points.unpersist()
  all_points_centroids.unpersist()
  concat_arrays.unpersist()


# compute percent change in cost
cost_tenth = cost_vec[10]
cost_zero = cost_vec[0]
cost_change = (1 - cost_tenth/cost_zero)*100

# plot cost as a function of iteration
iter_range = list(range(0,21))
costs = cost_vec[0:21]

plt.plot(iter_range,costs)
plt.ylabel('Cost (Initialized with c1)')
plt.xlabel('Iterations (i)')
plt.savefig('Cost_c1_Manhattan.png')
#plt.show()
