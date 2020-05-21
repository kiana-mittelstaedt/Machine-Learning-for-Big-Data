'''
Code for CSE547 HW3, Q2(c)
Kiana Mittelstaedt
I ran this using Google Colab, but it should run in Python on your machine
or in a Jupyter Notebook. Please just place the files 'youtube_community_top1000.txt'
and 'youtube_net.txt' into the correct directory to read them in.
Note: I changed the code significantly from the skeleton you provided. It was
difficult for me to interpret what your shell was asking for.
'''

import networkx as nx
import numpy as np
import pandas as pd
!pip install python-louvain
import community
import matplotlib.pyplot as plt
import os

def read_network(data):
    g = nx.read_edgelist(data, create_using=nx.Graph())
    return g

def read_true_community(data):
    with open(data) as f:
        c = [line.rstrip().split('\t') for line in f]
    return c

youtube_net = read_network('youtube_net.txt')
youtube_com = read_true_community('youtube_community_top1000.txt')

# compute density
# S is a community
# we need to find ns = the number of nodes in S, ms = the number of edges within S
density_vals = []
def density(coms):
    for i in range(len(coms)):
      S = coms[i]
      ns = len(S)
      sub = youtube_net.subgraph(S)
      ms = sub.number_of_edges()
      tmp = (2*ms)/(ns*(ns-1))
      density_vals.append(tmp)
    den = density_vals
    return den

# compute cut ratio
# we need cs = the number of edges crossing the boundary of community S, 
# ns = the number of nodes in S, and n = number of nodes in entire graph
n = len(youtube_net)
cut_values = []

def cut_ratio(g,coms):
  for i in range(len(coms)):
    S = coms[i]
    ns = len(S)
    cs = 0
    for j in S:
      neighbors = list(youtube_net.neighbors(j))
      for k in neighbors:
        if k not in S:
          cs += 1
        else:
          cs += 0
    cut = cs/(ns*(n-ns))
    cut_values.append(cut)
  ratio = cut_values
  return ratio

# compute modularity
m2 = 5975248 # this is just 2*the number of edges
def community_modularity(coms, g):
    if type(g) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")
    inc = 0
    deg = 0
    links = g.size()
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    community_mods = []
    for i in range(len(coms)):
      comm = youtube_com[i]
      mod_vec = []
      for node1 in comm:
        for node2 in comm:
          d1 = youtube_net.degree(node1)
          d2 = youtube_net.degree(node2)
          if youtube_net.has_edge(node1,node2) == True:
            A = 1
          else:
            A = 0
          Q = A - ((d1*d2)/m2)
          mod_vec.append(Q)
        comm_mod = 1/m2 * sum(mod_vec)
      
      community_mods.append(comm_mod)
    return community_mods

den_verify = density(youtube_com)
cut_verify = cut_ratio(youtube_net,youtube_com)
mod_verify = community_modularity(youtube_com,youtube_net)

def pquality_summary(graph, partition):
    mods = mod_verify
    crs = cut_verify
    dens = den_verify
    return [mods, crs, dens]

G = read_network('youtube_net.txt')
C = read_true_community("youtube_community_top1000.txt")
len_C = np.array([len(x) for x in C])
partition = C
mods, crs, dens = pquality_summary(G, partition)
mods_sort, den_sort_mods = zip(*sorted(zip(mods, dens), reverse=True))
crs_sort, den_sort_crs = zip(*sorted(zip(crs, dens), reverse=False))
den_sort_mods_avg = [np.mean(den_sort_mods[0:(i+1)]) for i in range(len(den_sort_mods))]
den_sort_crs_avg = [np.mean(den_sort_crs[0:(i+1)]) for i in range(len(den_sort_crs))]
ax = plt.gca()
ax.set_xlim(10**0, 10**3)
ax.set_xscale('log')
ax.plot(np.array([i for i in range(1000)]), np.array(den_sort_mods_avg), color='green', label='Modularity')
ax.plot(np.array([i for i in range(1000)]), np.array(den_sort_crs_avg), color='blue', label='Cut Ratio')
plt.title('Density')
plt.legend()
plt.xlabel('rank')
plt.ylabel('score')
plt.savefig("density.png")
