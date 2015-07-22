import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
def drawByTripleArry(data):
    data = np.asarray(data)
    G = nx.Graph()
    map(lambda x : G.add_edge(np.int(x[0])-1,np.int(x[1])-1,weight=x[2]),data)
    G = nx.kruskal_mst(G)
    pos = nx.spring_layout(G)

    #nx.draw_networkx_nodes(G,pos)
    #nx.draw_networkx(G)
    #plt.figure(1)
    plt.subplot(2,1,1)
    nx.draw_networkx(G)
    #nx.draw_networkx(G,pos=nx.spectral_layout(G))
    #plt.figure(2)
    #nx.draw_networkx(G,pos=pos)
    #plt.figure(3)
    #nx.draw_networkx(G,pos=nx.circular_layout(G))
    #plt.figure(4)
    #nx.draw_networkx(G,pos=nx.shell_layout(G))
    #plt.figure(5)
    plt.subplot(2,1,2)
    nx.draw_networkx(G,pos=nx.random_layout(G))
    plt.show()

if __name__ == '__main__':
    gdata = np.loadtxt('./firmmst.txt')
    drawByTripleArry(gdata)
    #print gdata
