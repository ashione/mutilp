import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from numpy import array
import numpy as np


mat = array([184, 222, 177, 216, 231,
             45, 123, 128, 200,
             129, 121, 203,
             46, 83,
             83])

dist_mat = mat

linkage_matrix = linkage(dist_mat, 'single')
print linkage_matrix

plt.figure(101)
plt.subplot(1, 2, 1)
plt.title("ascending")
dendrogram(linkage_matrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=array(['a', 'b', 'c', 'd', 'e', 'f']),
           distance_sort='ascending')

plt.subplot(1, 2, 2)
plt.title("descending")
dendrogram(linkage_matrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=array(['a', 'b', 'c', 'd', 'e', 'f']),
           distance_sort='descending')


def make_fake_data():
    amp = 1000.
    x = []
    y = []
    for i in range(0, 10):
        s = 20
        x.append(np.random.normal(30, s))
        y.append(np.random.normal(30, s))
    for i in range(0, 20):
        s = 2
        x.append(np.random.normal(150, s))
        y.append(np.random.normal(150, s))
    for i in range(0, 10):
        s = 5
        x.append(np.random.normal(-20, s))
        y.append(np.random.normal(50, s))

    #plt.figure(1)
    #plt.title('fake data')
    #plt.scatter(x, y)

    d = []
    for i in range(len(x) - 1):
        for j in range(i+1, len(x) - 1):
            d.append(np.sqrt(((x[i]-x[j])**2 + (y[i]-y[j])**2)))
    return d

#mat = make_fake_data()


#plt.figure(102)
#plt.title("Three Clusters")
#
#linkage_matrix = linkage(mat, 'single')
#print "three clusters"
#print linkage_matrix
#
#dendrogram(linkage_matrix,
#           truncate_mode='lastp',
#           color_threshold=1,
#           show_leaf_counts=True)
#
plt.show()
