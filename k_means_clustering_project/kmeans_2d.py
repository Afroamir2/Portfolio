import sys
import numpy as np
import matplotlib.pyplot as plt

# the name of the output file is a command line argument
if (len(sys.argv) < 2):
    print ("Command Usage : python3",sys.argv[0],"outfile")
    exit(1)
outfile = sys.argv[1]

(comment,rows_str,cols_str) = sys.stdin.readline().split()
rows = int(rows_str)

# read the data file
data = np.loadtxt(sys.stdin)

# plot the data
plt.gca().set_aspect('equal')
#plt.scatter(data[:,0],data[:,1],s=1,color='black')

# plot the means
rows_total = len(data)
num_means = rows_total - rows
if (num_means > 0):
    means = np.zeros(num_means,dtype='int')
    for k in range(num_means):
        means[k] = rows+k
    means_nums = range(len(means))
    point_means = np.zeros(len(data))
    distance_sq = np.zeros(len(means))
    for i in range(len(data)):
        for j in range(len(means)):
            distance_sq[j] = np.inner(data[i]-data[means[j]],
                    data[i]-data[means[j]])
        point_means[i] = np.argmin(distance_sq)
    plt.scatter (data[:,0],data[:,1],
            c=point_means,cmap="tab10",s=1)
    plt.scatter (data[means,0],data[means,1],
            c=means_nums,cmap="tab10",s=100)
    plt.scatter (data[means,0],data[means,1],
            s=100, facecolors='none', edgecolors='black')

# save the plot as an image
plt.savefig(outfile)
