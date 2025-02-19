import sys
import numpy as np
import matplotlib.pyplot as plt

if (len(sys.argv) < 2):
    print ("Not enough command line arguments")
    quit()

outfile = sys.argv[1]

# read the data file
data = np.loadtxt(sys.stdin)
data = data.astype(int)

plt.rcParams['figure.figsize'] = (20, 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
images = np.zeros((25,28,28),dtype=int);
for i in range(25):
    images[i] = np.asarray(data[i]).reshape(28,28)

f, axarr = plt.subplots(5,5)
for i in range(5):
    for j in range(5):
        axarr[i][j].imshow(images[i*5+j],cmap='gray',vmin=0, vmax=255, interpolation='none')
plt.savefig(outfile,bbox_inches='tight')
