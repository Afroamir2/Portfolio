#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include "vec.h"

// calculate the arg max
int calc_arg_max (double data[], int num_points, int dim, int centers[], int m,
		  int rank, int size) {
    int arg_max;
    double cost_sq = 0;
    for (int i=0+rank;i<num_points;i+=size) {
        double min_dist_sq = DBL_MAX;
        for (int j=0;j<m;j++) {
            double dist_sq = vec_dist_sq(data+i*dim,data+centers[j]*dim,dim);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        if (min_dist_sq > cost_sq) {
            cost_sq = min_dist_sq;
            arg_max = i;
        }
    }
    struct {double cost_sq; int arg_max; } rank_pair = { cost_sq,  arg_max}; 
    MPI_Allreduce(MPI_IN_PLACE, &rank_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    return rank_pair.arg_max;
}

// find the index of the cluster for the given point
int find_cluster (double kmeans[], double point[], int k, int dim) {
    int cluster;
    double min_dist_sq = DBL_MAX;
    for (int i=0;i<k;i++) {
        double dist_sq = vec_dist_sq(kmeans+i*dim,point,dim);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            cluster = i;
        }
    }
    return cluster;
}

// calculate the next kmeans
void calc_kmeans_next (double data[], int num_points, int dim, double kmeans[], double kmeans_next[], 
		       int k, int rank, int size) {
    int cluster_size[k];
    for (int i=0;i<k;i++) {
        cluster_size[i] = 0;
    }
    vec_zero(kmeans_next,k*dim);
    for (int i=0+rank;i<num_points;i+=size) {
        int cluster = find_cluster(kmeans,data+i*dim,k,dim);
        double* kmean = kmeans_next+cluster*dim;
        vec_add(kmean,data+i*dim,kmean,dim);
        cluster_size[cluster] += 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, kmeans_next, k*dim,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, cluster_size, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for (int i=0;i<k;i++) {
        double* kmean = kmeans_next+i*dim;	
        if (cluster_size[i] > 0) {
            vec_scalar_mult(kmean,1.0/cluster_size[i],kmean,dim);
        } else {
            printf ("error : cluster has no points!\n");
            exit(1);
        }
    }
}

// calculate kmeans using m steps of Lloyd's algorithm
void calc_kmeans (double data[], int num_points, int dim, double kmeans[], int k, int num_iter,
		  int rank, int size) {

    // find k centers using the farthest first algorithm
    int centers[k];
    centers[0] = 0;
    for (int m=1;m<k;m++) {
        centers[m] = calc_arg_max(data,num_points,dim,centers,m,rank,size);
    }

    // initialize kmeans using the k centers
    for (int i=0;i<k;i++) {
        vec_copy(kmeans+i*dim,data+centers[i]*dim,dim);
    }

    // update kmeans num_iter times
    double kmeans_next[k*dim];
    for (int i=0;i<num_iter;i++) {
        calc_kmeans_next(data,num_points,dim,kmeans,kmeans_next,k,rank,size);
        vec_copy(kmeans,kmeans_next,k*dim);
    }
}

int main (int argc, char* argv[]) {

    MPI_Init (&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // read k and m from command line
    if (argc < 3) {
        printf ("Command usage : %s %s %s\n",argv[0],"k","m");
        return 1;
    }
    int k = atoi(argv[1]);
    int m = atoi(argv[2]);

    // There are 60000 MNIST training images.  Each image is 28x28 = 784.
    int num_points = 60000; 
    int dim = 784;

    // dynamically allocate memory for the data matrix
    double* data = (double*)malloc(num_points*dim*sizeof(double));

    // read the binary mnist test file
    vec_read_bin(data,num_points*dim,"train-images-idx3-ubyte",16);

    // start the timer
    double start_time, end_time;
    start_time = MPI_Wtime();

    // calculate kmeans using m steps of Lloyd's algorithm
    double kmeans[k*dim];
    calc_kmeans(data,num_points,dim,kmeans,k,m,rank,size);


    // stop the timer
    end_time = MPI_Wtime();

    // output the results
    if (rank==0) {
#ifdef TIMING
	printf ("(%d,%.4f),",size,(end_time-start_time));
#else
	// print out size
	printf ("# size = %d\n",size);

	// print out wall time used
	printf ("# wall time used = %g sec\n",end_time-start_time);

	// print the results
	for (int i=0;i<k;i++) {
	    for (int j=0;j<dim;j++) {
		printf ("%.5f ",kmeans[i*dim+j]);
	    }
	    printf ("\n");
	}
#endif
    }

    // free the dynamically allocated memory
    free (data);

    MPI_Finalize();
}
