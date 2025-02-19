#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "vec.h"

// calculate the arg max
int calc_arg_max (double data[], int num_points, int dim, int centers[], int m) {
    int arg_max;
    double cost_sq = 0;
    for (int i=0;i<num_points;i++) {
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
    return arg_max;
}

// find the index of the cluster for the given point
int find_cluster (double kmeans[], double point[], int k, int dim) {
    int cluster = 0;
    double min_dist_sq = DBL_MAX;

    for (int i = 0; i < k; i++){
        double curr = vec_dist_sq(point, &kmeans[i*dim], dim);

        if(curr < min_dist_sq){
            min_dist_sq = curr;
            cluster = i;
        }


    }

    return cluster;
}

// calculate the next kmeans
void calc_kmeans_next (double data[], int num_points, int dim, double kmeans[], double kmeans_next[], int k) {
    double sums[k*dim];
    int counts[k];
    for (int i = 0; i < k * dim; i++){
        sums[i] = 0;
    } 
    for (int i = 0; i < k; i++){
        counts[i] = 0;
    }
    
    for (int i = 0; i < num_points; i++){
        int clusterIndex = find_cluster(kmeans, &data[i*dim], k, dim);
        counts[clusterIndex]++;
        for(int j = 0; j < dim; j++){
            sums[clusterIndex*dim+j] += data[i*dim+j];
        }
    }
    for (int i = 0; i < k; i++) {
    if (counts[i] == 0) {
        printf("Error: Empty cluster found. Exiting.\n");
        exit(1); 
    } else {
        for (int j = 0; j < dim; j++) {
            kmeans_next[i * dim + j] = sums[i*dim+j] / counts[i];
        }
    }
}


}

// calculate kmeans using m steps of Lloyd's algorithm
void calc_kmeans (double data[], int num_points, int dim, double kmeans[], int k, int m) {

    // find k centers using the farthest first algorithm
    int centers[k];
    centers[0] = 0;
    for (int m=1;m<k;m++) {
        centers[m] = calc_arg_max(data,num_points,dim,centers,m);
    }

    // initialize kmeans using the k centers
    for (int i=0;i<k;i++) {
        vec_copy(kmeans+i*dim,data+centers[i]*dim,dim);
    }

    // update kmeans m times
    double kmeans_next[k*dim];
    for (int i=0;i<m;i++) {
        calc_kmeans_next(data,num_points,dim,kmeans,kmeans_next,k);
        vec_copy(kmeans,kmeans_next,k*dim);
    }
}
