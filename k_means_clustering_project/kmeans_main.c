#include <stdio.h>
#include <stdlib.h>
#include "vec.h"
#include "kmeans.h"

int main (int argc, char* argv[]) {

    // get k and m from command line
    if (argc < 3) {
        printf ("Command usage : %s %s %s\n",argv[0],"k","m");
        return 1;
    }
    int k = atoi(argv[1]);
    int m = atoi(argv[2]);

    // read the number of points and the dimension of each point    
    int num_points, dim;
    if (scanf("%*c %d %d",&num_points, &dim) != 2) {
        printf ("error reading the number of points and the dimension\n");	
        return 1;
    }

    // dynamically allocate memory for the data array    
    double* data = (double*)malloc(num_points*dim*sizeof(double));

    // Read vectors from stdin and store them in the data array
    for (int i=0;i<num_points;i++) {
        if (vec_read_stdin(data+i*dim,dim) != dim) {
            printf ("error reading the next point from stdin\n");
            return 1;
        }
    }

    // calculate kmeans using m steps of Lloyd's algorithm
    double kmeans[k*dim];
    calc_kmeans(data,num_points,dim,kmeans,k,m);

    // print the results
    for (int i=0;i<k;i++) {
        for (int j=0;j<dim;j++) {
            printf ("%.5f ",kmeans[i*dim+j]);
        }
        printf ("\n");
    }

    // free the dynamically allocated memory
    free (data);
}
