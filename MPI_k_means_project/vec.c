#include <stdio.h>
#include <stdlib.h>
#include "vec.h"

/* calculates ||u-v||^2 */
double vec_dist_sq (double* u, double* v, int dim) {
    double dist_sq = 0;
    for (int i=0;i<dim;i++) {
        dist_sq += (u[i]-v[i])*(u[i]-v[i]);
    }
    return dist_sq;
}

/* w = u + v */
void vec_add (double* u, double* v, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        w[i] = u[i] + v[i];
    }
}

/* w = cv */
void vec_scalar_mult (double* v, double c, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        w[i] = v[i]*c;
    }
}

/* performs the deep copy v->data[i] = w->data[i] for all i */
void vec_copy (double* v, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        v[i] = w[i];
    }
}

/* zeros the vector v */
void vec_zero (double* v, int dim) {
    for (int i=0;i<dim;i++) {
        v[i] = 0;
    }
}

/* read a vector from a binary file of unsigned chars */
void vec_read_bin (double* data, int dim, char* filename, int header_size) {

    unsigned char header[header_size];
    FILE* fptr;

    /* open the binary file for reading */
    fptr = fopen(filename,"rb"); 

    /* need to check for null */
    if (fptr == 0) {
	printf ("Error opening binary data file.\n");
	exit(1);
    }

    /* allocate space for the temporary data buffer */
    unsigned char* buf = (unsigned char*)malloc(dim*sizeof(unsigned char));
    if (buf == 0) {
	printf ("Error allocating buffer to read data file.\n");
	exit(1);
    }

    /* skip over header */
    size_t num_read = 0;
    num_read = fread(header, sizeof(unsigned char), header_size, fptr);

    /* read data into temporary buffer */
    num_read = fread(buf, sizeof(unsigned char), dim, fptr);

    /* store data in the given vector (as doubles) */
    for (int i=0;i<dim;i++) {
	data[i] = (double)buf[i];
    }

    /* free the temporary data buffer */
    free (buf);

    /* close the binary file */
    fclose(fptr);

}
