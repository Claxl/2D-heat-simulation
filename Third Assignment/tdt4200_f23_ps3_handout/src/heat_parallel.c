#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;
int rank,size;
int dims[2] = {0};
int coord[2], period[2] = {0};

MPI_Comm comm_cart;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency,
    *local_size = NULL;


real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

#define T(x,y)                      temp[0][(y) * (N + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (N + 2) + (x)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int
main ( int argc, char **argv )
{
    // TODO 1:
    // - Initialize and finalize MPI.
    // - Create a cartesian communicator.
    // - Parse arguments in the rank 0 processes
    //   and broadcast to other processes
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if(rank == 0){
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }
    MPI_Dims_create(size,2,dims);
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, period, 0, &comm_cart );

    MPI_Bcast(&N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    
    MPI_Cart_coords(comm_cart,rank,2 ,coord);
    
   // printf("My coords are : %d %d I'm :%d\n", coord[0], coord[1], rank);
    
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.

        //boundary_condition();

       // time_step();

        if (( iteration % snapshot_frequency == 0 ) && (rank == 0))
        {
           printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

          //  domain_save ( iteration );
        }


        swap( &temp[0], &temp[1] );
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();
    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for ( int_t y = 1; y <= M/size; y++ )
    {
        for ( int_t x = 1; x <= N/size; x++ )
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


void
boundary_condition ( void )
{
    // TODO 4: Change the application of boundary conditions
    // to match the cartesian topology.

        for ( int_t x = 1; x <= N/dims[0]; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, M/dims[1]+1) = T(x, M/dims[1]-1);
    }

    //Only first rank has to manage upper boundary
    if (rank == 0)
    {
        for ( int_t y = 1; y <= M/dims[1]; y++ )
        {
            T(0, y) = T(2, y);
        }
    }

    //Only last rank has to manage lower boundary
    if (rank == (size - 1))
    {
        for ( int_t y = 1; y <= M/dims[1]; y++ )
        {
            T(N/dims[0]+1, y) = T(N/dims[0]-1, y);
        }
    }
}


void
domain_init ( void )
{
    // TODO 2:
    // - Find the number of columns and rows in each process' subgrid.
    // - Allocate memory for each process' subgrid.
    // - Find each process' offset to calculate the correct initial values.
    // Hint: you can get useful information from the cartesian communicator.
    // Note: you are allowed to assume that the grid size is divisible by
    // the number of processes.
    int_t local_col = M/dims[0];
    int_t local_row = N/dims[1];
    printf("My local size %d is : %d %d\n", rank, local_col, local_row);
   // int local_size = (local_col+2) * (local_row+2);
    temp[0] = malloc ( (local_col+2) * (local_row+2)* sizeof(real_t) );
    temp[1] = malloc ( (local_col+2) * (local_row+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (local_col+2) * (local_row+2) * sizeof(real_t) );

    dt = 0.1;
    int_t offset_x = local_row * coord[0];
    int_t offset_y = local_col * coord[1];
    printf("My %d offset is : %ld %ld\n",rank, offset_x, offset_y);
    for ( int_t y = 1; y <= local_col; y++ )
    {
        for ( int_t x = 1; x <= local_row; x++ )
        {
            real_t temperature = 30 + 30 * sin(((x+ offset_x) + (offset_y + y)) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + offset_x + offset_y+ y) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    // TODO 5: Use MPI I/O to save the state of the domain to file.
    // Hint: Creating MPI datatypes might be useful.

    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    for ( int_t iter = 1; iter <= N; iter++)
    {
        fwrite( temp[0] + (M+2) * iter + 1, sizeof(real_t), N, out );
    }
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
