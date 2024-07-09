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
int rank,size, north, south, east, west;
int dims[2] = {0};
int coord[2], period[2] = {0};

MPI_Comm comm_cart;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency,
    *local_size = NULL;

MPI_Datatype column_datatype /*North-South*/, row_datatype, grid, subgrid; //, all_column_datatype, all_row_datatype;

int_t local_col,
    local_row;
real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

MPI_Datatype
        grid,
        subgrid;

#define T(x,y)                      temp[0][(y) * (local_col + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (local_col + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (local_col + 2) + (x)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );
void create_types(void);
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
    MPI_Cart_shift(comm_cart, 0, 1, &south, &north); // down up
    MPI_Cart_shift(comm_cart, 1, 1, &east, &west); // left right
    printf("I'm %d my north: %d my south: %d my west: %d my east: %d\n", rank, north, south, west, east);
    printf("I'm %d my coords are : %d %d\n", rank, coord[0], coord[1]);
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );
    create_types();
    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.
        border_exchange();
        boundary_condition();

        time_step();

        if (( iteration % snapshot_frequency == 0 ) )
        {
           printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            domain_save ( iteration );
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


void create_types(void)
{
    MPI_Type_create_subarray(2, (int[2]){local_row + 2, local_col + 2}, (int[2]){local_row, local_col}, (int[2]){1, 1}, MPI_ORDER_C, MPI_DOUBLE, &subgrid);
    MPI_Type_create_subarray(2, (int[2]){N, M}, (int[2]){local_row, local_col}, (int[2]){ local_row * coord[0], coord[1] * local_col}, MPI_ORDER_C, MPI_DOUBLE, &grid);

    MPI_Type_commit(&subgrid);
    MPI_Type_commit(&grid);



    //Datatype for one column
    MPI_Type_vector(local_col,
                            1, //The block length, which is just one value
                            local_row+2, //Displacement between each value (need to skip one col ahead)
                            MPI_DOUBLE,
                            &column_datatype);
    MPI_Type_commit(&column_datatype);
    
    //Datatype for one row
    MPI_Type_contiguous(local_row, 
                        MPI_DOUBLE,
                        &row_datatype);
    MPI_Type_commit(&row_datatype);
    
}
void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for ( int_t y = 1; y <= local_col; y++ )
    {
        for ( int_t x = 1; x <= local_row; x++ )
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

void border_exchange( void ){
    // TODO 7: Implement border exchange.

        //North to south and South to north
        MPI_Sendrecv(&T(local_col,1), 1, column_datatype, west,0,&T(1,0), 1, column_datatype, east, 0,comm_cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&T(1, 1), 1, column_datatype, east,0,&T(local_col+1,1), 1, column_datatype, west, 0,comm_cart, MPI_STATUS_IGNORE);



        //east to west and west to east
        MPI_Sendrecv(&T(local_row,1),1, row_datatype, south, 0,&T(1,0),1, row_datatype, north, 0,comm_cart, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&T(1,1),1, row_datatype, north, 0,&T(local_row+1,0),1, row_datatype, south, 0,comm_cart, MPI_STATUS_IGNORE);
}

void
boundary_condition ( void )
{
    // TODO 4: Change the application of boundary conditions
    // to match the cartesian topology.

    /*
            N
        ________
        | 0 | 2 |
     W  |---|---|   E
        | 1 | 3 |
        ‾‾‾‾‾‾‾‾
            S    
    
    */
    if(rank == 0 || rank == 1){
        for(int_t x = 1;x <= local_col; x++){
            T(local_row+1, x) = T(local_row-1, x);
        }
    }else{
        for(int_t x = 1;x <= local_col; x++){
            T(0, x) = T(2, x);
        }
    }

    if(rank ==2 || rank == 3){
        for(int_t y = 1; y<= local_col; y++){
            T(y, local_row+1) = T(y, local_row-1);
        }
    }else{
        for(int_t y = 1; y<= local_col; y++){
            T(y, 0) = T(y, 2);
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
    local_col = M/dims[1];
    local_row = N/dims[0];
    //printf("My local size %d is : %ld %ld\n", rank, local_col, local_row);
   // int local_size = (local_col+2) * (local_row+2);
    temp[0] = malloc ( (local_col+2) * (local_row+2)* sizeof(real_t) );
    temp[1] = malloc ( (local_col+2) * (local_row+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (local_col+2) * (local_row+2) * sizeof(real_t) );
    //printf("%ld\n", (local_col+2) * (local_row+2));
    dt = 0.1;
    int_t offset_x = local_col * coord[1];
    int_t offset_y = local_row * coord[0];
    
    //printf("My %d offset is : %ld %ld\n",rank, offset_x, offset_y);
    for ( int_t y = 1; y <= local_row; y++ )
    {
        for ( int_t x = 1; x <= local_col; x++ )
        {
            real_t temperature = 30 + 30 * sin(((x+ offset_x) + (offset_y + y)) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + offset_x) + (offset_y+ y)) / 20.0)) / 605.0;

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
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    MPI_File out;
    MPI_File_open(comm_cart, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);

    MPI_File_set_view(out, 0, MPI_DOUBLE, grid, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, temp[0], 1, subgrid, MPI_STATUS_IGNORE);

    MPI_File_close(&out);
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
