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

MPI_Comm comm_cart;                // Cartesian communicator

// MPI Datatypes for file saving
MPI_Datatype column_datatype,
             row_datatype, 
             grid_datatype, 
             subgrid_datatype;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

int
    dims[2], coord[2],                          // Arrays to store dimensions and coordinates
    local_rows, local_cols,                               // Number of rows and columns for subprocess
    size,                                       // Process count
    rank, cart_rank,
    neighbour[4],
    local_size;                            

#define T(x,y)                      temp[0][(y) * (local_cols + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (local_cols + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (local_cols + 2) + (x)]
enum {NORTH, SOUTH, EAST, WEST};
void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void create_types (void);
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
void free_type(void){
    MPI_Type_free(&row_datatype);
    MPI_Type_free(&column_datatype);
    MPI_Type_free(&grid_datatype);
    MPI_Type_free(&subgrid_datatype);
}

int
main ( int argc, char **argv )
{   
    // TODO 1:
    // - Initialize and finalize MPI.
    // - Create a cartesian communicator.
    // - Parse arguments in the rank 0 processes
    //   and broadcast to other processes

    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

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

    MPI_Dims_create(size, 2, dims); //Get dimensions of the grid

    //printf("\ndims[0]= %d, dims[1]= %d \n", dims[0], dims[1]);

    // Create cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims,(int[2]) {0,0}, 0, &comm_cart);

    // Get my coordinates in the cartesian group
    MPI_Cart_coords(comm_cart, rank, 2, coord);

    // Get my rank in the cartesian group
    MPI_Cart_rank(comm_cart, coord, &cart_rank);


    // Broadcast the parameters to all processes
    MPI_Bcast(&N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // Get neighbours 
    MPI_Cart_shift(comm_cart, 1, 1, &neighbour[NORTH], &neighbour[SOUTH]);
    MPI_Cart_shift(comm_cart, 0, 1, &neighbour[WEST], &neighbour[EAST]);

    printf("I'm %d my north: %d my south: %d my west: %d my east: %d\n", rank, neighbour[NORTH], neighbour[SOUTH], neighbour[WEST], neighbour[EAST]);
    
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Create MPI Datatypes for saving and exchanging data
    create_types();

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.

        border_exchange();

        boundary_condition();

        time_step();

        if ( iteration % snapshot_frequency == 0 )
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
    free_type();
    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for ( int_t y = 1; y <= local_rows; y++ )
    {
        for ( int_t x = 1; x <= local_cols; x++ )
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

void border_exchange( void )
{
    //Sending lower bound to north 
    MPI_Sendrecv(
        &T(1, 1), 1, row_datatype, neighbour[NORTH], 0, 
        &T(1, local_rows + 1), 1, row_datatype, neighbour[SOUTH], 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE  
    );

    //Sending upper bound to south 
    MPI_Sendrecv(
        &T(1, local_rows), 1, row_datatype, neighbour[SOUTH], 0,
        &T(1, 0), 1, row_datatype, neighbour[NORTH], 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    //Sending left bound to west
    MPI_Sendrecv(
        &T(1, 1), 1, column_datatype, neighbour[WEST], 0,
        &T(local_cols + 1, 1), 1, column_datatype, neighbour[EAST], 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    //Sending right bound to east
    MPI_Sendrecv(
        &T(local_cols, 1), 1, column_datatype, neighbour[EAST], 0,
        &T(0, 1), 1, column_datatype, neighbour[WEST], 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

void boundary_condition( void )
{
    // Implement boundary conditions here
    
    // Lower edge
    if (coord[1] == dims[1]-1){
        for ( int_t x = 1; x <= local_cols; x++ ){
            T(x, local_rows + 1) = T(x, local_rows - 1);
        }
    }

    // Upper edge
    if (coord[1] == 0 ){
        for ( int_t x = 1; x <= local_cols; x++ ){
            T(x, 0) = T(x, 2);
        }
    }

    // Left edge
    if (coord[0] == 0){
        for ( int_t y = 1; y <= local_rows; y++ ){
            T(0, y) = T(2, y);
        }
    } 

    // Right edge
    if (coord[0] == dims[0]-1){
        for ( int_t y = 1; y <= local_rows; y++ ){
            T(local_cols + 1, y) = T(local_cols - 1, y);
        }
    }
        
}




void
domain_init ( void )
{

    //calculate subgrid
    local_cols = M/dims[0];
    local_rows = N/dims[1]; 
    local_size = (local_cols+2) * (local_rows+2);

    printf("rank %d local_cols: %d, local_rows: %d, local_size: %d\n", rank, local_cols, local_rows,local_size);
    // Calculate offset
    int offset_x = local_cols * coord[0];
    int offset_y = local_rows * coord[1];
    printf("rank %d offset_x: %d, offset_y: %d\n", rank, offset_x, offset_y);
    // Allocate memory for each subprocess 
    temp[0] = malloc ( local_size* sizeof(real_t) );
    temp[1] = malloc ( local_size * sizeof(real_t) );
    thermal_diffusivity = malloc ( local_size * sizeof(real_t) );

    dt = 0.1;


    for ( int_t y = 1; y <= local_rows; y++ )
    {
        for ( int_t x = 1; x <= local_cols; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + offset_x + y + offset_y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + offset_x) + (y + offset_y)) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }

}

void
create_types( void ){

    //A row is a contiguous vector of doubles sized by the number of columns of the subgrid.
    MPI_Type_contiguous(local_cols, MPI_DOUBLE, &row_datatype);
    MPI_Type_commit(&row_datatype);

    //A column is a vector of doubles sized by the number of rows of the subgrid.
    //The stride is the number that separates two elements of the vector.
    //In this case, the stride is the number of columns of the subgrid plus two, because we need to go to the next row.
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &column_datatype);
    MPI_Type_commit(&column_datatype);


    //A subgrid is a subarray of doubles sized by the number of rows and columns of the subgrid.
    MPI_Type_create_subarray(2,
                            (int[2]){local_rows + 2, local_cols + 2}, //This is the array size, we need to take also the boundaries
                            (int[2]){local_rows,local_cols}, //This is the subsize, so the size we need to look at
                            (int[2]){1, 1}, //Where we start to look at
                            MPI_ORDER_C, MPI_DOUBLE, &subgrid_datatype);

    //A grid is a subarray of doubles sized by the number of rows and columns of the grid.
    MPI_Type_create_subarray(2, (int[2]){N, M},//We need to take all the grid, because we will use this for setting the view
                            (int[2]){local_rows, local_cols}, 
                            (int[2]){ local_rows * coord[1], coord[0] * local_cols}, //We need to multiply the number of rows and columns by the coordinates to get the offset
                            MPI_ORDER_C, MPI_DOUBLE, &grid_datatype);

    MPI_Type_commit(&subgrid_datatype);
    MPI_Type_commit(&grid_datatype);


}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    int offset_x = local_cols * coord[0];
    int offset_y = local_rows * coord[1];
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    MPI_File out;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
    MPI_Offset offset = (offset_y * N + offset_x) * sizeof(real_t);

    MPI_File_set_view(out, 0, MPI_DOUBLE, grid_datatype, "native", MPI_INFO_NULL);
    
    MPI_File_write_all(out, temp[0], 1, subgrid_datatype, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
}

void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
