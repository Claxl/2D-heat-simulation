# TDT4200 Problem set 2: Introduction to MPI

## 2D heat simulation
In this exercise you will take a sequential implementation of the Finite Difference Method (FDM) for solving the 2D heat equation and write parallel versions using Pthreads and OpenMP. The sequential solution is provided, and can be found in `heat_sequential.c` Skeletons for your Pthreads and OpenMP implementations are provided in `heat_pthreads.c` and `heat_omp.c`, respectively. Information on solving this assignment is described in the lecture slides and in the problem set description.

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

Prepares the check.

### Sequential solution
**Compile**

`make sequential`

**Run**

`./sequential -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make sequential
./sequential -m 256 -n 256 -i 100000 -s 1000
```

**Compile and run**

You can also execute both of the above commands together with default values with `make run_sequential`

### Pthreads solution
**Compile**

`make pthreads`

**Run**

`./pthreads -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make pthreads
./pthreads -m 256 -n 256 -i 100000 -s 1000
```

**Compile and Run**

You can also execute both of the above commands together with default values with `make run_pthreads`.

### OpenMP solution
**Compile**

`make omp`

**Run**

`./omp -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make omp
./omp -m 256 -n 256 -i 100000 -s 1000
```

**Compile and Run**

You can also execute both of the above commands together with default values with `make run_omp`.

## Visualize
### Plots
`./plot_solution.sh -m [y_size] -n [x_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot_pthreads` or `make plot_omp` .

You can plot the sequential solution with `make plot_sequential`.

**Example**

`./plot_solution.sh -m 256 -n 256`

### Video
`make show_pthreads`
`make show_omp`

Compiles, runs, and plots the solutions with default values and creates a video using [ffmpeg](https://ffmpeg.org).

You can create a video from the sequential solution with `make show_sequential`.

## Check
`make check_pthreads`
`make check_omp`

Compiles and runs the solutions with default values and compares the output data to reference data.

You can check the sequential solution with `make check_sequential`.

## Options
Option | Description | Restrictions | Default value
:------------ | :------------ | :------------ | :------------
**-m** | Size of the y dimension | > 0 | 256
**-n** | Size of the x dimension | > 0 | 256
**-i** | Maximum number of iterations | > 0 | 100000
**-s** | Number of iterations between each time the state is saved to file | > 0 | 1000
**-np**| Number of processes (MPI option) | > 0 | 4

## Installing dependencies
**gnuplot**

Linux/Ubuntu:

```
sudo apt update
sudo apt install gnuplot
```

MacOSX:

```
brew update
brew install gnuplot
```

**ffmpeg**

Linux/Ubuntu:

```
sudo apt update
sudo apt install ffmpeg
```

MacOSX:

```
brew update
brew install ffmpeg
```
