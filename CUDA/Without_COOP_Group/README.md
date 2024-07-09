# TDT4200 Problem set 5: Introduction to CUDA

## 2D heat simulation
In this exercise you will take a seqential implementation of the Finite Difference Method (FDM) for solving the 2D heat equation and write a GPU version using CUDA. The sequential implementation is provided. Information on solving this assignment is described in the lecture slides and in the problem set description.

A sequential implementation is provided can be found in `heat_sequential.c` and should be kept as a reference. A copy of the sequential solution can be found in `heat_parallel.cu`, in which you should write your parallel implementation.

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

### Parallel solution
**Compile**

`make parallel`

**Run**

`./parallel -m [y_size] -n [x_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make parallel
./parallel -m 256 -n 256 -i 100000 -s 1000
```

**Compile and Run**

You can also execute both of the above commands together with default values with `make run`.

## Visualize
### Plots
`./plot_solution.sh -m [y_size] -n [x_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot` .

You can plot the sequential solution with `make plot_sequential`.

**Example**

`./plot_solution.sh -m 256 -n 256`

### Video
`make show`

Compiles, runs, and plots the solution with default values and creates a video using [ffmpeg](https://ffmpeg.org).

You can create a video from the sequential solution with `make show_sequential`.

## Check
`make check`

Compiles and runs the solution with default values and compares the output data to reference data.

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
If you do not have an Nvidia GPU available to use, or do not have the correct environment, you can use our the course servers through Snotra. See the document under *Sources and Syllabus* on Blackboard on how to access and use the cluster. You have to update the Makefile to use the correct CUDA compiler on Oppdal and Selbu. This is explained in comments in the Makefile.

**CUDA**

Linux/Ubuntu:

Use [this guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

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
