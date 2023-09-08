# TDT4200 Problem set 1

## 1D heat simulation
In this exercise you will work on a sequential implementation of implicit solvers for the 1D heat equation. The solvers are Jacobi, Gauss-Seidel and Red-black Gauss-Seidel. Information on solving this assignment is described in the lecture slides and in the problem set description.

The skeleton for your implementation can be found in `heat_sequential.c`.

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

Prepares the check.

### Compile
`make heat`

### Run
`./heat -n [x_size] -i [max_iteration] -s [snapshot_frequency] -t [solver_type]`

### Example
```
make heat
./heat -n 2048 -i 1000000 -s 10000 -t 1
```

### Compile and run
You can also execute both of the above commands together with default values with `make run`.

## Visualize
### Plots
`./plot_solution.sh -n [x_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot` .

**Example**

`./plot_solution.sh -n 2048`

### Video
`make show`

Compiles, runs, and plots the solution with default values and creates a video using [ffmpeg](https://ffmpeg.org).

## Check
`make check`

Compiles and runs the solution with default values and compares the output data to reference data.

## Options
Option | Description | Restrictions | Default value
:------------ | :------------ | :------------ | :------------
**-n** | Size of the x dimension | > 0 | 2048
**-i** | Maximum number of iterations | > 0 | 1000000
**-s** | Number of iterations between each time the state is saved to file | > 0 | 10000
**-t** | Solver type | [1, 2, 3] (1: Jacobi, 2: Gauss-Seidel, 3: Red-black Gauss-Seidel) | 1

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
