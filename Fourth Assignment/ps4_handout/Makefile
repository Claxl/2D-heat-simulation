PROGNAME:=heat

DATADIR:=data/
PLOTSDIR:=plots/
VIDEODIR:=video/

DATA:=$(DATADIR)*.dat
PLOTS:=$(PLOTSDIR)*.png
VIDEO:=$(VIDEODIR)*.mp4

CC:=gcc
CFLAGS+=-fopenmp
LDLIBS+=-lm -pthread

OMP_SRC_FILES:=src/$(PROGNAME)_omp.c src/argument_utils.c
PTHREADS_SRC_FILES:=src/$(PROGNAME)_pthreads.c src/argument_utils.c
SERIAL_SRC_FILES:=src/$(PROGNAME)_sequential.c src/argument_utils.c

.PHONY: clean purge setup run_sequential run_omp run_pthreads check_sequential check_omp check_pthreads plot_sequential plot_omp plot_pthreads show_sequential show_omp show_pthreads viz

pthreads: $(PTHREADS_SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@

omp: $(OMP_SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@

sequential: $(SERIAL_SRC_FILES)
	$(CC) $^ $(CFLAGS) -o $@ $(LDLIBS)

clean:
	-rm -f sequential omp pthreads

purge:
	-rm -f sequential omp pthreads $(DATA) $(PLOTS) $(VIDEO)

setup:
	-mkdir -p data plots video
	$(MAKE) -C check clean
	$(MAKE) -C check all

run_sequential: purge sequential
	./sequential

run_omp: purge omp
	./omp

run_pthreads: purge pthreads
	./pthreads

check_sequential: purge sequential
	./check/check_sequential_solution.sh

check_omp: purge omp
	./check/check_omp_solution.sh

check_pthreads: purge pthreads
	./check/check_pthreads_solution.sh

plot_sequential: purge run_sequential
	./plot_results.sh

plot_omp: purge run_omp
	./plot_results.sh

plot_pthreads: purge run_pthreads
	./plot_results.sh

show_sequential: purge run_sequential viz
show_omp: purge run_omp viz
show_pthreads: purge run_pthreads viz

viz:
	./plot_results.sh > /dev/null
	ffmpeg -y -i $(PLOTSDIR)%5d.png -vf format=yuv420p $(VIDEODIR)animation.mp4 &> /dev/null
	./open_video.sh &> /dev/null
