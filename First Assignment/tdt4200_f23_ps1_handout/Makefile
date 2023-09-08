PROGNAME:=heat
EXECUTABLE:=$(PROGNAME)

DATADIR:=data/
PLOTSDIR:=plots/
VIDEODIR:=video/

DATA:=$(DATADIR)*.dat
PLOTS:=$(PLOTSDIR)*.png
VIDEO:=$(VIDEODIR)*.mp4

CC=gcc
CFLAGS+= -std=c11 -O2 -g
LDLIBS+= -lm

SRC_FILES:=src/$(PROGNAME)_sequential.c src/argument_utils.c
INC_PATHS:=Iinc/

.PHONY: clean purge setup run check plot show viz

$(EXECUTABLE): $(SRC_FILES)
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(EXECUTABLE)

purge:
	rm -f $(EXECUTABLE) $(DATA) $(PLOTS) $(VIDEO)

setup:
	mkdir -p data plots video
	cd check && make

run: setup $(EXECUTABLE)
	./$(EXECUTABLE)

check: setup purge
	make heat
	./check/check_solution.sh

plot: setup purge run
	./plot_results.sh

show: setup purge run viz

viz:
	./plot_results.sh > /dev/null
	ffmpeg -y -i $(PLOTSDIR)%5d.png -vf format=yuv420p $(VIDEODIR)animation.mp4 &> /dev/null
	./open_video.sh &> /dev/null
