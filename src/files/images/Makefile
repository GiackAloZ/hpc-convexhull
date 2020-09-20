IMAGES:=$(addsuffix .png, $(INPUT_NAMES))

.PHONY: all

all: $(IMAGES)

%.png: ../$(INPUT_FOLDER)/%.in ../$(OUTPUT_FOLDER)/%.hull
	gnuplot -c plot-hull.gp $+ $@
