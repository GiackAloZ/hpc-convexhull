IMAGES:=$(addsuffix .png, $(INPUT_NAMES))

.PHONY: clean 

all: $(IMAGES)

%.png: ../$(INPUT_FOLDER)/%.in ../$(OUTPUT_FOLDER)/%.hull
	gnuplot -c plot-hull.gp $+ $@

clean:
	rm -rf *.png
