export INPUT_NAMES:=ace box1k box10k box100k box1M circ1k circ10k circ100k gaus100k gaus1M
export INPUT_FOLDER:=inputs
export OUTPUT_FOLDER:=outputs
export IMAGES_FOLDER:=images

.PHONY: all clean $(INPUT_FOLDER)

all: $(INPUT_FOLDER)

$(INPUT_FOLDER):
	$(MAKE) -s --directory $@

outputs:
	$(MAKE) -s --directory $(OUTPUT_FOLDER)

images:
	$(MAKE) -s --directory $(IMAGES_FOLDER)

clean: