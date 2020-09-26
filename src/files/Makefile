export INPUT_FOLDER:=inputs
export OUTPUT_FOLDER:=outputs
export IMAGES_FOLDER:=images

export INPUT_NAMES:=$(basename $(notdir $(wildcard $(INPUT_FOLDER)/*.in)))

.PHONY: inputs clean outputs images

inputs:
	$(MAKE) --directory $(INPUT_FOLDER)

outputs:
	$(MAKE) --directory $(OUTPUT_FOLDER)

images:
	$(MAKE) --directory $(IMAGES_FOLDER)

clean:
	$(MAKE) clean --directory $(OUTPUT_FOLDER)
	$(MAKE) clean --directory $(IMAGES_FOLDER)