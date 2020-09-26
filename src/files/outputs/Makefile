HULLS:=$(addsuffix .hull, $(INPUT_NAMES))

.PHONY: clean

all: $(HULLS)

%.hull: ../inputs/%.in
	$(COMMAND_EXE) ../../$(OUTPUT_EXE) < $+ > $@

clean:
	rm -rf *.hull