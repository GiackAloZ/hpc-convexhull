GEN_INPUT_NAMES:=box1k box10k box100k box1M circ1k circ10k circ100k
GEN_INPUT_FILES:=$(addsuffix .in, $(GEN_INPUT_NAMES))

ALL: $(GEN_INPUT_FILES)

box1k.in:
	rbox 1000 D2 > $@

box10k.in:
	rbox 10000 D2 > $@

box100k.in:
	rbox 100000 D2 > $@

box1M.in:
	rbox 1000000 D2 > $@

box10M.in:
	rbox 10000000 D2 > $@

circ1k.in:
	rbox s 1000 D2 > $@

circ10k.in:
	rbox s 10000 D2 > $@

circ100k.in:
	rbox s 100000 D2 > $@