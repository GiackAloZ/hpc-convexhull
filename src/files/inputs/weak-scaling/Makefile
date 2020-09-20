INPUT_NAMES:=circ30k circ42k circ60k circ90k circ103k
INPUT_FILES:=$(addsuffix .in, $(INPUT_NAMES))

ALL: $(INPUT_FILES)

circ30k.in:
	rbox s 30000 D2 > $@

circ42k.in:
	rbox s 42426 D2 > $@

circ60k.in:
	rbox s 60000 D2 > $@

circ90k.in:
	rbox s 90000 D2 > $@

circ103k.in:
	rbox s 103923 D2 > $@