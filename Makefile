# Mario to Sonic — build the emulator-side pieces.
#
# The only compiled artefact is the input plugin: everything above it is Python.
# We build against vendored m64p headers (vendor/m64p, taken from the simple64
# core source) so the build does not depend on libmupen64plus-dev being present.

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -Wall -Wextra -std=c11
BUILD   := build
PLUGIN  := $(BUILD)/mupen64plus-input-shm.so

.PHONY: all clean test test-player test-obs
all: $(PLUGIN)

$(PLUGIN): src/plug/input_shm.c | $(BUILD)
	$(CC) $(CFLAGS) -Ivendor/m64p -shared -o $@ $< -lrt
	@echo "built $@"

$(BUILD):
	@mkdir -p $(BUILD)

test: all
	python3 -m src.mk64.selftest

test-player: all
	python3 -m src.mk64.player_selftest

test-obs: all
	python3 -m src.obs.selftest

clean:
	rm -rf $(BUILD)
