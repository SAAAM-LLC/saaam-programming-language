# SAAAM Language - Native Runtime Build System
# Builds a dependency-free C runtime + optional CUDA kernel library.

CC ?= gcc
NVCC ?= nvcc
AR ?= ar

CFLAGS ?= -Wall -Wextra -O3 -std=c11
NVCCFLAGS ?= -O3 -arch=sm_75

LIB_NAME := libsaaam_runtime
STATIC_LIB := $(LIB_NAME).a

C_SOURCES := saaam_native_runtime.c
CUDA_SOURCES := saaam_cuda_kernels.cu
TEST_SOURCE := test_native_runtime.c

C_OBJECTS := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)

ifeq ($(OS),Windows_NT)
	SHARED_LIB := $(LIB_NAME).dll
	TEST_BIN := test_native_runtime.exe
	TEST_RUN := .\\$(TEST_BIN)
	RM := del /Q
	DEVNULL := NUL
else
	SHARED_LIB := $(LIB_NAME).so
	TEST_BIN := test_native_runtime
	TEST_RUN := ./$(TEST_BIN)
	RM := rm -f
	DEVNULL := /dev/null
	CFLAGS += -fPIC
	NVCCFLAGS += -Xcompiler -fPIC
endif

# CUDA detection works in both cmd.exe and sh environments.
CUDA_AVAILABLE := $(shell $(NVCC) --version >$(DEVNULL) 2>$(DEVNULL) && echo 1)

ifeq ($(CUDA_AVAILABLE),1)
	LINK := $(NVCC)
	CFLAGS += -DSAAAM_WITH_CUDA=1
	NVCCFLAGS += -DSAAAM_WITH_CUDA=1
else
	LINK := $(CC)
	# When CUDA isn't present, we skip CUDA objects entirely.
	CUDA_OBJECTS :=
endif

.PHONY: all clean test shared cuda_check

all: $(STATIC_LIB) test

shared: $(SHARED_LIB)

$(SHARED_LIB): $(C_OBJECTS) $(CUDA_OBJECTS)
	$(LINK) -shared -o $@ $^

$(STATIC_LIB): $(C_OBJECTS) $(CUDA_OBJECTS)
	$(AR) rcs $@ $^

%.o: %.c saaam_native_runtime.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu saaam_native_runtime.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test: $(TEST_BIN)
	$(TEST_RUN)

$(TEST_BIN): $(TEST_SOURCE) $(STATIC_LIB)
	$(LINK) -O3 -o $@ $(TEST_SOURCE) $(STATIC_LIB)

cuda_check:
ifeq ($(CUDA_AVAILABLE),1)
	@echo CUDA available
else
	@echo CUDA not found
endif

clean:
	-$(RM) $(C_OBJECTS) $(CUDA_OBJECTS) $(STATIC_LIB) $(SHARED_LIB) $(TEST_BIN) 2>$(DEVNULL)
