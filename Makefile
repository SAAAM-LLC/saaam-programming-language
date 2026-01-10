# SAAAM Language - Native Runtime Build System
# Builds C runtime, CUDA kernels, and creates shared library for Python FFI

CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -std=c11 -fPIC
NVCCFLAGS = -O3 -arch=sm_75 -Xcompiler -fPIC
LDFLAGS = -shared

# Source files
C_SOURCES = saaam_native_runtime.c
CUDA_SOURCES = saaam_cuda_kernels.cu
TEST_SOURCE = test_native_runtime.c

# Output files
LIB_NAME = libsaaam_runtime
SHARED_LIB = $(LIB_NAME).so
STATIC_LIB = $(LIB_NAME).a
TEST_BIN = test_native_runtime

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Platform-specific settings
ifeq ($(OS),Windows_NT)
	SHARED_LIB = $(LIB_NAME).dll
	TEST_BIN = test_native_runtime.exe
	RM = del /Q
	MKDIR = if not exist
else
	RM = rm -f
	MKDIR = mkdir -p
endif

# Detect CUDA availability
CUDA_AVAILABLE := $(shell which $(NVCC) 2>/dev/null)

.PHONY: all clean test help cuda_check

all: $(SHARED_LIB) $(TEST_BIN)
	@echo "🚀 SAAAM Native Runtime built successfully!"

# Build shared library
$(SHARED_LIB): $(C_OBJECTS) $(CUDA_OBJECTS)
	@echo "🔗 Linking shared library: $@"
ifdef CUDA_AVAILABLE
	$(NVCC) $(LDFLAGS) -o $@ $^
else
	$(CC) $(LDFLAGS) -o $@ $(C_OBJECTS)
	@echo "⚠️  Built without CUDA support (nvcc not found)"
endif

# Build static library
$(STATIC_LIB): $(C_OBJECTS) $(CUDA_OBJECTS)
	@echo "📦 Creating static library: $@"
	ar rcs $@ $^

# Compile C sources
%.o: %.c saaam_native_runtime.h
	@echo "🔨 Compiling C: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA sources
%.o: %.cu saaam_native_runtime.h
ifdef CUDA_AVAILABLE
	@echo "⚡ Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
	@echo "⚠️  Skipping CUDA compilation (nvcc not found): $<"
endif

# Build test program
$(TEST_BIN): $(TEST_SOURCE) $(SHARED_LIB)
	@echo "🧪 Building test program: $@"
ifdef CUDA_AVAILABLE
	$(NVCC) -O3 -o $@ $< -L. -lsaaam_runtime -Wl,-rpath,.
else
	$(CC) $(CFLAGS) -o $@ $< -L. -lsaaam_runtime -Wl,-rpath,.
endif

# Run tests
test: $(TEST_BIN)
	@echo "🧠 Running SAAAM Native Runtime Tests..."
	@echo ""
	@./$(TEST_BIN)

# Check CUDA availability
cuda_check:
ifdef CUDA_AVAILABLE
	@echo "✓ CUDA is available: $(CUDA_AVAILABLE)"
	@$(NVCC) --version
else
	@echo "✗ CUDA not found. Install CUDA Toolkit for GPU acceleration."
endif

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	$(RM) $(C_OBJECTS) $(CUDA_OBJECTS) $(SHARED_LIB) $(STATIC_LIB) $(TEST_BIN)
	@echo "✓ Clean complete"

# Help
help:
	@echo "SAAAM Native Runtime Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build shared library and test program (default)"
	@echo "  test        - Build and run tests"
	@echo "  clean       - Remove build artifacts"
	@echo "  cuda_check  - Check if CUDA is available"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Files:"
	@echo "  $(SHARED_LIB)  - Shared library for Python FFI"
	@echo "  $(STATIC_LIB)  - Static library"
	@echo "  $(TEST_BIN)     - Test executable"
	@echo ""
	@echo "Requirements:"
	@echo "  - GCC or compatible C compiler"
	@echo "  - CUDA Toolkit (optional, for GPU acceleration)"
	@echo ""
	@echo "Usage:"
	@echo "  make           # Build everything"
	@echo "  make test      # Run tests"
	@echo "  make clean     # Clean build"
