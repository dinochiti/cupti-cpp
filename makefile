# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -lcupti -arch=sm_61 -O3 -Xcompiler -O3 -Xcompiler -Wall

# Executable name
TARGETDIR = bin/
TARGET = cc

# Source files
SRCS = src/*.cpp src/*.cu

# Utilities
MKDIR = mkdir -p
RMDIR = rmdir

# Build rule
all:
	$(MKDIR) $(TARGETDIR)
	$(NVCC) $(SRCS) $(NVCCFLAGS) -o $(TARGETDIR)$(TARGET)

# Clean rule
clean:
	$(RM) $(TARGETDIR)*
	$(RMDIR) $(TARGETDIR)

# Phony targets
.PHONY: all clean
