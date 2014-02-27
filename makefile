# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-5.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_SDK_PATH   ?=$(CUDA_PATH)/samples/common/inc
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# CUDA code generation flags
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM30)

# OS-specific build flags
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64 

NVCCFLAGS := -m64 

TARGET := release

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. -I$(CUDA_SDK_PATH)

# Target rules
all: radixSort

radixSort.o: radixSortBack.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

radixSort: radixSort.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS) 

clean:
	rm -f radixSortThrust radixSortThrust.o 
