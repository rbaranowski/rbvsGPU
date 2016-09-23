include config.mk

CUDA_OBJS := pearson_cor_gpu.o factor_model_gpu.o utility_functions_gpu.o
C_OBJS := factor_model.o  pearson_cor.o utility_functions.o

 
#compiler/preprocessor options
INCS := -I. -I"$(CUDA_INC)" -I"$(R_INC)"
PARAMS := $(DEVICEOPTS) -Xcompiler '$(CPICFLAGS)'

#linker options
LD_PARAMS := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LIBS :=  -L"$(R_LIB)" -L"$(CUDA_LIB)" -lcurand -lcublas




TARGETS := rbvsGPU.so

#NVCC := $(CUDA_HOME)/bin/nvcc 
NVCC := $(CUDA_HOME)/bin/nvcc -D_FORCE_INLINES -gencode arch=compute_52,code=sm_52 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 
#NVCC := $(CUDA_HOME)/bin/nopvcc -arch=compute_20 -code=sm_20

all: $(TARGETS) 

$(TARGETS): $(CUDA_OBJS) $(C_OBJS)
	$(NVCC) -shared  $(LIBS) $(CUDA_OBJS) $(C_OBJS) -o $@

$(CUDA_OBJS): %.o: %.cu
	$(NVCC) -c $(INCS) $(LIBS) $(PARAMS) $^ -o $@

$(C_OBJS): %.o: %.c
	gcc  -c $(INCS) $(PKG_CFLAGS) $(CPICFLAGS) $^ -o $@

clean:
	rm -rf *o *so

.PHONY: all clean
