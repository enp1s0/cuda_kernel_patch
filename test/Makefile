NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../include -lcuda -lnvrtc -arch=sm_80
TARGET=cukf.test

$(TARGET):main.cu ../include/cuda_kernel_patch.hpp
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
