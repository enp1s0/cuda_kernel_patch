NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../include -lcuda -lnvrtc
TARGET=cukf.test

$(TARGET):main.cu ../include/cuda_kernel_fusing.hpp
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
