NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../include
TARGET=cukf.test

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
