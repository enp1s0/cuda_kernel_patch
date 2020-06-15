#include <nvrtc.h>
#include <cuda.h>
#include <cuda_kernel_fusing.hpp>

constexpr std::size_t N = 1lu << 29;
constexpr std::size_t block_size = 1lu << 8;

int main() {
	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr, const unsigned inner_loop",
			"const unsigned tid, float* const dp",
			"const unsigned tid = threadIdx.x; float* dp = dst_ptr + inner_loop * tid;",
			""
			);

	kernel_constructor.debug_print_arguments();
	kernel_constructor.add_device_function(
			"device_func_init",
			R"(
{
	const float* const sp = src_ptr + inner_loop * tid;
	for (unsigned i = 0; i < inner_loop; i++) {
		dp[i] = sp[i];
	}
}
)"
			);
	kernel_constructor.add_device_function(
			"device_func_0",
			R"(
{
	for (unsigned i = 0; i < inner_loop; i++) {
		dp[i] *= 4;
	}
}
)"
			);
	kernel_constructor.add_device_function(
			"device_func_1",
			R"(
{
	for (unsigned i = 0; i < inner_loop; i++) {
		dp[i] *= 2;
	}
}
)"
			);

	std::printf("# -- kernel code\n");
	const std::string kernel_code = kernel_constructor.generate_kernel_code({
				"device_func_init",
				"device_func_0",
				"device_func_1",
				"device_func_0",
				"device_func_1",
				});
	std::cout << kernel_code << std::endl;

	const char* include_paths[] = {
		"/usr/local/cuda-11.0/include/"
	};
	constexpr unsigned num_headers = 1;
	const char* headers_list[] = {
		"cooperative_groups.h"
	};

	nvrtcProgram program;
	nvrtcCreateProgram(&program,
			kernel_code.c_str(),
			"kernel.cu",
			num_headers,
			include_paths,
			headers_list
			);
	const char *options[] = {
		"--gpu-architecture=compute_75",
	};
	nvrtcResult result = nvrtcCompileProgram(program, 1, options);
	size_t log_size;
	nvrtcGetProgramLogSize(program,&log_size);
	char *log = new char[log_size];
	nvrtcGetProgramLog(program,log);
	std::cout<<log<<std::endl;
	delete [] log;
	if(result != NVRTC_SUCCESS){
		std::cerr<<"Compilation failed"<<std::endl;
		return 1;
	}

	// Get PTX
	std::size_t ptx_size;
	nvrtcGetPTXSize(program, &ptx_size);
	char *ptx = new char [ptx_size];
	nvrtcGetPTX(program, ptx);
	nvrtcDestroyProgram(&program);

	// Create kernel image
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction;
	cuInit(0);
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&cuContext, 0, cuDevice);
	cuModuleLoadDataEx(&cuModule, ptx, 0, 0, 0);
	cuModuleGetFunction(&cuFunction, cuModule, "cukf_main");
	delete [] ptx;

	// Launch
	float *dx, *dy;
	cudaMalloc(&dx, sizeof(float) * N);
	cudaMalloc(&dy, sizeof(float) * N);

	unsigned inner_loop = 4;

	void *args[] = {&dy,&dx,&inner_loop};
	cuLaunchKernel(cuFunction,
			N / block_size / inner_loop,1,1,
			block_size,1,1,
			0, NULL,
			args, 0);
	cuCtxSynchronize();

	cudaFree(dx);
	cudaFree(dy);

	cuModuleUnload(cuModule);
	cuCtxDestroy(cuContext);
}
