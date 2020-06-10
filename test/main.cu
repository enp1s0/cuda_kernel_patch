#include <cuda_kernel_fusing.hpp>

int main() {
	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr",
			"const unsigned tid",
			"const unsigned tid = threadIdx.x;",
			""
			);

	kernel_constructor.debug_print_arguments();
	kernel_constructor.add_device_function(
			"device_func_0",
			R"(
{
	dst_ptr[tid] = src_ptr[tid];
}
)"
			);

	std::printf("# -- kernel code\n");
	std::printf("%s\n", kernel_constructor.generate_kernel_code({"device_func_0"}).c_str());
}
