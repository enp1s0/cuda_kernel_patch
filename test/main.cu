#include <cuda_kernel_fusing.hpp>

int main() {
	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr",
			"const unsigned tid, const float& a",
			"const unsigned tid = threadIdx.x; float a = 1.0f;",
			"dst_ptr[tid] = a;"
			);

	kernel_constructor.debug_print_arguments();
	kernel_constructor.add_device_function(
			"device_func_0",
			R"(
{
	a *= src_ptr[tid];
}
)"
			);
	kernel_constructor.add_device_function(
			"device_func_1",
			R"(
{
	a /= src_ptr[tid];
}
)"
			);

	std::printf("# -- kernel code\n");
	std::printf("%s\n", kernel_constructor.generate_kernel_code({
				"device_func_0",
				"device_func_1",
				"device_func_0",
				"device_func_1",
				}).c_str());
}
