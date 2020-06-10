#include <cuda_kernel_fusing.hpp>

int main() {
	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr",
			"tid",
			"const std::size_t tid = threadIdx.x",
			""
			);

	kernel_constructor.debug_print_arguments();
}
