#ifndef __CUDA_KERNEL_FUSING_HPP__
#define __CUDA_KERNEL_FUSING_HPP__
#include <string>
#include <vector>
#include <utility>

namespace cuda_kernel_fusing {
namespace utils {
std::string get_argument_name(const std::string argument) {
	const std::size_t end_pos = argument.find_last_not_of(' ');
	const std::string argument_0(argument, 0, (end_pos != std::string::npos ? end_pos : argument.size()) + 1);
	const std::size_t start_pos = argument_0.find_last_of(' ');
	const std::string argument_1(argument_0, (start_pos != std::string::npos ? (start_pos + 1) : 0));

	return argument_1;
}

std::vector<std::string> get_argument_names(const std::string argument_string) {
	std::vector<std::string> argument_names;

	std::size_t start_pos = 0;
	std::size_t end_pos = argument_string.find_first_of(',');

	while (start_pos < argument_string.size()) {
		std::string sub_str(argument_string, start_pos, end_pos - start_pos);

		argument_names.push_back(get_argument_name(sub_str));
		start_pos = end_pos + 1;
		end_pos = argument_string.find_first_of(',', start_pos);

		if (end_pos == std::string::npos) {
			break;
		}
	}

	return argument_names;
}
} // namespace utils
} // namespace cuda_kernel_fusing

#endif /* end of include guard */
