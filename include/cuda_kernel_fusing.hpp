#ifndef __CUDA_KERNEL_FUSING_HPP__
#define __CUDA_KERNEL_FUSING_HPP__
#include <iostream>
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

class kernel_constructor {
	const std::string global_argument_string;
	const std::vector<std::string> global_argument_names;
	const std::vector<std::string> appending_device_argument_names;

	const std::string preprocess_string;
	const std::string postprocess_string;

public:
	kernel_constructor(const std::string global_argument_string, const std::string device_argument_string, const std::string preprocess_string, const std::string postprocess_string)
		: global_argument_string(global_argument_string),
		preprocess_string(preprocess_string),
		postprocess_string(postprocess_string),
		global_argument_names(utils::get_argument_names(global_argument_string)),
		appending_device_argument_names(utils::get_argument_names(global_argument_string))
	{}

	// Debug functions
	void debug_print_arguments() const {
		std::printf("# %15s : %s\n", "global args", global_argument_string.c_str());
		for (const auto str : global_argument_names) {
			std::printf("- %s\n", str.c_str());
		}
		std::printf("# %15s :\n", "device args (A)");
		for (const auto str : appending_device_argument_names) {
			std::printf("- %s\n", str.c_str());
		}
		std::printf("# %15s :\n", "Preprocess");
		std::printf("```cpp\n%s\n```\n", preprocess_string.c_str());
		std::printf("# %15s :\n", "Postprocess");
		std::printf("```cpp\n%s\n```\n", postprocess_string.c_str());
	}
};
} // namespace cuda_kernel_fusing

#endif /* end of include guard */
