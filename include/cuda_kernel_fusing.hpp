#ifndef __CUDA_KERNEL_FUSING_HPP__
#define __CUDA_KERNEL_FUSING_HPP__
#include <algorithm>
#include <iostream>
#include <string>
#include <map>
#include <utility>
#include <vector>

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
	if (end_pos == std::string::npos) {
		end_pos = argument_string.size();
	}

	while (start_pos < argument_string.size()) {
		std::string sub_str(argument_string, start_pos, end_pos - start_pos);

		argument_names.push_back(get_argument_name(sub_str));
		start_pos = end_pos + 1;
		end_pos = argument_string.find_first_of(',', start_pos);

		if (end_pos == std::string::npos) {
			end_pos = argument_string.size();
		}
	}

	return argument_names;
}

std::vector<std::string> append_string_vector(std::vector<std::string> first, const std::vector<std::string> second) {
	first.insert(first.end(), second.begin(), second.end());
	return first;
}

std::string serialize_string_vector(const std::vector<std::string> string_vector) {
	if (string_vector.size() == 0) {
		return "";
	}

	std::string serialized_string = "";
	std::size_t i = 0;
	for (const auto str : string_vector) {
		if (i++ != 0) {
			serialized_string += ", ";
		}
		serialized_string += str;
	}

	return serialized_string;
}
} // namespace utils

class kernel_constructor {
	const std::string global_argument_string;
	const std::string device_argument_string;

	const std::string device_calling_argument_string;

	const std::string preprocess_string;
	const std::string postprocess_string;

	std::map<std::string, std::string> device_functions;

public:
	kernel_constructor(const std::string global_argument_string, const std::string device_argument_string, const std::string preprocess_string, const std::string postprocess_string)
		: global_argument_string(global_argument_string),
		device_argument_string(global_argument_string + "," + device_argument_string),
		preprocess_string(preprocess_string),
		postprocess_string(postprocess_string),
		  device_calling_argument_string(utils::serialize_string_vector(utils::append_string_vector(utils::get_argument_names(global_argument_string), utils::get_argument_names(device_argument_string))))
	{}

	void add_device_function(const std::string function_name, const std::string function_definition) {
		device_functions.insert(std::make_pair(function_name,
					"__device__ void " + function_name + "(" + device_argument_string + ")" + function_definition
					));
	}

	std::string generate_kernel_code(const std::vector<std::string> device_function_names) const {
		std::string kernel_code = "";

		// Add the definition of device functions
		std::vector<std::string> unique_device_function_names(device_function_names.size());
		std::copy(device_function_names.begin(), device_function_names.end(), unique_device_function_names.begin());
		std::sort(unique_device_function_names.begin(), unique_device_function_names.end());
		unique_device_function_names.erase(std::unique(unique_device_function_names.begin(), unique_device_function_names.end()), unique_device_function_names.end());
		for (const std::string device_function_name : unique_device_function_names) {
			kernel_code += device_functions.at(device_function_name) + "\n";
		}
		
		// Add a global function
		kernel_code += "extern \"C\" __global__ void cukf_main(" + global_argument_string + ") {\n";
		kernel_code += preprocess_string + "\n";
		for (const auto device_function_name : device_function_names) {
			kernel_code += device_function_name + "(" + device_calling_argument_string + ");\n";
		}
		kernel_code += postprocess_string + "\n";
		kernel_code += "}\n";

		return kernel_code;
	}


	// Debug functions
	void debug_print_arguments() const {
		std::printf("# %15s : %s\n", "device args", device_argument_string.c_str());
		std::printf("# %15s :\n", "Preprocess");
		std::printf("```cpp\n%s\n```\n", preprocess_string.c_str());
		std::printf("# %15s :\n", "Postprocess");
		std::printf("```cpp\n%s\n```\n", postprocess_string.c_str());
	}
};
} // namespace cuda_kernel_fusing

#endif /* end of include guard */
