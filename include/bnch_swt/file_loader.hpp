/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/benchmarksuite

#pragma once

#include <bnch_swt/config.hpp>
#include <filesystem>
#include <sstream>
#include <fstream>

namespace bnch_swt {

	class file_loader {
	  public:
		constexpr file_loader() {
		}

		static std::string load_file(const std::string& file_path) {
			std::string directory{ file_path.substr(0, file_path.find_last_of("/") + 1) };
			if (!std::filesystem::exists(directory)) {
				std::filesystem::create_directories(directory);
			}
			if (!std::filesystem::exists(static_cast<std::string>(file_path))) {
				std::ofstream create_file{ file_path.data() };
				create_file.close();
			}
			std::ifstream the_stream{ file_path.data(), std::ios::binary | std::ios::in };
			std::stringstream input_stream{};
			input_stream << the_stream.rdbuf();
			the_stream.close();
			return input_stream.str();
		}

		static void save_file(const std::string& file_to_save, const std::string& file_path, bool retry = true) {
			std::ofstream the_stream{ file_path.data(), std::ios::binary | std::ios::out | std::ios::trunc };
			the_stream.write(file_to_save.data(), static_cast<int64_t>(file_to_save.size()));
			if (the_stream.is_open()) {
				std::cout << "File succesfully written to: " << file_path << std::endl;
			} else {
				std::string directory{ file_path.substr(0, file_path.find_last_of("/") + 1) };
				if (!std::filesystem::exists(directory) && retry) {
					std::filesystem::create_directories(directory);
					return save_file(file_to_save, file_path, false);
				}
				std::cerr << "File failed to be written to: " << file_path << std::endl;
			}
			the_stream.close();
		}
	};

}
