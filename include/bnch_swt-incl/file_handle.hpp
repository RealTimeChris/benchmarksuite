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

#include <bnch_swt-incl/config.hpp>
#include <filesystem>
#include <sstream>
#include <fstream>

namespace bnch_swt {

	class file_handle {
	  public:
		static void save_file(const std::string& data, const std::string& path) {
			std::filesystem::path abs_path = std::filesystem::absolute(path);
			std::filesystem::create_directories(abs_path.parent_path());
			std::fstream stream{ abs_path, std::ios::out | std::ios::trunc };
			if (stream.is_open()) {
				stream << data;
				stream.flush();
				bool ok = stream.good();
				stream.close();
				std::cout << (ok ? "Saved: " : "Write error: ") << abs_path.string() << std::endl;
			} else {
				std::cout << "Failed to open for writing: " << abs_path.string() << std::endl;
			}
		}

		static std::string get(const std::string& path) {
			std::fstream stream{ std::filesystem::absolute(path), std::ios::in };
			if (stream.is_open()) {
				return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
			} else {
				throw std::runtime_error{ "Sorry, but we failed to load the file at: " + path };
			}
			return {};
		}
	};

}
