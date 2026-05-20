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

	class file_handle {
	  public:
		explicit file_handle(const std::string& path) : path(std::filesystem::absolute(path)) {
			stream.open(this->path, std::ios::in);
			if (stream.is_open()) {
				contents = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
				stream.close();
			}
		}

		~file_handle() {
			if (!dirty)
				return;
			stream.open(path, std::ios::out | std::ios::trunc);
			if (stream.is_open()) {
				stream << contents;
				stream.flush();
				bool ok = stream.good();
				stream.close();
				std::cout << (ok ? "Saved: " : "Write error: ") << path.string() << std::endl;
			} else {
				std::cout << "Failed to open for writing: " << path.string() << std::endl;
			}
		}

		file_handle(const file_handle&)			   = delete;
		file_handle& operator=(const file_handle&) = delete;

		std::string& get() {
			dirty = true;
			return contents;
		}

		const std::string& get() const {
			return contents;
		}

	  private:
		std::filesystem::path path{};
		std::string contents{};
		std::fstream stream{};
		bool dirty{ false };
	};

}
