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

#include <bnch_swt-incl/concepts.hpp>
#include <type_traits>
#include <concepts>
#include <cstdint>
#include <variant>

namespace bnch_swt {

	[[maybe_unused]] inline static std::string get_time() {
		std::string new_time_string{};
		new_time_string.resize(1024);
#if BNCH_SWT_PLATFORM_WINDOWS
		std::time_t result = std::time(nullptr);
		std::tm result_two{};
		localtime_s(&result_two, &result);
#else
		std::time_t result = std::time(nullptr);
		std::tm result_two{ *localtime(&result) };
#endif
		new_time_string.resize(strftime(new_time_string.data(), 1024, "%b %d, %Y", &result_two));
		return new_time_string;
	}

	[[maybe_unused]] inline static std::string url_encode(std::string value) {
		std::ostringstream escaped;
		escaped.fill('0');
		escaped << std::hex;

		for (char c: value) {
			if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
				escaped << c;
			} else if (c == ':') {
				escaped << '%' << std::setw(2) << static_cast<int32_t>(static_cast<unsigned char>(' '));
			} else {
				escaped << '%' << std::setw(2) << static_cast<int32_t>(static_cast<unsigned char>(c));
			}
		}

		return escaped.str();
	}

	[[maybe_unused]] inline static std::string get_current_path_impl() {
		return static_cast<std::string>(bnch_swt::system_info_data<bnch_swt::benchmark_types::cpu>::os_id) + "-" +
			static_cast<std::string>(bnch_swt::system_info_data<bnch_swt::benchmark_types::cpu>::compiler_id);
	}

	[[maybe_unused]] inline static int32_t execute_python_script(const std::string& script_path, const std::string& argument_01, const std::string& argument_02) {
#if BNCH_SWT_PLATFORM_WINDOWS
		static constexpr std::string_view python_name{ "python" };
#else
		static constexpr std::string_view python_name{ "python3" };
#endif
		auto quote = [](const std::string& s) {
			std::string out{ "\"" };
			for (char c: s) {
				if (c == '"' || c == '\\') {
					out += '\\';
				}
				out += c;
			}
			out += '"';
			return out;
		};
		std::string command{ python_name };
		command += ' ';
		command += quote(script_path);
		command += ' ';
		command += quote(argument_01);
		command += ' ';
		command += quote(argument_02);
		int32_t raw_result = system(command.c_str());
#if BNCH_SWT_PLATFORM_WINDOWS
		int32_t exit_code = raw_result;
#else
		int32_t exit_code = (raw_result == -1) ? -1 : (WIFEXITED(raw_result) ? WEXITSTATUS(raw_result) : 128 + WTERMSIG(raw_result));
#endif
		if (exit_code != 0) {
			std::cout << "Error: Failed to execute Python script. Command exited with code " << exit_code << std::endl;
		}
		return exit_code;
	}
}
