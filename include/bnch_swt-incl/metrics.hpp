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

#include <bnch_swt-incl/event_counter.hpp>
#include <unordered_map>
#include <string_view>
#include <cstdint>
#include <span>

namespace bnch_swt {

	namespace internal {

		template<benchmark_types> BNCH_SWT_HOST std::string get_device_info();

		template<> BNCH_SWT_HOST std::string get_device_info<benchmark_types::cpu>() {
#if defined(__x86_64__) || defined(_M_AMD64)
			std::array<uint32_t, 12> regs{};
			auto cpuid = [](uint32_t leaf, uint32_t* res) {
	#if BNCH_SWT_COMPILER_MSVC
				__cpuidex(std::bit_cast<int*>(res), static_cast<int>(leaf), 0);
	#elif BNCH_SWT_COMPILER_GCC || BNCH_SWT_COMPILER_CLANG
				__asm__ volatile("cpuid" : "=a"(res[0]), "=b"(res[1]), "=c"(res[2]), "=d"(res[3]) : "a"(leaf), "c"(0));
	#endif
			};

			uint32_t ext_info[4];
			cpuid(0x80000000, ext_info);
			if (ext_info[0] < 0x80000004) {
				return "Unknown x86_64 CPU";
			}

			for (uint32_t i = 0; i < 3; ++i) {
				cpuid(0x80000002 + i, regs.data() + (i * 4));
			}

			char brand[49]{};
			std::memcpy(brand, regs.data(), 48);

			std::string result(brand);
			auto last_char = result.find_last_not_of(" \t\n\r\f\v");
			if (last_char != std::string::npos) {
				result.resize(last_char + 1);
			}

			return result;

#elif BNCH_SWT_PLATFORM_MAC || defined(__FreeBSD__)
			char buffer[256]{};
			size_t size = sizeof(buffer);
			if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
				return std::string(buffer);
			}
			return "Unknown CPU";
#elif BNCH_SWT_PLATFORM_LINUX
			std::ifstream cpuinfo("/proc/cpuinfo");
			std::string line;
			while (std::getline(cpuinfo, line)) {
				if (line.find("model name") == 0 || line.find("Hardware") == 0 || line.find("Model") == 0) {
					auto colon = line.find(':');
					if (colon != std::string::npos) {
						auto start = line.find_first_not_of(" \t", colon + 1);
						if (start != std::string::npos)
							return line.substr(start);
					}
				}
			}
			return "Unknown Linux CPU";
#elif BNCH_SWT_PLATFORM_WINDOWS
			HKEY hkey;
			if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
				char buffer[256]{};
				DWORD size = sizeof(buffer);
				if (RegQueryValueExA(hkey, "ProcessorNameString", nullptr, nullptr, std::bit_cast<LPBYTE>(buffer), &size) == ERROR_SUCCESS) {
					RegCloseKey(hkey);
					return std::string(buffer);
				}
				RegCloseKey(hkey);
			}
			return "Unknown Windows CPU";
#else

			return "Unsupported Architecture";
#endif
		}

		[[nodiscard]] consteval auto get_os_name() {
#if BNCH_SWT_PLATFORM_WINDOWS
			return string_literal{ "Windows" };
#elif BNCH_SWT_PLATFORM_MAC
			return string_literal{ "macOS" };
#elif BNCH_SWT_PLATFORM_LINUX
			return string_literal{ "Linux" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		[[nodiscard]] consteval auto get_compiler_name() {
#if BNCH_SWT_COMPILER_CLANG
			return string_literal{ "Clang" };
#elif BNCH_SWT_COMPILER_GCC
			return string_literal{ "GCC" };
#elif BNCH_SWT_COMPILER_MSVC
			return string_literal{ "MSVC" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		static constexpr bnch_swt::string_literal operating_system_name{ get_os_name() };
		static constexpr bnch_swt::string_literal operating_system_version{ BNCH_SWT_OPERATING_SYSTEM_VERSION };
		static constexpr bnch_swt::string_literal compiler_id{ get_compiler_name() };
		static constexpr bnch_swt::string_literal compiler_version{ BNCH_SWT_COMPILER_VERSION };

	}

	enum class position_type : uint8_t { win, tie, loss, none };

	template<benchmark_types benchmark_type> struct system_info_data {
		inline static constexpr std::string_view compiler_version{ internal::compiler_version };
		inline static constexpr std::string_view compiler_id{ internal::compiler_id };
		inline static constexpr std::string_view os_version{ internal::operating_system_version };
		inline static constexpr std::string_view os_id{ internal::operating_system_name };
		inline static constexpr std::string_view device_type{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };
		inline static constexpr std::string_view instruction_set_name{ benchmark_type == benchmark_types::cpu ? BNCH_SWT_INSTRUCTION_SET_NAME : "CUDA" };
		static std::string_view device_name() noexcept {
			static const std::string* name{ new std::string{ internal::get_device_info<benchmark_type>() + "-" + static_cast<std::string>(instruction_set_name) } };
			return *name;
		}
	};	

}
