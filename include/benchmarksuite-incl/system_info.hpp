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

#include <benchmarksuite-incl/event_counter.hpp>

namespace benchmarksuite {

	namespace internal {

		template<benchmark_types> BNCH_SWT_HOST std::string get_device_info();

		template<> BNCH_SWT_HOST std::string get_device_info<benchmark_types::cpu>() {
			std::string result{};

#if BNCH_SWT_ARCH_X64
			{
				std::array<char, 49> brand{};
				bool have_brand{};

	#if BNCH_SWT_COMPILER_MSVC
				std::array<int32_t, 4> regs{};
				__cpuid(regs.data(), static_cast<int32_t>(0x80000000));
				have_brand = static_cast<uint32_t>(regs[0]) >= 0x80000004u;
				if (have_brand) {
					for (uint32_t x = 0; x < 3; ++x) {
						__cpuid(regs.data(), static_cast<int32_t>(0x80000002u + x));
						std::memcpy(brand.data() + (x * 16ull), regs.data(), 16ull);
					}
				}
	#else
				uint32_t eax{}, ebx{}, ecx{}, edx{};
				have_brand = __get_cpuid_max(0x80000000u, nullptr) >= 0x80000004u;
				if (have_brand) {
					for (uint32_t x = 0; x < 3; ++x) {
						__cpuid(0x80000002u + x, eax, ebx, ecx, edx);
						std::memcpy(brand.data() + (x * 16ull), &eax, 4ull);
						std::memcpy(brand.data() + (x * 16ull) + 4ull, &ebx, 4ull);
						std::memcpy(brand.data() + (x * 16ull) + 8ull, &ecx, 4ull);
						std::memcpy(brand.data() + (x * 16ull) + 12ull, &edx, 4ull);
					}
				}
	#endif

				if (have_brand) {
					brand[48] = '\0';
					result	  = std::string{ brand.data() };
				}
			}
#elif BNCH_SWT_ARCH_ARM64
	#if BNCH_SWT_PLATFORM_MAC
			{
				size_t length{};
				if (sysctlbyname("machdep.cpu.brand_string", nullptr, &length, nullptr, 0) == 0 && length > 0) {
					result.resize(length);
					if (sysctlbyname("machdep.cpu.brand_string", result.data(), &length, nullptr, 0) != 0) {
						result.clear();
					} else {
						result.resize(std::strlen(result.c_str()));
					}
				}
			}
	#elif BNCH_SWT_PLATFORM_LINUX || BNCH_SWT_PLATFORM_ANDROID
			{
				std::ifstream stream{ "/proc/cpuinfo" };
				std::string line{};
				while (std::getline(stream, line)) {
					const size_t colon = line.find(':');
					if (colon == std::string::npos) {
						continue;
					}
					const std::string key = line.substr(0, colon);
					if (key.find("model name") == 0 || key.find("Model") == 0 || key.find("Hardware") == 0 || key.find("CPU part") == 0) {
						result			   = line.substr(colon + 1ull);
						const size_t first = result.find_first_not_of(" \t");
						const size_t last  = result.find_last_not_of(" \t\r\n");
						result			   = (first == std::string::npos) ? std::string{} : result.substr(first, (last - first) + 1ull);
						if (!result.empty() && key.find("model name") == 0) {
							break;
						}
					}
				}
			}
	#elif BNCH_SWT_PLATFORM_WINDOWS
			{
				HKEY key{};
				if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) == ERROR_SUCCESS) {
					std::array<char, 256> buffer{};
					DWORD size = static_cast<DWORD>(buffer.size());
					DWORD type{};
					if (RegQueryValueExA(key, "ProcessorNameString", nullptr, &type, reinterpret_cast<LPBYTE>(buffer.data()), &size) == ERROR_SUCCESS && type == REG_SZ) {
						result = std::string{ buffer.data() };
					}
					RegCloseKey(key);
				}
			}
	#endif
#endif
			if (result.empty()) {
				result = "Unknown CPU";
			}
			const size_t first = result.find_first_not_of(" \t");
			const size_t last  = result.find_last_not_of(" \t\r\n");
			result			   = (first == std::string::npos) ? std::string{ "Unknown CPU" } : result.substr(first, (last - first) + 1ull);
			return result;
		}

		[[nodiscard]] consteval auto get_os_name() {
#if BNCH_SWT_PLATFORM_WINDOWS
			return string_literal{ "Windows" };
#elif BNCH_SWT_PLATFORM_MAC
			return string_literal{ "macOS" };
#elif BNCH_SWT_PLATFORM_LINUX
			return string_literal{ "Linux" };
#elif BNCH_SWT_PLATFORM_ANDROID
			return string_literal{ "Android" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		[[nodiscard]] consteval auto get_compiler_name() {
#if BNCH_SWT_COMPILER_CUDA
			return string_literal{ "CUDA" };
#elif BNCH_SWT_COMPILER_CLANG
			return string_literal{ "Clang" };
#elif BNCH_SWT_COMPILER_GCC
			return string_literal{ "GCC" };
#elif BNCH_SWT_COMPILER_MSVC
			return string_literal{ "MSVC" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		static constexpr benchmarksuite::string_literal operating_system_name{ get_os_name() };
		static constexpr benchmarksuite::string_literal operating_system_version{ BNCH_SWT_OPERATING_SYSTEM_VERSION };
		static constexpr benchmarksuite::string_literal compiler_id{ get_compiler_name() };
		static constexpr benchmarksuite::string_literal compiler_version{ BNCH_SWT_COMPILER_VERSION };

	}

	template<benchmark_types benchmark_type> struct system_info_data {
		inline static constexpr std::string_view compiler_version{ internal::compiler_version };
		inline static constexpr std::string_view compiler_id{ internal::compiler_id };
		inline static constexpr std::string_view os_version{ internal::operating_system_version };
		inline static constexpr std::string_view os_id{ internal::operating_system_name };
		inline static constexpr std::string_view device_type{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };
		inline static constexpr std::string_view instruction_set_name{ benchmark_type == benchmark_types::cpu ? BNCH_SWT_INSTRUCTION_SET_NAME : "CUDA" };
		BNCH_SWT_HOST static std::string_view device_name() noexcept {
			using leaked_string = std::string;
			static const leaked_string& name{ *new leaked_string{ internal::get_device_info<benchmark_type>() + "-" + static_cast<std::string>(instruction_set_name) } };
			return name;
		}
	};

}
