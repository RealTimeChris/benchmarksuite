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

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <Windows.h>
#endif

#if BNCH_SWT_PLATFORM_MAC
	#include <sys/sysctl.h>
	#include <sys/types.h>
	#include <string>
#endif

#if BNCH_SWT_ARCH_ARM64
	#if defined(__linux__)
		#include <sys/auxv.h>
		#include <asm/hwcap.h>
	#endif
#else
	#if BNCH_SWT_COMPILER_MSVC
		#include <intrin.h>
	#elif BNCH_SWT_COMPILER_GCC || BNCH_SWT_COMPILER_CLANG
		#include <cpuid.h>
	#endif
#endif

namespace bnch_swt {

	enum class instruction_set {
		FALLBACK = 0x0,
		AVX2	 = 0x1,
		AVX512f	 = 0x2,
		NEON	 = 0x4,
		SVE2	 = 0x8,
	};

	enum class cache_level {
		one	  = 1,
		two	  = 2,
		three = 3,
	};

#if defined(__aarch64__) || defined(_M_ARM64)
	inline static uint32_t detect_supported_architectures() {
		uint32_t host_isa = static_cast<uint32_t>(instruction_set::NEON);

	#if defined(__linux__) && defined(HWCAP_SVE)
		unsigned long hwcap = getauxval(AT_HWCAP);
		if (hwcap & HWCAP_SVE) {
			host_isa |= static_cast<uint32_t>(instruction_set::SVE2);
		}
	#endif

		return host_isa;
	}

#elif defined(__x86_64__) || defined(_M_X64)
	static constexpr uint32_t cpuid_avx2_bit	 = 1ul << 5;
	static constexpr uint32_t cpuid_avx512_bit	 = 1ul << 16;
	static constexpr uint64_t cpuid_avx256_saved = 1ULL << 2;
	static constexpr uint64_t cpuid_avx512_saved = 7ULL << 5;
	static constexpr uint32_t cpuid_osx_save	 = (1ul << 26) | (1ul << 27);

	inline static void cpuid(int32_t* eax, int32_t* ebx, int32_t* ecx, int32_t* edx) {
	#if BNCH_SWT_COMPILER_MSVC
		int32_t cpu_info[4];
		__cpuidex(cpu_info, *eax, *ecx);
		*eax = cpu_info[0];
		*ebx = cpu_info[1];
		*ecx = cpu_info[2];
		*edx = cpu_info[3];
	#else
		uint32_t a = *eax, b, c = *ecx, d;
		asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "c"(c));
		*eax = a;
		*ebx = b;
		*ecx = c;
		*edx = d;
	#endif
	}

	inline static uint64_t xgetbv() {
	#if BNCH_SWT_COMPILER_MSVC
		return _xgetbv(0);
	#else
		uint32_t eax, edx;
		asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
		return (( uint64_t )edx << 32) | eax;
	#endif
	}

	inline static int32_t detect_supported_architectures() {
		std::int32_t eax	   = 0;
		std::int32_t ebx	   = 0;
		std::int32_t ecx	   = 0;
		std::int32_t edx	   = 0;
		std::int32_t host_isa  = static_cast<uint32_t>(instruction_set::FALLBACK);

		eax = 0x1;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);

		if ((ecx & cpuid_osx_save) != cpuid_osx_save) {
			return host_isa;
		}

		uint64_t xcr0 = xgetbv();
		if ((xcr0 & cpuid_avx256_saved) == 0) {
			return host_isa;
		}

		eax = 0x7;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);

		if (ebx & cpuid_avx2_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
		}

		if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
			return host_isa;
		}

		if (ebx & cpuid_avx512_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512f);
		}

		return host_isa;
	}

#else
	inline static uint32_t detect_supported_architectures() {
		return static_cast<uint32_t>(instruction_set::FALLBACK);
	}
#endif

	inline uint64_t get_cache_size(cache_level level) {
#if defined(_WIN32) || defined(_WIN64)
		DWORD bufferSize = 0;
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};
		GetLogicalProcessorInformation(nullptr, &bufferSize);
		if (bufferSize == 0)
			return 0;
		buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

		if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
			return 0;
		}

		for (const auto& i: buffer) {
			if (i.Relationship == RelationCache && i.Cache.Level == static_cast<int32_t>(level)) {
				if (level == cache_level::one && i.Cache.Type == CacheData) {
					return i.Cache.Size;
				} else if (level != cache_level::one && i.Cache.Type == CacheUnified) {
					return i.Cache.Size;
				}
			}
		}
		return 0;

#elif defined(__linux__) || defined(__ANDROID__)
		auto get_cache_size_from_file = [](const std::string& index) -> uint64_t {
			const std::string cacheFilePath = "/sys/devices/system/cpu/cpu0/cache/index" + index + "/size";
			std::ifstream file(cacheFilePath);
			if (!file.is_open())
				return 0ULL;

			std::string sizeStr;
			file >> sizeStr;
			uint64_t size = std::stoul(sizeStr);
			if (sizeStr.find('K') != std::string::npos)
				size *= 1024;
			else if (sizeStr.find('M') != std::string::npos)
				size *= 1024 * 1024;
			return static_cast<uint64_t>(size);
		};

		if (level == cache_level::one)
			return get_cache_size_from_file("0");
		std::string idx = (level == cache_level::two) ? "2" : "3";
		return get_cache_size_from_file(idx);

#elif defined(__APPLE__)
		auto get_cache_size_for_mac = [](const char* cacheType) {
			uint64_t cacheSize = 0;
			size_t size		   = sizeof(cacheSize);
			std::string query  = std::string("hw.") + cacheType + "cachesize";
			if (sysctlbyname(query.c_str(), &cacheSize, &size, nullptr, 0) != 0)
				return 0ULL;
			return cacheSize;
		};

		if (level == cache_level::one)
			return get_cache_size_for_mac("l1d");
		if (level == cache_level::two)
			return get_cache_size_for_mac("l2");
		return get_cache_size_for_mac("l3");
#endif

		return 0;
	}

	enum class host_cxx_compilers {
		clang,
		gnu,
		msvc,
	};
	
	struct cpu_properties {
		uint64_t thread_count{};
		uint64_t l1_cache_size{};
		uint64_t l2_cache_size{};
		uint64_t l3_cache_size{};
		uint64_t cpu_arch_index{};
		uint64_t cpu_alignment{};
	};

	inline static cpu_properties get_device_properties() {
		const uint32_t thread_count	 = std::thread::hardware_concurrency();
		const uint64_t l1_cache_size = get_cache_size(cache_level::one);
		const uint64_t l2_cache_size = get_cache_size(cache_level::two);
		const uint64_t l3_cache_size = get_cache_size(cache_level::three);

		cpu_properties return_value;
		return_value.thread_count = thread_count;
		return_value.l1_cache_size = l1_cache_size;
		return_value.l2_cache_size = l2_cache_size;
		return_value.l3_cache_size = l3_cache_size;
		return return_value;
	}

	inline static cpu_properties& get_cpu_properties() {
		cpu_properties* cpu_props{ new cpu_properties{ get_device_properties() } };
		return *cpu_props;
	}
}
