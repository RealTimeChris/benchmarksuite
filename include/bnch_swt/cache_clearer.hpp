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
/// Sep 1, 2024
#pragma once

#include <bnch_swt/benchmarksuite_cpu_properties.hpp>
#include <bnch_swt/config.hpp>
#include <iostream>
#include <fstream>

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <Windows.h>
	#include <intrin.h>
#elif BNCH_SWT_PLATFORM_LINUX
	#include <unistd.h>
	#include <vector>
	#include <string>
	#if defined(__i386__) || defined(__x86_64__)
		#include <immintrin.h>
	#endif
#elif BNCH_SWT_PLATFORM_MAC
	#include <libkern/OSCacheControl.h>
	#include <sys/sysctl.h>
	#include <unistd.h>
	#include <vector>
#endif

namespace bnch_swt::internal {

	enum class cache_level {
		one	  = 1,
		two	  = 2,
		three = 3,
	};

	BNCH_SWT_HOST size_t get_cache_line_size() {
#if BNCH_SWT_PLATFORM_WINDOWS
		DWORD buffer_size = 0;
		GetLogicalProcessorInformation(nullptr, &buffer_size);
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
		if (!GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		for (const auto& info: buffer) {
			if (info.Relationship == RelationCache && info.Cache.Level == 1) {
				return info.Cache.LineSize;
			}
		}
#elif BNCH_SWT_PLATFORM_LINUX
		long line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
		if (line_size <= 0) {
			std::cerr << "Failed to retrieve cache line size using sysconf!" << std::endl;
			return 0;
		}
		return static_cast<size_t>(line_size);
#elif BNCH_SWT_PLATFORM_MAC
		size_t line_size = 0;
		size_t size		 = sizeof(line_size);
		if (sysctlbyname("hw.cachelinesize", &line_size, &size, nullptr, 0) != 0) {
			std::cerr << "Failed to retrieve cache line size using sysctl!" << std::endl;
			return 0;
		}
		return line_size;
#else
		std::cerr << "Unsupported platform!" << std::endl;
		return 0;
#endif
		return 0;
	}

	BNCH_SWT_HOST static void flush_cache(void* ptr, size_t size, [[maybe_unused]] size_t cache_line_size, bool clear_instruction_cache = false) {
#if BNCH_SWT_PLATFORM_MAC
		if (clear_instruction_cache) {
			sys_icache_invalidate(ptr, size);
		} else {
			sys_dcache_flush(ptr, size);
		}
#else
		char* buffer = static_cast<char*>(ptr);
	#if BNCH_SWT_PLATFORM_WINDOWS
		for (size_t i = 0; i < size; i += cache_line_size) {
			_mm_clflush(buffer + i);
		}
		_mm_sfence();

		if (clear_instruction_cache) {
			if (!FlushInstructionCache(GetCurrentProcess(), buffer, size)) {
				std::cerr << "Failed to flush instruction cache!" << std::endl;
			}
		}
	#elif BNCH_SWT_PLATFORM_LINUX
		#if BNCH_SWT_ARCH_X64
		for (size_t i = 0; i < size; i += cache_line_size) {
			__builtin_ia32_clflush(buffer + i);
		}
		#elif BNCH_SWT_ARCH_ARM || BNCH_SWT_ARCH_ARM64
		for (size_t i = 0; i < size; i += cache_line_size) {
			#if BNCH_SWT_ARCH_ARM64
			__asm__ __volatile__("dc civac, %0" : : "r"(buffer + i) : "memory");
			#else
			__builtin___clear_cache(buffer + i, buffer + i + cache_line_size);
			#endif
		}
		__asm__ __volatile__("dsb sy" : : : "memory");
		#endif

		if (clear_instruction_cache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
	#elif BNCH_SWT_PLATFORM_ANDROID
		if (clear_instruction_cache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
	#endif
#endif
	}

	template<benchmark_types benchmark_type> class cache_clearer {
		size_t cache_line_size{ get_cache_line_size() };
		std::array<size_t, 3> cache_sizes{ { cpu_properties::l1_cache_size, cpu_properties::l2_cache_size, cpu_properties::l3_cache_size } };

		size_t max_cache_size{ std::max({ cache_sizes[0], cache_sizes[1], cache_sizes[2] }) };

		std::vector<char> evict_buffer{ [&] {
			std::vector<char> return_values{};
			if (max_cache_size > 0) {
				return_values.resize(max_cache_size * 4 + cache_line_size);
			}
			return return_values;
		}() };

		BNCH_SWT_HOST void evict_cache(size_t cache_level) {
			if (cache_level >= 1 && cache_level <= 3 && cache_sizes[cache_level - 1] > 0 && !evict_buffer.empty()) {
				size_t target_size = cache_sizes[cache_level - 1] * 4;

				const size_t stride = 4093;
				volatile char sink	= 0;

				for (size_t offset = 0; offset < target_size; offset += cache_line_size) {
					size_t idx		  = (offset * stride) % evict_buffer.size();
					evict_buffer[idx] = static_cast<char>(idx);
					sink			  = sink + evict_buffer[idx];
				}

				flush_cache(evict_buffer.data(), evict_buffer.size(), cache_line_size);
				if (cache_level == 1) {
					flush_cache(evict_buffer.data(), evict_buffer.size(), cache_line_size, true);
				}
			}
		}

	  public:
		BNCH_SWT_HOST void evict_caches() {
			evict_cache(3);
			evict_cache(2);
			evict_cache(1);
		}
	};
}
