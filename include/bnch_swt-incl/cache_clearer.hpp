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

#pragma once

#include <bnch_swt-incl/benchmarksuite_cpu_properties.hpp>
#include <bnch_swt-incl/config.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

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

	template<benchmark_types benchmark_type> struct cache_clearer {
	  protected:
		BNCH_SWT_HOST static size_t get_cache_line_size_impl() {
#if BNCH_SWT_PLATFORM_WINDOWS
			DWORD buffer_size = 0;
			GetLogicalProcessorInformation(nullptr, &buffer_size);

			std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer_raw(static_cast<uint64_t>(buffer_size) / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

			if (!GetLogicalProcessorInformation(buffer_raw.data(), &buffer_size)) {
				return 64;
			}

			size_t num_elements = buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
			const auto* buffer	= buffer_raw.data();

			for (size_t i = 0; i < num_elements; ++i) {
				const auto& info = buffer[i];
				if (info.Relationship == RelationCache && info.Cache.Level == 1) {
					return info.Cache.LineSize;
				}
			}
			return 64;

#elif BNCH_SWT_PLATFORM_LINUX
			long line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
			if (line_size <= 0) {
				std::cerr << "Failed to retrieve cache line size using sysconf! Falling back to 64." << std::endl;
				return 64;
			}
			return static_cast<size_t>(line_size);
#elif BNCH_SWT_PLATFORM_MAC
			size_t line_size = 0;
			size_t size		 = sizeof(line_size);
			if (sysctlbyname("hw.cachelinesize", &line_size, &size, nullptr, 0) != 0) {
				std::cerr << "Failed to retrieve cache line size using sysctl! Falling back to 64." << std::endl;
				return 64;
			}
			return line_size;
#else
			std::cerr << "Unsupported platform! Falling back to 64." << std::endl;
			return 64;
#endif
		}

		BNCH_SWT_HOST static void memory_fence() {
#if BNCH_SWT_PLATFORM_WINDOWS || (BNCH_SWT_PLATFORM_LINUX && BNCH_SWT_ARCH_X64)
			_mm_mfence();
#elif BNCH_SWT_ARCH_ARM64
			__asm__ __volatile__("dmb sy" : : : "memory");
#else
			std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
		}

		BNCH_SWT_HOST static void flush_cache_line(void* ptr, bool clear_instruction = false) {
			if (!ptr)
				return;

#if BNCH_SWT_PLATFORM_MAC
			if (clear_instruction) {
				sys_icache_invalidate(ptr, get_cache_line_size_impl());
			} else {
				sys_dcache_flush(ptr, get_cache_line_size_impl());
			}
#elif BNCH_SWT_PLATFORM_WINDOWS
			_mm_clflush(ptr);
			if (clear_instruction) {
				FlushInstructionCache(GetCurrentProcess(), ptr, get_cache_line_size_impl());
			}
#elif BNCH_SWT_PLATFORM_LINUX
	#if BNCH_SWT_ARCH_X64
			__builtin_ia32_clflush(ptr);
	#elif BNCH_SWT_ARCH_ARM64
			__asm__ __volatile__("dc civac, %0" : : "r"(ptr) : "memory");
	#endif

			if (clear_instruction) {
				if (ptr != nullptr) {
					__builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + get_cache_line_size_impl());
				}
			}
#endif
		}
		inline static size_t get_cache_line_size() {
			static const size_t cache_line_size = get_cache_line_size_impl();
			return cache_line_size;
		} 
		static constexpr std::array<size_t, 3> cache_sizes{ { cpu_properties::l1_cache_size, cpu_properties::l2_cache_size, cpu_properties::l3_cache_size } };
		static constexpr size_t biggest_cache_size{ [] {
			size_t return_value{};
			for (size_t size: cache_sizes) {
				if (size > return_value) {
					return_value = size;
				}
			}
			return return_value > 0 ? return_value : 8 * 1024 * 1024;
		}() };

		static constexpr size_t eviction_multiplier = 2;
		static constexpr size_t working_set_size	= biggest_cache_size * eviction_multiplier;

		struct aligned_vector {
			std::vector<char> data;
			char* aligned_ptr;
			size_t size;

			aligned_vector(size_t requested_size) {
				data.resize(requested_size + 64ULL);
				uintptr_t addr	  = std::bit_cast<uintptr_t>(data.data());
				uintptr_t aligned = (addr + 63ULL) & ~63ULL;
				aligned_ptr		  = std::bit_cast<char*>(aligned);
				size			  = requested_size;
			}

			char* get() {
				return aligned_ptr;
			}
			size_t get_size() const {
				return size;
			}
		};

		aligned_vector evict_buffer{ working_set_size > 0 ? working_set_size : 32 * 1024 * 1024 };

		uint64_t rng_state{ static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) };

		inline size_t fast_rng(size_t max_val) {
			rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
			return static_cast<size_t>(rng_state % max_val);
		}

		inline static std::vector<size_t> get_accessed_indices() {
			std::vector<size_t> return_value;
			return_value.resize((biggest_cache_size * eviction_multiplier) / get_cache_line_size());
			return return_value;
		}

		std::vector<size_t> accessed_indices{ get_accessed_indices() };

		BNCH_SWT_HOST void random_access_evict(size_t cache_level) {
			if (cache_level < 1 || cache_level > 3)
				return;
			if (cache_sizes[cache_level - 1] == 0)
				return;

			char* buffer_start = evict_buffer.get();
			size_t buffer_size = evict_buffer.get_size();

			if (buffer_size == 0)
				return;

			const size_t target_cache_lines = (cache_sizes[cache_level - 1] * eviction_multiplier) / get_cache_line_size();
			const size_t buffer_cache_lines = buffer_size / get_cache_line_size();

			if (buffer_cache_lines == 0)
				return;

			for (size_t i = 0; i < target_cache_lines; ++i) {
				size_t line_idx		= fast_rng(buffer_cache_lines);
				accessed_indices[i] = line_idx;

				char* cache_line = buffer_start + (line_idx * get_cache_line_size());

				for (size_t offset = 0; offset < get_cache_line_size(); offset += 8) {
					if (offset + 8 <= get_cache_line_size()) {
						uint64_t* ptr = std::bit_cast<uint64_t*>(cache_line + offset);
						*ptr		  = static_cast<uint64_t>(line_idx) ^ static_cast<uint64_t>(offset);
					}
				}
			}

			memory_fence();

			for (size_t i = target_cache_lines - 1; i > 0; --i) {
				size_t j = fast_rng(i + 1);
				std::swap(accessed_indices[i], accessed_indices[j]);
			}

			volatile uint64_t sink = 0;
			for (size_t i = 0; i < target_cache_lines; ++i) {
				size_t line_idx	 = accessed_indices[i];
				char* cache_line = buffer_start + (line_idx * get_cache_line_size());

				for (size_t offset = 0; offset < get_cache_line_size(); offset += 8) {
					if (offset + 8 <= get_cache_line_size()) {
						uint64_t* ptr = std::bit_cast<uint64_t*>(cache_line + offset);
						sink += *ptr;
					}
				}

				flush_cache_line(cache_line, false);
			}

			bnch_swt::do_not_optimize_away(sink);
			memory_fence();

			if (cache_level == 1) {
				for (size_t i = target_cache_lines - 1; i > 0; --i) {
					size_t j = fast_rng(i + 1);
					std::swap(accessed_indices[i], accessed_indices[j]);
				}

				for (size_t i = 0; i < target_cache_lines && i < accessed_indices.size(); ++i) {
					char* cache_line = buffer_start + (accessed_indices[i] * get_cache_line_size());
					flush_cache_line(cache_line, true);
				}
				memory_fence();
			}
		}

		BNCH_SWT_HOST void aggressive_evict(cache_level level) {
			constexpr int num_passes = 3;
			for (int pass = 0; pass < num_passes; ++pass) {
				random_access_evict(static_cast<size_t>(level));

#if BNCH_SWT_ARCH_X64
				_mm_pause();
#endif
			}
		}

	  public:
		cache_clearer() {
			char* buffer_start = evict_buffer.get();
			size_t buffer_size = evict_buffer.get_size();

			if (buffer_start && buffer_size > 0) {
				for (size_t i = 0; i < buffer_size; i += get_cache_line_size()) {
					buffer_start[i] = static_cast<char>(i);
					flush_cache_line(&buffer_start[i], false);
				}
				memory_fence();
			}
		}

		BNCH_SWT_HOST void evict_caches() {
			aggressive_evict(cache_level::three);
			aggressive_evict(cache_level::two);
			aggressive_evict(cache_level::one);
			memory_fence();
		}

		BNCH_SWT_HOST void nuclear_evict() {
			evict_caches();

			char* buffer_start			= evict_buffer.get();
			size_t buffer_size			= evict_buffer.get_size();
			const size_t total_elements = buffer_size / sizeof(uint64_t);

			if (total_elements > 0) {
				volatile uint64_t sink = 0;

				for (size_t i = 0; i < total_elements * 2; ++i) {
					size_t idx	  = fast_rng(total_elements);
					uint64_t* ptr = std::bit_cast<uint64_t*>(buffer_start + (idx * sizeof(uint64_t)));
					sink += *ptr;
					flush_cache_line(ptr, false);

					size_t stride = fast_rng(4096);
					idx			  = (idx + stride) % total_elements;
					ptr			  = std::bit_cast<uint64_t*>(buffer_start + (idx * sizeof(uint64_t)));
					sink += *ptr;
					flush_cache_line(ptr, false);
				}

				bnch_swt::do_not_optimize_away(sink);
				memory_fence();
			}
		}
	};

	template<> struct cache_clearer<benchmark_types::cuda> {
		BNCH_SWT_HOST void evict_caches() {
		}

		BNCH_SWT_HOST void nuclear_evict() {
		}
	};

}
