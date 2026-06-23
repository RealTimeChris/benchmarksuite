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

// Sampled mostly from https://github.com/fastfloat/fast_float
#pragma once

#include <bnch_swt-incl/config.hpp>

#if BNCH_SWT_PLATFORM_LINUX

	#include <linux/perf_event.h>
	#include <asm/unistd.h>
	#include <sys/ioctl.h>
	#include <unistd.h>
	#include <cstring>
	#include <vector>

namespace bnch_swt::internal {

	inline static uint64_t rdtsc() {
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return static_cast<unsigned long>(a) | (static_cast<unsigned long>(d) << 32);
	}

	template<benchmark_types benchmark_types, typename function_type> struct iteration_metric_collector {
		template<typename metric_type, typename... arg_types> BNCH_SWT_NOINLINE static void impl(metric_type& iteration_data, arg_types&&... args) {
			const auto start_clock				= clock_type::now();
			const volatile uint64_t cycle_start = rdtsc();
			iteration_data.bytes_processed		= static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...));
			const volatile uint64_t cycle_end	= rdtsc();
			const auto end_clock				= clock_type::now();
			iteration_data.time_in_ns			= (end_clock - start_clock).count();
			iteration_data.cycles.emplace(cycle_end - cycle_start);
		}
	};

}

#endif
