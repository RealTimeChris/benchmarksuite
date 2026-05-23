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

#include <optional>
#include <cstdint>
#include <chrono>

#if BNCH_SWT_COMPILER_CUDA
	#define BNCH_SWT_ALIGN(x) __align__(x)
	#include <cuda_fp16.h>
	#include <cuda_bf16.h>
#else
	#define BNCH_SWT_ALIGN(x) alignas(x)
#endif

namespace bnch_swt {

	using nanoseconds  = std::chrono::duration<double, std::nano>;
	using milliseconds = std::chrono::duration<double, std::milli>;

	struct steady_clock {
		using clock_type			  = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
		using seconds				  = std::chrono::duration<double>;
		using time_point_type_nano	  = std::chrono::time_point<clock_type, nanoseconds>;
		using time_point_type_seconds = std::chrono::time_point<clock_type, seconds>;
		using rep					  = double;
		using period				  = std::nano;
		using duration				  = std::chrono::duration<rep, period>;
		using time_point			  = std::chrono::time_point<steady_clock>;

		static constexpr bool is_steady = true;

		static time_point now() noexcept {
			auto system_now			  = std::chrono::steady_clock::now();
			auto duration_since_epoch = system_now.time_since_epoch();

			auto casted_duration = std::chrono::duration_cast<duration>(duration_since_epoch);

			return time_point(casted_duration);
		}
	};

	using clock_type = steady_clock;

	enum class benchmark_types {
		cpu,
		cuda,
	};

	namespace internal {

		template<typename event_count, benchmark_types, uint64_t count> struct event_collector_type;

	}

	template<typename value_type> using base_t = std::remove_cvref_t<value_type>;

	struct iteration_metrics {
		std::optional<uint64_t> cycles;
		uint64_t bytes_processed;
		double time_in_ns;
	};

}
