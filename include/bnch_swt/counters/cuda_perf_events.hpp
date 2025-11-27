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
/// Dec 6, 2024
// Sampled mostly from https://github.com/fastfloat/fast_float
#pragma once

#include <bnch_swt/config.hpp>

#if BNCH_SWT_COMPILER_CUDA

	#include <cuda_runtime.h>
	#include <cuda.h>

namespace bnch_swt::internal {

	template<typename function_type, typename... args_types> BNCH_SWT_GLOBAL static void profiling_wrapper(args_types... args) {
		function_type::impl(args...);
	}
	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cuda, count> : public std::vector<event_count> {
		std::vector<cudaEvent_t> start_events{};
		std::vector<cudaEvent_t> stop_events{};
		uint64_t current_index{};
		bool working{};
		uint64_t* d_metrics{};
		BNCH_SWT_HOST event_collector_type() : std::vector<event_count>(count), working(true), current_index(0) {
			start_events.resize(count);
			stop_events.resize(count);
			for (uint64_t i = 0; i < count; ++i) {
				if (cudaEventCreate(&start_events[i]) != cudaSuccess || cudaEventCreate(&stop_events[i]) != cudaSuccess) {
					working = false;
					return;
				}
			}
			if (cudaMalloc(&d_metrics, sizeof(uint64_t) * 16) != cudaSuccess) {
				working = false;
			}
		}
		BNCH_SWT_HOST ~event_collector_type() {
			for (auto& evt: start_events) {
				cudaEventDestroy(evt);
			}
			for (auto& evt: stop_events) {
				cudaEventDestroy(evt);
			}
			if (d_metrics) {
				cudaFree(d_metrics);
			}
		}
		BNCH_SWT_HOST bool has_events() const {
			return working;
		}
		template<typename function_type, typename... args_types> BNCH_SWT_HOST void run(dim3 grid, dim3 block, uint64_t shared_mem, uint64_t bytes_processed, args_types... args) {
			if (!working || current_index >= count) {
				return;
			}
			cudaEventRecord(start_events[current_index]);
			profiling_wrapper<function_type><<<grid, block, shared_mem>>>(args...);
			cudaEventRecord(stop_events[current_index]);
			cudaDeviceSynchronize();
			float ms = 0;
			cudaEventElapsedTime(&ms, start_events[current_index], stop_events[current_index]);
			std::vector<event_count>::operator[](current_index).elapsed			  = std::chrono::duration<double, std::milli>(ms);
			std::vector<event_count>::operator[](current_index).cuda_event_ms_val = ms;
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(bytes_processed);
			int clock_rate_khz;
			cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
			uint64_t cycles = static_cast<uint64_t>(ms * 1e-3 * clock_rate_khz * 1000.0);
			std::vector<event_count>::operator[](current_index).cycles_val.emplace(cycles);
			++current_index;
		}
		BNCH_SWT_HOST void set_bytes_processed(uint64_t bytes) {
			if (current_index > 0) {
				std::vector<event_count>::operator[](current_index - 1).bytes_processed_val.emplace(bytes);
			}
		}
	};

}

#endif
