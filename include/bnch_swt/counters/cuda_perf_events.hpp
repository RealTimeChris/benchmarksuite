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

#if BNCH_SWT_COMPILER_CUDA

	#include <cuda_runtime.h>
	#include <cuda.h>

namespace bnch_swt::internal {

	struct cuda_timer {
		BNCH_SWT_HOST cuda_timer() noexcept {
			if (cudaEventCreate(&start_val) != cudaSuccess) {
				return;
			}
			if (cudaEventCreate(&stop_val) != cudaSuccess) {
				return;
			}
		}

		BNCH_SWT_HOST void start() noexcept {
			cudaEventRecord(start_val, 0);
		}

		BNCH_SWT_HOST void stop() noexcept {
			cudaEventRecord(stop_val, 0);
			cudaEventSynchronize(stop_val);
		}

		BNCH_SWT_HOST double get_time() noexcept {
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start_val, stop_val);
			return static_cast<double>(milliseconds);
		}

		BNCH_SWT_HOST ~cuda_timer() noexcept {
			cudaEventDestroy(start_val);
			cudaEventDestroy(stop_val);
		}

	  protected:
		cudaEvent_t start_val{}, stop_val{};
	};

	template<typename function_type, typename... args_types> BNCH_SWT_GLOBAL static void profiling_wrapper(args_types... args) {
		function_type::impl(args...);
	}

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cuda, count> : public std::vector<event_count> {
		std::vector<cuda_timer> events{};
		uint64_t current_index{};

		BNCH_SWT_HOST event_collector_type() : std::vector<event_count>(count), current_index(0) {
			events.resize(count);
		}

		BNCH_SWT_HOST ~event_collector_type() {
		}

		template<typename function_type, typename... args_types> BNCH_SWT_HOST void run(dim3 grid, dim3 block, uint64_t shared_mem, uint64_t bytes_processed, args_types... args) {
			if (current_index >= count) {
				return;
			}
			events[current_index].start();
			profiling_wrapper<function_type><<<grid, block, shared_mem>>>(args...);
			events[current_index].stop();
			double ms{ events[current_index].get_time() };
			std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(duration_type(ms));
			std::vector<event_count>::operator[](current_index).cuda_event_ms_val.emplace(ms);
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(bytes_processed);
			int clock_rate_khz;
			cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
			uint64_t cycles = static_cast<uint64_t>(ms * 1e-3 * clock_rate_khz * 1000.0);
			std::vector<event_count>::operator[](current_index).cycles_val.emplace(cycles);
			++current_index;
		}

		template<auto function, typename... args_types> BNCH_SWT_HOST void run(dim3 grid, dim3 block, uint64_t shared_mem, uint64_t bytes_processed, args_types... args) {
			if (current_index >= count) {
				return;
			}
			events[current_index].start();
			function<<<grid, block, shared_mem>>>(args...);
			events[current_index].stop();
			double ms{ events[current_index].get_time() };
			std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(duration_type(ms));
			std::vector<event_count>::operator[](current_index).cuda_event_ms_val.emplace(ms);
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
