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
#include <source_location>

#if BNCH_SWT_COMPILER_CUDA

	#include <cuda_runtime.h>
	#include <cuda.h>

namespace bnch_swt {

	namespace internal {

		static constexpr const char* get_function_name(std::source_location location = std::source_location::current()) {
			return location.function_name();
		}

		static constexpr uint64_t get_line(std::source_location location = std::source_location::current()) {
			return location.line();
		}

		template<benchmark_types> BNCH_SWT_HOST std::string get_device_info();

		BNCH_SWT_HOST bool check_cuda_status(const char* function_name = get_function_name(), uint64_t line = get_line()) {
			if (auto result = cudaGetLastError(); result) {
				std::cout << "In Function: " << function_name << ", On Line: " << line << ", Cuda Error : " << cudaGetErrorString(result) << std::endl;
				return false;
			} else {
				return true;
			}
		}

		template<> BNCH_SWT_HOST std::string get_device_info<benchmark_types::cuda>() {
			int device_count  = 0;
			cudaError_t error = cudaGetDeviceCount(&device_count);

			if (error != cudaSuccess || device_count == 0) {
				return "Unknown NVIDIA GPU or No Driver";
			}

			cudaDeviceProp prop;
			error = cudaGetDeviceProperties(&prop, 0);

			if (error != cudaSuccess) {
				return "Error retrieving GPU properties";
			}

			return std::string(prop.name);
		}

		struct cuda_timer {
			cuda_timer(const cuda_timer&)			 = delete;
			cuda_timer& operator=(const cuda_timer&) = delete;

			BNCH_SWT_HOST cuda_timer(cuda_timer&& other) noexcept : start_val(other.start_val), stop_val(other.stop_val) {
				other.start_val = {};
				other.stop_val	= {};
			}

			BNCH_SWT_HOST cuda_timer& operator=(cuda_timer&& other) noexcept {
				if (this != &other) {
					cudaEventDestroy(start_val);
					check_cuda_status();
					cudaEventDestroy(stop_val);
					check_cuda_status();
					start_val		= other.start_val;
					stop_val		= other.stop_val;
					other.start_val = {};
					other.stop_val	= {};
				}
				return *this;
			}

			BNCH_SWT_HOST cuda_timer() noexcept {
				cudaEventCreate(&start_val);
				check_cuda_status();
				cudaEventCreate(&stop_val);
				check_cuda_status();
			}

			BNCH_SWT_HOST void start() noexcept {
				cudaEventRecord(start_val, 0);
			}

			BNCH_SWT_HOST void stop() noexcept {
				cudaEventRecord(stop_val, 0);
				check_cuda_status();
				cudaEventSynchronize(stop_val);
				check_cuda_status();
			}

			BNCH_SWT_HOST double get_time() noexcept {
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start_val, stop_val);
				return static_cast<double>(milliseconds);
			}

			BNCH_SWT_HOST ~cuda_timer() noexcept {
				cudaEventDestroy(start_val);
				check_cuda_status();
				cudaEventDestroy(stop_val);
				check_cuda_status();
			}

		  protected:
			cudaEvent_t start_val{}, stop_val{};
		};

		template<typename function_type, typename... args_types> BNCH_SWT_GLOBAL static void profiling_wrapper(args_types&&... args) {
			function_type::impl(args...);
		}

		template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cuda, count> : public std::vector<event_count> {
			std::vector<cuda_timer> events{};
			uint64_t current_index{};

			BNCH_SWT_HOST void reset() {
				current_index = 0;
			}

			BNCH_SWT_HOST event_collector_type() : std::vector<event_count>(count), current_index(0) {
				events.resize(count);
			}

			BNCH_SWT_HOST ~event_collector_type() {
			}

			template<typename function_type, typename... args_types> BNCH_SWT_HOST void run_from_host(uint64_t bytes_processed, args_types&&... args) {
				if (current_index >= count) {
					return;
				}
				events[current_index].start();
				cudaDeviceSynchronize();
				check_cuda_status();
				function_type::impl(args...);
				check_cuda_status();
				events[current_index].stop();
				double ms{ events[current_index].get_time() };
				std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(ms);
				std::vector<event_count>::operator[](current_index).cuda_event_ms_val.emplace(ms);
				std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(bytes_processed);
				uint64_t nanoseconds = static_cast<uint64_t>(ms * 1e6);
				std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(nanoseconds);
				int clock_rate_khz;
				cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
				check_cuda_status();
				uint64_t cycles = static_cast<uint64_t>(ms * 1e-3 * clock_rate_khz * 1000.0);
				std::vector<event_count>::operator[](current_index).cycles_val.emplace(cycles);
				++current_index;
			}

			template<function_pointer_types auto function, typename... args_types>
			BNCH_SWT_HOST void run_cooperative(dim3 grid, dim3 block, uint64_t shared_mem, cudaStream_t stream, uint64_t bytes_processed, args_types&&... args_new) {
				if (current_index >= count) {
					return;
				}
				if constexpr (sizeof...(args_new) > 0) {
					void* args[] = { ( void* )std::addressof(args_new)... };
					cudaDeviceSynchronize();
					events[current_index].start();
					cudaLaunchCooperativeKernel(function, grid, block, args, shared_mem, stream);
					check_cuda_status();
				} else {
					events[current_index].start();
					cudaLaunchCooperativeKernel(function, grid, block, nullptr, shared_mem, stream);
					check_cuda_status();
				}
				events[current_index].stop();
				double ms{ events[current_index].get_time() };
				std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(ms);
				std::vector<event_count>::operator[](current_index).cuda_event_ms_val.emplace(ms);
				std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(bytes_processed);
				uint64_t nanoseconds = static_cast<uint64_t>(ms * 1e6);
				std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(nanoseconds);
				int clock_rate_khz;
				cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
				check_cuda_status();
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

}

#endif
