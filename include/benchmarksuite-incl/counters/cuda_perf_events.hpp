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

#include <benchmarksuite-incl/config.hpp>

#if BNCH_SWT_COMPILER_CUDA

namespace benchmarksuite {

	namespace internal {

		BNCH_SWT_HOST static constexpr const char* get_function_name(std::source_location location = std::source_location::current()) {
			return location.function_name();
		}

		BNCH_SWT_HOST static constexpr uint64_t get_line(std::source_location location = std::source_location::current()) {
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
				check_cuda_status();
			}

			BNCH_SWT_HOST void stop() noexcept {
				cudaDeviceSynchronize();
				check_cuda_status();
				cudaEventRecord(stop_val, 0);
				check_cuda_status();
				cudaEventSynchronize(stop_val);
				check_cuda_status();
			}

			BNCH_SWT_HOST double get_time() noexcept {
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start_val, stop_val);
				check_cuda_status();
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

		template<typename function_type> struct iteration_metric_collector<benchmark_types::cuda, function_type> {
			template<typename metric_type, typename... arg_types> BNCH_SWT_NOINLINE static void impl(metric_type& iteration_data, arg_types&&... args) {
				internal::cuda_timer timer{};
				timer.start();
				iteration_data.bytes_processed = static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...));
				internal::check_cuda_status();
				timer.stop();
				double ms				  = timer.get_time();
				iteration_data.time_in_ns = static_cast<double>(ms * 1e6);

				int clock_rate_khz = 0;
				cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
				internal::check_cuda_status();
				uint64_t cycles = static_cast<uint64_t>(ms * 1e-3 * clock_rate_khz * 1000.0);
				iteration_data.cycles.emplace(cycles);
			}
		};
	}

}

#endif
