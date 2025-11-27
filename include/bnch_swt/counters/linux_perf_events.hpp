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

#if BNCH_SWT_PLATFORM_LINUX

	#include <linux/perf_event.h>
	#include <asm/unistd.h>
	#include <sys/ioctl.h>
	#include <unistd.h>
	#include <cstring>
	#include <vector>

namespace bnch_swt::internal {

	BNCH_SWT_HOST uint64_t rdtsc() {
	#if defined(__x86_64__)
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return static_cast<unsigned long>(a) | (static_cast<unsigned long>(d) << 32);
	#elif defined(__i386__)
		uint64_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		return 0;
	#endif
	}

	class linux_events {
	  protected:
		std::vector<uint64_t> temp_result_vec{};
		std::vector<uint64_t> ids{};
		perf_event_attr attribs{};
		uint64_t num_events{};
		bool working{};
		int32_t fd{};

	  public:
		BNCH_SWT_HOST explicit linux_events(std::vector<int32_t> config_vec) : working(true) {
			memset(&attribs, 0, sizeof(attribs));
			attribs.type		   = PERF_TYPE_HARDWARE;
			attribs.size		   = sizeof(attribs);
			attribs.disabled	   = 1;
			attribs.exclude_kernel = 1;
			attribs.exclude_hv	   = 1;

			attribs.sample_period	  = 0;
			attribs.read_format		  = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
			const int32_t pid		  = 0;
			const int32_t cpu		  = -1;
			const unsigned long flags = 0;

			int32_t group = -1;
			num_events	  = config_vec.size();
			ids.resize(config_vec.size());
			uint32_t i = 0;
			for (auto config: config_vec) {
				attribs.config = static_cast<long long unsigned int>(config);
				int32_t _fd	   = static_cast<int32_t>(syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags));
				if (_fd == -1) {
					report_error("perf_event_open");
				}
				ioctl(_fd, PERF_EVENT_IOC_ID, &ids[i++]);
				if (group == -1) {
					group = _fd;
					fd	  = _fd;
				}
			}

			temp_result_vec.resize(num_events * 2 + 1);
		}

		BNCH_SWT_HOST ~linux_events() {
			if (fd != -1) {
				close(fd);
			}
		}

		BNCH_SWT_HOST void run() {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_RESET)");
				}

				if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_ENABLE)");
				}
			}
		}

		BNCH_SWT_HOST void end(std::vector<uint64_t>& results) {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
					report_error("ioctl(PERF_EVENT_IOC_DISABLE)");
				}

				if (read(fd, temp_result_vec.data(), temp_result_vec.size() * 8) == -1) {
					report_error("read");
				}
			}

			for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
				results[i / 2] = temp_result_vec[i];
			}
			for (uint32_t i = 2; i < temp_result_vec.size(); i += 2) {
				if (ids[i / 2 - 1] != temp_result_vec[i]) {
					report_error("event mismatch");
				}
			}
		}

		bool is_working() {
			return working;
		}

	  protected:
		BNCH_SWT_HOST void report_error(const std::string&) {
			working = false;
		}
	};

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cpu, count> : public linux_events, public std::vector<event_count> {
		std::vector<uint64_t> results{};
		uint64_t current_index{};
		BNCH_SWT_HOST event_collector_type()
			: linux_events{ std::vector<int32_t>{ PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
				  PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES } },
			  std::vector<event_count>{ count } {
		}

		BNCH_SWT_HOST bool has_events() {
			return linux_events::is_working();
		}

		template<typename function_type, typename... arg_types> BNCH_SWT_HOST void run(arg_types&&... args) {
			if (has_events()) {
				linux_events::run();
			}
			uint64_t result;
			const auto start_clock		  = clock_type::now();
			volatile uint64_t cycle_start = rdtsc();
			result						  = static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...));
			volatile uint64_t cycle_end	  = rdtsc();
			const auto end_clock		  = clock_type::now();
			std::vector<event_count>::operator[](current_index).cycles_val.emplace(cycle_end - cycle_start);
			std::vector<event_count>::operator[](current_index).elapsed = end_clock - start_clock;
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(result);
			if (has_events()) {
				if (results.size() != linux_events::temp_result_vec.size()) {
					results.resize(linux_events::temp_result_vec.size());
				}
				linux_events::end(results);
				std::vector<event_count>::operator[](current_index).instructions_val.emplace(results[1]);
				std::vector<event_count>::operator[](current_index).branches_val.emplace(results[2]);
				std::vector<event_count>::operator[](current_index).branch_misses_val.emplace(results[3]);
				std::vector<event_count>::operator[](current_index).cache_references_val.emplace(results[4]);
				std::vector<event_count>::operator[](current_index).cache_misses_val.emplace(results[5]);
			}
			++current_index;
			return;
		}
	};
}

#endif
