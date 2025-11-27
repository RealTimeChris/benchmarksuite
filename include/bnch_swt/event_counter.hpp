// bnch_swt/event_counter.hpp
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
#pragma once

#include <bnch_swt/config.hpp>
#include <bnch_swt/counters/apple_arm_perf_events.hpp>
#include <bnch_swt/counters/windows_perf_events.hpp>
#include <bnch_swt/counters/linux_perf_events.hpp>
#include <bnch_swt/counters/andriod_events.hpp>
#include <bnch_swt/counters/cuda_perf_events.hpp>
#include <optional>
#include <chrono>

namespace bnch_swt::internal {

	template<benchmark_types benchmark_type> struct event_count;
	template<> struct event_count<benchmark_types::cpu> {
		template<typename value_type, benchmark_types, uint64_t count> friend struct event_collector_type;
		BNCH_SWT_HOST event_count() noexcept {
		}
		BNCH_SWT_HOST double elapsed_ns() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}
		BNCH_SWT_HOST bool bytes_processed(uint64_t& bytes_processed_new) const noexcept {
			if (bytes_processed_val.has_value()) {
				bytes_processed_new = bytes_processed_val.value();
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool cycles(double& cycles_new) const {
			if (cycles_val.has_value()) {
				cycles_new = static_cast<double>(cycles_val.value());
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool instructions(double& instructions_new) const noexcept {
			if (instructions_val.has_value()) {
				instructions_new = static_cast<double>(instructions_val.value());
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool branches(double& branches_new) const noexcept {
			if (branches_val.has_value()) {
				branches_new = static_cast<double>(branches_val.value());
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool branch_misses(double& branch_misses_new) const noexcept {
			if (branch_misses_val.has_value()) {
				branch_misses_new = static_cast<double>(branch_misses_val.value());
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool cache_misses(double& cache_misses_new) const noexcept {
			if (cache_misses_val.has_value()) {
				cache_misses_new = static_cast<double>(cache_misses_val.value());
				return true;
			} else {
				return false;
			}
		}
		BNCH_SWT_HOST bool cache_references(double& cache_references_new) const noexcept {
			if (cache_references_val.has_value()) {
				cache_references_new = static_cast<double>(cache_references_val.value());
				return true;
			} else {
				return false;
			}
		}

	  protected:
		std::optional<uint64_t> cache_references_val{};
		std::optional<uint64_t> bytes_processed_val{};
		std::optional<uint64_t> branch_misses_val{};
		std::optional<uint64_t> instructions_val{};
		std::optional<uint64_t> cache_misses_val{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> branches_val{};
		std::optional<uint64_t> cycles_val{};
	};
	template<> struct event_count<benchmark_types::cuda> {
		BNCH_SWT_HOST event_count() noexcept {
		}
		BNCH_SWT_HOST double elapsed_ns() const noexcept {
			return std::chrono::duration<double, std::nano>(elapsed).count();
		}
		BNCH_SWT_HOST double cuda_event_ms() const noexcept {
			return cuda_event_ms_val;
		}
		BNCH_SWT_HOST bool bytes_processed(uint64_t& bytes_processed_new) const noexcept {
			if (bytes_processed_val.has_value()) {
				bytes_processed_new = bytes_processed_val.value();
				return true;
			}
			return false;
		}
		BNCH_SWT_HOST bool cycles(double& cycles_new) const {
			if (cycles_val.has_value()) {
				cycles_new = static_cast<double>(cycles_val.value());
				return true;
			}
			return false;
		}

	  protected:
		template<typename event_count, benchmark_types, uint64_t count> friend struct event_collector_type;
		std::optional<uint64_t> bytes_processed_val{};
		std::chrono::duration<double> elapsed{};
		std::optional<uint64_t> cycles_val{};
		double cuda_event_ms_val{};
	};
	template<uint64_t count, benchmark_types benchmark_type> using event_collector = event_collector_type<event_count<benchmark_type>, benchmark_type, count>;

}
