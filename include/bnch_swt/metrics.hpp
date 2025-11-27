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

#include <bnch_swt/event_counter.hpp>
#include <cstdint>
#include <span>

namespace bnch_swt {

	BNCH_SWT_HOST double calculate_throughput_mbps(double nanoseconds, double bytes_processed) {
		constexpr double bytes_per_mb	  = 1024.0 * 1024.0;
		constexpr double nanos_per_second = 1e9;
		double megabytes				  = bytes_processed / bytes_per_mb;
		double seconds					  = nanoseconds / nanos_per_second;
		if (seconds == 0.0) {
			return 0.0;
		}
		return megabytes / seconds;
	}

	BNCH_SWT_HOST double calculate_units_ps(double nanoseconds, double bytes_processed) {
		return (bytes_processed * 1000000000.0) / nanoseconds;
	}

	template<benchmark_types benchmark_type> struct performance_metrics;

	template<> struct performance_metrics<benchmark_types::cpu> {
		double throughput_percentage_deviation{ std::numeric_limits<double>::max() };
		std::optional<double> cache_references_per_execution{};
		std::optional<double> instructions_per_execution{};
		std::optional<double> branch_misses_per_execution{};
		std::optional<double> cache_misses_per_execution{};
		std::optional<double> instructions_per_cycle{};
		std::optional<double> branches_per_execution{};
		std::optional<double> instructions_per_byte{};
		std::optional<double> cycles_per_execution{};
		std::optional<double> cycles_per_byte{};
		std::optional<double> frequency_ghz{};

		uint64_t measured_iteration_count{};
		uint64_t total_iteration_count{};
		double throughput_mb_per_sec{};
		uint64_t bytes_processed{};
		std::string name{};
		double time_in_ns{};

		BNCH_SWT_HOST bool operator>(const performance_metrics& other) const {
			return throughput_mb_per_sec > other.throughput_mb_per_sec;
		}

		template<string_literal benchmark_name_new, bool mbps = true>
		BNCH_SWT_HOST static performance_metrics collect_metrics(std::span<internal::event_count<benchmark_types::cpu>>&& events_newer, uint64_t total_iteration_count) {
			static constexpr string_literal benchmark_name{ benchmark_name_new };

			if (events_newer.empty()) {
				return {};
			}

			performance_metrics metrics{};
			metrics.name					 = benchmark_name.operator std::string();
			metrics.measured_iteration_count = events_newer.size();
			metrics.total_iteration_count	 = total_iteration_count;

			double throughput_total{};
			double throughput_min{ std::numeric_limits<double>::max() };
			uint64_t valid_throughput_count{ 0 };
			uint64_t bytes_processed_total{};
			double ns_total{};
			double cycles_total{};
			double instructions_total{};
			double branches_total{};
			double branch_misses_total{};
			double cache_references_total{};
			double cache_misses_total{};

			for (const auto& e: events_newer) {
				double ns = e.elapsed_ns();
				ns_total += ns;

				uint64_t bytes_processed{};
				if (e.bytes_processed(bytes_processed)) {
					bytes_processed_total += bytes_processed;

					double throughput{};
					if constexpr (mbps) {
						throughput = calculate_throughput_mbps(ns, static_cast<double>(bytes_processed));
					} else {
						throughput = calculate_units_ps(ns, static_cast<double>(bytes_processed));
					}

					if (throughput > 0.0) {
						throughput_total += throughput;
						throughput_min = std::min(throughput, throughput_min);
						++valid_throughput_count;
					}
				}

				double value{};
				if (e.cycles(value)) {
					cycles_total += value;
				}
				if (e.instructions(value)) {
					instructions_total += value;
				}
				if (e.branches(value)) {
					branches_total += value;
				}
				if (e.branch_misses(value)) {
					branch_misses_total += value;
				}
				if (e.cache_references(value)) {
					cache_references_total += value;
				}
				if (e.cache_misses(value)) {
					cache_misses_total += value;
				}
			}

			const double inv_size			   = 1.0 / static_cast<double>(events_newer.size());
			const uint64_t bytes_processed_avg = bytes_processed_total / events_newer.size();
			const double ns_avg				   = ns_total * inv_size;
			const double cycles_avg			   = cycles_total * inv_size;
			const double instructions_avg	   = instructions_total * inv_size;
			const double branches_avg		   = branches_total * inv_size;
			const double branch_misses_avg	   = branch_misses_total * inv_size;
			const double cache_references_avg  = cache_references_total * inv_size;
			const double cache_misses_avg	   = cache_misses_total * inv_size;

			metrics.time_in_ns = ns_avg;

			constexpr double epsilon = 1e-6;

			const double throughput_avg = valid_throughput_count > 0 ? throughput_total / static_cast<double>(valid_throughput_count) : 0.0;
			if (valid_throughput_count > 0 && throughput_avg > epsilon) {
				metrics.bytes_processed					= bytes_processed_avg;
				metrics.throughput_mb_per_sec			= throughput_avg;
				metrics.throughput_percentage_deviation = ((throughput_avg - throughput_min) * 100.0) / throughput_avg;
			}

			if (std::abs(ns_avg) > epsilon) {
				metrics.bytes_processed					= bytes_processed_avg;
				metrics.throughput_mb_per_sec			= throughput_avg;
				metrics.throughput_percentage_deviation = ((throughput_avg - throughput_min) * 100.0) / throughput_avg;
			}

			if (std::abs(cycles_avg) > epsilon) {
				if (bytes_processed_avg > 0) {
					metrics.cycles_per_byte.emplace(cycles_avg / static_cast<double>(bytes_processed_avg));
				}
				metrics.cycles_per_execution.emplace(cycles_avg);
				metrics.frequency_ghz.emplace(cycles_avg / ns_avg);
			}

			if (std::abs(instructions_avg) > epsilon) {
				if (bytes_processed_avg > 0) {
					metrics.instructions_per_byte.emplace(instructions_avg / static_cast<double>(bytes_processed_avg));
				}
				if (std::abs(cycles_avg) > epsilon) {
					metrics.instructions_per_cycle.emplace(instructions_avg / cycles_avg);
				}
				metrics.instructions_per_execution.emplace(instructions_avg);
			}

			if (std::abs(branches_avg) > epsilon) {
				metrics.branches_per_execution.emplace(branches_avg);
				metrics.branch_misses_per_execution.emplace(branch_misses_avg);
			}

			if (std::abs(cache_misses_avg) > epsilon) {
				metrics.cache_misses_per_execution.emplace(cache_misses_avg);
			}

			if (std::abs(cache_references_avg) > epsilon) {
				metrics.cache_references_per_execution.emplace(cache_references_avg);
			}

			return metrics;
		}
	};

	template<> struct performance_metrics<benchmark_types::cuda> {
		double throughput_percentage_deviation{ std::numeric_limits<double>::max() };
		std::optional<double> cycles_per_execution{};
		std::optional<double> cuda_event_ms_avg{};
		std::optional<double> cycles_per_byte{};

		uint64_t measured_iteration_count{};
		uint64_t total_iteration_count{};
		double throughput_mb_per_sec{};
		uint64_t bytes_processed{};
		std::string name{};
		double time_in_ns{};

		BNCH_SWT_HOST bool operator>(const performance_metrics& other) const {
			return throughput_mb_per_sec > other.throughput_mb_per_sec;
		}

		BNCH_SWT_HOST bool operator<(const performance_metrics& other) const {
			return throughput_mb_per_sec < other.throughput_mb_per_sec;
		}

		template<string_literal benchmark_name_new, bool mbps = true>
		BNCH_SWT_HOST static performance_metrics collect_metrics(std::span<internal::event_count<benchmark_types::cuda>>&& events_newer, uint64_t total_iteration_count) {
			static constexpr string_literal benchmark_name{ benchmark_name_new };
			performance_metrics metrics{};
			metrics.name					 = benchmark_name.operator std::string();
			metrics.measured_iteration_count = events_newer.size();
			metrics.total_iteration_count	 = total_iteration_count;
			double throughput{};
			double throughput_total{};
			double throughput_avg{};
			double throughput_min{ std::numeric_limits<double>::max() };
			uint64_t bytes_processed{};
			uint64_t bytes_processed_total{};
			uint64_t bytes_processed_avg{};
			double ns{};
			double ns_total{};
			double ns_avg{};
			double ms{};
			double ms_total{};
			double ms_avg{};
			double cycles{};
			double cycles_total{};
			double cycles_avg{};
			for (const internal::event_count<benchmark_types::cuda>& e: events_newer) {
				ns = e.elapsed_ns();
				ns_total += ns;
				ms = e.cuda_event_ms();
				ms_total += ms;

				if (e.bytes_processed(bytes_processed)) {
					bytes_processed_total += bytes_processed;
					if constexpr (mbps) {
						throughput = calculate_throughput_mbps(ns, static_cast<double>(bytes_processed));
					} else {
						throughput = calculate_units_ps(ns, static_cast<double>(bytes_processed));
					}
					throughput_total += throughput;
					throughput_min = throughput < throughput_min ? throughput : throughput_min;
				}

				if (e.cycles(cycles)) {
					cycles_total += cycles;
				}
			}
			if (events_newer.size() > 0) {
				bytes_processed_avg = bytes_processed_total / events_newer.size();
				ns_avg				= ns_total / static_cast<double>(events_newer.size());
				ms_avg				= ms_total / static_cast<double>(events_newer.size());
				throughput_avg		= throughput_total / static_cast<double>(events_newer.size());
				cycles_avg			= cycles_total / static_cast<double>(events_newer.size());
				metrics.time_in_ns	= ns_avg;
			} else {
				return {};
			}

			constexpr double epsilon = 1e-6;
			if (std::abs(ns_avg) > epsilon) {
				metrics.bytes_processed					= bytes_processed_avg;
				metrics.throughput_mb_per_sec			= throughput_avg;
				metrics.throughput_percentage_deviation = ((throughput_avg - throughput_min) * 100.0) / throughput_avg;
				metrics.cuda_event_ms_avg.emplace(ms_avg);
			}
			if (std::abs(cycles_avg) > epsilon) {
				if (metrics.bytes_processed > 0) {
					metrics.cycles_per_byte.emplace(cycles_avg / static_cast<double>(metrics.bytes_processed));
				}
				metrics.cycles_per_execution.emplace(cycles_total / static_cast<double>(events_newer.size()));
			}
			return metrics;
		}
	};
}
