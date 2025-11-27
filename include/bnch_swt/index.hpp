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

#include <bnch_swt/benchmarksuite_gpu_properties.hpp>
#include <bnch_swt/benchmarksuite_cpu_properties.hpp>
#include <bnch_swt/random_generators.hpp>
#include <bnch_swt/string_literal.hpp>
#include <bnch_swt/do_not_optimize.hpp>
#include <bnch_swt/event_counter.hpp>
#include <bnch_swt/cache_clearer.hpp>
#include <bnch_swt/file_loader.hpp>
#include <bnch_swt/printable.hpp>
#include <bnch_swt/metrics.hpp>
#include <bnch_swt/config.hpp>
#include <unordered_map>

namespace bnch_swt {

	template<benchmark_types benchmark_type, const std::string_view& stage_name_new, const std::string_view& metric_name_new> struct result_printer;

	template<typename value_type> BNCH_SWT_HOST static auto print_metric(std::string_view label, const value_type& value_new) {
		static constexpr uint64_t LABEL_WIDTH = 60;

		if constexpr (internal::optional_t<value_type>) {
			if (value_new.has_value()) {
				std::cout << std::left << std::setw(LABEL_WIDTH) << label << ": ";
				std::cout << value_new.value();
				std::cout << std::endl;
			}
		} else {
			std::cout << std::left << std::setw(LABEL_WIDTH) << label << ": ";
			std::cout << value_new;
			std::cout << std::endl;
		}
	}

	template<const std::string_view& stage_name_new, const std::string_view& metric_name_new> struct result_printer<benchmark_types::cpu, stage_name_new, metric_name_new> {
		BNCH_SWT_HOST static void impl(const std::vector<performance_metrics<benchmark_types::cpu>>& results_new, bool show_comparison = true, bool show_metrics = true) {
			std::cout << "CPU Performance Metrics for: " << stage_name_new << std::endl;

			if (show_metrics) {
				static const std::string throughput_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string("Throughput (") + std::string(metric_name_new) + "/s)";
					} else {
						return std::string("Throughput (MB/s)");
					}
				}();

				static const std::string metric_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string(metric_name_new) + "s Processed";
					} else {
						return std::string("Bytes Processed");
					}
				}();

				static const std::string cycle_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string("Cycles per ") + std::string(metric_name_new);
					} else {
						return std::string("Cycles per Byte");
					}
				}();

				static const std::string instruction_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string("Instructions per ") + std::string(metric_name_new);
					} else {
						return std::string("Instructions per Byte");
					}
				}();

				for (const auto& value: results_new) {
					std::cout << "Metrics for: " << value.name << std::endl;
					std::cout << std::fixed << std::setprecision(2);

					print_metric("Total Iterations to Stabilize", value.total_iteration_count);
					print_metric("Measured Iterations", value.measured_iteration_count);
					print_metric(metric_label, value.bytes_processed);

					print_metric("Nanoseconds per Execution", value.time_in_ns);
					print_metric("Frequency (GHz)", value.frequency_ghz);

					print_metric(throughput_label, value.throughput_mb_per_sec);

					print_metric("Throughput Percentage Deviation (+/-%)", value.throughput_percentage_deviation);
					print_metric("Cycles per Execution", value.cycles_per_execution);
					print_metric(cycle_label, value.cycles_per_byte);

					print_metric("Instructions per Execution", value.instructions_per_execution);
					print_metric("Instructions per Cycle", value.instructions_per_cycle);
					print_metric(instruction_label, value.instructions_per_byte);

					print_metric("Branches per Execution", value.branches_per_execution);
					print_metric("Branch Misses per Execution", value.branch_misses_per_execution);
					print_metric("Cache References per Execution", value.cache_references_per_execution);
					print_metric("Cache Misses per Execution", value.cache_misses_per_execution);

					std::cout << "----------------------------------------" << std::endl;
				}
			}

			if (show_comparison && results_new.size() > 1) {
				for (uint64_t x = 0; x < results_new.size() - 1; ++x) {
					double difference = ((results_new[x].throughput_mb_per_sec - results_new[x + 1].throughput_mb_per_sec) / results_new[x + 1].throughput_mb_per_sec) * 100.0;

					std::cout << "Library " << results_new[x].name << " is faster than library " << results_new[x + 1].name << " by " << difference << "%." << std::endl;
				}
			}
		}
	};

	template<const std::string_view& stage_name_new, const std::string_view& metric_name_new> struct result_printer<benchmark_types::cuda, stage_name_new, metric_name_new> {
		BNCH_SWT_HOST static void impl(const std::vector<performance_metrics<benchmark_types::cuda>>& results_new, bool show_comparison = true, bool show_metrics = true) {
			std::cout << "GPU Performance Metrics for: " << stage_name_new << std::endl;

			if (show_metrics) {
				static const std::string throughput_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string("Throughput (") + std::string(metric_name_new) + "/s)";
					} else {
						return std::string("Throughput (MB/s)");
					}
				}();

				static const std::string metric_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string(metric_name_new) + "s Processed";
					} else {
						return std::string("Bytes Processed");
					}
				}();

				static const std::string cycle_label = []() {
					if constexpr (metric_name_new.size() > 0) {
						return std::string("GPU Cycles per ") + std::string(metric_name_new);
					} else {
						return std::string("GPU Cycles per Byte");
					}
				}();

				for (const auto& value: results_new) {
					std::cout << "Metrics for: " << value.name << std::endl;
					std::cout << std::fixed << std::setprecision(2);

					print_metric("Total Iterations to Stabilize", value.total_iteration_count);
					print_metric("Measured Iterations", value.measured_iteration_count);
					print_metric(metric_label, value.bytes_processed);

					print_metric("Milliseconds per Execution", value.cuda_event_ms_avg);
					print_metric("Nanoseconds per Execution", value.time_in_ns);

					print_metric(throughput_label, value.throughput_mb_per_sec);

					print_metric("Throughput Percentage Deviation (+/-%)", value.throughput_percentage_deviation);
					print_metric("Cycles per Execution", value.cycles_per_execution);
					print_metric(cycle_label, value.cycles_per_byte);

					std::cout << "(CPU metrics like instructions/branches/cache are not available on GPU)" << std::endl;

					std::cout << "----------------------------------------" << std::endl;
				}
			}

			if (show_comparison && results_new.size() > 1) {
				for (uint64_t x = 0; x < results_new.size() - 1; ++x) {
					double difference = ((results_new[x].throughput_mb_per_sec - results_new[x + 1].throughput_mb_per_sec) / results_new[x + 1].throughput_mb_per_sec) * 100.0;

					std::cout << "Kernel " << results_new[x].name << " is faster than kernel " << results_new[x + 1].name << " by " << difference << "%." << std::endl;
				}
			}
		}
	};

	template<string_literal stage_name_new, uint64_t max_execution_count = 200, uint64_t measured_iteration_count = 25, benchmark_types benchmark_type = benchmark_types::cpu,
		bool clear_cpu_cache_between_each_iteration = false, string_literal metric_name_new = string_literal<1>{}>
	struct benchmark_stage {
		static_assert(max_execution_count % measured_iteration_count == 0, "Sorry, but please enter a max_execution_count that is divisible by measured_iteration_count.");

		BNCH_SWT_HOST static auto& get_results_internal() {
			static thread_local std::unordered_map<std::string_view, performance_metrics<benchmark_type>> results{};
			return results;
		}

		static constexpr bool use_non_mbps_metric{ metric_name_new.size() == 0 };

		BNCH_SWT_HOST static void print_results(bool show_comparison = true, bool show_metrics = true) {
			std::vector<performance_metrics<benchmark_type>> results_new{};
			for (const auto& [key, value]: get_results_internal()) {
				results_new.emplace_back(value);
			}
			if (results_new.size() > 0) {
				std::sort(results_new.begin(), results_new.end(), std::greater<performance_metrics<benchmark_type>>{});
				static constexpr std::string_view stage_name_newer{ stage_name_new.operator std::string_view() };
				static constexpr std::string_view metric_name_newer{ metric_name_new.operator std::string_view() };
				result_printer<benchmark_type, stage_name_newer, metric_name_newer>::impl(results_new, show_comparison, show_metrics);
			}
		}

		BNCH_SWT_HOST static auto get_results() {
			std::vector<performance_metrics<benchmark_type>> results_new{};
			for (const auto& [key, value]: get_results_internal()) {
				results_new.emplace_back(value);
			}
			if (results_new.size() > 0) {
				std::sort(results_new.begin(), results_new.end(), std::greater<performance_metrics<benchmark_type>>{});
			}
			return results_new;
		}

		template<string_literal subject_name_new, typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_HOST static performance_metrics<benchmark_type> run_benchmark(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				static_assert(std::convertible_to<std::invoke_result_t<decltype(function_type::impl), arg_types...>, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}
			internal::event_collector<max_execution_count, benchmark_type> events{};
			internal::cache_clearer<benchmark_type> cache_clearer{};
			performance_metrics<benchmark_type> lowest_results{};
			performance_metrics<benchmark_type> results_temp{};
			uint64_t current_global_index{ measured_iteration_count };
			for (uint64_t x = 0; x < max_execution_count; ++x) {
				if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
					cache_clearer.evict_caches();
				}
				events.template run<function_type>(std::forward<arg_types>(args)...);
			}
			std::span<internal::event_count<benchmark_type>> new_ptr{ static_cast<std::vector<internal::event_count<benchmark_type>>&>(events) };
			static constexpr uint64_t final_measured_iteration_count{ max_execution_count - measured_iteration_count > 0 ? max_execution_count - measured_iteration_count : 1 };
			for (uint64_t x = 0; x < final_measured_iteration_count; ++x, ++current_global_index) {
				results_temp   = performance_metrics<benchmark_type>::template collect_metrics<subject_name, use_non_mbps_metric>(new_ptr.subspan(x, measured_iteration_count),
					  current_global_index);
				lowest_results = results_temp.throughput_percentage_deviation < lowest_results.throughput_percentage_deviation ? results_temp : lowest_results;
			}
			get_results_internal()[subject_name.operator std::string_view()] = lowest_results;
			return get_results_internal()[subject_name.operator std::string_view()];
		}
	};

}
