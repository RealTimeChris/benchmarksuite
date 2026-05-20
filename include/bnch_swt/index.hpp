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
#include <unordered_set>
#include <unordered_map>
#include <iostream>

namespace bnch_swt {

	struct stage_config {
		uint64_t max_execution_count{ 200 };
		uint64_t measured_iteration_count{ 10 };
		benchmark_types benchmark_type{ benchmark_types::cpu };
		bool clear_cpu_cache_between_each_iteration{ false };
		bool clear_cpu_cache_before_all_iterations{ true };
		double desired_percentage_deviation{ 1.0 };
		double max_time_seconds{ 5.5 };
	};

	namespace internal {

		static constexpr bnch_swt::string_literal operating_system_name{ BNCH_SWT_OPERATING_SYSTEM_NAME };
		static constexpr bnch_swt::string_literal operating_system_version{ BNCH_SWT_OPERATING_SYSTEM_VERSION };
		static constexpr bnch_swt::string_literal compiler_id{ BNCH_SWT_COMPILER_ID };
		static constexpr bnch_swt::string_literal compiler_version{ BNCH_SWT_COMPILER_VERSION };

		template<benchmark_types> BNCH_SWT_HOST std::string get_device_info();

		template<> BNCH_SWT_HOST std::string get_device_info<benchmark_types::cpu>() {
#if defined(__x86_64__) || defined(_M_AMD64)
			std::array<uint32_t, 12> regs{};
			auto cpuid = [](uint32_t leaf, uint32_t* res) {
	#if defined(_MSC_VER)
				__cpuidex(std::bit_cast<int*>(res), static_cast<int>(leaf), 0);
	#elif defined(__GNUC__) || defined(__clang__)
				__asm__ volatile("cpuid" : "=a"(res[0]), "=b"(res[1]), "=c"(res[2]), "=d"(res[3]) : "a"(leaf), "c"(0));
	#endif
			};

			uint32_t ext_info[4];
			cpuid(0x80000000, ext_info);
			if (ext_info[0] < 0x80000004) {
				return "Unknown x86_64 CPU";
			}

			for (uint32_t i = 0; i < 3; ++i) {
				cpuid(0x80000002 + i, regs.data() + (i * 4));
			}

			char brand[49]{};
			std::memcpy(brand, regs.data(), 48);

			std::string result(brand);
			auto last_char = result.find_last_not_of(" \t\n\r\f\v");
			if (last_char != std::string::npos) {
				result.resize(last_char + 1);
			}

			return result;

#elif defined(__APPLE__) || defined(__FreeBSD__)
			char buffer[256]{};
			size_t size = sizeof(buffer);
			if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
				return std::string(buffer);
			}
			return "Unknown CPU";
#elif defined(__linux__)
			std::ifstream cpuinfo("/proc/cpuinfo");
			std::string line;
			while (std::getline(cpuinfo, line)) {
				if (line.find("model name") == 0 || line.find("Hardware") == 0 || line.find("Model") == 0) {
					auto colon = line.find(':');
					if (colon != std::string::npos) {
						auto start = line.find_first_not_of(" \t", colon + 1);
						if (start != std::string::npos)
							return line.substr(start);
					}
				}
			}
			return "Unknown Linux CPU";
#elif defined(_WIN32)
			HKEY hkey;
			if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
				char buffer[256]{};
				DWORD size = sizeof(buffer);
				if (RegQueryValueExA(hkey, "ProcessorNameString", nullptr, nullptr, std::bit_cast<LPBYTE>(buffer), &size) == ERROR_SUCCESS) {
					RegCloseKey(hkey);
					return std::string(buffer);
				}
				RegCloseKey(hkey);
			}
			return "Unknown Windows CPU";
#else

			return "Unsupported Architecture";
#endif
		}

	}

	template<auto function> struct functor_executor {
		template<internal::not_invocable... arg_types> BNCH_SWT_HOST static decltype(auto) impl(arg_types&&... args) {
			return function(std::forward<arg_types>(args)...);
		}
	};

	template<auto stage_name, benchmark_types benchmark_type> struct results_holder {};

	template<string_literal stage_name_new, string_literal test_name_new, string_literal subject_name, bnch_swt::stage_config stage_config_new,
		bnch_swt::benchmark_types benchmark_type, bool use_non_mbps_metric>
	struct measurement_context {
		static constexpr uint64_t max_execution_count				 = stage_config_new.max_execution_count;
		static constexpr uint64_t measured_iteration_count			 = stage_config_new.measured_iteration_count;
		static constexpr bool clear_cpu_cache_between_each_iteration = stage_config_new.clear_cpu_cache_between_each_iteration;
		static constexpr bool clear_cpu_cache_before_all_iterations	 = stage_config_new.clear_cpu_cache_before_all_iterations;
		static constexpr double desired_percentage_deviation		 = stage_config_new.desired_percentage_deviation;
		static constexpr double max_time_seconds					 = stage_config_new.max_time_seconds;

		using stage_results_type = stage_results<stage_name_new, benchmark_type>;
		static constexpr string_literal stage_name{ stage_name_new };
		static constexpr string_literal test_name{ test_name_new };

		template<typename function_type, internal::not_invocable... arg_types> BNCH_SWT_HOST static auto& run_adaptive_impl(arg_types&&... args) {
			internal::event_collector<max_execution_count, benchmark_type> events{};
			internal::cache_clearer<benchmark_type> cache_clearer{};

			if constexpr (clear_cpu_cache_before_all_iterations && benchmark_type == benchmark_types::cpu) {
				cache_clearer.evict_caches();
			}

			auto start_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			auto end_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			uint64_t total_iterations_target = measured_iteration_count * 2;
			uint64_t current_iteration_count = 0;

			performance_metrics<benchmark_type> overall_best{};
			overall_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

			while (total_iterations_target <= max_execution_count && (end_time - start_time).count() < max_time_seconds) {
				events.reset();
				for (uint64_t x = 0; x < total_iterations_target && current_iteration_count < max_execution_count && (end_time - start_time).count() < max_time_seconds;
					++current_iteration_count) {
					if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
						cache_clearer.evict_caches();
					}
					events.template run<function_type>(std::forward<arg_types>(args)...);

					end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				}

				std::span<internal::event_count<benchmark_type>> all_events{ static_cast<std::vector<internal::event_count<benchmark_type>>&>(events) };

				performance_metrics<benchmark_type> round_best{};
				round_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

				for (uint64_t window_start = 0; window_start <= current_iteration_count - measured_iteration_count; ++window_start) {
					auto temp_result = performance_metrics<benchmark_type>::template collect_metrics<subject_name, use_non_mbps_metric>(
						all_events.subspan(window_start, measured_iteration_count), window_start, current_iteration_count);

					if (temp_result.throughput_percentage_deviation < round_best.throughput_percentage_deviation) {
						round_best = temp_result;
					}
				}


				if (round_best.throughput_percentage_deviation < overall_best.throughput_percentage_deviation) {
					overall_best = round_best;
				}

				if (round_best.throughput_percentage_deviation <= desired_percentage_deviation) {
					stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
					return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
				}

				end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				total_iterations_target *= 2;
			}

			stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
			return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
		}

		template<typename function_type, internal::not_invocable... arg_types>
		BNCH_SWT_HOST static auto& run_adaptive_from_host_impl(uint64_t bytes_processed, arg_types&&... args) {
			internal::event_collector<max_execution_count, benchmark_type> events{};

			auto start_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			auto end_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			uint64_t total_iterations_target = measured_iteration_count * 2;
			uint64_t current_iteration_count = 0;

			performance_metrics<benchmark_type> overall_best{};
			overall_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

			while (total_iterations_target <= max_execution_count && (end_time - start_time).count() < max_time_seconds) {
				events.reset();
				for (uint64_t x = 0; x < total_iterations_target && current_iteration_count < max_execution_count && (end_time - start_time).count() < max_time_seconds;
					++current_iteration_count) {
					events.template run_from_host<function_type>(bytes_processed, args...);
					end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				}

				std::span<internal::event_count<benchmark_type>> all_events{ static_cast<std::vector<internal::event_count<benchmark_type>>&>(events) };

				performance_metrics<benchmark_type> round_best{};
				round_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

				for (uint64_t window_start = 0; window_start <= current_iteration_count - measured_iteration_count; ++window_start) {
					auto temp_result = performance_metrics<benchmark_type>::template collect_metrics<subject_name, use_non_mbps_metric>(
						all_events.subspan(window_start, measured_iteration_count), window_start, current_iteration_count);

					if (temp_result.throughput_percentage_deviation < round_best.throughput_percentage_deviation) {
						round_best = temp_result;
					}
				}

				if (round_best.throughput_percentage_deviation < overall_best.throughput_percentage_deviation) {
					overall_best = round_best;
				}

				if (round_best.throughput_percentage_deviation <= desired_percentage_deviation) {
					stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
					return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
				}

				end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				total_iterations_target *= 2;
			}

			stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
			return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
		}

		template<internal::function_pointer_types auto function, internal::not_invocable... arg_types>
		BNCH_SWT_HOST static auto& run_adaptive_cooperative_impl(arg_types&&... args) {
			internal::event_collector<max_execution_count, benchmark_type> events{};

			auto start_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			auto end_time					 = std::chrono::time_point_cast<seconds>(std::chrono::steady_clock::now());
			uint64_t total_iterations_target = measured_iteration_count * 2;
			uint64_t current_iteration_count = 0;

			performance_metrics<benchmark_type> overall_best{};
			overall_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

			while (total_iterations_target <= max_execution_count && (end_time - start_time).count() < max_time_seconds) {
				events.reset();
				for (uint64_t x = 0; x < total_iterations_target && current_iteration_count < max_execution_count && (end_time - start_time).count() < max_time_seconds;
					++current_iteration_count) {
					events.template run_cooperative<function>(std::forward<arg_types>(args)...);
					end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				}

				std::span<internal::event_count<benchmark_type>> all_events{ static_cast<std::vector<internal::event_count<benchmark_type>>&>(events) };

				performance_metrics<benchmark_type> round_best{};
				round_best.throughput_percentage_deviation = std::numeric_limits<double>::max();

				for (uint64_t window_start = 0; window_start <= current_iteration_count - measured_iteration_count; ++window_start) {
					auto temp_result = performance_metrics<benchmark_type>::template collect_metrics<subject_name, use_non_mbps_metric>(
						all_events.subspan(window_start, measured_iteration_count), window_start, current_iteration_count);

					if (temp_result.throughput_percentage_deviation < round_best.throughput_percentage_deviation) {
						round_best = temp_result;
					}
				}

				if (round_best.throughput_percentage_deviation < overall_best.throughput_percentage_deviation) {
					overall_best = round_best;
				}

				if (round_best.throughput_percentage_deviation <= desired_percentage_deviation) {
					stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
					return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
				}

				end_time = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::steady_clock::now());
				total_iterations_target *= 2;
			}

			stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = overall_best;
			return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
		}

		template<typename function_type, internal::not_invocable... arg_types> BNCH_SWT_HOST static auto& run_adaptive_from_host(uint64_t bytes_processed, arg_types&&... args) {
			return run_adaptive_from_host_impl<function_type>(bytes_processed, std::forward<arg_types>(args)...);
		}

		template<auto function, internal::not_invocable... arg_types>
		BNCH_SWT_HOST static auto& run_adaptive_cooperative(arg_types&&... args) {
			return run_adaptive_cooperative_impl<function>(std::forward<arg_types>(args)...);
		}

		template<auto function, internal::not_invocable... arg_types> BNCH_SWT_HOST static auto& run_adaptive(arg_types&&... args) {
			return run_adaptive_impl<functor_executor<function>>(std::forward<arg_types>(args)...);
		}

		template<typename function_type, internal::not_invocable... arg_types> BNCH_SWT_HOST static auto& run_adaptive(arg_types&&... args) {
			return run_adaptive_impl<function_type>(std::forward<arg_types>(args)...);
		}
	};

	struct library_win_count {
		uint64_t win_count{};
		std::string name{};

		BNCH_SWT_HOST bool operator>(const library_win_count& other) const {
			return win_count > other.win_count;
		}
	};

	template<benchmark_types benchmark_type, auto, auto, performance_metrics_presence<benchmark_type> metrics_presence> struct result_printer;

	template<bool printed, typename value_type> BNCH_SWT_HOST static auto print_metric(std::string_view label, const value_type& value_new) {
		if constexpr (printed) {
			if constexpr (internal::optional_t<value_type>) {
				if (value_new.has_value()) {
					std::cout << std::left << label << ": ";
					std::cout << value_new.value();
					std::cout << std::endl;
				}
			} else {
				std::cout << std::left << label << ": ";
				std::cout << value_new;
				std::cout << std::endl;
			}
		}
	}

	template<auto stage_name_newer, auto metric_name_new, performance_metrics_presence<benchmark_types::cpu> metrics_presence>
	struct result_printer<benchmark_types::cpu, stage_name_newer, metric_name_new, metrics_presence> {
		static constexpr auto stage_name_new{ stage_name_newer };
		BNCH_SWT_HOST static void print_results(const performance_metrics<benchmark_types::cpu>& value, bool show_metrics = true) {
			if (show_metrics) {
				static constexpr string_literal throughput_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						string_literal throughput_label_new{ "Throughput (" + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) + "/s)" };
						return throughput_label_new;
					} else {
						return string_literal{ "Throughput (MB/s)" };
					}
				}();
				static constexpr std::string_view throughput_label{ throughput_label_raw.operator std::string_view() };

				static constexpr string_literal metric_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						string_literal metric_label_new{ string_literal{ internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) + "s Processed" } };
						return metric_label_new;
					} else {
						return string_literal{ "Bytes Processed" };
					}
				}();
				static constexpr std::string_view metric_label{ metric_label_raw.operator std::string_view() };

				static constexpr string_literal cycle_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						return string_literal{ "Cycles per " + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) };
					} else {
						return string_literal{ "Cycles per Byte" };
					}
				}();
				static constexpr std::string_view cycle_label{ cycle_label_raw.operator std::string_view() };

				static constexpr string_literal instruction_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						return string_literal{ "Instructions per " + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) };
					} else {
						return string_literal{ "Instructions per Byte" };
					}
				}();
				static constexpr std::string_view instruction_label{ instruction_label_raw.operator std::string_view() };
				std::cout << std::fixed << std::setprecision(2);
				print_metric<metrics_presence.total_iteration_count>("Total Iterations", value.total_iteration_count);
				print_metric<metrics_presence.iterations_to_stabilize>("Total Iterations to Stabilize", value.iterations_to_stabilize);
				print_metric<metrics_presence.measured_iteration_count>("Measured Iterations", value.measured_iteration_count);
				print_metric<metrics_presence.bytes_processed>(metric_label, value.bytes_processed);
				print_metric<metrics_presence.time_in_ns>("Nanoseconds per Execution", value.time_in_ns);
				print_metric<metrics_presence.frequency_ghz>("Frequency (GHz)", value.frequency_ghz);
				print_metric<metrics_presence.throughput_mb_per_sec>(throughput_label, value.throughput_mb_per_sec);
				print_metric<metrics_presence.throughput_percentage_deviation>("Throughput Percentage Deviation (+/-%)", value.throughput_percentage_deviation);
				print_metric<metrics_presence.cycles_per_execution>("Cycles per Execution", value.cycles_per_execution);
				print_metric<metrics_presence.cycles_per_byte>(cycle_label, value.cycles_per_byte);
				print_metric<metrics_presence.instructions_per_execution>("Instructions per Execution", value.instructions_per_execution);
				print_metric<metrics_presence.instructions_per_cycle>("Instructions per Cycle", value.instructions_per_cycle);
				print_metric<metrics_presence.instructions_per_byte>(instruction_label, value.instructions_per_byte);
				print_metric<metrics_presence.branches_per_execution>("Branches per Execution", value.branches_per_execution);
				print_metric<metrics_presence.branch_misses_per_execution>("Branch Misses per Execution", value.branch_misses_per_execution);
				print_metric<metrics_presence.cache_references_per_execution>("Cache References per Execution", value.cache_references_per_execution);
				print_metric<metrics_presence.cache_misses_per_execution>("Cache Misses per Execution", value.cache_misses_per_execution);
				if constexpr (has_any_metric_enabled(metrics_presence)) {
					std::cout << "----------------------------------------" << std::endl;
				}
			}
		}
	};

	template<auto stage_name_new, auto metric_name_new, performance_metrics_presence<benchmark_types::cuda> metrics_presence>
	struct result_printer<benchmark_types::cuda, stage_name_new, metric_name_new, metrics_presence> {
		BNCH_SWT_HOST static void print_results(const performance_metrics<benchmark_types::cuda>& value, bool show_metrics = true) {
			if (show_metrics) {
				static constexpr string_literal throughput_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						string_literal throughput_label_new{ "Throughput (" + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) + "/s)" };
						return throughput_label_new;
					} else {
						return string_literal{ "Throughput (MB/s)" };
					}
				}();
				static constexpr std::string_view throughput_label{ throughput_label_raw.operator std::string_view() };

				static constexpr string_literal metric_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						string_literal metric_label_new{ string_literal{ internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) + "s Processed" } };
						return metric_label_new;
					} else {
						return string_literal{ "Bytes Processed" };
					}
				}();
				static constexpr std::string_view metric_label{ metric_label_raw.operator std::string_view() };

				static constexpr string_literal cycle_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						return string_literal{ "Cycles per " + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) };
					} else {
						return string_literal{ "Cycles per Byte" };
					}
				}();
				static constexpr std::string_view cycle_label{ cycle_label_raw.operator std::string_view() };

				static constexpr string_literal instruction_label_raw = []() {
					if constexpr (metric_name_new.size() > 0) {
						return string_literal{ "Instructions per " + internal::string_literal_from_view<metric_name_new.size()>(metric_name_new) };
					} else {
						return string_literal{ "Instructions per Byte" };
					}
				}();
				static constexpr std::string_view instruction_label{ instruction_label_raw.operator std::string_view() };
				std::cout << std::fixed << std::setprecision(2);
				print_metric<metrics_presence.total_iteration_count>("Total Iterations", value.total_iteration_count);
				print_metric<metrics_presence.iterations_to_stabilize>("Total Iterations to Stabilize", value.iterations_to_stabilize);
				print_metric<metrics_presence.measured_iteration_count>("Measured Iterations", value.measured_iteration_count);
				print_metric<metrics_presence.bytes_processed>(metric_label, value.bytes_processed);
				print_metric<metrics_presence.cuda_event_ms_avg>("Milliseconds per Execution", value.cuda_event_ms_avg);
				print_metric<metrics_presence.time_in_ns>("Nanoseconds per Execution", value.time_in_ns);
				print_metric<metrics_presence.throughput_mb_per_sec>(throughput_label, value.throughput_mb_per_sec);
				print_metric<metrics_presence.throughput_percentage_deviation>("Throughput Percentage Deviation (+/-%)", value.throughput_percentage_deviation);
				print_metric<metrics_presence.cycles_per_execution>("Cycles per Execution", value.cycles_per_execution);
				print_metric<metrics_presence.cycles_per_byte>(cycle_label, value.cycles_per_byte);
				std::cout << "(CPU metrics like instructions/branches/cache are not available on GPU)" << std::endl;
				if constexpr (has_any_metric_enabled(metrics_presence)) {
					std::cout << "----------------------------------------" << std::endl;
				}
			}
		}
	};

	template<string_literal stage_name_new, stage_config stage_config_new = stage_config{}, string_literal metric_name_new = string_literal<1>{}> struct benchmark_stage {
		static constexpr uint64_t max_execution_count{ stage_config_new.max_execution_count };
		static constexpr uint64_t measured_iteration_count{ stage_config_new.measured_iteration_count };
		static constexpr benchmark_types benchmark_type{ stage_config_new.benchmark_type };
		static constexpr bool clear_cpu_cache_between_each_iteration{ stage_config_new.clear_cpu_cache_between_each_iteration };
		static constexpr bool clear_cpu_cache_before_all_iterations{ stage_config_new.clear_cpu_cache_before_all_iterations };
		using stage_results_type = stage_results<stage_name_new, benchmark_type>;
		static constexpr string_literal stage_name{ stage_name_new };
		static constexpr string_literal device_type{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };
		static_assert(max_execution_count % measured_iteration_count == 0, "Sorry, but please enter a max_execution_count that is divisible by measured_iteration_count.");

		static constexpr bool use_non_mbps_metric{ metric_name_new.size() == 0 };

		template<performance_metrics_presence<benchmark_type> metrics_presence = performance_metrics_presence<benchmark_type>{}>
		BNCH_SWT_HOST static void print_results(bool show_metrics = true) {
			std::cout << "----------------------------------------" << std::endl;
			std::cout << device_type << " Performance Metrics for Stage: " << stage_name << std::endl;
			std::cout << "Running on: " << internal::get_device_info<benchmark_type>() << std::endl;
			std::cout << "OS: " << internal::operating_system_name << " " << internal::operating_system_version << std::endl;
			std::cout << "Compiler: " << internal::compiler_id << " " << internal::compiler_version << std::endl;

			using stage_t								  = bnch_swt::stage_results<stage_name, benchmark_type>;
			static constexpr double confidence_multiplier = 1.96;
			static constexpr double epsilon				  = 1e-12;

			std::unordered_map<std::string, uint32_t> total_wins;
			std::unordered_map<std::string, uint32_t> total_ties;
			std::unordered_map<std::string, uint32_t> total_second_places;

			auto ranges_overlap = [&](const auto* a, const auto* b) -> bool {
				double a_stddev = std::abs(a->throughput_percentage_deviation) / 100.0;
				double b_stddev = std::abs(b->throughput_percentage_deviation) / 100.0;

				if (a_stddev < epsilon || b_stddev < epsilon) {
					return std::abs(a->throughput_mb_per_sec - b->throughput_mb_per_sec) < epsilon;
				}

				double a_min = a->throughput_mb_per_sec * (1.0 - a_stddev * confidence_multiplier);
				double a_max = a->throughput_mb_per_sec * (1.0 + a_stddev * confidence_multiplier);
				double b_min = b->throughput_mb_per_sec * (1.0 - b_stddev * confidence_multiplier);
				double b_max = b->throughput_mb_per_sec * (1.0 + b_stddev * confidence_multiplier);

				return !(a_max + epsilon < b_min || b_max + epsilon < a_min);
			};

			for (const auto& library_results: stage_t::results) {
				if (library_results.results.empty()) {
					continue;
				}

				std::vector<const typename stage_t::performance_metrics_type*> sorted;
				for (const auto& [lib_name, metrics]: library_results.results) {
					sorted.push_back(&metrics);
				}

				std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
					return a->throughput_mb_per_sec > b->throughput_mb_per_sec;
				});

				std::vector<std::vector<size_t>> groups;
				std::vector<bool> grouped(sorted.size(), false);

				for (size_t i = 0; i < sorted.size(); ++i) {
					if (grouped[i])
						continue;

					std::vector<size_t> group;
					group.push_back(i);
					grouped[i] = true;

					for (size_t j = i + 1; j < sorted.size(); ++j) {
						if (grouped[j])
							continue;

						bool overlaps_group = false;
						for (size_t member: group) {
							if (ranges_overlap(sorted[member], sorted[j])) {
								overlaps_group = true;
								break;
							}
						}

						if (overlaps_group) {
							group.push_back(j);
							grouped[j] = true;
						}
					}

					groups.push_back(std::move(group));
				}

				std::sort(groups.begin(), groups.end(), [&](const auto& a, const auto& b) {
					double a_throughput = 0, b_throughput = 0;
					for (size_t idx: a)
						a_throughput += sorted[idx]->throughput_mb_per_sec;
					for (size_t idx: b)
						b_throughput += sorted[idx]->throughput_mb_per_sec;
					return (a_throughput / a.size()) > (b_throughput / b.size());
				});

				if (groups.empty())
					continue;

				if (groups[0].size() > 1) {
					for (size_t idx: groups[0]) {
						total_ties[sorted[idx]->library_name]++;
					}
				} else {
					total_wins[sorted[groups[0][0]]->library_name]++;
				}

				if (groups.size() > 1 && groups[1].size() == 1) {
					total_second_places[sorted[groups[1][0]]->library_name]++;
				}

				std::cout << "----------------------------------------" << std::endl;
				std::cout << "Test: " << library_results.test_name << std::endl;
				std::cout << "----------------------------------------" << std::endl;

				size_t rank = 1;
				for (size_t g = 0; g < groups.size(); ++g) {
					const auto& group = groups[g];
					bool is_tied	  = group.size() > 1;

					for (size_t j = 0; j < group.size(); ++j) {
						const auto* current = sorted[group[j]];
						double deviation	= std::abs(current->throughput_percentage_deviation);

						std::cout << rank << ". " << current->library_name << " (" << std::fixed << std::setprecision(2) << current->throughput_mb_per_sec << " MB/s +/-"
								  << deviation << "%)";

						if (is_tied) {
							std::cout << " [STATISTICAL TIE";
							if (group.size() > 2)
								std::cout << " (" << group.size() << "-way)";
							std::cout << "]";
						}

						if (g + 1 < groups.size()) {
							const auto* next  = sorted[groups[g + 1][0]];
							double diff		  = current->throughput_mb_per_sec - next->throughput_mb_per_sec;
							double pct_faster = (next->throughput_mb_per_sec > 1e-9) ? (diff / next->throughput_mb_per_sec) * 100.0 : 0.0;
							std::cout << " | ~" << std::setprecision(1) << pct_faster << "% faster than " << next->library_name;
						}

						static constexpr string_literal metric_name_newer{ metric_name_new };
						std::cout << std::endl;
						result_printer<benchmark_type, stage_name, metric_name_newer, metrics_presence>::print_results(*current, show_metrics);
					}

					rank += group.size();
				}
			}

			std::vector<library_win_count> total_wins_vector;
			for (const auto& [lib, wins]: total_wins) {
				total_wins_vector.emplace_back(library_win_count{ .win_count = wins, .name = lib });
			}
			std::sort(total_wins_vector.begin(), total_wins_vector.end(), std::greater<library_win_count>{});

			std::cout << "\n=== STATISTICAL SUMMARY FOR " << stage_name.operator std::string() << " ===" << std::endl;
			std::cout << "(95% confidence intervals, statistical ties don't count as wins)" << std::endl;

			for (const auto& value: total_wins_vector) {
				std::cout << value.name << ": " << value.win_count << " wins";
				if (total_second_places.count(value.name)) {
					std::cout << " (" << total_second_places[value.name] << " second places)";
				}
				std::cout << std::endl;
			}

			if (!total_ties.empty()) {
				std::cout << "\n=== STATISTICAL TIES (no clear winner) ===" << std::endl;
				for (const auto& [lib, ties]: total_ties) {
					std::cout << lib << ": " << ties << " tests where statistically tied for first" << std::endl;
				}
			}
		}

		template<performance_metrics_presence<benchmark_type> metrics_presence = performance_metrics_presence<benchmark_type>{}>
		BNCH_SWT_HOST static std::string generate_markdown(const std::string& results_title, const std::string& file_path = "") {
			std::stringstream return_value{};
			return_value << "# " << results_title << " Benchmark Results  " << std::endl;
			return_value << "**Platform:** " << internal::get_device_info<benchmark_type>() << "  " << std::endl;
			return_value << "**OS:** " << internal::operating_system_name.operator std::string() << " " << internal::operating_system_version.operator std::string() << "  "
						 << std::endl;
			return_value << "**Compiler:** " << internal::compiler_id.operator std::string() << " " << internal::compiler_version.operator std::string() << "  " << std::endl
						 << std::endl;
			return_value << "---" << std::endl << std::endl;

			using stage_t								  = bnch_swt::stage_results<stage_name, benchmark_type>;
			static constexpr double confidence_multiplier = 1.96;
			static constexpr double epsilon				  = 1e-12;

			std::unordered_map<std::string, uint32_t> total_wins;
			std::unordered_map<std::string, uint32_t> total_ties;
			std::unordered_map<std::string, uint32_t> total_second_places;

			auto ranges_overlap = [&](const auto* a, const auto* b) -> bool {
				double a_stddev = std::abs(a->throughput_percentage_deviation) / 100.0;
				double b_stddev = std::abs(b->throughput_percentage_deviation) / 100.0;
				if (a_stddev < epsilon || b_stddev < epsilon) {
					return std::abs(a->throughput_mb_per_sec - b->throughput_mb_per_sec) < epsilon;
				}
				double a_min = a->throughput_mb_per_sec * (1.0 - a_stddev * confidence_multiplier);
				double a_max = a->throughput_mb_per_sec * (1.0 + a_stddev * confidence_multiplier);
				double b_min = b->throughput_mb_per_sec * (1.0 - b_stddev * confidence_multiplier);
				double b_max = b->throughput_mb_per_sec * (1.0 + b_stddev * confidence_multiplier);
				return !(a_max + epsilon < b_min || b_max + epsilon < a_min);
			};

			struct processed_row {
				std::string test_name;
				std::vector<std::vector<size_t>> groups;
				std::vector<const typename stage_t::performance_metrics_type*> sorted;
			};

			std::vector<processed_row> rows;
			size_t max_competitors = 0;

			for (const auto& library_results: stage_t::results) {
				if (library_results.results.empty())
					continue;

				std::vector<const typename stage_t::performance_metrics_type*> sorted;
				for (const auto& [lib_name, metrics]: library_results.results) {
					sorted.push_back(&metrics);
				}
				std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
					return a->throughput_mb_per_sec > b->throughput_mb_per_sec;
				});

				std::vector<std::vector<size_t>> groups;
				std::vector<bool> grouped(sorted.size(), false);
				for (size_t i = 0; i < sorted.size(); ++i) {
					if (grouped[i])
						continue;
					std::vector<size_t> group;
					group.push_back(i);
					grouped[i] = true;
					for (size_t j = i + 1; j < sorted.size(); ++j) {
						if (grouped[j])
							continue;
						bool overlaps_group = false;
						for (size_t member: group) {
							if (ranges_overlap(sorted[member], sorted[j])) {
								overlaps_group = true;
								break;
							}
						}
						if (overlaps_group) {
							group.push_back(j);
							grouped[j] = true;
						}
					}
					groups.push_back(std::move(group));
				}

				std::sort(groups.begin(), groups.end(), [&](const auto& a, const auto& b) {
					double a_t = 0, b_t = 0;
					for (size_t idx: a)
						a_t += sorted[idx]->throughput_mb_per_sec;
					for (size_t idx: b)
						b_t += sorted[idx]->throughput_mb_per_sec;
					return (a_t / a.size()) > (b_t / b.size());
				});

				if (!groups.empty()) {
					if (groups[0].size() > 1) {
						for (size_t idx: groups[0])
							total_ties[sorted[idx]->library_name]++;
					} else {
						total_wins[sorted[groups[0][0]]->library_name]++;
					}
					if (groups.size() > 1 && groups[1].size() == 1)
						total_second_places[sorted[groups[1][0]]->library_name]++;
				}

				max_competitors = std::max(max_competitors, sorted.size());
				rows.push_back(processed_row{ static_cast<std::string>(library_results.test_name), std::move(groups), std::move(sorted) });
			}

			std::vector<library_win_count> total_wins_vector;
			for (const auto& [lib, wins]: total_wins)
				total_wins_vector.emplace_back(library_win_count{ .win_count = wins, .name = lib });
			std::sort(total_wins_vector.begin(), total_wins_vector.end(), std::greater<library_win_count>{});

			uint32_t max_wins = total_wins_vector.empty() ? 0 : total_wins_vector[0].win_count;

			return_value << "## " << stage_name.operator std::string() << "\n\n";

			return_value << "### " << stage_name.operator std::string() << " Statistical Summary\n\n";
			return_value << "| Library | Outright Wins | 2nd Place | Statistical Ties for 1st |\n";
			return_value << "|---|---|---|---|\n";

			std::unordered_set<std::string> seen;
			auto write_summary_row = [&](const std::string& lib) {
				if (!seen.insert(lib).second)
					return;
				uint32_t wins	= total_wins.count(lib) ? total_wins.at(lib) : 0;
				uint32_t second = total_second_places.count(lib) ? total_second_places.at(lib) : 0;
				uint32_t ties	= total_ties.count(lib) ? total_ties.at(lib) : 0;
				bool is_top		= (wins == max_wins && max_wins > 0);
				return_value << "| " << (is_top ? "**" : "") << lib << (is_top ? "**" : "") << " | ";
				if (wins > 0) {
					if (is_top)
						return_value << "**" << wins << "**";
					else
						return_value << wins;
				} else {
					return_value << "-";
				}
				return_value << " | ";
				return_value << (second > 0 ? std::to_string(second) : "-") << " | ";
				return_value << (ties > 0 ? std::to_string(ties) : "-") << " |\n";
			};

			for (const auto& entry: total_wins_vector)
				write_summary_row(entry.name);
			for (const auto& [lib, v]: total_ties)
				write_summary_row(lib);
			for (const auto& [lib, v]: total_second_places)
				write_summary_row(lib);

			return_value << "\n---\n\n";

			return_value << "## " << stage_name.operator std::string() << "\n\n";

			return_value << "| Test |";
			for (size_t c = 0; c < max_competitors; ++c) {
				if (c == 0)
					return_value << " 1st |";
				else if (c == 1)
					return_value << " 2nd |";
				else if (c == 2)
					return_value << " 3rd |";
				else
					return_value << " " << (c + 1) << "th |";
			}
			return_value << "\n|---|";
			for (size_t c = 0; c < max_competitors; ++c)
				return_value << "---|";
			return_value << "\n";

			for (const auto& row: rows) {
				return_value << "| " << row.test_name << " |";
				for (size_t g = 0; g < row.groups.size(); ++g) {
					const auto& group = row.groups[g];
					bool is_tied	  = group.size() > 1;
					for (size_t j = 0; j < group.size(); ++j) {
						const auto* current = row.sorted[group[j]];
						return_value << " ";
						if (g == 0 && !is_tied) {
							return_value << "**" << current->library_name << " " << std::fixed << std::setprecision(0) << current->throughput_mb_per_sec << " MB/s**";
							if (row.groups.size() > 1) {
								const auto* next = row.sorted[row.groups[1][0]];
								double pct		 = (next->throughput_mb_per_sec > 1e-9)
										  ? ((current->throughput_mb_per_sec - next->throughput_mb_per_sec) / next->throughput_mb_per_sec) * 100.0
										  : 0.0;
								return_value << " (+" << std::setprecision(1) << pct << "% over " << next->library_name << ")";
							}
						} else {
							return_value << current->library_name << " " << std::fixed << std::setprecision(0) << current->throughput_mb_per_sec << " MB/s";
							if (is_tied) {
								if (group.size() == 2)
									return_value << " `[TIE]`";
								else
									return_value << " `[" << group.size() << "-way TIE]`";
							}
						}
						return_value << " |";
					}
				}
				return_value << "\n";
			}

			if (!file_path.empty()) {
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle file{ file_path + "/" + file_name + "-" + stage_name.operator std::string() + ".md" };
				file.get() = return_value.str();
			}
			return return_value.str();
		}

		BNCH_SWT_HOST static decltype(auto) clear_all_results() {
			stage_results<stage_name, benchmark_type>::results.clear();
		}

		BNCH_SWT_HOST static decltype(auto) get_all_results() {
			return stage_results<stage_name, benchmark_type>::results;
		}

		BNCH_SWT_HOST static decltype(auto) get_test_results(std::string_view test_name) {
			return stage_results_type::get_results_internal(test_name);
		}

		template<string_literal test_name_new, string_literal subject_name_new, typename function_type, internal::not_invocable... arg_types>
		static auto& run_benchmark(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				using return_type = decltype(function_type::impl(std::declval<arg_types>()...));
				static_assert(std::convertible_to<return_type, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}

			measurement_context<stage_name, test_name, subject_name, stage_config_new, benchmark_type, use_non_mbps_metric> ctx{};
			return ctx.template run_adaptive<function_type>(std::forward<arg_types>(args)...);
		}

		template<string_literal test_name_new, string_literal subject_name_new, auto function, internal::not_invocable... arg_types>
		static auto& run_benchmark(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				using return_type = decltype(function(std::declval<arg_types>()...));
				static_assert(std::convertible_to<return_type, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}

			measurement_context<stage_name, test_name, subject_name, stage_config_new, benchmark_type, use_non_mbps_metric> ctx{};
			return ctx.template run_adaptive<function>(std::forward<arg_types>(args)...);
		}

		template<string_literal test_name_new, string_literal subject_name_new, auto function_type, internal::not_invocable... arg_types>
		static auto& run_benchmark_from_host(uint64_t bytes_processed, arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			measurement_context<stage_name, test_name, subject_name, stage_config_new, benchmark_type, use_non_mbps_metric> ctx{};
			return ctx.template run_adaptive_from_host<function_type>(bytes_processed, std::forward<arg_types>(args)...);
		}

		template<string_literal test_name_new, string_literal subject_name_new, typename function_type, internal::not_invocable... arg_types>
		static auto& run_benchmark_from_host(uint64_t bytes_processed, arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			measurement_context<stage_name, test_name, subject_name, stage_config_new, benchmark_type, use_non_mbps_metric> ctx{};
			return ctx.template run_adaptive_from_host<function_type>(bytes_processed, std::forward<arg_types>(args)...);
		}

		template<string_literal test_name_new, string_literal subject_name_new, internal::function_pointer_types auto function, internal::not_invocable... arg_types>
		static auto& run_benchmark_cooperative(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			measurement_context<stage_name, test_name, subject_name, stage_config_new, benchmark_type, use_non_mbps_metric> ctx{};
			return ctx.template run_adaptive_cooperative<function>(std::forward<arg_types>(args)...);
		}
	};

}
