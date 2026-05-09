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
#include <iostream>

namespace bnch_swt {

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
				__cpuidex(reinterpret_cast<int*>(res), static_cast<int>(leaf), 0);
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
				if (RegQueryValueExA(hkey, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(buffer), &size) == ERROR_SUCCESS) {
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

		template<auto stage_name, benchmark_types benchmark_type> struct results_holder {
		};

		template<string_literal stage_name_new, string_literal test_name_new, string_literal subject_name, uint64_t max_execution_count, uint64_t measured_iteration_count,
			benchmark_types benchmark_type, bool use_non_mbps_metric>
		struct measurement_context {
			using stage_results_type =  stage_results<stage_name_new,benchmark_type>;
			static constexpr string_literal stage_name{ stage_name_new };
			static constexpr string_literal test_name{ test_name_new };
			internal::event_collector<max_execution_count, benchmark_type> events{};
			internal::cache_clearer<benchmark_type> cache_clearer{};

			BNCH_SWT_HOST auto& finalize() {
				performance_metrics<benchmark_type> lowest_results{};
				performance_metrics<benchmark_type> results_temp{};
				std::span<internal::event_count<benchmark_type>> new_ptr{ static_cast<std::vector<internal::event_count<benchmark_type>>&>(events) };
				static constexpr uint64_t final_measured_iteration_count{ max_execution_count - measured_iteration_count > 0 ? max_execution_count - measured_iteration_count : 1 };
				uint64_t current_global_index{ measured_iteration_count };
				for (uint64_t x = 0; x < final_measured_iteration_count; ++x, ++current_global_index) {
					results_temp = performance_metrics<benchmark_type>::template collect_metrics<stage_name, subject_name, use_non_mbps_metric>(
						new_ptr.subspan(x, measured_iteration_count), current_global_index, max_execution_count);
					lowest_results = results_temp.throughput_percentage_deviation < lowest_results.throughput_percentage_deviation ? results_temp : lowest_results;
				}
				stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()] = lowest_results;
				return stage_results_type::get_results_internal(test_name.operator std::string_view())[subject_name.operator std::string_view()];
			}
		};

	}

	struct library_win_count {
		uint64_t win_count{};
		std::string name{};

		BNCH_SWT_HOST bool operator>(const library_win_count& other) const {
			return win_count > other.win_count;
		}
	};

	template<benchmark_types benchmark_type, auto, auto, performance_metrics_presence<benchmark_type> metrics_presence> struct result_printer;

	template<bool printed, typename value_type> BNCH_SWT_HOST static auto print_metric(std::string_view label, const value_type& value_new) {
		static constexpr uint64_t label_width = 60;
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
		BNCH_SWT_HOST static void print_result(const performance_metrics<benchmark_types::cpu>& value, bool show_comparison = true, bool show_metrics = true) {
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
		BNCH_SWT_HOST static void impl(const std::vector<performance_metrics<benchmark_types::cuda>>& results_new, bool show_comparison = true, bool show_metrics = true) {
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

				for (const auto& value: results_new) {
					print_metric<has_any_metric_enabled(metrics_presence)>("Metrics for: ", value.library_name);
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

			if (show_comparison && results_new.size() > 1) {
				for (uint64_t x = 0; x < results_new.size() - 1; ++x) {
					double difference = ((results_new[x].throughput_mb_per_sec - results_new[x + 1].throughput_mb_per_sec) / results_new[x + 1].throughput_mb_per_sec) * 100.0;

					std::cout << "Kernel " << results_new[x].library_name << " is faster than kernel " << results_new[x + 1].library_name << " by " << difference << "%."
							  << std::endl;
				}
			}
		}
	};

	template<string_literal stage_name_new, uint64_t max_execution_count = 200, uint64_t measured_iteration_count = 25, benchmark_types benchmark_type = benchmark_types::cpu,
		bool clear_cpu_cache_between_each_iteration = false, string_literal metric_name_new = string_literal<1>{}>
	struct benchmark_stage {
		using stage_results_type =  stage_results<stage_name_new,benchmark_type>;
		static constexpr string_literal stage_name{ stage_name_new };
		static constexpr string_literal device_type{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };
		static_assert(max_execution_count % measured_iteration_count == 0, "Sorry, but please enter a max_execution_count that is divisible by measured_iteration_count.");

		static constexpr bool use_non_mbps_metric{ metric_name_new.size() == 0 };

		template<performance_metrics_presence<benchmark_type> metrics_presence = performance_metrics_presence<benchmark_type>{}>
		BNCH_SWT_HOST static void print_results(bool show_comparison = true, bool show_metrics = true) {
			std::cout << "----------------------------------------" << std::endl;
			std::cout << device_type << " Performance Metrics for Stage: " << stage_name_new << std::endl;
			std::cout << "Running on: " << internal::get_device_info<benchmark_type>() << std::endl;
			std::cout << "OS: " << internal::operating_system_name << " " << internal::operating_system_version << std::endl;
			std::cout << "Compiler: " << internal::compiler_id << " " << internal::compiler_version << std::endl;
			using stage_t							  = bnch_swt::stage_results<stage_name, benchmark_type>;
			static constexpr double tie_threshold_pct = 1.0;
			std::unordered_map<std::string, uint32_t> total_wins;
			std::unordered_map<std::string, uint32_t> total_ties;
			for (const auto& library_results: stage_t::results) {
				if (library_results.results.empty()) {
					continue;
				}
				std::vector<const typename stage_t::performance_metrics_type*> sorted;
				for (const auto& [lib_name, metrics]: library_results.results) {
					sorted.push_back(&metrics);
				}
				std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
					return *a > *b;
				});
				std::vector<std::vector<size_t>> groups;
				size_t i = 0;
				while (i < sorted.size()) {
					double ref = sorted[i]->throughput_mb_per_sec;
					std::vector<size_t> group;
					group.push_back(i);
					while (i + 1 < sorted.size()) {
						double diff_pct = (ref > 1e-9) ? ((ref - sorted[i + 1]->throughput_mb_per_sec) / ref) * 100.0 : 0.0;
						if (diff_pct <= tie_threshold_pct) {
							++i;
							group.push_back(i);
						} else {
							break;
						}
					}
					groups.push_back(std::move(group));
					++i;
				}
				if (groups[0].size() > 1) {
					for (size_t idx: groups[0]) {
						total_ties[sorted[idx]->library_name]++;
					}
				} else {
					total_wins[sorted[groups[0][0]]->library_name]++;
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
						std::cout << rank << ". " << current->library_name << " (" << current->throughput_mb_per_sec << " MB/s)";
						if (is_tied) {
							std::string tied_with;
							for (size_t k = 0; k < group.size(); ++k) {
								if (k == j)
									continue;
								if (!tied_with.empty())
									tied_with += ", ";
								tied_with += sorted[group[k]]->library_name;
							}
							std::cout << " | TIED with " << tied_with;
						}
						if (g + 1 < groups.size()) {
							const auto* next  = sorted[groups[g + 1][0]];
							double diff		  = current->throughput_mb_per_sec - next->throughput_mb_per_sec;
							double pct_faster = (next->throughput_mb_per_sec > 1e-9) ? (diff / next->throughput_mb_per_sec) * 100.0 : 0.0;
							std::cout << " | " << pct_faster << "% faster than " << next->library_name;
						}

						static constexpr string_literal stage_name_newer{ stage_name_new };
						static constexpr string_literal metric_name_newer{ metric_name_new };
						std::cout << std::endl;
						result_printer<benchmark_type, stage_name_newer, metric_name_newer, metrics_presence>::print_result(*current, show_comparison, show_metrics);
					}
					rank += group.size();
				}
			}
			std::vector<library_win_count> total_wins_vector;
			for (const auto& [lib, wins]: total_wins) {
				total_wins_vector.emplace_back(library_win_count{ .win_count = wins, .name = lib });
			}
			std::sort(total_wins_vector.begin(), total_wins_vector.end(), std::greater<library_win_count>{});
			std::cout << "=== TOTAL WINS FOR " << stage_name.operator std::string() << " ===\n";
			for (const auto& value: total_wins_vector) {
				std::cout << value.name << ": " << value.win_count << std::endl;
			}
			if (!total_ties.empty()) {
				std::cout << "=== TOTAL TIES ===\n";
				for (const auto& [lib, ties]: total_ties) {
					std::cout << lib << ": " << ties << std::endl;
				}
			}
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
				if constexpr (sizeof...(args) > 0) {
					static_assert(std::convertible_to<std::invoke_result_t<decltype(function_type::template impl<arg_types...>), arg_types...>, uint64_t>,
						"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
				} else {
					static_assert(std::convertible_to<std::invoke_result_t<decltype(function_type::impl), arg_types...>, uint64_t>,
						"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
				}
			}
			internal::measurement_context<stage_name, test_name, subject_name, max_execution_count, measured_iteration_count, benchmark_type, use_non_mbps_metric> ctx{};
			for (uint64_t x = 0; x < max_execution_count; ++x) {
				if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
					ctx.cache_clearer.evict_caches();
				}
				ctx.events.template run<function_type>(std::forward<arg_types>(args)...);
			}
			return ctx.finalize();
		}

		template<string_literal test_name_new, string_literal subject_name_new, auto function, internal::not_invocable... arg_types>
		static auto& run_benchmark(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				static_assert(std::convertible_to<std::invoke_result_t<decltype(function), arg_types...>, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}
			internal::measurement_context<stage_name, test_name, subject_name, max_execution_count, measured_iteration_count, benchmark_type, use_non_mbps_metric> ctx{};
			for (uint64_t x = 0; x < max_execution_count; ++x) {
				if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
					ctx.cache_clearer.evict_caches();
				}
				ctx.events.template run<function>(std::forward<arg_types>(args)...);
			}
			return ctx.finalize();
		}

		template<string_literal test_name_new, string_literal subject_name_new, typename function, typename... arg_types>
		static auto& run_from_host(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				static_assert(std::convertible_to<std::invoke_result_t<function, arg_types...>, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}
			internal::measurement_context<stage_name, test_name, subject_name, max_execution_count, measured_iteration_count, benchmark_type, use_non_mbps_metric> ctx{};
			for (uint64_t x = 0; x < max_execution_count; ++x) {
				if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
					ctx.cache_clearer.evict_caches();
				}
				ctx.events.template run<function>(std::forward<arg_types>(args)...);
			}
			return ctx.finalize();
		}

		template<string_literal test_name_new, string_literal subject_name_new, auto function, internal::not_invocable... arg_types>
		static performance_metrics<benchmark_type> run_benchmark_cooperative(arg_types&&... args) {
			static constexpr string_literal subject_name{ subject_name_new };
			static constexpr string_literal test_name{ test_name_new };
			if constexpr (benchmark_type == benchmark_types::cpu) {
				static_assert(std::convertible_to<std::invoke_result_t<decltype(function), arg_types...>, uint64_t>,
					"Sorry, but the lambda passed to run_benchmark() must return a uint64_t, reflecting the number of bytes processed!");
			}
			internal::measurement_context<stage_name, test_name, subject_name, max_execution_count, measured_iteration_count, benchmark_type, use_non_mbps_metric> ctx{};
			for (uint64_t x = 0; x < max_execution_count; ++x) {
				if constexpr (clear_cpu_cache_between_each_iteration && benchmark_type == benchmark_types::cpu) {
					ctx.cache_clearer.evict_caches();
				}
				ctx.events.template run<function>(std::forward<arg_types>(args)...);
			}
			return ctx.finalize();
		}
	};

}
