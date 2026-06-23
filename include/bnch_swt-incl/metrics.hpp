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

#include <bnch_swt-incl/event_counter.hpp>
#include <unordered_map>
#include <string_view>
#include <cstdint>
#include <span>

namespace bnch_swt {

	namespace internal {

		template<benchmark_types> BNCH_SWT_HOST std::string get_device_info();

		template<> BNCH_SWT_HOST std::string get_device_info<benchmark_types::cpu>() {
#if defined(__x86_64__) || defined(_M_AMD64)
			std::array<uint32_t, 12> regs{};
			auto cpuid = [](uint32_t leaf, uint32_t* res) {
	#if BNCH_SWT_COMPILER_MSVC
				__cpuidex(std::bit_cast<int*>(res), static_cast<int>(leaf), 0);
	#elif BNCH_SWT_COMPILER_GCC || BNCH_SWT_COMPILER_CLANG
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

#elif BNCH_SWT_PLATFORM_MAC || defined(__FreeBSD__)
			char buffer[256]{};
			size_t size = sizeof(buffer);
			if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
				return std::string(buffer);
			}
			return "Unknown CPU";
#elif BNCH_SWT_PLATFORM_LINUX
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
#elif BNCH_SWT_PLATFORM_WINDOWS
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

		[[nodiscard]] consteval auto get_os_name() {
#if BNCH_SWT_PLATFORM_WINDOWS
			return string_literal{ "Windows" };
#elif BNCH_SWT_PLATFORM_MAC
			return string_literal{ "macOS" };
#elif BNCH_SWT_PLATFORM_LINUX
			return string_literal{ "Linux" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		[[nodiscard]] consteval auto get_compiler_name() {
#if BNCH_SWT_COMPILER_CLANG
			return string_literal{ "Clang" };
#elif BNCH_SWT_COMPILER_GCC
			return string_literal{ "GCC" };
#elif BNCH_SWT_COMPILER_MSVC
			return string_literal{ "MSVC" };
#else
			return string_literal{ "Unknown" };
#endif
		}

		static constexpr bnch_swt::string_literal operating_system_name{ get_os_name() };
		static constexpr bnch_swt::string_literal operating_system_version{ BNCH_SWT_OPERATING_SYSTEM_VERSION };
		static constexpr bnch_swt::string_literal compiler_id{ get_compiler_name() };
		static constexpr bnch_swt::string_literal compiler_version{ BNCH_SWT_COMPILER_VERSION };

	}

	static inline double calculate_throughput_mbps(double nanoseconds, double bytes_processed) {
		constexpr double bytes_per_mb	  = 1024.0 * 1024.0;
		constexpr double nanos_per_second = 1e9;
		double megabytes				  = bytes_processed / bytes_per_mb;
		double seconds					  = nanoseconds / nanos_per_second;
		if (seconds == 0.0) {
			return 0.0;
		}
		return megabytes / seconds;
	}

	static inline double calculate_units_ps(double nanoseconds, double bytes_processed) {
		return (bytes_processed * 1000000000.0) / nanoseconds;
	}

	template<benchmark_types benchmark_type> struct performance_metrics;

	template<string_literal stage_name_new, benchmark_types benchmark_type> struct raw_stage_results {
		using performance_metrics_type = performance_metrics<benchmark_type>;
		inline static std::unordered_map<std::string_view, std::unordered_map<std::string_view, performance_metrics_type>> test_results{};

		BNCH_SWT_HOST static auto& get_test_results(std::string_view test_name) {
			return test_results[test_name];
		}
	};

	template<> struct performance_metrics<benchmark_types::cpu> {
		double throughput_percentage_deviation{ std::numeric_limits<double>::max() };
		double throughput_stability_deviation{ std::numeric_limits<double>::max() };
		std::optional<double> cache_references_per_execution{};
		std::optional<double> branch_misses_per_execution{};
		std::optional<double> instructions_per_execution{};
		std::optional<double> cache_misses_per_execution{};
		std::optional<double> instructions_per_cycle{};
		std::optional<double> branches_per_execution{};
		std::optional<double> instructions_per_byte{};
		std::optional<double> cycles_per_execution{};
		std::optional<double> cycles_per_byte{};
		std::optional<double> frequency_ghz{};
		uint64_t measured_execution_count{};
		uint64_t iterations_to_stabilize{};
		uint64_t total_execution_count{};
		double throughput_mb_per_sec{};
		std::string library_name{};
		uint64_t bytes_processed{};
		std::string cpu_name{};
		double time_in_ns{};

		BNCH_SWT_HOST performance_metrics() {
		}

		BNCH_SWT_HOST bool operator>(const performance_metrics& other) const {
			return throughput_mb_per_sec > other.throughput_mb_per_sec;
		}

		template<string_literal library_name_new, bool mbps = true>
		BNCH_SWT_HOST static performance_metrics collect_metrics(std::span<internal::event_count<benchmark_types::cpu>>&& events_newer, uint64_t iterations_to_stabilize,
			uint64_t total_execution_count, const std::string& cpu_name_new) {
			static constexpr string_literal library_name{ library_name_new };

			if (events_newer.empty()) {
				return {};
			}

			performance_metrics metrics{};
			metrics.library_name			 = library_name.operator std::string();
			metrics.measured_execution_count = events_newer.size();
			metrics.total_execution_count	 = total_execution_count;
			metrics.cpu_name				 = cpu_name_new;
			metrics.iterations_to_stabilize	 = iterations_to_stabilize;

			std::vector<double> throughputs{};
			throughputs.reserve(events_newer.size());

			uint64_t bytes_processed_total{};
			double throughput_total{};
			double ns_total{};
			double cycles_total{};
			double instructions_total{};
			double branches_total{};
			double branch_misses_total{};
			double cache_references_total{};
			double cache_misses_total{};

			for (const auto& e: events_newer) {
				double ns{};
				if (e.elapsed_ns(ns)) {
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
							throughputs.push_back(throughput);
						}
					}
					double value{};
					if (e.cycles(value))
						cycles_total += value;
					if (e.instructions(value))
						instructions_total += value;
					if (e.branches(value))
						branches_total += value;
					if (e.branch_misses(value))
						branch_misses_total += value;
					if (e.cache_references(value))
						cache_references_total += value;
					if (e.cache_misses(value))
						cache_misses_total += value;
				}
			}

			constexpr double epsilon = 1e-6;

			const double inv_n				   = 1.0 / static_cast<double>(events_newer.size());
			const double ns_avg				   = ns_total * inv_n;
			const double cycles_avg			   = cycles_total * inv_n;
			const double instructions_avg	   = instructions_total * inv_n;
			const double branches_avg		   = branches_total * inv_n;
			const double branch_misses_avg	   = branch_misses_total * inv_n;
			const double cache_refs_avg		   = cache_references_total * inv_n;
			const double cache_misses_avg	   = cache_misses_total * inv_n;
			const uint64_t bytes_processed_avg = bytes_processed_total / events_newer.size();

			metrics.time_in_ns = ns_avg;

			const uint64_t valid_throughput_count = throughputs.size();
			const double throughput_avg			  = valid_throughput_count > 0 ? throughput_total / static_cast<double>(valid_throughput_count) : 0.0;

			if (valid_throughput_count > 0 && throughput_avg > epsilon) {
				double variance{};
				double throughput_max{};
				double throughput_min{ std::numeric_limits<double>::max() };

				if (valid_throughput_count > 1) {
					for (double throughput: throughputs) {
						const double diff = throughput - throughput_avg;
						variance += diff * diff;
						throughput_min = std::min(throughput, throughput_min);
						throughput_max = std::max(throughput, throughput_max);
					}
					variance /= static_cast<double>(valid_throughput_count - 1);
				} else {
					throughput_min = throughputs.front();
					throughput_max = throughputs.front();
				}

				const double stddev			= std::sqrt(variance);
				const double standard_error = stddev / std::sqrt(static_cast<double>(valid_throughput_count));

				metrics.bytes_processed					= bytes_processed_avg;
				metrics.throughput_mb_per_sec			= throughput_avg;
				metrics.throughput_percentage_deviation = (standard_error / throughput_avg) * 100.0;
				metrics.throughput_stability_deviation	= ((throughput_max - throughput_min) / throughput_avg) * 100.0;
			}

			if (std::abs(cycles_avg) > epsilon) {
				if (bytes_processed_avg > 0)
					metrics.cycles_per_byte.emplace(cycles_avg / static_cast<double>(bytes_processed_avg));
				metrics.cycles_per_execution.emplace(cycles_avg);
				if (ns_avg > epsilon)
					metrics.frequency_ghz.emplace(cycles_avg / ns_avg);
			}

			if (std::abs(instructions_avg) > epsilon) {
				if (bytes_processed_avg > 0)
					metrics.instructions_per_byte.emplace(instructions_avg / static_cast<double>(bytes_processed_avg));
				if (std::abs(cycles_avg) > epsilon)
					metrics.instructions_per_cycle.emplace(instructions_avg / cycles_avg);
				metrics.instructions_per_execution.emplace(instructions_avg);
			}

			if (std::abs(branches_avg) > epsilon) {
				metrics.branches_per_execution.emplace(branches_avg);
				metrics.branch_misses_per_execution.emplace(branch_misses_avg);
			}

			if (std::abs(cache_misses_avg) > epsilon)
				metrics.cache_misses_per_execution.emplace(cache_misses_avg);

			if (std::abs(cache_refs_avg) > epsilon)
				metrics.cache_references_per_execution.emplace(cache_refs_avg);

			return metrics;
		}
	};

	template<> struct performance_metrics<benchmark_types::cuda> {
		double throughput_percentage_deviation{ std::numeric_limits<double>::max() };
		double throughput_stability_deviation{ std::numeric_limits<double>::max() };
		std::optional<double> cycles_per_execution{};
		std::optional<double> cuda_event_ms_avg{};
		std::optional<double> cycles_per_byte{};
		uint64_t measured_execution_count{};
		uint64_t iterations_to_stabilize{};
		uint64_t total_execution_count{};
		double throughput_mb_per_sec{};
		std::string library_name{};
		uint64_t bytes_processed{};
		std::string gpu_name{};
		double time_in_ns{};

		BNCH_SWT_HOST performance_metrics() {
		}

		BNCH_SWT_HOST bool operator>(const performance_metrics& other) const {
			return throughput_mb_per_sec > other.throughput_mb_per_sec;
		}

		BNCH_SWT_HOST bool operator<(const performance_metrics& other) const {
			return throughput_mb_per_sec < other.throughput_mb_per_sec;
		}

		template<string_literal library_name_new, bool mbps = true>
		BNCH_SWT_HOST static performance_metrics collect_metrics(std::span<internal::event_count<benchmark_types::cuda>>&& events_newer, uint64_t iterations_to_stabilize,
			uint64_t total_execution_count, const std::string& gpu_name_new) {
			static constexpr string_literal library_name{ library_name_new };

			if (events_newer.empty()) {
				return {};
			}

			performance_metrics metrics{};
			metrics.gpu_name				 = gpu_name_new;
			metrics.library_name			 = library_name;
			metrics.measured_execution_count = events_newer.size();
			metrics.total_execution_count	 = total_execution_count;
			metrics.iterations_to_stabilize	 = iterations_to_stabilize;

			std::vector<double> throughputs{};
			throughputs.reserve(events_newer.size());

			uint64_t bytes_processed_total{};
			double throughput_total{};
			double ns_total{};
			double cycles_total{};

			for (const auto& e: events_newer) {
				double ns{};
				if (e.elapsed_ns(ns)) {
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
							throughputs.push_back(throughput);
						}
					}
					double value{};
					if (e.cycles(value))
						cycles_total += value;
				}
			}

			constexpr double epsilon = 1e-6;

			const double inv_n				   = 1.0 / static_cast<double>(events_newer.size());
			const double ns_avg				   = ns_total * inv_n;
			const double cycles_avg			   = cycles_total * inv_n;
			const uint64_t bytes_processed_avg = bytes_processed_total / events_newer.size();

			metrics.time_in_ns = ns_avg;

			const uint64_t valid_throughput_count = throughputs.size();
			const double throughput_avg			  = valid_throughput_count > 0 ? throughput_total / static_cast<double>(valid_throughput_count) : 0.0;

			if (valid_throughput_count > 0 && throughput_avg > epsilon) {
				double variance{};
				double throughput_max{};
				double throughput_min{ std::numeric_limits<double>::max() };

				if (valid_throughput_count > 1) {
					for (double throughput: throughputs) {
						const double diff = throughput - throughput_avg;
						variance += diff * diff;
						throughput_min = std::min(throughput, throughput_min);
						throughput_max = std::max(throughput, throughput_max);
					}
					variance /= static_cast<double>(valid_throughput_count - 1);
				} else {
					throughput_min = throughputs.front();
					throughput_max = throughputs.front();
				}

				const double stddev			= std::sqrt(variance);
				const double standard_error = stddev / std::sqrt(static_cast<double>(valid_throughput_count));

				metrics.bytes_processed					= bytes_processed_avg;
				metrics.throughput_mb_per_sec			= throughput_avg;
				metrics.throughput_percentage_deviation = (standard_error / throughput_avg) * 100.0;
				metrics.throughput_stability_deviation	= ((throughput_max - throughput_min) / throughput_avg) * 100.0;
			}

			if (std::abs(cycles_avg) > epsilon) {
				if (bytes_processed_avg > 0)
					metrics.cycles_per_byte.emplace(cycles_avg / static_cast<double>(bytes_processed_avg));
				metrics.cycles_per_execution.emplace(cycles_avg);
			}

			return metrics;
		}
	};

	template<benchmark_types benchmark_type> struct performance_metrics_presence {};

	template<> struct performance_metrics_presence<benchmark_types::cpu> {
		bool throughput_percentage_deviation{ true };
		bool cache_references_per_execution{ false };
		bool branch_misses_per_execution{ false };
		bool instructions_per_execution{ false };
		bool cache_misses_per_execution{ false };
		bool measured_execution_count{ true };
		bool iterations_to_stabilize{ true };
		bool instructions_per_cycle{ false };
		bool branches_per_execution{ false };
		bool instructions_per_byte{ false };
		bool total_execution_count{ false };
		bool throughput_mb_per_sec{ true };
		bool cycles_per_execution{ false };
		bool bytes_processed{ true };
		bool cycles_per_byte{ false };
		bool frequency_ghz{ false };
		bool time_in_ns{ false };
	};

	template<benchmark_types benchmark_type> constexpr bool has_any_metric_enabled(const performance_metrics_presence<benchmark_type>& m) {
		if constexpr (benchmark_type == benchmark_types::cpu) {
			auto fields = std::tie(m.throughput_percentage_deviation, m.cache_references_per_execution, m.branch_misses_per_execution, m.instructions_per_execution,
				m.cache_misses_per_execution, m.measured_execution_count, m.iterations_to_stabilize, m.instructions_per_cycle, m.branches_per_execution, m.instructions_per_byte,
				m.total_execution_count, m.throughput_mb_per_sec, m.cycles_per_execution, m.bytes_processed, m.cycles_per_byte, m.frequency_ghz, m.time_in_ns);

			return fields != std::make_tuple(false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false);
		} else {
			auto fields = std::tie(m.throughput_percentage_deviation, m.measured_execution_count, m.iterations_to_stabilize, m.total_execution_count, m.throughput_mb_per_sec,
				m.cycles_per_execution, m.cuda_event_ms_avg, m.cycles_per_byte, m.bytes_processed, m.time_in_ns);

			return fields != std::make_tuple(false, false, false, false, false, false, false, false, false, false);
		}
	}

	template<> struct performance_metrics_presence<benchmark_types::cuda> {
		bool throughput_percentage_deviation{ true };
		bool measured_execution_count{ true };
		bool iterations_to_stabilize{ true };
		bool total_execution_count{ true };
		bool throughput_mb_per_sec{ true };
		bool cycles_per_execution{ false };
		bool cuda_event_ms_avg{ false };
		bool cycles_per_byte{ true };
		bool bytes_processed{ true };
		bool time_in_ns{ false };
	};

	enum class position_type : uint8_t { win, tie, loss, none };

	template<benchmark_types benchmark_type> struct library_result : public performance_metrics<benchmark_type> {
		position_type position_type_val{ position_type::none };
		uint64_t position{};
	};

	template<benchmark_types benchmark_type> struct system_info_data {
		inline static constexpr std::string_view compiler_version{ internal::compiler_version };
		inline static constexpr std::string_view compiler_id{ internal::compiler_id };
		inline static constexpr std::string_view os_version{ internal::operating_system_version };
		inline static constexpr std::string_view os_id{ internal::operating_system_name };
		inline static constexpr std::string_view device_type{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };
		static std::string_view device_name() noexcept {
			static const std::string* name{ new std::string{ internal::get_device_info<benchmark_type>() } };
			return *name;
		}
	};

	template<benchmark_types benchmark_type, performance_metrics_presence<benchmark_type> metrics_presence> struct test_results {
		using library_result_type = library_result<benchmark_type>;
		using sys_info			  = system_info_data<benchmark_type>;

		static constexpr string_literal device_type_string{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };

		std::vector<library_result_type> results{};
		std::string stage_name_str{};
		std::string test_name{};

		static constexpr double confidence_multiplier = 1.96;
		static constexpr double epsilon				  = 1e-12;

		template<string_literal stage_name> static test_results build(std::string_view test_name_in) {
			using raw_t = raw_stage_results<stage_name, benchmark_type>;

			test_results tr{};
			tr.stage_name_str = stage_name.operator std::string();
			tr.test_name	  = test_name_in;

			const auto& raw = raw_t::get_test_results(test_name_in);
			if (raw.empty())
				return tr;

			auto ranges_overlap = [](const performance_metrics<benchmark_type>& a, const performance_metrics<benchmark_type>& b) -> bool {
				double a_stddev = std::abs(a.throughput_percentage_deviation) / 100.0;
				double b_stddev = std::abs(b.throughput_percentage_deviation) / 100.0;
				if (a_stddev < epsilon || b_stddev < epsilon)
					return std::abs(a.throughput_mb_per_sec - b.throughput_mb_per_sec) < epsilon;
				double a_min = a.throughput_mb_per_sec * (1.0 - a_stddev * confidence_multiplier);
				double a_max = a.throughput_mb_per_sec * (1.0 + a_stddev * confidence_multiplier);
				double b_min = b.throughput_mb_per_sec * (1.0 - b_stddev * confidence_multiplier);
				double b_max = b.throughput_mb_per_sec * (1.0 + b_stddev * confidence_multiplier);
				return !(a_max + epsilon < b_min || b_max + epsilon < a_min);
			};

			std::vector<const performance_metrics<benchmark_type>*> sorted{};
			for (const auto& [lib_name, metrics]: raw)
				sorted.push_back(&metrics);
			std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
				return a->throughput_mb_per_sec > b->throughput_mb_per_sec;
			});

			std::vector<std::vector<size_t>> groups{};
			std::vector<bool> grouped(sorted.size(), false);
			for (size_t i = 0; i < sorted.size(); ++i) {
				if (grouped[i])
					continue;
				std::vector<size_t> group{};
				group.push_back(i);
				grouped[i] = true;
				for (size_t j = i + 1; j < sorted.size(); ++j) {
					if (grouped[j])
						continue;
					bool overlaps_group = false;
					for (size_t member: group) {
						if (ranges_overlap(*sorted[member], *sorted[j])) {
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

			std::unordered_map<std::string, position_type> pos_map{};
			std::unordered_map<std::string, uint64_t> rank_map{};
			uint64_t rank = 1;
			for (const auto& group: groups) {
				const bool is_tie = group.size() > 1;
				for (size_t idx: group) {
					const std::string lib = sorted[idx]->library_name;
					pos_map[lib]		  = (rank == 1) ? (is_tie ? position_type::tie : position_type::win) : position_type::loss;
					rank_map[lib]		  = rank;
				}
				rank += group.size();
			}

			for (const auto* m: sorted) {
				library_result_type lr{};
				static_cast<performance_metrics<benchmark_type>&>(lr) = *m;
				lr.position_type_val								  = pos_map[m->library_name];
				lr.position											  = rank_map[m->library_name];
				tr.results.push_back(std::move(lr));
			}

			return tr;
		}

		std::string csv_preamble() const {
			std::stringstream ss{};
			ss << "# " << test_name << " Test Results " << "\n ";
			ss << "#**" << device_type_string << ":** " << sys_info::device_name() << "\n";
			ss << "#**OS:** " << sys_info::os_id << "-" << sys_info::os_version << "\n";
			ss << "#**Compiler:** " << sys_info::compiler_id << "-" << sys_info::compiler_version << "\n\n";
			return ss.str();
		}

		static std::string csv_header() {
			std::string h = "Library";
			if constexpr (metrics_presence.throughput_mb_per_sec)
				h += ",Throughput (MB/s)";
			if constexpr (metrics_presence.throughput_percentage_deviation)
				h += ",Throughput Deviation (%)";
			if constexpr (metrics_presence.time_in_ns)
				h += ",Time (ns)";
			if constexpr (metrics_presence.bytes_processed)
				h += ",File Size (Bytes)";
			if constexpr (metrics_presence.measured_execution_count)
				h += ",Measured Executions";
			if constexpr (metrics_presence.iterations_to_stabilize)
				h += ",Iterations to Stabilize";
			if constexpr (metrics_presence.total_execution_count)
				h += ",Total Executions";
			if constexpr (metrics_presence.cycles_per_byte)
				h += ",Cycles/Byte";
			if constexpr (metrics_presence.cycles_per_execution)
				h += ",Cycles/Execution";
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					h += ",Frequency (GHz)";
				if constexpr (metrics_presence.instructions_per_byte)
					h += ",Instructions/Byte";
				if constexpr (metrics_presence.instructions_per_cycle)
					h += ",Instructions/Cycle";
				if constexpr (metrics_presence.instructions_per_execution)
					h += ",Instructions/Execution";
				if constexpr (metrics_presence.branches_per_execution)
					h += ",Branches/Execution";
				if constexpr (metrics_presence.branch_misses_per_execution)
					h += ",Branch Misses/Execution";
				if constexpr (metrics_presence.cache_references_per_execution)
					h += ",Cache References/Execution";
				if constexpr (metrics_presence.cache_misses_per_execution)
					h += ",Cache Misses/Execution";
			}
			h += ",Position";
			return h;
		}

		static std::string result_to_csv_line(const library_result_type& r) {
			std::stringstream ss{};
			ss << r.library_name;
			if constexpr (metrics_presence.throughput_mb_per_sec)
				ss << "," << r.throughput_mb_per_sec;
			if constexpr (metrics_presence.throughput_percentage_deviation)
				ss << "," << r.throughput_percentage_deviation;
			if constexpr (metrics_presence.time_in_ns)
				ss << "," << r.time_in_ns;
			if constexpr (metrics_presence.bytes_processed)
				ss << "," << r.bytes_processed;
			if constexpr (metrics_presence.measured_execution_count)
				ss << "," << r.measured_execution_count;
			if constexpr (metrics_presence.iterations_to_stabilize)
				ss << "," << r.iterations_to_stabilize;
			if constexpr (metrics_presence.total_execution_count)
				ss << "," << r.total_execution_count;
			if constexpr (metrics_presence.cycles_per_byte)
				ss << "," << (r.cycles_per_byte.has_value() ? std::to_string(*r.cycles_per_byte) : "");
			if constexpr (metrics_presence.cycles_per_execution)
				ss << "," << (r.cycles_per_execution.has_value() ? std::to_string(*r.cycles_per_execution) : "");
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					ss << "," << (r.frequency_ghz.has_value() ? std::to_string(*r.frequency_ghz) : "");
				if constexpr (metrics_presence.instructions_per_byte)
					ss << "," << (r.instructions_per_byte.has_value() ? std::to_string(*r.instructions_per_byte) : "");
				if constexpr (metrics_presence.instructions_per_cycle)
					ss << "," << (r.instructions_per_cycle.has_value() ? std::to_string(*r.instructions_per_cycle) : "");
				if constexpr (metrics_presence.instructions_per_execution)
					ss << "," << (r.instructions_per_execution.has_value() ? std::to_string(*r.instructions_per_execution) : "");
				if constexpr (metrics_presence.branches_per_execution)
					ss << "," << (r.branches_per_execution.has_value() ? std::to_string(*r.branches_per_execution) : "");
				if constexpr (metrics_presence.branch_misses_per_execution)
					ss << "," << (r.branch_misses_per_execution.has_value() ? std::to_string(*r.branch_misses_per_execution) : "");
				if constexpr (metrics_presence.cache_references_per_execution)
					ss << "," << (r.cache_references_per_execution.has_value() ? std::to_string(*r.cache_references_per_execution) : "");
				if constexpr (metrics_presence.cache_misses_per_execution)
					ss << "," << (r.cache_misses_per_execution.has_value() ? std::to_string(*r.cache_misses_per_execution) : "");
			}
			switch (r.position_type_val) {
				case position_type::win:
					ss << ",Win";
					break;
				case position_type::tie:
					ss << ",Tie";
					break;
				case position_type::loss:
					ss << ",Loss";
					break;
				default:
					ss << ",";
					break;
			}
			ss << "\n";
			return ss.str();
		}

		std::string to_csv(bool include_preamble = true, const std::string& file_path = "") const {
			std::stringstream ss{};
			if (include_preamble) {
				ss << csv_preamble();
			}
			ss << csv_header() << "\n";
			for (const auto& r: results) {
				ss << result_to_csv_line(r);
			}

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string safe_name = test_name;
				std::replace(safe_name.begin(), safe_name.end(), '.', '_');
				std::replace(safe_name.begin(), safe_name.end(), '/', '_');
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + stage_name_str + ".csv");
			}

			return text;
		}

		std::string md_preamble() const {
			std::stringstream ss{};
			ss << "**" << device_type_string << ":** " << sys_info::device_name() << "  \n";
			ss << "**OS:** " << sys_info::os_id << "-" << sys_info::os_version << "  \n";
			ss << "**Compiler:** " << sys_info::compiler_id << "-" << sys_info::compiler_version << "  \n\n";
			return ss.str();
		}

		static std::string md_header_row() {
			std::string h = "| Library |";
			if constexpr (metrics_presence.throughput_mb_per_sec)
				h += " Throughput (MB/s) |";
			if constexpr (metrics_presence.throughput_percentage_deviation)
				h += " Percentage Deviation (+/-%) |";
			if constexpr (metrics_presence.bytes_processed)
				h += " File Size (Bytes) |";
			if constexpr (metrics_presence.time_in_ns)
				h += " Time (ns) |";
			if constexpr (metrics_presence.measured_execution_count)
				h += " Measured Executions |";
			if constexpr (metrics_presence.iterations_to_stabilize)
				h += " Iterations to Stabilize |";
			if constexpr (metrics_presence.total_execution_count)
				h += " Total Executions |";
			if constexpr (metrics_presence.cycles_per_byte)
				h += " Cycles/Byte |";
			if constexpr (metrics_presence.cycles_per_execution)
				h += " Cycles/Execution |";
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					h += " Frequency (GHz) |";
				if constexpr (metrics_presence.instructions_per_byte)
					h += " Instructions/Byte |";
				if constexpr (metrics_presence.instructions_per_cycle)
					h += " Instructions/Cycle |";
				if constexpr (metrics_presence.instructions_per_execution)
					h += " Instructions/Execution |";
				if constexpr (metrics_presence.branches_per_execution)
					h += " Branches/Execution |";
				if constexpr (metrics_presence.branch_misses_per_execution)
					h += " Branch Misses/Execution |";
				if constexpr (metrics_presence.cache_references_per_execution)
					h += " Cache References/Execution |";
				if constexpr (metrics_presence.cache_misses_per_execution)
					h += " Cache Misses/Execution |";
			}
			return h;
		}

		static std::string md_separator_row() {
			std::string s = "| ------- |";
			if constexpr (metrics_presence.throughput_mb_per_sec)
				s += " ----------- |";
			if constexpr (metrics_presence.throughput_percentage_deviation)
				s += " -------------------- |";
			if constexpr (metrics_presence.bytes_processed)
				s += " --------------- |";
			if constexpr (metrics_presence.time_in_ns)
				s += " --------- |";
			if constexpr (metrics_presence.measured_execution_count)
				s += " ------------------- |";
			if constexpr (metrics_presence.iterations_to_stabilize)
				s += " ----------------------- |";
			if constexpr (metrics_presence.total_execution_count)
				s += " ---------------- |";
			if constexpr (metrics_presence.cycles_per_byte)
				s += " ----------- |";
			if constexpr (metrics_presence.cycles_per_execution)
				s += " ----------------- |";
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					s += " --------------- |";
				if constexpr (metrics_presence.instructions_per_byte)
					s += " ----------------- |";
				if constexpr (metrics_presence.instructions_per_cycle)
					s += " ------------------ |";
				if constexpr (metrics_presence.instructions_per_execution)
					s += " ---------------------- |";
				if constexpr (metrics_presence.branches_per_execution)
					s += " ------------------- |";
				if constexpr (metrics_presence.branch_misses_per_execution)
					s += " ----------------------- |";
				if constexpr (metrics_presence.cache_references_per_execution)
					s += " --------------------------- |";
				if constexpr (metrics_presence.cache_misses_per_execution)
					s += " -------------------- |";
			}
			return s;
		}

		static std::string result_to_md_row(const library_result_type& r) {
			std::string lib_cell = r.library_name;
			if (r.position_type_val == position_type::tie)
				lib_cell += " **STATISTICAL TIE**";

			std::stringstream ss{};
			ss << "| " << lib_cell << " |";
			if constexpr (metrics_presence.throughput_mb_per_sec)
				ss << " " << r.throughput_mb_per_sec << " |";
			if constexpr (metrics_presence.throughput_percentage_deviation)
				ss << " " << r.throughput_percentage_deviation << " |";
			if constexpr (metrics_presence.bytes_processed)
				ss << " " << r.bytes_processed << " |";
			if constexpr (metrics_presence.time_in_ns)
				ss << " " << r.time_in_ns << " |";
			if constexpr (metrics_presence.measured_execution_count)
				ss << " " << r.measured_execution_count << " |";
			if constexpr (metrics_presence.iterations_to_stabilize)
				ss << " " << r.iterations_to_stabilize << " |";
			if constexpr (metrics_presence.total_execution_count)
				ss << " " << r.total_execution_count << " |";
			if constexpr (metrics_presence.cycles_per_byte)
				ss << " " << (r.cycles_per_byte.has_value() ? std::to_string(*r.cycles_per_byte) : "") << " |";
			if constexpr (metrics_presence.cycles_per_execution)
				ss << " " << (r.cycles_per_execution.has_value() ? std::to_string(*r.cycles_per_execution) : "") << " |";
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					ss << " " << (r.frequency_ghz.has_value() ? std::to_string(*r.frequency_ghz) : "") << " |";
				if constexpr (metrics_presence.instructions_per_byte)
					ss << " " << (r.instructions_per_byte.has_value() ? std::to_string(*r.instructions_per_byte) : "") << " |";
				if constexpr (metrics_presence.instructions_per_cycle)
					ss << " " << (r.instructions_per_cycle.has_value() ? std::to_string(*r.instructions_per_cycle) : "") << " |";
				if constexpr (metrics_presence.instructions_per_execution)
					ss << " " << (r.instructions_per_execution.has_value() ? std::to_string(*r.instructions_per_execution) : "") << " |";
				if constexpr (metrics_presence.branches_per_execution)
					ss << " " << (r.branches_per_execution.has_value() ? std::to_string(*r.branches_per_execution) : "") << " |";
				if constexpr (metrics_presence.branch_misses_per_execution)
					ss << " " << (r.branch_misses_per_execution.has_value() ? std::to_string(*r.branch_misses_per_execution) : "") << " |";
				if constexpr (metrics_presence.cache_references_per_execution)
					ss << " " << (r.cache_references_per_execution.has_value() ? std::to_string(*r.cache_references_per_execution) : "") << " |";
				if constexpr (metrics_presence.cache_misses_per_execution)
					ss << " " << (r.cache_misses_per_execution.has_value() ? std::to_string(*r.cache_misses_per_execution) : "") << " |";
			}
			ss << "\n";
			return ss.str();
		}

		std::string to_markdown(bool include_preamble = true, bool include_test_title = true, const std::string& file_path = "") const {
			std::stringstream ss{};
			if (include_test_title) {
				ss << "### " << test_name << " Test Results\n";
			}
			if (include_preamble) {
				ss << md_preamble();
			}
			ss << md_header_row() << "\n";
			ss << md_separator_row() << "\n";
			for (const auto& r: results) {
				ss << result_to_md_row(r);
			}

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string safe_name = test_name;
				std::replace(safe_name.begin(), safe_name.end(), '.', '_');
				std::replace(safe_name.begin(), safe_name.end(), '/', '_');
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + stage_name_str + ".md");
			}

			return text;
		}

		void print(bool include_preamble = true) const {
			std::cout << to_markdown(include_preamble);
		}
	};

	template<benchmark_types benchmark_type, performance_metrics_presence<benchmark_type> metrics_presence> struct stage_results {
		using test_results_type	  = test_results<benchmark_type, metrics_presence>;
		using library_result_type = library_result<benchmark_type>;

		static constexpr string_literal device_type_string{ benchmark_type == benchmark_types::cpu ? "CPU" : "GPU" };

		using sys_info = system_info_data<benchmark_type>;
		std::vector<test_results_type> tests{};
		std::string stage_name_str{};

		std::unordered_map<std::string, uint32_t> total_wins{};
		std::unordered_map<std::string, uint32_t> total_ties{};
		std::unordered_map<std::string, uint32_t> total_losses{};

		struct accum {
			uint64_t count{};
			double throughput_mb_per_sec{};
			double throughput_percentage_deviation{};
			double time_in_ns{};
			double bytes_processed{};
			double measured_execution_count{};
			double iterations_to_stabilize{};
			double total_execution_count{};
			double cycles_per_byte{};
			uint64_t cycles_per_byte_n{};
			double cycles_per_execution{};
			uint64_t cycles_per_execution_n{};
			double frequency_ghz{};
			uint64_t frequency_ghz_n{};
			double instructions_per_byte{};
			uint64_t instructions_per_byte_n{};
			double instructions_per_cycle{};
			uint64_t instructions_per_cycle_n{};
			double instructions_per_execution{};
			uint64_t instructions_per_execution_n{};
			double branches_per_execution{};
			uint64_t branches_per_execution_n{};
			double branch_misses_per_execution{};
			uint64_t branch_misses_per_execution_n{};
			double cache_references_per_execution{};
			uint64_t cache_references_per_execution_n{};
			double cache_misses_per_execution{};
			uint64_t cache_misses_per_execution_n{};

			void accumulate(const library_result<benchmark_type>& r) {
				++count;
				if constexpr (metrics_presence.throughput_mb_per_sec)
					throughput_mb_per_sec += r.throughput_mb_per_sec;
				if constexpr (metrics_presence.throughput_percentage_deviation)
					throughput_percentage_deviation += r.throughput_percentage_deviation;
				if constexpr (metrics_presence.time_in_ns)
					time_in_ns += r.time_in_ns;
				if constexpr (metrics_presence.bytes_processed)
					bytes_processed += static_cast<double>(r.bytes_processed);
				if constexpr (metrics_presence.measured_execution_count)
					measured_execution_count += static_cast<double>(r.measured_execution_count);
				if constexpr (metrics_presence.iterations_to_stabilize)
					iterations_to_stabilize += static_cast<double>(r.iterations_to_stabilize);
				if constexpr (metrics_presence.total_execution_count)
					total_execution_count += static_cast<double>(r.total_execution_count);
				if constexpr (metrics_presence.cycles_per_byte)
					if (r.cycles_per_byte.has_value()) {
						cycles_per_byte += *r.cycles_per_byte;
						++cycles_per_byte_n;
					}
				if constexpr (metrics_presence.cycles_per_execution)
					if (r.cycles_per_execution.has_value()) {
						cycles_per_execution += *r.cycles_per_execution;
						++cycles_per_execution_n;
					}
				if constexpr (benchmark_type == benchmark_types::cpu) {
					if constexpr (metrics_presence.frequency_ghz)
						if (r.frequency_ghz.has_value()) {
							frequency_ghz += *r.frequency_ghz;
							++frequency_ghz_n;
						}
					if constexpr (metrics_presence.instructions_per_byte)
						if (r.instructions_per_byte.has_value()) {
							instructions_per_byte += *r.instructions_per_byte;
							++instructions_per_byte_n;
						}
					if constexpr (metrics_presence.instructions_per_cycle)
						if (r.instructions_per_cycle.has_value()) {
							instructions_per_cycle += *r.instructions_per_cycle;
							++instructions_per_cycle_n;
						}
					if constexpr (metrics_presence.instructions_per_execution)
						if (r.instructions_per_execution.has_value()) {
							instructions_per_execution += *r.instructions_per_execution;
							++instructions_per_execution_n;
						}
					if constexpr (metrics_presence.branches_per_execution)
						if (r.branches_per_execution.has_value()) {
							branches_per_execution += *r.branches_per_execution;
							++branches_per_execution_n;
						}
					if constexpr (metrics_presence.branch_misses_per_execution)
						if (r.branch_misses_per_execution.has_value()) {
							branch_misses_per_execution += *r.branch_misses_per_execution;
							++branch_misses_per_execution_n;
						}
					if constexpr (metrics_presence.cache_references_per_execution)
						if (r.cache_references_per_execution.has_value()) {
							cache_references_per_execution += *r.cache_references_per_execution;
							++cache_references_per_execution_n;
						}
					if constexpr (metrics_presence.cache_misses_per_execution)
						if (r.cache_misses_per_execution.has_value()) {
							cache_misses_per_execution += *r.cache_misses_per_execution;
							++cache_misses_per_execution_n;
						}
				}
			}
		};

		std::unordered_map<std::string, accum> accumulators{};

		template<string_literal stage_name> static stage_results build() {
			using raw_t = raw_stage_results<stage_name, benchmark_type>;

			stage_results sr{};
			sr.stage_name_str = stage_name.operator std::string();

			for (const auto& [test_name_sv, _]: raw_t::test_results) {
				auto tr = test_results_type::template build<stage_name>(static_cast<std::string>(test_name_sv));

				for (const auto& r: tr.results) {
					switch (r.position_type_val) {
						case position_type::win:
							sr.total_wins[r.library_name]++;
							break;
						case position_type::tie:
							sr.total_ties[r.library_name]++;
							break;
						case position_type::loss:
							sr.total_losses[r.library_name]++;
							break;
						default:
							break;
					}
					sr.accumulators[r.library_name].accumulate(r);
				}

				sr.tests.push_back(std::move(tr));
			}

			return sr;
		}

		std::vector<std::string> sorted_libs() const {
			std::vector<std::string> libs{};
			for (const auto& [lib, _]: accumulators)
				libs.push_back(lib);
			std::sort(libs.begin(), libs.end(), [&](const std::string& a, const std::string& b) {
				uint32_t a_wins = total_wins.count(a) ? total_wins.at(a) : 0;
				uint32_t b_wins = total_wins.count(b) ? total_wins.at(b) : 0;
				return a_wins > b_wins;
			});
			return libs;
		}

		std::string csv_preamble() const {
			std::stringstream ss{};
			ss << "# " << stage_name_str << " Stage Results" << "\n";
			ss << "#**" << device_type_string << ":** " << sys_info::device_name() << "\n";
			ss << "#**OS:** " << sys_info::os_id << "-" << sys_info::os_version << "\n";
			ss << "#**Compiler:** " << sys_info::compiler_id << "-" << sys_info::compiler_version << "\n\n";
			return ss.str();
		}

		std::string md_preamble() const {
			std::stringstream ss{};
			ss << stage_name_str << " Stage Results" << "\n";
			ss << "**" << device_type_string << ":** " << sys_info::device_name() << "  \n";
			ss << "**OS:** " << sys_info::os_id << "-" << sys_info::os_version << "  \n";
			ss << "**Compiler:** " << sys_info::compiler_id << "-" << sys_info::compiler_version << "  \n\n";
			return ss.str();
		}

		static std::string stage_csv_header() {
			std::string h = "Library";
			if constexpr (metrics_presence.throughput_mb_per_sec)
				h += ",Average Throughput (MB/s)";
			if constexpr (metrics_presence.throughput_percentage_deviation)
				h += ",Average Throughput Deviation (%)";
			if constexpr (metrics_presence.time_in_ns)
				h += ",Average Time (ns)";
			if constexpr (metrics_presence.bytes_processed)
				h += ",Average File Size (Bytes)";
			if constexpr (metrics_presence.measured_execution_count)
				h += ",Average Measured Executions";
			if constexpr (metrics_presence.iterations_to_stabilize)
				h += ",Average Iterations to Stabilize";
			if constexpr (metrics_presence.total_execution_count)
				h += ",Average Total Executions";
			if constexpr (metrics_presence.cycles_per_byte)
				h += ",Average Cycles/Byte";
			if constexpr (metrics_presence.cycles_per_execution)
				h += ",Average Cycles/Execution";
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					h += ",Average Frequency (GHz)";
				if constexpr (metrics_presence.instructions_per_byte)
					h += ",Average Instructions/Byte";
				if constexpr (metrics_presence.instructions_per_cycle)
					h += ",Average Instructions/Cycle";
				if constexpr (metrics_presence.instructions_per_execution)
					h += ",Average Instructions/Execution";
				if constexpr (metrics_presence.branches_per_execution)
					h += ",Average Branches/Execution";
				if constexpr (metrics_presence.branch_misses_per_execution)
					h += ",Average Branch Misses/Execution";
				if constexpr (metrics_presence.cache_references_per_execution)
					h += ",Average Cache References/Execution";
				if constexpr (metrics_presence.cache_misses_per_execution)
					h += ",Average Cache Misses/Execution";
			}
			h += ",Wins,Ties,Losses";
			return h;
		}

		std::string accum_to_csv_line(const std::string& lib) const {
			const auto& a = accumulators.at(lib);
			if (a.count == 0)
				return {};
			const double inv = 1.0 / static_cast<double>(a.count);

			std::stringstream ss{};
			ss << lib;
			if constexpr (metrics_presence.throughput_mb_per_sec)
				ss << "," << (a.throughput_mb_per_sec * inv);
			if constexpr (metrics_presence.throughput_percentage_deviation)
				ss << "," << (a.throughput_percentage_deviation * inv);
			if constexpr (metrics_presence.time_in_ns)
				ss << "," << (a.time_in_ns * inv);
			if constexpr (metrics_presence.bytes_processed)
				ss << "," << (a.bytes_processed * inv);
			if constexpr (metrics_presence.measured_execution_count)
				ss << "," << (a.measured_execution_count * inv);
			if constexpr (metrics_presence.iterations_to_stabilize)
				ss << "," << (a.iterations_to_stabilize * inv);
			if constexpr (metrics_presence.total_execution_count)
				ss << "," << (a.total_execution_count * inv);
			if constexpr (metrics_presence.cycles_per_byte)
				ss << "," << (a.cycles_per_byte_n > 0 ? std::to_string(a.cycles_per_byte / static_cast<double>(a.cycles_per_byte_n)) : "");
			if constexpr (metrics_presence.cycles_per_execution)
				ss << "," << (a.cycles_per_execution_n > 0 ? std::to_string(a.cycles_per_execution / static_cast<double>(a.cycles_per_execution_n)) : "");
			if constexpr (benchmark_type == benchmark_types::cpu) {
				if constexpr (metrics_presence.frequency_ghz)
					ss << "," << (a.frequency_ghz_n > 0 ? std::to_string(a.frequency_ghz / static_cast<double>(a.frequency_ghz_n)) : "");
				if constexpr (metrics_presence.instructions_per_byte)
					ss << "," << (a.instructions_per_byte_n > 0 ? std::to_string(a.instructions_per_byte / static_cast<double>(a.instructions_per_byte_n)) : "");
				if constexpr (metrics_presence.instructions_per_cycle)
					ss << "," << (a.instructions_per_cycle_n > 0 ? std::to_string(a.instructions_per_cycle / static_cast<double>(a.instructions_per_cycle_n)) : "");
				if constexpr (metrics_presence.instructions_per_execution)
					ss << "," << (a.instructions_per_execution_n > 0 ? std::to_string(a.instructions_per_execution / static_cast<double>(a.instructions_per_execution_n)) : "");
				if constexpr (metrics_presence.branches_per_execution)
					ss << "," << (a.branches_per_execution_n > 0 ? std::to_string(a.branches_per_execution / static_cast<double>(a.branches_per_execution_n)) : "");
				if constexpr (metrics_presence.branch_misses_per_execution)
					ss << "," << (a.branch_misses_per_execution_n > 0 ? std::to_string(a.branch_misses_per_execution / static_cast<double>(a.branch_misses_per_execution_n)) : "");
				if constexpr (metrics_presence.cache_references_per_execution)
					ss << ","
					   << (a.cache_references_per_execution_n > 0 ? std::to_string(a.cache_references_per_execution / static_cast<double>(a.cache_references_per_execution_n))
																  : "");
				if constexpr (metrics_presence.cache_misses_per_execution)
					ss << "," << (a.cache_misses_per_execution_n > 0 ? std::to_string(a.cache_misses_per_execution / static_cast<double>(a.cache_misses_per_execution_n)) : "");
			}
			uint32_t wins	= total_wins.count(lib) ? total_wins.at(lib) : 0;
			uint32_t ties	= total_ties.count(lib) ? total_ties.at(lib) : 0;
			uint32_t losses = total_losses.count(lib) ? total_losses.at(lib) : 0;
			ss << "," << wins << "," << ties << "," << losses << "\n";
			return ss.str();
		}

		std::string to_csv(const std::string& file_path = "") const {
			std::stringstream ss{};
			ss << csv_preamble();
			ss << stage_csv_header() << "\n";
			for (const auto& lib: sorted_libs())
				ss << accum_to_csv_line(lib);

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + stage_name_str + ".csv");
			}

			return text;
		}

		std::string to_markdown(const std::string& file_path = "") const {
			std::stringstream ss{};
			ss << md_preamble();
			for (const auto& tr: tests)
				ss << tr.to_markdown(false) << "\n";

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + stage_name_str + ".md");
			}

			return text;
		}

		void print() const {
			std::cout << to_markdown();
		}
	};

}
