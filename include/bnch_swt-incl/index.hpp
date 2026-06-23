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

#include <bnch_swt-incl/config.hpp>
#include <bnch_swt-incl/benchmarksuite_cpu_properties.hpp>
#include <bnch_swt-incl/benchmarksuite_gpu_properties.hpp>
#include <bnch_swt-incl/random_generators.hpp>
#include <bnch_swt-incl/do_not_optimize.hpp>
#include <bnch_swt-incl/thread_affinity.hpp>
#include <bnch_swt-incl/string_literal.hpp>
#include <bnch_swt-incl/event_counter.hpp>
#include <bnch_swt-incl/cache_clearer.hpp>
#include <bnch_swt-incl/file_handle.hpp>
#include <bnch_swt-incl/printable.hpp>
#include <bnch_swt-incl/metrics.hpp>
#include <bnch_swt-incl/utility.hpp>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <numeric>

namespace bnch_swt {

	struct epoch_state_data {
		uint64_t total_iterations_run;
		uint64_t target_count;
		double start_time_ns;
	};

	struct stage_config_data {
		bool clear_cpu_caches_before_iterations{ true };
		uint64_t measured_iteration_count{ 100 };
		uint64_t max_iteration_count{ 1000 };
		double convergence_threshold{ 1.0 };
		benchmark_types benchmark_type{};
		uint64_t max_time_in_s{ 5 };
		double rse_threshold{ 2.5 };
		uint64_t max_k{ 100000 };
		uint64_t min_k{ 30 };
	};

	struct printable_milliseconds {
		double value;
		static constexpr const char* units{ "ms" };
		printable_milliseconds(double ns_value = 0) : value{ static_cast<double>(ns_value) / 1e6 } {
		}

		inline friend std::ostream& operator<<(std::ostream& os BNCH_SWT_LIFETIME_BOUND, const printable_milliseconds& ms) {
			os << ms.value << units;
			return os;
		}
	};

	struct library_benchmark_data {
		printable_milliseconds final_ms_spent;
		uint64_t final_sample_size;
		double final_rse{ 100.0 };
		uint64_t bytes_processed;
		double final_throughput;
		double final_variance;
		double final_mean;
		bool converged;
	};

	struct library_completion_data : public library_benchmark_data {
		bnch_swt::position_type position_type_val{ bnch_swt::position_type::none };
		std::string_view name;
		uint64_t position;
		bool operator>(const library_completion_data& rhs) const {
			return position > rhs.position;
		}
	};

	inline static library_benchmark_data calculate_epoch_stats(const iteration_metrics* iteration_data, uint64_t iterations, uint64_t k) {
		if (k == 0) {
			return {};
		}

		uint64_t start_idx = iterations - k;

		double time_sum		 = 0.0;
		uint64_t total_bytes = 0;
		for (uint64_t i = start_idx; i < iterations; ++i) {
			time_sum += iteration_data[i].time_in_ns;
			total_bytes += iteration_data[i].bytes_processed;
		}

		double mean = time_sum / static_cast<double>(k);

		double variance_sum = 0.0;
		for (uint64_t i = start_idx; i < iterations; ++i) {
			double diff = iteration_data[i].time_in_ns - mean;
			variance_sum += diff * diff;
		}

		double variance = 0.0;
		if (k > 1) {
			variance = variance_sum / static_cast<double>(k - 1);
		}

		double std_dev	 = std::sqrt(variance);
		double std_error = std_dev / std::sqrt(static_cast<double>(k));
		double rse		 = (mean > 0.0) ? (std_error / mean) * 100.0 : 0.0;

		double total_time_seconds = time_sum / 1'000'000'000.0;
		double total_megabytes	  = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
		double throughput		  = (total_time_seconds > 0.0) ? (total_megabytes / total_time_seconds) : 0.0;

		library_benchmark_data results{};
		results.final_sample_size = k;
		results.final_mean		  = mean;
		results.final_variance	  = variance;
		results.final_rse		  = rse;
		results.final_throughput  = throughput;
		results.converged		  = false;
		results.bytes_processed	  = total_bytes / k;

		return results;
	}

	template<auto function> struct function_holder {
		template<typename... arg_types> static uint64_t impl(arg_types&&... args) {
			return function(std::forward<arg_types>(args)...);
		}
	};

	template<typename functor_type, stage_config_data stage_config, typename... arg_types>
	[[maybe_unused]] BNCH_SWT_NOINLINE static library_benchmark_data impl(uint64_t iterations, arg_types&&... args) {
		static_assert(stage_config.max_iteration_count > 0, "MAX ITERATIONS MUST BE GREATER THAN ZERO TO ALLOCATE THE BUFFER");
		static_assert(stage_config.measured_iteration_count > 0, "MEASURED ITERATION COUNT MUST START AT GREATER THAN ZERO");
		static_assert(stage_config.measured_iteration_count <= stage_config.max_iteration_count, "INITIAL WORKLOAD STEPS CANNOT EXCEED THE ABSOLUTE MAXIMUM BUFFER LIMIT");
		static_assert(stage_config.min_k <= stage_config.max_k, "MINIMUM WINDOW SIZE FLOOR (MIN_K) CANNOT BE GREATER THAN THE CEILING (MAX_K)");
		static_assert(stage_config.min_k <= stage_config.max_iteration_count, "STATISTICAL WINDOW FLOOR CANNOT BE LARGER THAN THE TOTAL ALLOCATED MEMORY BUFFER");
		static_assert(stage_config.min_k > 1, "MIN_K MUST BE GREATER THAN 1 TO ALLOW VALID BESSEL CORRECTION VARIANCE MATH (K - 1)");
		static_assert(stage_config.rse_threshold > 0.0, "RELATIVE STANDARD ERROR THRESHOLD MUST BE POSITIVE");
		static_assert(stage_config.convergence_threshold >= 0.0, "CONVERGENCE THRESHOLD CANNOT BE NEGATIVE");
		static_assert(stage_config.max_time_in_s > 0, "MAX TIME LIMIT MUST BE GREATER THAN ZERO");
		std::vector<iteration_metrics> iteration_data;
		iteration_data.resize(stage_config.max_iteration_count);

		if (iterations > stage_config.max_iteration_count) {
			iterations = stage_config.max_iteration_count;
		}

		functor_type functor_runner{};
		for (uint64_t x = 0; x < iterations; ++x) {
			auto& metrics = iteration_data[x];
			functor_runner.impl(metrics, std::forward<arg_types>(args)...);
		}

		uint64_t k = std::max(iterations / 10, stage_config.min_k);
		k		   = std::min(k, stage_config.max_k);

		if (k > iterations) {
			k = iterations;
		}

		return calculate_epoch_stats(iteration_data.data(), iterations, k);
	}

	struct test_data {
		std::unordered_map<std::string_view, library_benchmark_data> results{};
		std::string test_name;
	};

	template<benchmark_types benchmark_type> struct final_test_results {
		std::vector<library_completion_data> sorted_results{};
		system_info_data<benchmark_type> system_info;
		std::string_view stage_name_str;
		std::string test_name;

		std::string csv_preamble() const {
			std::stringstream ss{};
			ss << "# " << test_name << " Test Results " << std::endl;
			ss << "#**" << system_info.device_type << ":** " << system_info.device_name() << std::endl;
			ss << "#**OS:** " << system_info.os_id << "-" << system_info.os_version << std::endl;
			ss << "#**Compiler:** " << system_info.compiler_id << "-" << system_info.compiler_version << std::endl << std::endl;
			return ss.str();
		}

		static std::string csv_header() {
			std::string h = "Library";
			h += ",Throughput (MB/s)";
			h += ",RSE (%)";
			h += ",Window Duration";
			h += ",File Size (Bytes)";
			h += ",Window Samples (k)";
			h += ",Variance";
			h += ",Latency / Run (ns)";
			h += ",Position";
			return h;
		}

		static std::string result_to_csv_line(const library_completion_data& r) {
			std::stringstream ss{};
			ss << r.name;
			ss << "," << r.final_throughput;
			ss << "," << r.final_rse;
			ss << "," << r.final_ms_spent;
			ss << "," << r.bytes_processed;
			ss << "," << r.final_sample_size;
			ss << "," << r.final_variance;
			ss << "," << r.final_mean;
			switch (static_cast<uint64_t>(r.position_type_val)) {
				case static_cast<uint64_t>(position_type::win):
					ss << ",Win";
					break;
				case static_cast<uint64_t>(position_type::tie):
					ss << ",Tie";
					break;
				case static_cast<uint64_t>(position_type::loss):
					ss << ",Loss";
					break;
				case static_cast<uint64_t>(position_type::none):
					[[fallthrough]];
				default:
					ss << ",";
					break;
			}
			ss << std::endl;
			return ss.str();
		}

		std::string to_csv(bool include_preamble = true, const std::string& file_path = "") const {
			std::stringstream ss{};
			if (include_preamble) {
				ss << csv_preamble();
			}
			ss << csv_header() << std::endl;
			for (const auto& r: sorted_results) {
				ss << result_to_csv_line(r);
			}

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string safe_name = static_cast<std::string>(test_name);
				std::replace(safe_name.begin(), safe_name.end(), '.', '_');
				std::replace(safe_name.begin(), safe_name.end(), '/', '_');
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + static_cast<std::string>(stage_name_str) + ".csv");
			}

			return text;
		}

		std::string md_preamble() const {
			std::stringstream ss{};
			ss << "# " << test_name << " Test Results" << std::endl << std::endl;
			ss << "**" << system_info.device_type << ":** " << system_info.device_name() << "  " << std::endl;
			ss << "**OS:** " << system_info.os_id << "-" << system_info.os_version << "  " << std::endl;
			ss << "**Compiler:** " << system_info.compiler_id << "-" << system_info.compiler_version << "  " << std::endl << std::endl;
			return ss.str();
		}

		static std::string md_header_row() {
			std::string h = "| Library";
			h += " | Throughput (MB/s)";
			h += " | RSE (%)";
			h += " | Window Duration";
			h += " | File Size (Bytes)";
			h += " | Window Samples (k)";
			h += " | Variance";
			h += " | Latency / Run (ns)";
			h += " | Position |";
			return h;
		}

		static std::string md_separator_row() {
			std::string s = "| -------";
			s += " | -----------";
			s += " | -------";
			s += " | ---------";
			s += " | ---------------";
			s += " | --------------------";
			s += " | ----------";
			s += " | ----";
			s += " | -------- |";
			return s;
		}

		static std::string result_to_md_row(const library_completion_data& r) {
			std::string lib_cell = static_cast<std::string>(r.name);
			if (r.position_type_val == position_type::tie)
				lib_cell += " **STATISTICAL TIE**";

			std::stringstream ss{};
			ss << "| " << lib_cell;
			ss << " | " << r.final_throughput;
			ss << " | " << r.final_rse;
			ss << " | " << r.final_ms_spent;
			ss << " | " << r.bytes_processed;
			ss << " | " << r.final_sample_size;
			ss << " | " << r.final_variance;
			ss << " | " << r.final_mean;
			ss << " | " << r.position;
			switch (static_cast<uint64_t>(r.position_type_val)) {
				case static_cast<uint64_t>(position_type::win):
					ss << "(Win)";
					break;
				case static_cast<uint64_t>(position_type::tie):
					ss << "(Tie)";
					break;
				case static_cast<uint64_t>(position_type::loss):
					ss << "(Loss)";
					break;
				case static_cast<uint64_t>(position_type::none):
					[[fallthrough]];
				default:
					break;
			}
			ss << " |" << std::endl;
			return ss.str();
		}

		std::string to_markdown(bool include_preamble = true, bool include_test_title = true, const std::string& file_path = "") const {
			std::stringstream ss{};
			if (include_test_title) {
				ss << "### " << test_name << " Test Results" << std::endl;
			}
			if (include_preamble) {
				ss << md_preamble();
			}
			ss << md_header_row() << std::endl;
			ss << md_separator_row() << std::endl;
			for (const auto& r: sorted_results) {
				ss << result_to_md_row(r);
			}

			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string safe_name = static_cast<std::string>(test_name);
				std::replace(safe_name.begin(), safe_name.end(), '.', '_');
				std::replace(safe_name.begin(), safe_name.end(), '/', '_');
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + static_cast<std::string>(stage_name_str) + ".md");
			}

			return text;
		}

		void print(bool include_preamble = true) const {
			std::cout << to_markdown(include_preamble);
		}
	};

	struct stage_data {
		std::unordered_map<std::string_view, test_data> results{};
		std::string_view stage_name_str;
	};

	struct library_positions {
		std::string lib_name;
		uint64_t losses;
		uint64_t wins;
		uint64_t ties;
		bool operator>(const library_positions& rhs) const {
			return wins > rhs.wins;
		}
	};

	template<benchmark_types benchmark_type> struct stage_results_data {
		std::vector<library_positions> lib_positions;
		std::vector<final_test_results<benchmark_type>> results{};
		system_info_data<benchmark_type> system_info;
		std::string_view stage_name_str;
		std::string csv_preamble() const {
			std::stringstream ss{};
			ss << "# " << stage_name_str << " Stage Results" << "\n";
			ss << "#**" << system_info.device_type << ":** " << system_info.device_name() << "\n";
			ss << "#**OS:** " << system_info.os_id << "-" << system_info.os_version << "\n";
			ss << "#**Compiler:** " << system_info.compiler_id << "-" << system_info.compiler_version << "\n\n";
			return ss.str();
		}

		static std::string csv_header() {
			std::string h = "Library";
			h += ",Wins";
			h += ",Ties";
			h += ",Losses";
			return h;
		}

		std::string accum_to_csv_line(const library_positions& lib) const {
			std::stringstream ss{};
			uint64_t wins	= lib.wins;
			uint64_t ties	= lib.ties;
			uint64_t losses = lib.losses;
			ss << lib.lib_name << "," << wins << "," << ties << "," << losses << "\n";
			return ss.str();
		}

		std::string to_csv(const std::string& file_path = "") const {
			std::stringstream ss{};
			ss << csv_preamble();
			ss << csv_header() << "\n";
			for (const auto& lib: lib_positions) {
				ss << accum_to_csv_line(lib);
			}


			const std::string text = ss.str();

			if (!file_path.empty()) {
				std::string file_name{ internal::operating_system_name.operator std::string() + "-" + internal::compiler_id.operator std::string() };
				file_handle::save_file(text, file_path + "/" + file_name + "-" + static_cast<std::string>(stage_name_str) + ".md");
			}

			return text;
		}
	};

	template<typename functor_type_new, stage_config_data stage_config, typename... arg_types> inline library_benchmark_data run_adaptive_benchmark(arg_types&&... args) {
		static constexpr double max_time_in_ns{ static_cast<double>(stage_config.max_time_in_s) * 1e9 };
		using functor_type = internal::iteration_metric_collector<stage_config.benchmark_type, functor_type_new>;
		epoch_state_data state{};
		state.target_count		   = stage_config.measured_iteration_count;
		state.total_iterations_run = 0;
		state.start_time_ns		   = bnch_swt::clock_type::now().time_since_epoch().count();
		library_benchmark_data current_best_stats{};
		library_benchmark_data current_stats{};
		double previous_mean = -1.0;
		double elapsed_time	 = 0;
		while (state.total_iterations_run + state.target_count < stage_config.max_iteration_count) {
			current_stats = impl<functor_type, stage_config>(state.target_count, std::forward<arg_types>(args)...);
			state.total_iterations_run += state.target_count;

			double current_time_ns = bnch_swt::clock_type::now().time_since_epoch().count();
			elapsed_time		   = current_time_ns - state.start_time_ns;

			bool rse_converged	= (current_stats.final_rse <= stage_config.rse_threshold) && (current_stats.final_variance > 0.0);
			bool mean_converged = false;

			if (previous_mean >= 0.0) {
				double mean_delta		 = std::abs(current_stats.final_mean - previous_mean);
				double percentage_change = (previous_mean > 0.0) ? (mean_delta / previous_mean) : 0.0;
				if (percentage_change <= stage_config.convergence_threshold) {
					mean_converged = true;
				}
			}

			if (rse_converged && mean_converged) {
				current_stats.converged		 = true;
				current_stats.final_ms_spent = elapsed_time;
				current_best_stats			 = current_stats;
				return current_best_stats;
			}

			if (rse_converged && mean_converged && current_stats.final_mean > 0.0) {
				current_best_stats = current_stats;
			}

			if (elapsed_time >= max_time_in_ns || state.total_iterations_run + state.target_count >= stage_config.max_iteration_count) {
				if (current_best_stats.final_mean == 0.0) {
					current_best_stats = current_stats;
				}
				current_best_stats.converged	  = false;
				current_best_stats.final_ms_spent = elapsed_time;
				return current_best_stats;
			}

			previous_mean = current_stats.final_mean;

			state.target_count *= 2;
			if (state.target_count > stage_config.max_iteration_count) {
				state.target_count = stage_config.max_iteration_count;
			}
		}
		return current_best_stats;
	}

	inline bool is_statistically_tied(const library_benchmark_data& a, const library_benchmark_data& b) {
		double n1 = static_cast<double>(a.final_sample_size);
		double n2 = static_cast<double>(b.final_sample_size);

		if (n1 <= 1.0 || n2 <= 1.0) {
			return true;
		}

		double v1 = a.final_variance;
		double v2 = b.final_variance;

		double se1		   = v1 / n1;
		double se2		   = v2 / n2;
		double denominator = std::sqrt(se1 + se2);

		if (denominator == 0.0) {
			return true;
		}

		double t_stat = std::abs(a.final_mean - b.final_mean) / denominator;

		double num = (se1 + se2) * (se1 + se2);
		double den = ((se1 * se1) / (n1 - 1.0)) + ((se2 * se2) / (n2 - 1.0));
		double df  = num / den;

		double t_critical = 1.95996 + (2.4132 / df) + (4.4212 / (df * df));

		return t_stat < t_critical;
	}

	template<benchmark_types benchmark_type> inline final_test_results<benchmark_type> process_test_rankings(const test_data& raw_data) {
		std::vector<library_completion_data> leaderboard;
		leaderboard.reserve(raw_data.results.size());
		final_test_results<benchmark_type> return_value;
		system_info_data<benchmark_type> system_info;
		return_value.test_name	 = raw_data.test_name;
		return_value.system_info = system_info;
		for (const auto& [lib_name, bench]: raw_data.results) {
			library_completion_data item{};
			static_cast<library_benchmark_data&>(item) = bench;
			item.name								   = lib_name;
			if (item.converged) {
				leaderboard.push_back(item);
			}
		}

		std::sort(leaderboard.begin(), leaderboard.end(), [](const library_completion_data& a, const library_completion_data& b) {
			return a.final_throughput > b.final_throughput;
		});

		if (leaderboard.empty()) {
			return return_value;
		}

		uint64_t current_rank	= 1;
		leaderboard[0].position = current_rank;

		for (size_t i = 1; i < leaderboard.size(); ++i) {
			if (is_statistically_tied(leaderboard[i - 1], leaderboard[i])) {
				leaderboard[i].position = leaderboard[i - 1].position;
			} else {
				current_rank			= i + 1;
				leaderboard[i].position = current_rank;
			}
		}

		for (size_t i = 0; i < leaderboard.size(); ++i) {
			uint64_t r = leaderboard[i].position;

			bool ties_with_someone = false;
			if (i > 0 && leaderboard[i - 1].position == r)
				ties_with_someone = true;
			if (i < leaderboard.size() - 1 && leaderboard[i + 1].position == r)
				ties_with_someone = true;

			if (r == 1) {
				leaderboard[i].position_type_val = ties_with_someone ? bnch_swt::position_type::tie : bnch_swt::position_type::win;
			} else {
				leaderboard[i].position_type_val = ties_with_someone ? bnch_swt::position_type::tie : bnch_swt::position_type::loss;
			}
		}

		return_value.sorted_results = std::move(leaderboard);

		return return_value;
	}

	template<benchmark_types benchmark_type> inline stage_results_data<benchmark_type> process_stage_rankings(const stage_data& raw_data) {
		std::vector<final_test_results<benchmark_type>> leaderboard;
		leaderboard.reserve(raw_data.results.size());
		std::unordered_map<std::string, library_positions> lib_positions{};
		stage_results_data<benchmark_type> return_value;
		system_info_data<benchmark_type> system_info;
		return_value.stage_name_str = raw_data.stage_name_str;
		return_value.system_info	= system_info;
		for (const auto& [test_name, bench]: raw_data.results) {
			leaderboard.push_back(process_test_rankings<benchmark_type>(bench));
		}

		for (size_t i = 0; i < leaderboard.size(); ++i) {
			for (auto& value: leaderboard[i].sorted_results) {
				library_completion_data& val_temp = value;
				if (val_temp.position_type_val == position_type::win) {
					++lib_positions[static_cast<std::string>(val_temp.name)].wins;
				} else if (val_temp.position_type_val == position_type::loss) {
					++lib_positions[static_cast<std::string>(val_temp.name)].losses;
				} else if (val_temp.position_type_val == position_type::tie) {
					++lib_positions[static_cast<std::string>(val_temp.name)].ties;
				}
			}
		}
		for (auto& [k, v]: lib_positions) {
			library_positions lib_pos;
			lib_pos.lib_name = k;
			lib_pos.wins	 = v.wins;
			lib_pos.ties	 = v.ties;
			lib_pos.losses	 = v.losses;
			return_value.lib_positions.emplace_back(lib_pos);
		}
		std::sort(return_value.lib_positions.begin(), return_value.lib_positions.end(), std::greater<library_positions>{});
		for (const auto& entry: leaderboard) {
			return_value.results.emplace_back(entry);
		}

		return return_value;
	}

	template<string_literal stage_name_new, const stage_config_data stage_config> struct benchmark_stage {
		inline static stage_data& get_raw_test_data() {
			static stage_data* raw_test_data{ new stage_data{} };
			return *raw_test_data;
		}

		template<bnch_swt::string_literal test_name_new, bnch_swt::string_literal library_name_new, typename functor_type, typename... arg_types>
		static void run_benchmark(arg_types&&... args) {
			[[maybe_unused]] static constexpr string_literal stage_name{ stage_name_new };
			static constexpr string_literal test_name{ test_name_new };
			static constexpr string_literal library_name{ library_name_new };
			test_data& test_data_val = get_raw_test_data().results[test_name.operator std::string_view()];
			test_data_val.test_name	 = test_name.operator std::string();
			if constexpr (stage_config.clear_cpu_caches_before_iterations) {
				internal::cache_clearer<stage_config.benchmark_type> cclearer{};
				cclearer.evict_caches();
			}
			test_data_val.results[library_name.operator std::string_view()] = run_adaptive_benchmark<functor_type, stage_config>(std::forward<arg_types>(args)...);
		}

		template<bnch_swt::string_literal test_name_new, bnch_swt::string_literal library_name_new, auto function, typename... arg_types>
		static void run_benchmark(arg_types&&... args) {
			[[maybe_unused]] static constexpr string_literal stage_name{ stage_name_new };
			static constexpr string_literal test_name{ test_name_new };
			static constexpr string_literal library_name{ library_name_new };
			using function_type												= function_holder<function>;
			test_data& test_data_val										= get_raw_test_data().results[test_name.operator std::string_view()];
			test_data_val.test_name											= test_name.operator std::string();
			test_data_val.results[library_name.operator std::string_view()] = run_adaptive_benchmark<function_type, stage_config>(std::forward<arg_types>(args)...);
		}

		inline static auto get_test_results(const std::string& test_name) {
			return process_test_rankings<stage_config.benchmark_type>(get_raw_test_data().results[test_name]);
		}

		inline static auto get_all_results() {
			return process_stage_rankings<stage_config.benchmark_type>(get_raw_test_data());
		}
	};

}
