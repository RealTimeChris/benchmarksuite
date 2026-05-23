#include <rt-ut>
#include <bnch_swt>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace bnch_swt;
using rt_ut::unit_test;

namespace {

	bool approx(double a, double b, double eps = 1e-6) {
		return std::abs(a - b) <= eps;
	}

	library_benchmark_data make_bench(double mean, double variance, uint64_t sample_size, double throughput, bool converged) {
		library_benchmark_data d{};
		d.final_mean		= mean;
		d.final_variance	= variance;
		d.final_sample_size = sample_size;
		d.final_throughput	= throughput;
		d.converged			= converged;
		return d;
	}

	library_completion_data make_completion(std::string_view name, double throughput, position_type pt, uint64_t position) {
		library_completion_data d{};
		d.final_throughput	= throughput;
		d.name				= name;
		d.position			= position;
		d.position_type_val = pt;
		return d;
	}

	std::optional<library_completion_data> find_lib(const std::vector<library_completion_data>& v, std::string_view name) {
		for (const auto& item: v) {
			if (item.name == name) {
				return item;
			}
		}
		return std::nullopt;
	}

	std::optional<library_positions> find_pos(const std::vector<library_positions>& v, std::string_view name) {
		for (const auto& item: v) {
			if (item.lib_name == name) {
				return item;
			}
		}
		return std::nullopt;
	}

	iteration_metrics make_metric(double time_ns, uint64_t bytes) {
		iteration_metrics value{};
		value.time_in_ns	  = time_ns;
		value.bytes_processed = bytes;
		return value;
	}

	bool contains(const std::string& haystack, std::string_view needle) {
		return haystack.find(needle) != std::string::npos;
	}

}

int main() {
	size_t total{};
	size_t passed{};
	auto check = [&](bool result) {
		++total;
		if (result) {
			++passed;
		}
	};

	std::cout << "==== printable_milliseconds ====" << std::endl;
	check(unit_test<"pm_converts_ns_to_ms", true, false>::run([] {
		return approx(printable_milliseconds{ 5'000'000.0 }.value, 5.0);
	}));

	check(unit_test<"pm_zero_is_zero", true, false>::run([] {
		return approx(printable_milliseconds{ 0.0 }.value, 0.0);
	}));

	check(unit_test<"pm_fractional", true, false>::run([] {
		return approx(printable_milliseconds{ 1'500'000.0 }.value, 1.5);
	}));

	check(unit_test<"pm_stream_has_units", true, false>::run([] {
		std::stringstream ss{};
		ss << printable_milliseconds{ 2'000'000.0 };
		return contains(ss.str(), "2") && contains(ss.str(), "ms");
	}));

	std::cout << "==== position_type enum ====" << std::endl;
	check(unit_test<"pt_win_is_0", true, false>::assert_eq<uint8_t{ 0 }>([] {
		return static_cast<uint8_t>(position_type::win);
	}));

	check(unit_test<"pt_tie_is_1", true, false>::assert_eq<uint8_t{ 1 }>([] {
		return static_cast<uint8_t>(position_type::tie);
	}));

	check(unit_test<"pt_loss_is_2", true, false>::assert_eq<uint8_t{ 2 }>([] {
		return static_cast<uint8_t>(position_type::loss);
	}));

	check(unit_test<"pt_none_is_3", true, false>::assert_eq<uint8_t{ 3 }>([] {
		return static_cast<uint8_t>(position_type::none);
	}));

	std::cout << "==== library_positions::operator> ====" << std::endl;
	check(unit_test<"libpos_more_wins_is_greater", true, false>::run([] {
		library_positions a{ "a", 1, 10, 0 };
		library_positions b{ "b", 1, 5, 0 };
		return a > b;
	}));

	check(unit_test<"libpos_equal_wins_not_greater", true, false>::run([] {
		library_positions a{ "a", 1, 5, 0 };
		library_positions b{ "b", 9, 5, 9 };
		return !(a > b);
	}));

	std::cout << "==== library_completion_data::operator> ====" << std::endl;
	check(unit_test<"compl_higher_position_is_greater", true, false>::run([] {
		auto a = make_completion("a", 100.0, position_type::loss, 3);
		auto b = make_completion("b", 999.0, position_type::win, 1);
		return a > b;
	}));

	std::cout << "==== is_statistically_tied ====" << std::endl;
	check(unit_test<"tie_identical_inputs", true, false>::run([] {
		auto a = make_bench(100.0, 4.0, 50, 0.0, true);
		auto b = make_bench(100.0, 4.0, 50, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_n1_le_1_returns_true", true, false>::run([] {
		auto a = make_bench(100.0, 4.0, 1, 0.0, true);
		auto b = make_bench(200.0, 4.0, 50, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_n2_le_1_returns_true", true, false>::run([] {
		auto a = make_bench(100.0, 4.0, 50, 0.0, true);
		auto b = make_bench(200.0, 4.0, 1, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_huge_gap_tiny_var_not_tied", true, false>::run([] {
		auto a = make_bench(100.0, 1.0, 100, 0.0, true);
		auto b = make_bench(120.0, 1.0, 100, 0.0, true);
		return !is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_close_means_wide_spread_tied", true, false>::run([] {
		auto a = make_bench(100.0, 100.0, 50, 0.0, true);
		auto b = make_bench(101.0, 100.0, 50, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_symmetric_argument_order", true, false>::run([] {
		auto a = make_bench(100.0, 100.0, 50, 0.0, true);
		auto b = make_bench(101.0, 100.0, 50, 0.0, true);
		return is_statistically_tied(a, b) == is_statistically_tied(b, a);
	}));

	check(unit_test<"tie_zero_var_same_mean_true", true, false>::run([] {
		auto a = make_bench(100.0, 0.0, 50, 0.0, true);
		auto b = make_bench(100.0, 0.0, 50, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	check(unit_test<"tie_zero_var_diff_mean_currently_true", true, false>::run([] {
		auto a = make_bench(100.0, 0.0, 50, 0.0, true);
		auto b = make_bench(200.0, 0.0, 50, 0.0, true);
		return is_statistically_tied(a, b);
	}));

	std::cout << "==== calculate_epoch_stats ====" << std::endl;
	check(unit_test<"stats_k_zero_returns_default", true, false>::run([] {
		std::vector<iteration_metrics> data(4);
		auto r = calculate_epoch_stats(data.data(), 4, 0);
		return r.final_sample_size == 0 && approx(r.final_mean, 0.0) && approx(r.final_rse, 100.0);
	}));

	check(unit_test<"stats_constant_mean_and_variance", true, false>::run([] {
		std::vector<iteration_metrics> data;
		for (int i = 0; i < 5; ++i) {
			data.push_back(make_metric(100.0, 1024));
		}
		auto r = calculate_epoch_stats(data.data(), 5, 5);
		return approx(r.final_mean, 100.0) && approx(r.final_variance, 0.0) && r.final_sample_size == 5;
	}));

	check(unit_test<"stats_constant_bytes_processed_is_per_iter_mean", true, false>::run([] {
		std::vector<iteration_metrics> data;
		for (int i = 0; i < 5; ++i) {
			data.push_back(make_metric(100.0, 1024));
		}
		auto r = calculate_epoch_stats(data.data(), 5, 5);
		return r.bytes_processed == 1024;
	}));

	check(unit_test<"stats_throughput_known_value", true, false>::run([] {
		std::vector<iteration_metrics> data;
		for (int i = 0; i < 5; ++i) {
			data.push_back(make_metric(100.0, 1024));
		}
		auto r = calculate_epoch_stats(data.data(), 5, 5);
		return approx(r.final_throughput, 9765.625, 1e-2);
	}));

	check(unit_test<"stats_bessel_variance", true, false>::run([] {
		std::vector<iteration_metrics> data;
		data.push_back(make_metric(90.0, 1));

		data.push_back(make_metric(110.0, 1));

		data.push_back(make_metric(90.0, 1));

		data.push_back(make_metric(110.0, 1));

		auto r = calculate_epoch_stats(data.data(), 4, 4);
		return approx(r.final_mean, 100.0) && approx(r.final_variance, 400.0 / 3.0, 1e-9);
	}));

	check(unit_test<"stats_window_uses_only_last_k", true, false>::run([] {
		std::vector<iteration_metrics> data;
		for (int i = 0; i < 7; ++i) {
			data.push_back(make_metric(999.0, 1));
		}
		for (int i = 0; i < 3; ++i) {
			data.push_back(make_metric(100.0, 2048));
		}
		auto r = calculate_epoch_stats(data.data(), 10, 3);
		return approx(r.final_mean, 100.0) && r.bytes_processed == 2048 && r.final_sample_size == 3;
	}));

	check(unit_test<"stats_k_one_variance_zero", true, false>::run([] {
		std::vector<iteration_metrics> data;
		data.push_back(make_metric(42.0, 8));

		auto r = calculate_epoch_stats(data.data(), 1, 1);
		return approx(r.final_mean, 42.0) && approx(r.final_variance, 0.0) && approx(r.final_rse, 0.0);
	}));

	std::cout << "==== process_test_rankings ====" << std::endl;
	check(unit_test<"rank_empty_yields_empty", true, false>::run([] {
		test_data td{};
		td.test_name = "Empty";
		auto res	 = process_test_rankings<benchmark_types::cpu>(td);
		return res.sorted_results.empty();
	}));

	check(unit_test<"rank_excludes_unconverged", true, false>::run([] {
		test_data td{};
		td.test_name		= "ConvergenceFilter";
		td.results["Real"]	= make_bench(10.0, 0.01, 100, 1000.0, true);
		td.results["Ghost"] = make_bench(10.0, 0.01, 100, 2000.0, false);
		auto res			= process_test_rankings<benchmark_types::cpu>(td);
		return res.sorted_results.size() == 1 && res.sorted_results[0].name == "Real";
	}));

	check(unit_test<"rank_clear_winner_and_loser", true, false>::run([] {
		test_data td{};
		td.test_name	   = "WinLoss";
		td.results["Fast"] = make_bench(10.0, 0.01, 100, 1000.0, true);
		td.results["Slow"] = make_bench(20.0, 0.01, 100, 500.0, true);
		auto res		   = process_test_rankings<benchmark_types::cpu>(td);
		if (res.sorted_results.size() != 2) {
			return false;
		}
		auto fast = find_lib(res.sorted_results, "Fast");
		auto slow = find_lib(res.sorted_results, "Slow");
		if (!fast || !slow) {
			return false;
		}
		return res.sorted_results[0].name == "Fast" && fast->position == 1 && fast->position_type_val == position_type::win && slow->position == 2 &&
			slow->position_type_val == position_type::loss;
	}));

	check(unit_test<"rank_statistical_tie_shares_rank", true, false>::run([] {
		test_data td{};
		td.test_name	= "TiePair";
		td.results["A"] = make_bench(100.0, 50.0, 40, 1000.0, true);
		td.results["B"] = make_bench(101.0, 50.0, 40, 990.0, true);
		auto res		= process_test_rankings<benchmark_types::cpu>(td);
		auto a			= find_lib(res.sorted_results, "A");
		auto b			= find_lib(res.sorted_results, "B");
		if (!a || !b) {
			return false;
		}
		return a->position == 1 && b->position == 1 && a->position_type_val == position_type::tie && b->position_type_val == position_type::tie;
	}));

	check(unit_test<"rank_top_throughput_sorted_first", true, false>::run([] {
		test_data td{};
		td.test_name	  = "SortOrder";
		td.results["Mid"] = make_bench(15.0, 0.01, 100, 750.0, true);
		td.results["Top"] = make_bench(10.0, 0.01, 100, 1500.0, true);
		td.results["Low"] = make_bench(30.0, 0.01, 100, 300.0, true);
		auto res		  = process_test_rankings<benchmark_types::cpu>(td);
		return res.sorted_results.size() == 3 && res.sorted_results[0].name == "Top";
	}));

	std::cout << "==== process_stage_rankings ====" << std::endl;
	check(unit_test<"stage_aggregates_wins_and_losses", true, false>::run([] {
		test_data t1{};
		t1.test_name	   = "T1";
		t1.results["Fast"] = make_bench(10.0, 0.01, 100, 1000.0, true);
		t1.results["Slow"] = make_bench(20.0, 0.01, 100, 500.0, true);
		test_data t2{};
		t2.test_name	   = "T2";
		t2.results["Fast"] = make_bench(10.0, 0.01, 100, 1200.0, true);
		t2.results["Slow"] = make_bench(25.0, 0.01, 100, 400.0, true);
		stage_data sd{};
		sd.stage_name_str = "StageX";
		sd.results["T1"]  = t1;
		sd.results["T2"]  = t2;
		auto res		  = process_stage_rankings<benchmark_types::cpu>(sd);
		auto fast		  = find_pos(res.lib_positions, "Fast");
		auto slow		  = find_pos(res.lib_positions, "Slow");
		if (!fast || !slow) {
			return false;
		}
		return fast->wins == 2 && fast->losses == 0 && slow->wins == 0 && slow->losses == 2;
	}));

	check(unit_test<"stage_sorted_by_wins_desc", true, false>::run([] {
		test_data t1{};
		t1.test_name	   = "T1";
		t1.results["Fast"] = make_bench(10.0, 0.01, 100, 1000.0, true);
		t1.results["Slow"] = make_bench(20.0, 0.01, 100, 500.0, true);
		stage_data sd{};
		sd.stage_name_str = "StageY";
		sd.results["T1"]  = t1;
		auto res		  = process_stage_rankings<benchmark_types::cpu>(sd);
		return !res.lib_positions.empty() && res.lib_positions.front().lib_name == "Fast";
	}));

	check(unit_test<"stage_counts_ties", true, false>::run([] {
		test_data t1{};
		t1.test_name	= "T1";
		t1.results["A"] = make_bench(100.0, 50.0, 40, 1000.0, true);
		t1.results["B"] = make_bench(101.0, 50.0, 40, 990.0, true);
		stage_data sd{};
		sd.stage_name_str = "StageZ";
		sd.results["T1"]  = t1;
		auto res		  = process_stage_rankings<benchmark_types::cpu>(sd);
		auto a			  = find_pos(res.lib_positions, "A");
		auto b			  = find_pos(res.lib_positions, "B");
		if (!a || !b) {
			return false;
		}
		return a->ties == 1 && b->ties == 1;
	}));

	std::cout << "==== output formatting ====" << std::endl;
	check(unit_test<"fmt_csv_header_fields", true, false>::run([] {
		auto h = final_test_results::csv_header();
		return contains(h, "Library") && contains(h, "Throughput") && contains(h, "Position");
	}));

	check(unit_test<"fmt_md_header_starts_pipe", true, false>::run([] {
		return final_test_results::md_header_row().rfind("| Library", 0) == 0;
	}));

	check(unit_test<"fmt_md_separator_has_dashes", true, false>::run([] {
		return contains(final_test_results::md_separator_row(), "---");
	}));

	check(unit_test<"fmt_csv_line_win_label", true, false>::run([] {
		auto r	  = make_completion("LibWin", 123.0, position_type::win, 1);
		auto line = final_test_results::result_to_csv_line(r);
		return contains(line, "LibWin") && contains(line, "Win");
	}));

	check(unit_test<"fmt_csv_line_tie_label", true, false>::run([] {
		auto r	  = make_completion("LibTie", 99.0, position_type::tie, 1);
		auto line = final_test_results::result_to_csv_line(r);
		return contains(line, "Tie");
	}));

	check(unit_test<"fmt_md_row_marks_tie", true, false>::run([] {
		auto r	 = make_completion("LibTie", 99.0, position_type::tie, 1);
		auto row = final_test_results::result_to_md_row(r);
		return contains(row, "STATISTICAL TIE");
	}));

	check(unit_test<"fmt_to_csv_emits_rows", true, false>::run([] {
		final_test_results ftr{};
		ftr.test_name = "Demo";
		ftr.sorted_results.push_back(make_completion("OnlyLib", 555.0, position_type::win, 1));

		auto csv = ftr.to_csv(false);
		return contains(csv, "Library") && contains(csv, "OnlyLib");
	}));

	check(unit_test<"fmt_to_markdown_emits_rows", true, false>::run([] {
		final_test_results ftr{};
		ftr.test_name = "Demo";
		ftr.sorted_results.push_back(make_completion("MdLib", 777.0, position_type::win, 1));

		auto md = ftr.to_markdown(false, false);
		return contains(md, "MdLib") && contains(md, "| Library");
	}));

	check(unit_test<"fmt_stage_csv_header_fields", true, false>::run([] {
		auto h = stage_results_data::csv_header();
		return contains(h, "Wins") && contains(h, "Ties") && contains(h, "Losses");
	}));

	check(unit_test<"fmt_stage_accum_line", true, false>::run([] {
		stage_results_data srd{};
		library_positions lp{ "AggLib", 2, 3, 1 };
		auto line = srd.accum_to_csv_line(lp);
		return contains(line, "AggLib") && contains(line, "3");
	}));

	check(unit_test<"fmt_stage_to_csv", true, false>::run([] {
		stage_results_data srd{};
		srd.lib_positions.push_back(library_positions{ "StageLib", 0, 4, 0 });
		auto csv = srd.to_csv();
		return contains(csv, "Library") && contains(csv, "StageLib");
	}));

	std::cout << "==== stage_config_data defaults ====" << std::endl;
	check(unit_test<"cfg_default_measured_count", true, false>::assert_eq<uint64_t{ 100 }>([] {
		return stage_config_data{}.measured_iteration_count;
	}));

	check(unit_test<"cfg_default_max_count", true, false>::assert_eq<uint64_t{ 1000 }>([] {
		return stage_config_data{}.max_iteration_count;
	}));

	check(unit_test<"cfg_default_min_k", true, false>::assert_eq<uint64_t{ 30 }>([] {
		return stage_config_data{}.min_k;
	}));

	check(unit_test<"cfg_default_max_k", true, false>::assert_eq<uint64_t{ 100000 }>([] {
		return stage_config_data{}.max_k;
	}));

	check(unit_test<"cfg_default_rse_threshold", true, false>::run([] {
		return approx(stage_config_data{}.rse_threshold, 2.5);
	}));

	check(unit_test<"cfg_default_convergence_threshold", true, false>::run([] {
		return approx(stage_config_data{}.convergence_threshold, 1.0);
	}));

	check(unit_test<"cfg_default_max_time", true, false>::assert_eq<uint64_t{ 5 }>([] {
		return stage_config_data{}.max_time_in_s;
	}));

	check(unit_test<"cfg_default_clears_caches", true, false>::run([] {
		return stage_config_data{}.clear_cpu_caches_before_iterations == true;
	}));

	std::cout << "==== function_holder ====" << std::endl;
	check(unit_test<"fh_forwards_and_returns", true, false>::assert_eq<uint64_t{ 42 }>([] {
		return function_holder<+[](uint64_t x) -> uint64_t {
			return x * 2;
		}>::impl(uint64_t{ 21 });
	}));

	std::cout << "==== system info smoke ====" << std::endl;
	check(unit_test<"sys_device_name_non_empty", true, false>::run([] {
		return !get_device_name<benchmark_types::cpu>().empty();
	}));

	std::cout << std::endl << "==== SUMMARY ====" << std::endl;
	std::cout << passed << " / " << total << " checks passed." << std::endl;
	return passed == total ? 0 : 1;
}
