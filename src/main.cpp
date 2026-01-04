/*
	MIT License
	Copyright (c) 2024 RealTimeChris
*/

#include "DToStr.hpp"
#include <bnch_swt/index.hpp>
#include <charconv>
#include <string>
#include <vector>
#include <random>

template<typename value_type> std::vector<value_type> generate_float_vector(size_t vector_size) {
	std::vector<value_type> result;
	result.reserve(vector_size);

	std::mt19937_64 rng(std::random_device{}());
	// Distribution for significand
	std::uniform_real_distribution<value_type> dist(0.00001, 1000000.0);
	// Distribution for occasional very small/large numbers
	std::uniform_int_distribution<int> exp_dist(-30, 30);

	for (size_t i = 0; i < vector_size; ++i) {
		value_type val = dist(rng) * std::pow(10.0, exp_dist(rng));
		if (i % 2 == 0)
			val = -val;
		result.push_back(val);
	}
	return result;
}

static constexpr auto max_iterations{ 20 };
static constexpr auto measured_iterations{ max_iterations / 5 };

template<typename value_type> struct benchmark_jsonifier_float_to_chars {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomFloats, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			auto* end = bnch_swt::to_chars<value_type>::impl(resultsTest[current_index][x].data(), randomFloats[current_index][x]);
			auto len  = static_cast<size_t>(end - resultsTest[current_index][x].data());
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += len;
		}
		return currentCount;
	}
};

template<typename value_type> struct benchmark_std_float_to_chars {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomFloats, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			auto [ptr, ec] =
				std::to_chars(resultsTest[current_index][x].data(), resultsTest[current_index][x].data() + resultsTest[current_index][x].size(), randomFloats[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += static_cast<size_t>(ptr - resultsTest[current_index][x].data());
		}
		return currentCount;
	}
};

template<typename value_type> struct benchmark_std_float_to_string {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomFloats, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			resultsTest[current_index][x] = std::to_string(randomFloats[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += resultsTest[current_index][x].size();
		}
		return currentCount;
	}
};

template<bnch_swt::string_literal stage_name, typename benchmark_type, bnch_swt::string_literal benchmark_name>
BNCH_SWT_HOST void run_and_validate_float(auto& resultsTest, const auto& randomFloats, uint64_t count, uint64_t& current_index) {
	using benchmark = bnch_swt::benchmark_stage<stage_name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu,
		[]() consteval {
			return bnch_swt::performance_metrics_presence<bnch_swt::benchmark_types::cpu>{ .throughput_mb_per_sec = true,
				.bytes_processed																				  = true,
				.cycles_per_byte																				  = true,
				.name																							  = true };
		}(),
		true>;

	benchmark::template run_benchmark<benchmark_name, benchmark_type>(resultsTest, randomFloats, count, current_index);

	// Validation for floats is tricky due to precision/representation differences,
	// but we check if the string is non-empty and valid.
	for (uint64_t y = 0; y < count; ++y) {
		if (resultsTest[current_index][y].empty()) {
			std::cout << benchmark_name.operator std::string_view() << " failed to serialize a float!" << std::endl;
			return;
		}
	}
}

template<typename value_type, bnch_swt::string_literal name, uint64_t count> inline void testFloatFunction() {
	std::vector<std::vector<value_type>> randomFloats{};
	randomFloats.resize(max_iterations);
	for (uint64_t x = 0; x < max_iterations; ++x) {
		randomFloats[x] = generate_float_vector<value_type>(count);
	}

	using benchmark = bnch_swt::benchmark_stage<name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu,
		[]() consteval {
			return bnch_swt::performance_metrics_presence<bnch_swt::benchmark_types::cpu>{ .throughput_mb_per_sec = true,
				.bytes_processed																				  = true,
				.cycles_per_byte																				  = true,
				.name																							  = true };
		}(),
		true>;

	std::vector<std::vector<std::string>> resultsTest01{};
	std::vector<std::vector<std::string>> resultsTest02{};
	std::vector<std::vector<std::string>> resultsTest03{};
	resultsTest01.resize(max_iterations);
	resultsTest02.resize(max_iterations);
	resultsTest03.resize(max_iterations);

	for (uint64_t x = 0; x < max_iterations; ++x) {
		resultsTest01[x].resize(count);
		resultsTest02[x].resize(count);
		resultsTest03[x].resize(count);

		for (uint64_t y = 0; y < count; ++y) {
			// Pre-allocate buffer space to avoid allocation overhead during the hot loop
			resultsTest01[x][y].resize(48);
			resultsTest02[x][y].resize(48);
			resultsTest03[x][y].resize(48);
		}
	}

	uint64_t currentIndex{};
	run_and_validate_float<name, benchmark_std_float_to_string<value_type>, "std::to_string">(resultsTest01, randomFloats, count, currentIndex);
	currentIndex = 0;
	run_and_validate_float<name, benchmark_std_float_to_chars<value_type>, "std::to_chars">(resultsTest02, randomFloats, count, currentIndex);
	currentIndex = 0;
	run_and_validate_float<name, benchmark_jsonifier_float_to_chars<value_type>, "toChars_Float">(resultsTest03, randomFloats, count, currentIndex);
	currentIndex = 0;
	benchmark::print_results(true, true);
}

template<typename T> void verify_float_roundtrip(T val, const std::string& label) {
	char buf[64];
	auto* end = bnch_swt::to_chars<T>::impl(buf, val);

	T parsed_val;
	auto [ptr, ec] = std::from_chars(buf, end, parsed_val);

	if (ec != std::errc{} || std::bit_cast<uint64_t>(static_cast<double>(val)) != std::bit_cast<uint64_t>(static_cast<double>(parsed_val))) {
		// Handle NaN specifically as bit patterns can vary
		if (std::isnan(val) && std::isnan(parsed_val))
			return;

		std::cerr << "ROUND-TRIP FAILURE [" << label << "]\n"
				  << "Original: " << std::scientific << std::setprecision(17) << val << "\n"
				  << "String  : " << std::string(buf, end - buf) << "\n"
				  << "Parsed  : " << parsed_val << "\n"
				  << "Bits Orig: " << std::hex << std::bit_cast<uint64_t>(static_cast<double>(val)) << "\n"
				  << "Bits Pars: " << std::hex << std::bit_cast<uint64_t>(static_cast<double>(parsed_val)) << std::dec << "\n";
		std::exit(1);
	}
}

void run_rigorous_float_tests() {
	std::cout << "Starting Rigorous Round-Trip Verification...\n";

	std::vector<double> edge_cases = { 0.0, -0.0, 1.0, -1.0, std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), std::numeric_limits<double>::epsilon(),
		std::pow(2.0, -52),// Smallest subnormal significand
		0.3333333333333333, 1.2345678901234567e-20, 1.2345678901234567e+20 };

	for (auto v: edge_cases) {
		verify_float_roundtrip<double>(v, "Edge Case Double");
		verify_float_roundtrip<float>(static_cast<float>(v), "Edge Case Float");
	}

	std::mt19937_64 rng(42);
	std::uniform_real_distribution<double> dist(-1e30, 1e30);
	for (int i = 0; i < 1000000; ++i) {
		verify_float_roundtrip<double>(dist(rng), "Fuzz Double");
	}

	std::cout << "All Round-Trip Tests Passed!\n";
}

int32_t main() {
	run_rigorous_float_tests();
}