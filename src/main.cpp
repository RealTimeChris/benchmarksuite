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
#include <bnch_swt/index.hpp>
#include <random>

template<typename value_type>
concept uns32_t = std::unsigned_integral<value_type> && sizeof(value_type) == 4;

template<typename value_type>
concept uns64_t = std::unsigned_integral<value_type> && sizeof(value_type) == 8;

template<uint64_t digits> consteval uint64_t max_value_for_digits() noexcept {
	uint64_t power = 1;
	for (uint64_t i = 0; i < digits; ++i) {
		power *= 10;
	}
	return power - 1;
}

template<typename value_type, value_type maxLength> std::vector<value_type> generate_random_integers(uint64_t count) {
	std::vector<value_type> randomNumbers;
	for (uint64_t i = 0; i < count; ++i) {
		randomNumbers.emplace_back(bnch_swt::random_generator<value_type>::impl(static_cast<value_type>(max_value_for_digits<maxLength>())));
	}
	return randomNumbers;
}

template<uns32_t value_type> BNCH_SWT_HOST uint8_t rtc_digit_count(const value_type inputValue) {
	static constexpr uint8_t digitCounts_32[]{ 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
	static constexpr uint32_t digitCountThresholds_32[]{ 0u, 9u, 99u, 999u, 9999u, 99999u, 999999u, 9999999u, 99999999u, 999999999u, 4294967295u };
	const uint8_t originalDigitCount{ digitCounts_32[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint8_t>(inputValue > digitCountThresholds_32[originalDigitCount]);
}

template<uns64_t value_type> BNCH_SWT_HOST uint8_t rtc_digit_count(const value_type inputValue) {
	static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
	static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull, 99999999999ull,
		999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull, 9999999999999999999ull };
	const uint8_t originalDigitCount{ digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint8_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

BNCH_SWT_HOST int int_log2(uint64_t x) {
	return 63 - std::countl_zero(x | 1);
}

BNCH_SWT_HOST int lemire_digit_count(uint32_t x) {
	static constexpr uint32_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999 };
	int y						  = (9 * int_log2(x)) >> 5;
	y += x > table[y];
	return y + 1;
}

BNCH_SWT_HOST int lemire_digit_count(uint64_t x) {
	static constexpr uint64_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, 9999999999, 99999999999, 999999999999, 9999999999999, 99999999999999,
		999999999999999ULL, 9999999999999999ULL, 99999999999999999ULL, 999999999999999999ULL, 9999999999999999999ULL };
	int y						  = (19 * int_log2(x) >> 6);
	y += x > table[y];
	return y + 1;
}

BNCH_SWT_HOST int fast_digit_count(uint32_t x) {
	static constexpr uint32_t table[32] = {
		9ul,
		9ul,
		9ul,
		9ul,
		99ul,
		99ul,
		99ul,
		999ul,
		999ul,
		999ul,
		9999ul,
		9999ul,
		9999ul,
		9999ul,
		99999ul,
		99999ul,
		99999ul,
		999999ul,
		999999ul,
		999999ul,
		9999999ul,
		9999999ul,
		9999999ul,
		9999999ul,
		99999999ul,
		99999999ul,
		99999999ul,
		999999999ul,
		999999999ul,
		999999999ul,
		4294967295ul,
		4294967295ul,
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

BNCH_SWT_HOST int fast_digit_count(uint64_t x) {
	static constexpr uint64_t table[64] = {
		9ull,
		9ull,
		9ull,
		9ull,
		99ull,
		99ull,
		99ull,
		999ull,
		999ull,
		999ull,
		9999ull,
		9999ull,
		9999ull,
		9999ull,
		99999ull,
		99999ull,
		99999ull,
		999999ull,
		999999ull,
		999999ull,
		9999999ull,
		9999999ull,
		9999999ull,
		9999999ull,
		99999999ull,
		99999999ull,
		99999999ull,
		999999999ull,
		999999999ull,
		999999999ull,
		9999999999ull,
		9999999999ull,
		9999999999ull,
		9999999999ull,
		99999999999ull,
		99999999999ull,
		99999999999ull,
		999999999999ull,
		999999999999ull,
		999999999999ull,
		9999999999999ull,
		9999999999999ull,
		9999999999999ull,
		9999999999999ull,
		99999999999999ull,
		99999999999999ull,
		99999999999999ull,
		999999999999999ull,
		999999999999999ull,
		999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		99999999999999999ull,
		99999999999999999ull,
		99999999999999999ull,
		999999999999999999ull,
		999999999999999999ull,
		999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

static constexpr uint64_t total_iterations{ 20 };
static constexpr uint64_t measured_iterations{ 5 };

template<uint64_t count, bnch_swt::string_literal name, uint64_t digit_length, typename value_type> BNCH_SWT_HOST void test_function() {
	std::vector<value_type> random_integers{ generate_random_integers<value_type, digit_length>(count) };
	std::vector<uint64_t> counts{};
	std::vector<value_type> results{};
	counts.resize(count);
	results.resize(count);

	for (uint64_t x = 0; x < count; ++x) {
		counts[x] = fast_digit_count(random_integers[x]);
	}

	using benchmark_type = bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, total_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, false>;

	static constexpr bnch_swt::string_literal bit_size{ bnch_swt::internal::to_string_literal<sizeof(value_type) * 8>() };

	struct lemire_digit_count_type {
		BNCH_SWT_HOST static uint64_t impl(std::vector<value_type>& random_integers, std::vector<value_type>& results) {
			value_type currentCount{};
			for (uint64_t x = 0; x < count; ++x) {
				auto newCount = lemire_digit_count(random_integers[x]);
				results[x]	  = newCount;
				currentCount += static_cast<value_type>(newCount);
				bnch_swt::do_not_optimize_away(currentCount);
			}
			return currentCount;
		}
	};

	struct fast_digit_count_type {
		BNCH_SWT_HOST static uint64_t impl(std::vector<value_type>& random_integers, std::vector<value_type>& results) {
			value_type currentCount{};
			for (uint64_t x = 0; x < count; ++x) {
				auto newCount = fast_digit_count(random_integers[x]);
				results[x]	  = newCount;
				currentCount += static_cast<value_type>(newCount);
				bnch_swt::do_not_optimize_away(currentCount);
			}
			return currentCount;
		}
	};

	struct rtc_digit_count_type {
		BNCH_SWT_HOST static uint64_t impl(std::vector<value_type>& random_integers, std::vector<value_type>& results) {
			value_type currentCount{};
			for (uint64_t x = 0; x < count; ++x) {
				auto newCount = rtc_digit_count(random_integers[x]);
				results[x]	  = newCount;
				currentCount += static_cast<value_type>(newCount);
				bnch_swt::do_not_optimize_away(currentCount);
			}
			return currentCount;
		}
	};

	benchmark_type::template run_benchmark<"lemire-digit-count-" + bit_size, lemire_digit_count_type>(random_integers, results);
	for (uint64_t y = 0; y < count; ++y) {
		if (results[y] != counts[y]) {
			std::cout << "lemire-digit-count-" << sizeof(value_type) * 8 << " failed to count the integers of value : " << random_integers[y]
					  << ",instead it counted : " << results[y] << ", when it should be: " << counts[y] << std::endl;
		}
	}

	benchmark_type::template run_benchmark<"fast-digit-count-" + bit_size, fast_digit_count_type>(random_integers, results);
	for (uint64_t y = 0; y < count; ++y) {
		if (results[y] != counts[y]) {
			std::cout << "fast-digit-count-" << sizeof(value_type) * 8 << " failed to count the integers of value : " << random_integers[y]
					  << ",instead it counted : " << results[y] << ", when it should be: " << counts[y] << std::endl;
		}
	}

	benchmark_type::template run_benchmark<"rtc-digit-count-" + bit_size, rtc_digit_count_type>(random_integers, results);
	for (uint64_t y = 0; y < count; ++y) {
		if (results[y] != counts[y]) {
			std::cout << "rtc-digit-count-" << sizeof(value_type) * 8 << " failed to count the integers of value : " << random_integers[y] << ",instead it counted : " << results[y]
					  << ", when it should be: " << counts[y] << std::endl;
		}
	}

	benchmark_type::print_results(true, true);
}

int main() {
	test_function<10000, "uint32-test-10000-length-0-10", 10, uint32_t>();
	test_function<10000, "uint64-test-10000-length-0-20", 20, uint64_t>();
	return 0;
}