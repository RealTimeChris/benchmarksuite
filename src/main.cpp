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
#include <void-numerics>
#include <bnch_swt/index.hpp>
#include <source_location>
#include <atomic>
#include <thread>

inline static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10,
	10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

inline static constexpr uint64_t digitCountThresholds[]{ 0ULL, 9ULL, 99ULL, 999ULL, 9999ULL, 99999ULL, 999999ULL, 9999999ULL, 99999999ULL, 999999999ULL, 9999999999ULL,
	99999999999ULL, 999999999999ULL, 9999999999999ULL, 99999999999999ULL, 999999999999999ULL, 9999999999999999ULL, 99999999999999999ULL, 999999999999999999ULL,
	9999999999999999999ULL };

BNCH_SWT_HOST static uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

BNCH_SWT_NOINLINE void uint8_test() {
	std::string test_string{};
	{
		uint8_t value{ bnch_swt::random_generator<uint8_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void int8_test() {
	std::string test_string{};
	{
		int8_t value{ bnch_swt::random_generator<int8_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void uint16_test() {
	std::string test_string{};
	{
		uint16_t value{ bnch_swt::random_generator<uint16_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void int16_test() {
	std::string test_string{};
	{
		int16_t value{ bnch_swt::random_generator<int16_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void uint32_test() {
	std::string test_string{};
	{
		uint32_t value{ bnch_swt::random_generator<uint32_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void int32_test() {
	std::string test_string{};
	{
		int32_t value{ bnch_swt::random_generator<int32_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void uint64_test() {
	std::string test_string{};
	{
		uint64_t value{ bnch_swt::random_generator<uint64_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

BNCH_SWT_NOINLINE void int64_test() {
	std::string test_string{};
	{
		int64_t value{ bnch_swt::random_generator<int64_t>{}.impl() };
		test_string.resize(fastDigitCount(value) + 1);
		vn::to_chars(test_string.data(), test_string.data() + test_string.size(), value);
		bnch_swt::do_not_optimize_away(test_string);
	}
}

int main() {
	uint8_test();
	int8_test();
	uint16_test();
	int16_test();
	uint32_test();
	int32_test();
	uint64_test();
	int64_test();
	return 0;
}