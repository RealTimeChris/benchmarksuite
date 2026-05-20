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
#include <bnch_swt/index.hpp>
#include <source_location>
#include <atomic>
#include <thread>

static constexpr char values[]{ "12345678901234567890" };

BNCH_SWT_ALIGN(64) static constexpr const char* aligned_ptr{ values };
static constexpr const char* ptr{ values };

BNCH_SWT_NOINLINE void test_function_01(char* string) {
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	auto value = rand();
	std::memcpy(string, aligned_ptr, value % 8);
}

BNCH_SWT_NOINLINE void test_function_02(char* string) {
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	auto value = rand();
	std::memcpy(string, aligned_ptr, value % 8);
}

int main() {
	std::string value{};
	value.resize(128);
	test_function_02(value.data());
	bnch_swt::do_not_optimize_away(value);
	test_function_01(value.data());
	bnch_swt::do_not_optimize_away(value);
	return 0;
}