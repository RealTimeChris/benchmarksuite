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

static constexpr uint64_t total_iterations{ 10 };
static constexpr uint64_t measured_iterations{ 10 };

int main() {

	struct test_struct_no_pause {
		BNCH_SWT_HOST static uint64_t impl() {
			[[maybe_unused]] auto start = std::chrono::high_resolution_clock::now();
			[[maybe_unused]] auto end = std::chrono::high_resolution_clock::now();
			return 200000ull;
		}
	};

	struct test_struct_pause {
		BNCH_SWT_HOST static uint64_t impl() {
			[[maybe_unused]] auto start = std::chrono::high_resolution_clock::now();
			[[maybe_unused]] auto end = std::chrono::high_resolution_clock::now();
			return 200000ull;
		}
	};

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::run_benchmark<"no-yield", test_struct_no_pause>();
	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::run_benchmark<"yield", test_struct_pause>();

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::print_results();
	return 0;
}
