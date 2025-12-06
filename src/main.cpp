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

static constexpr uint64_t total_iterations{ 10000 };
static constexpr uint64_t measured_iterations{ 100 };
static constexpr uint64_t wait_notify_cycles{ 1000 };

struct test_atomic_uint64 {
	BNCH_SWT_HOST static uint64_t impl() {
		std::atomic<int64_t> flag{ 0 };
		std::thread waiter([&]() {
			int64_t value{};
			for (int64_t i = 0; i < wait_notify_cycles; ++i) {
				int64_t expected = i;
				++value;
				flag.wait(expected);
				bnch_swt::do_not_optimize_away(value);
			}
		});
		int64_t value{};
		for (int64_t i = 1; i <= wait_notify_cycles; ++i) {
			flag.store(i, std::memory_order_release);
			flag.notify_one();
			value = flag.load();
			bnch_swt::do_not_optimize_away(value);
		}
		waiter.join();
		return 20000;
	}
};

struct test_atomic_signed_lock_free {
	BNCH_SWT_HOST static uint64_t impl() {
		std::atomic_signed_lock_free flag{ 0 };
		std::thread waiter([&]() {
			typename std::atomic_signed_lock_free::value_type value{};
			for (typename std::atomic_signed_lock_free::value_type i = 0; i < wait_notify_cycles; ++i) {
				typename std::atomic_signed_lock_free::value_type expected = i;
				++value;
				flag.wait(expected);
				bnch_swt::do_not_optimize_away(value);
			}
		});
		typename std::atomic_signed_lock_free::value_type value{};
		for (typename std::atomic_signed_lock_free::value_type i = 1; i <= wait_notify_cycles; ++i) {
			flag.store(i, std::memory_order_release);
			flag.notify_one();
			value = flag.load();
			bnch_swt::do_not_optimize_away(value);
		}
		waiter.join();
		return 20000;
	}
};

int main() {
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_uint64", test_atomic_uint64>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_signed_lock_free", test_atomic_signed_lock_free>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::print_results();

	return 0;
}