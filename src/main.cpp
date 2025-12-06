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
static constexpr uint64_t measured_iterations{ 10 };
static constexpr uint64_t wait_notify_cycles{ 1000 };

template<typename value_type> void test_function() {
	std::cout << std::source_location::current().function_name() << std::endl;
}

template<typename value_type, template<typename...> typename template_type> static constexpr bool is_specialization_v{ false };

template<template<typename...> typename template_type, typename... args> static constexpr bool is_specialization_v<template_type<args...>, template_type>{ true };

template<typename value_type, template<typename...> typename template_type>
concept is_specialization = is_specialization_v<value_type, template_type>;

template<typename value_type>
concept is_vector = is_specialization<value_type, std::vector>;

template<typename value_type>
concept is_duration = is_specialization<value_type, std::chrono::duration>;

int main() {
	std::cout << "IS IT?: " << is_duration<std::chrono ::duration<double, std::milli>> << std::endl;
	std::cout << "IS IT?: " << is_duration<int32_t> << std::endl;
	test_function<std::atomic_signed_lock_free::value_type>();

	struct test_atomic_uint64 {
		BNCH_SWT_HOST static uint64_t impl() {
			std::atomic<uint64_t> flag{ 0 };
			auto start = std::chrono::high_resolution_clock::now();
			std::thread waiter([&]() {
				uint64_t value{};
				for (uint64_t i = 0; i < wait_notify_cycles; ++i) {
					uint64_t expected = i;
					++value;
					flag.wait(expected);
					bnch_swt::do_not_optimize_away(value);
				}
			});
			uint64_t value{};
			for (uint64_t i = 1; i <= wait_notify_cycles; ++i) {
				flag.store(i, std::memory_order_release);
				flag.notify_one();
				bnch_swt::do_not_optimize_away(value);
			}
			waiter.join();
			auto end = std::chrono::high_resolution_clock::now();
			return 20000;
		}
	};

	struct test_atomic_signed_lock_free {
		BNCH_SWT_HOST static uint64_t impl() {
			std::atomic_signed_lock_free flag{ 0 };
			auto start = std::chrono::high_resolution_clock::now();
			std::thread waiter([&]() {
				uint64_t value{};
				for (uint64_t i = 0; i < wait_notify_cycles; ++i) {
					auto expected = static_cast<std::atomic_signed_lock_free::value_type>(i);
					++value;
					flag.wait(expected);
					bnch_swt::do_not_optimize_away(value);
				}
			});

			uint64_t value{};
			for (uint64_t i = 1; i <= wait_notify_cycles; ++i) {
				flag.store(static_cast<std::atomic_signed_lock_free::value_type>(i), std::memory_order_release);
				flag.notify_one();
				bnch_swt::do_not_optimize_away(value);
			}

			waiter.join();
			auto end = std::chrono::high_resolution_clock::now();
			return 20000;
		}
	};

	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_uint64", test_atomic_uint64>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_signed_lock_free", test_atomic_signed_lock_free>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::print_results();

	return 0;
}