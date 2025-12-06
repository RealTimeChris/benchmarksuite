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

template<uint64_t max_length_new> struct BNCH_SWT_ALIGN(64) network_acces_interface {
	static constexpr uint64_t max_length{ max_length_new };
	char* ptr_start{};
	uint64_t length{};
};

struct network_core_config {
	uint64_t max_request_length{};
	uint64_t max_request_count{};
	uint64_t max_response_length{};
	uint64_t max_response_count{};

	static constexpr uint64_t align_to_cache(uint64_t size) {
		return (size + 63) & ~63;
	}

	constexpr uint64_t get_aligned_request_size() const {
		return align_to_cache(max_request_length);
	}

	constexpr uint64_t get_aligned_response_size() const {
		return align_to_cache(max_response_length);
	}

	constexpr uint64_t get_total_request_byte_size() const {
		return max_request_count * get_aligned_request_size();
	}

	constexpr uint64_t get_total_response_byte_size() const {
		return max_response_count * get_aligned_response_size();
	}
};

template<network_core_config config = network_core_config{}> struct network_core {
	using req_interface_t  = network_acces_interface<config.max_request_length>;
	using resp_interface_t = network_acces_interface<config.max_response_length>;

	std::vector<char> out_data;
	std::vector<char> in_data;

	std::vector<resp_interface_t> out_interfaces;
	std::vector<req_interface_t> in_interfaces;
	BNCH_SWT_ALIGN(64) std::atomic<resp_interface_t*> out_head_ptr {};
	BNCH_SWT_ALIGN(64) std::atomic<req_interface_t*> in_head_ptr {};

	BNCH_SWT_HOST network_core() {
		static constexpr uint64_t response_byte_size{ config.get_total_response_byte_size() };
		static constexpr uint64_t request_byte_size{ config.get_total_request_byte_size() };
		out_data.resize(response_byte_size);
		in_data.resize(request_byte_size);

		out_interfaces.resize(config.max_response_count);
		in_interfaces.resize(config.max_request_count);

		static constexpr uint64_t req_stride  = config.get_aligned_request_size();
		static constexpr uint64_t resp_stride = config.get_aligned_response_size();

		for (uint64_t x = 0; x < config.max_response_count; ++x) {
			out_interfaces[x].ptr_start = out_data.data() + (x * resp_stride);
			out_interfaces[x].length	= 0;
		}

		for (uint64_t x = 0; x < config.max_request_count; ++x) {
			in_interfaces[x].ptr_start = in_data.data() + (x * req_stride);
			in_interfaces[x].length	   = 0;
		}
	}

	BNCH_SWT_HOST void register_with_kernel(int io_uring_fd) {
	}
};

int main() {
	auto new_value_01 = std::atomic<int64_t>{}.load();
	auto new_value_02 = std::atomic_signed_lock_free{}.load();
	std::cout << "std::atomic<uint64_t>::value_type: " << typeid(new_value_01).name() << std::endl;
	std::cout << "std::atomic_signed_lock_free::value_type: " << typeid(new_value_02).name() << std::endl;

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
	network_core core{};

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

	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_uint64", test_atomic_uint64>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_signed_lock_free", test_atomic_signed_lock_free>();
	bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::print_results();

	return 0;
}