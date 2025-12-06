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

template<typename value_type>
concept uint32_types = sizeof(std::remove_cvref_t<value_type>) == 4;

template<typename value_type>
concept uint16_types = sizeof(std::remove_cvref_t<value_type>) == 2;

template<typename value_type>
concept uint64_types = sizeof(std::remove_cvref_t<value_type>) == 8;

// -------------------------------------------------------------------------
// CONSTEXPR HELPER
// -------------------------------------------------------------------------
template<typename T> BNCH_SWT_HOST constexpr T lzcnt_constexpr(T value) noexcept {
	if (value == 0)
		return static_cast<T>(sizeof(T) * 8);

	T count		 = 0;
	T total_bits = static_cast<T>(sizeof(T) * 8);
	T msb_mask	 = static_cast<T>(1) << (total_bits - 1);

	while ((value & msb_mask) == 0) {
		value <<= 1;
		++count;
	}
	return count;
}

// -------------------------------------------------------------------------
// 16-BIT IMPLEMENTATION (CUDA SAFE)
// -------------------------------------------------------------------------
template<uint16_types value_type> BNCH_SWT_HOST constexpr value_type lzcnt(const value_type value) noexcept {
	// C++20: The compiler WILL optimize this branch away.
	if (std::is_constant_evaluated()) {
		return lzcnt_constexpr(value);
	}

#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
	return static_cast<value_type>(__clz(static_cast<int>(value))) - 16;
#elif BNCH_SWT_COMPILER_MSVC
	#if BNCH_SWT_ARCH_ARM64
	unsigned int leading_zero = 0;
	if (_BitScanReverse32(&leading_zero, value)) {
		return 15 - static_cast<value_type>(leading_zero);
	}
	#else
	return static_cast<value_type>(_lzcnt_u32(value) - 16);
	#endif
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
	if (value == 0)
		return 16;
	return static_cast<value_type>(__builtin_clz(static_cast<unsigned int>(value))) - 16;
#else
	return lzcnt_constexpr(value);
#endif
}

template<uint32_types value_type> BNCH_SWT_HOST constexpr value_type lzcnt(const value_type value) noexcept {
	if (std::is_constant_evaluated()) {
		return lzcnt_constexpr(value);
	}

#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
	return static_cast<value_type>(__clz(static_cast<int>(value)));
#elif BNCH_SWT_COMPILER_MSVC
	#if BNCH_SWT_ARCH_ARM64
	unsigned int leading_zero = 0;
	if (_BitScanReverse32(&leading_zero, value)) {
		return 31 - static_cast<value_type>(leading_zero);
	}
	#else
	return _lzcnt_u32(value);
	#endif
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
	if (value == 0)
		return 32;
	return static_cast<value_type>(__builtin_clz(static_cast<unsigned int>(value)));
#else
	return lzcnt_constexpr(value);
#endif
}

// -------------------------------------------------------------------------
// 64-BIT IMPLEMENTATION (CUDA SAFE)
// -------------------------------------------------------------------------
template<uint64_types value_type> BNCH_SWT_HOST constexpr value_type lzcnt(const value_type value) noexcept {
	if (std::is_constant_evaluated()) {
		return lzcnt_constexpr(value);
	}

#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
	return static_cast<value_type>(__clzll(static_cast<long long>(value)));
#elif BNCH_SWT_COMPILER_MSVC
	#if BNCH_SWT_ARCH_ARM64
	unsigned long leading_zero = 0;
	if (_BitScanReverse64(&leading_zero, value)) {
		return 63 - static_cast<value_type>(leading_zero);
	}
	#else
	return _lzcnt_u64(value);
	#endif
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
	if (value == 0)
		return 64;
	return static_cast<value_type>(__builtin_clzll(static_cast<unsigned long long>(value)));
#else
	return lzcnt_constexpr(value);
#endif
}

int main() {
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	uint16_t value_new16{ static_cast<uint16_t>(rand()) };
	uint32_t value_new32{ static_cast<uint32_t>(rand()) };
	uint64_t value_new64{ static_cast<uint64_t>(rand()) };
	uint16_t value16{ lzcnt(value_new16) };
	std::cout << "CURRENT VALUE: " << value16 << std::endl;
	uint32_t value32{ lzcnt(value_new32) };
	std::cout << "CURRENT VALUE: " << value32 << std::endl;
	uint64_t value64{ lzcnt(value_new64) };
	std::cout << "CURRENT VALUE: " << value64 << std::endl;
	

	//bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::run_benchmark<"atomic_uint64", test_atomic_uint64>();
	//bnch_swt::benchmark_stage<"wait_notify_benchmark", total_iterations, measured_iterations>::print_results();

	return 0;
}