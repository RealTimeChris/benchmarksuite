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
/// https://github.com/RealTimeChris/BenchmarkSuite
#include <bnch_swt/index.hpp>
#include <source_location>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstddef>

#include <string_view>

// Your alignment macro (adjust as needed)
#ifndef BNCH_SWT_ALIGN
	#define BNCH_SWT_ALIGN(alignment) alignas(alignment)
#endif

#ifndef BNCH_SWT_HOST
	#define BNCH_SWT_HOST __forceinline__ __host__
#endif

#ifndef BNCH_SWT_DEVICE
	#define BNCH_SWT_DEVICE __forceinline__ __device__
#endif

#ifndef BNCH_SWT_HOST_DEVICE
	#define BNCH_SWT_HOST_DEVICE __forceinline__ __host__ __device__
#endif

template<typename value_type, value_type...> struct uint_type;

template<typename value_type>
concept uint64_types = std::is_integral_v<value_type> && sizeof(value_type) == 8;

template<typename value_type>
concept uint32_types = std::is_integral_v<value_type> && sizeof(value_type) == 4;

template<typename value_type>
concept uint_types = std::is_unsigned_v<value_type>;

template<typename value_type> constexpr value_type lzcnt_constexpr(value_type value) noexcept {
	if (value == 0) {
		return sizeof(value_type) * 8;
	}

	value_type count = 0;
	value_type mask	 = value_type(1) << (sizeof(value_type) * 8 - 1);

	while ((value & mask) == 0) {
		++count;
		mask >>= 1;
	}

	return count;
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
	} else {
		return 32;
	}
	#else
	return _lzcnt_u32(value);
	#endif
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GNU
	return (value == 0) ? 32 : static_cast<value_type>(__builtin_clz(static_cast<unsigned int>(value)));
#else
	return lzcnt_constexpr(value);
#endif
}

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
	} else {
		return 64;
	}
	#else
	return _lzcnt_u64(value);
	#endif
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GNU
	return (value == 0) ? 64 : static_cast<value_type>(__builtin_clzll(static_cast<uint64_t>(value)));
#else
	return lzcnt_constexpr(value);
#endif
}

template<typename value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) uint_pair {
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> multiplicand{};
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> shift{};
};

struct u128 {
	uint64_t hi;
	uint64_t lo;
	BNCH_SWT_HOST friend constexpr u128 operator+(const u128& a, const u128& b) {
		uint64_t n_lo = a.lo + b.lo;
		uint64_t n_hi = a.hi + b.hi + (n_lo < a.lo ? 1ULL : 0ULL);
		return { n_hi, n_lo };
	}

	BNCH_SWT_HOST friend constexpr u128 operator-(const u128& a, const u128& b) {
		uint64_t n_hi = a.hi - b.hi - (a.lo < b.lo ? 1ULL : 0ULL);
		return { n_hi, a.lo - b.lo };
	}

	BNCH_SWT_HOST friend constexpr u128 operator<<(const u128& v, uint64_t s) {
		if (s == 0ULL)
			return v;
		if (s >= 128ULL)
			return { 0ULL, 0ULL };
		if (s >= 64ULL)
			return { v.lo << (s - 64ULL), 0ULL };
		return { (v.hi << s) | (v.lo >> (64ULL - s)), v.lo << s };
	}

	BNCH_SWT_HOST friend constexpr bool operator>=(const u128& a, const u128& b) {
		return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
	}
};

template<uint64_types value_type, value_type divisor_newer = 0> BNCH_SWT_HOST constexpr uint_pair<value_type> collect_values(value_type divisor_newest = 0) {
	value_type divisor_new{};
	if (std::is_constant_evaluated()) {
		divisor_new = divisor_newer;
	} else {
		divisor_new = divisor_newest;
	}
	if (divisor_new == 1ULL) {
		return uint_pair<value_type>{ { 1ULL }, { 0ULL } };
	}

	value_type div_m1 = divisor_new - 1ULL;

	value_type lz = (div_m1 == 0ULL) ? 128ULL : lzcnt(div_m1) + 64ULL;

	if (lz > 127ULL) {
		return uint_pair<value_type>{ { 1ULL }, { 0ULL } };
	}

	const value_type l		   = 127ULL - lz;
	const value_type shift_val = 64ULL + l;

	u128 rem		   = { 0ULL, 0ULL };
	u128 quo		   = { 0ULL, 0ULL };
	const u128 num	   = u128{ 0ULL, 1ULL } << shift_val;
	const u128 divisor = { 0ULL, divisor_new };

	const u128 target = num + u128{ 0ULL, divisor_new - 1ULL };

	for (int64_t i = 127LL; i >= 0LL; --i) {
		rem = rem << 1ULL;
		if ((i >= 64 && (target.hi & (1ULL << (i - 64)))) || (i < 64 && (target.lo & (1ULL << i)))) {
			rem.lo |= 1ULL;
		}
		if (rem >= divisor) {
			rem = rem - divisor;
			if (i >= 64LL)
				quo.hi |= (1ULL << (i - 64LL));
			else
				quo.lo |= (1ULL << i);
		}
	}

	return uint_pair<value_type>{ { quo.lo }, { shift_val } };
}

template<uint32_types value_type, value_type divisor_newer = 0> BNCH_SWT_HOST constexpr uint_pair<value_type> collect_values(value_type divisor_newest = 0) {
	value_type divisor_new{};
	if (std::is_constant_evaluated()) {
		divisor_new = divisor_newer;
	} else {
		divisor_new = divisor_newest;
	}
	if (divisor_new == 1U) {
		return uint_pair<value_type>{ { 1U }, { 0U } };
	}
	const uint32_t div_minus_1{ divisor_new - 1U };
	const uint32_t lz{ lzcnt(div_minus_1) };
	if (lz > 31U) {
		return uint_pair<value_type>{ { 1U }, { 0U } };
	}
	const uint32_t l{ 31U - lz };
	const uint64_t numerator{ 1ULL << (32ULL + l) };
	const uint32_t m_128{ static_cast<uint32_t>((numerator + divisor_new - 1UL) / divisor_new) };
	return uint_pair<value_type>{ { m_128 }, { 32U + l } };
}

template<typename value_type, value_type const_value_new> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) const_aligned_uint {
	static constexpr bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> const_value{ const_value_new };
};

template<typename value_type, value_type const_value_new> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) aligned_uint : public const_aligned_uint<value_type, const_value_new> {
	BNCH_SWT_HOST_DEVICE constexpr aligned_uint() {
	}
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> value{};
};

template<typename value_type, bool, value_type> struct div_mod_logic;

template<uint64_t size> struct get_int_type_by_size {
	using type = std::conditional_t<size == 8ULL, uint64_t, std::conditional_t<size == 4ULL, uint32_t, std::conditional_t<size == 2ULL, uint16_t, uint8_t>>>;
};

template<uint64_t size> using get_int_type_by_size_t = get_int_type_by_size<size>::type;

template<uint_types value_type> BNCH_SWT_HOST_DEVICE static value_type host_umulhi_impl(value_type u, value_type v) noexcept {
	using half_int_type = get_int_type_by_size_t<(sizeof(value_type) / 2ULL)>;
	static constexpr uint64_t bits_to_shift{ (sizeof(value_type) * 8ULL) / 2ULL };
	static constexpr half_int_type max_value{ std::numeric_limits<half_int_type>::max() };

	value_type u1		  = u >> bits_to_shift;
	value_type u0		  = u & max_value;
	value_type v1		  = v >> bits_to_shift;
	value_type v0		  = v & max_value;
	value_type mid_prod_1 = u1 * v0;
	value_type mid_prod_2 = u0 * v1;

	value_type low_prod		= u0 * v0;
	value_type carry_to_mid = low_prod >> bits_to_shift;

	value_type mid_sum	   = (mid_prod_1 & max_value) + (mid_prod_2 & max_value) + carry_to_mid;
	value_type carry_to_hi = (mid_prod_1 >> bits_to_shift) + (mid_prod_2 >> bits_to_shift) + (mid_sum >> bits_to_shift);

	return (u1 * v1) + carry_to_hi;
}

template<uint_types value_type> BNCH_SWT_HOST_DEVICE static value_type host_umulhi(value_type u, value_type v) noexcept {
#if BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
	const __uint128_t product = static_cast<__uint128_t>(u) * static_cast<__uint128_t>(v);
	return static_cast<value_type>(product >> (sizeof(value_type) * 8));
#elif BNCH_SWT_COMPILER_MSVC
	value_type high_part;
	_umul128(u, v, &high_part);
	return high_part;
#else
	return host_umulhi_impl(u, v);
#endif
}

template<typename value_type, value_type divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic<value_type, true, divisor> : public aligned_uint<value_type, divisor>,
																																		public uint_pair<value_type> {
	BNCH_SWT_HOST_DEVICE constexpr div_mod_logic() {
	}

	BNCH_SWT_HOST_DEVICE operator value_type&() {
		return aligned_uint<value_type, divisor>::value.value;
	}

	BNCH_SWT_HOST void collect_values(value_type d) {
		aligned_uint<value_type, divisor>::value.emplace(d);
		*static_cast<uint_pair<value_type>*>(this) = ::collect_values<value_type>(d);
	}

	template<typename other_type> BNCH_SWT_HOST_DEVICE friend value_type operator/(const other_type& lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	template<typename other_type> BNCH_SWT_HOST_DEVICE friend value_type operator%(const other_type& lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}

  protected:
	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const {
		if (aligned_uint<value_type, divisor>::value.value == 1) {
			return val;
		}
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
		if constexpr (std::same_as<value_type, uint64_t>) {
			return __umul64hi(val, uint_pair<value_type>::multiplicand) >> (uint_pair<value_type>::shift - 64ULL);
		} else {
			return __umulhi(val, uint_pair<value_type>::multiplicand) >> (uint_pair<value_type>::shift - 32ULL);
		}
#else
		if constexpr (std::same_as<value_type, uint64_t>) {
			uint64_t high_part = host_umulhi(uint_pair<value_type>::multiplicand.value, val);
			return high_part >> (uint_pair<value_type>::shift - 64ULL);
		} else {
			uint64_t product = static_cast<uint64_t>(val) * uint_pair<value_type>::multiplicand;
			return static_cast<value_type>(product >> uint_pair<value_type>::shift);
		}
#endif
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		return val - (div(val) * aligned_uint<value_type, divisor>::value.value);
	}
};

template<typename value_type> BNCH_SWT_HOST_DEVICE consteval bool is_power_of_2(value_type N) {
	return N > 0 && (N & (N - 1)) == 0;
}

template<typename value_type> BNCH_SWT_HOST_DEVICE consteval value_type log2_ct(value_type N) {
	value_type result = 0;
	value_type value  = N;
	while (value >>= 1) {
		++result;
	}
	return result;
}

template<typename value_type, value_type divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic<value_type, false, divisor>
	: public const_aligned_uint<value_type, divisor> {
	static constexpr uint_pair<value_type> multiplicand_and_shift{ collect_values<value_type, divisor>() };

	BNCH_SWT_HOST_DEVICE operator value_type() const {
		return const_aligned_uint<value_type, divisor>::const_value.value;
	}

	template<typename other_type> BNCH_SWT_HOST_DEVICE friend value_type operator/(const other_type& lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	template<typename other_type> BNCH_SWT_HOST_DEVICE friend value_type operator%(const other_type& lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}

  protected:
	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const {
		if constexpr (divisor == 1ULL) {
			return val;
		}
		if constexpr (is_power_of_2(divisor)) {
			static constexpr value_type shift_amount{ log2_ct(divisor) };
			return val >> shift_amount;
		} else {
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
			if constexpr (std::same_as<value_type, uint64_t>) {
				return __umul64hi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 64ULL);
			} else {
				return __umulhi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 32ULL);
			}
#else
			if constexpr (std::same_as<value_type, uint64_t>) {
				uint64_t high_part = host_umulhi(multiplicand_and_shift.multiplicand.value, val);
				uint64_t result;
				if constexpr (multiplicand_and_shift.shift >= 64ULL) {
					result = high_part >> (multiplicand_and_shift.shift - 64ULL);
				} else {
					uint64_t low_part = multiplicand_and_shift.multiplicand * val;
					result			  = (high_part << (64ULL - multiplicand_and_shift.shift)) | (low_part >> multiplicand_and_shift.shift);
				}
				return result;
			} else {
				return static_cast<value_type>((static_cast<uint64_t>(val) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
			}
#endif
		}
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		if constexpr (is_power_of_2(divisor)) {
			return val & (divisor - 1);
		} else {
			return val - (div(val) * divisor);
		}
	}
};

template<typename value_type, value_type static_divisor> struct division {
	BNCH_SWT_DEVICE static value_type div(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			static constexpr value_type shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			static constexpr div_mod_logic<value_type, false, static_divisor> mul_shift{};
			return value / mul_shift;
		}
	}
};

template<typename value_type, value_type static_divisor> struct modulo {
	BNCH_SWT_DEVICE static value_type mod(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1ULL);
		} else {
			static constexpr div_mod_logic<value_type, false, static_divisor> mul_shift{};
			return value % mul_shift;
		}
	}
};

constexpr uint64_t TOTAL_ITERATIONS	   = 200;
constexpr uint64_t MEASURED_ITERATIONS = 20;
constexpr size_t N_ELEMENTS			   = 4096ULL * 256ULL;

template<typename value_type> void prepare_data(std::vector<value_type>& host_input, value_type*& d_input, value_type*& d_output_native, value_type*& d_output_magic, value_type*& d_iteration_counter, size_t n, size_t total_iterations) {
	size_t total_elements = n * total_iterations;
	host_input.resize(total_elements);

	for (value_type iter = 0; iter < total_iterations; ++iter) {
		for (size_t i = 0; i < n; ++i) {
			host_input[iter * n + i] = (iter * 999999ULL + i * 1234567ULL + 12345ULL);
		}
	}

	cudaMalloc(&d_input, total_elements * sizeof(value_type));
	cudaMalloc(&d_output_native, n * sizeof(value_type));
	cudaMalloc(&d_output_magic, n * sizeof(value_type));
	cudaMalloc(&d_iteration_counter, sizeof(value_type));

	cudaMemcpy(d_input, host_input.data(), total_elements * sizeof(value_type), cudaMemcpyHostToDevice);

	value_type initial_counter = 0;
	cudaMemcpy(d_iteration_counter, &initial_counter, sizeof(value_type), cudaMemcpyHostToDevice);
}

template<typename value_type> void cleanup(value_type* d_input, value_type* d_output_native, value_type* d_output_magic, value_type* d_iteration_counter) {
	cudaFree(d_input);
	cudaFree(d_output_native);
	cudaFree(d_output_magic);
	cudaFree(d_iteration_counter);
}

template<typename value_type>
__global__ void native_div_kernel(const value_type* __restrict__ input, value_type* __restrict__ output, size_t divisor, size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	if (idx >= n) {
		return;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += input[offset + idx] / divisor;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += input[offset + idx] / divisor;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += input[offset + idx] / divisor;
	}
}

template<typename value_type> __global__ void magic_div_kernel_rt(const value_type* __restrict__ input, value_type* __restrict__ output,
	const div_mod_logic<value_type, true, 0>& __restrict__ magic_new, size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	if (idx >= n) {
		return;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += (input[offset + idx]) / magic_new;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += (input[offset + idx]) / magic_new;
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += (input[offset + idx]) / magic_new;
	}
}

template<uint64_t divisor, typename value_type>
__global__ void magic_div_kernel_ct(const value_type* __restrict__ input, value_type* __restrict__ output, size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	if (idx >= n) {
		return;
	}
	static constexpr div_mod_logic<value_type, false, 2048> magic{};

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += division<value_type, divisor>::div(input[offset + idx]);
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += division<value_type, divisor>::div(input[offset + idx]);
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += division<value_type, divisor>::div(input[offset + idx]);
	}
}

template<uint64_t TEST_DIVISOR> BNCH_SWT_HOST void test_function() {
	cudaDeviceReset();
	{
		std::vector<uint64_t> host_input;
		uint64_t *d_input = nullptr, *d_output_native = nullptr, *d_output_magic = nullptr, *d_iteration_counter = nullptr;

		prepare_data(host_input, d_input, d_output_native, d_output_magic, d_iteration_counter, N_ELEMENTS, TOTAL_ITERATIONS);

		using MagicType = div_mod_logic<uint64_t, true, 0>;
		MagicType magic_div;
		magic_div.collect_values(TEST_DIVISOR);


		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";

		constexpr int BLOCK_SIZE = 256;
		const int GRID_SIZE		 = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 grid(GRID_SIZE);
		dim3 block(BLOCK_SIZE);

		constexpr uint64_t shared_mem	 = 0;
		const uint64_t bytes_transferred = N_ELEMENTS * sizeof(uint64_t);

		std::cout << "\n=== Running division benchmark (64-bit) ===\n"
				  << "Elements per iteration: " << N_ELEMENTS << "\n"
				  << "Total iterations: " << TOTAL_ITERATIONS << "\n"
				  << "Total unique datasets: " << (N_ELEMENTS * TOTAL_ITERATIONS) << "\n"
				  << "Divisor: " << TEST_DIVISOR << "\n\n";

		using Bench = bnch_swt::benchmark_stage<"division-benchmark-64-bit", TOTAL_ITERATIONS, MEASURED_ITERATIONS, bnch_swt::benchmark_types::cuda, false>;

		uint64_t reset_counter = 0;

		std::cout << "=== BASELINE BENCHMARKS ===\n\n";

		//cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		//native_div_kernel<<<grid, block>>>(d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS, d_iteration_counter);
		if (auto error = cudaGetLastError(); error != cudaSuccess) {
			std::cout << "Warmup ERROR: " << cudaGetErrorString(error) << "\n";
		}
		cudaDeviceSynchronize();
		static constexpr auto native_div_kernel_ptr = &native_div_kernel<uint64_t>;
		static constexpr auto magic_div_kernel_ct_ptr = &magic_div_kernel_ct<TEST_DIVISOR, uint64_t>;
		static constexpr auto magic_div_kernel_rt_ptr = &magic_div_kernel_rt<uint64_t>;

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);
		cudaDeviceSynchronize();
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt", magic_div_kernel_rt_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, magic_div, N_ELEMENTS,
			d_iteration_counter);
		cudaDeviceSynchronize();
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-ct", magic_div_kernel_ct_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, N_ELEMENTS,
			d_iteration_counter);
		Bench::print_results();
		cleanup(d_input, d_output_native, d_output_magic, d_iteration_counter);
	}

	{
		std::vector<uint32_t> host_input;
		uint32_t *d_input = nullptr, *d_output_native = nullptr, *d_output_magic = nullptr, *d_iteration_counter = nullptr;

		prepare_data(host_input, d_input, d_output_native, d_output_magic, d_iteration_counter, N_ELEMENTS, TOTAL_ITERATIONS);

		using MagicType = div_mod_logic<uint32_t, true, 0>;
		MagicType magic_div;
		magic_div.collect_values(TEST_DIVISOR);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";

		constexpr int BLOCK_SIZE = 256;
		const int GRID_SIZE		 = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 grid(GRID_SIZE);
		dim3 block(BLOCK_SIZE);

		constexpr uint32_t shared_mem	 = 0;
		const uint32_t bytes_transferred = N_ELEMENTS * sizeof(uint32_t);

		std::cout << "\n=== Running division benchmark (32-bit) ===\n"
				  << "Elements per iteration: " << N_ELEMENTS << "\n"
				  << "Total iterations: " << TOTAL_ITERATIONS << "\n"
				  << "Total unique datasets: " << (N_ELEMENTS * TOTAL_ITERATIONS) << "\n"
				  << "Divisor: " << TEST_DIVISOR << "\n\n";

		using Bench = bnch_swt::benchmark_stage<"division-benchmark-32-bit", TOTAL_ITERATIONS, MEASURED_ITERATIONS, bnch_swt::benchmark_types::cuda, false>;

		uint32_t reset_counter = 0;

		std::cout << "=== BASELINE BENCHMARKS ===\n\n";

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		native_div_kernel<<<grid, block>>>(d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS, d_iteration_counter);
		if (auto error = cudaGetLastError(); error != cudaSuccess) {
			std::cout << "Warmup ERROR: " << cudaGetErrorString(error) << "\n";
		}
		cudaDeviceSynchronize();
		static constexpr auto native_div_kernel_ptr	  = &native_div_kernel<uint32_t>;
		static constexpr auto magic_div_kernel_ct_ptr = &magic_div_kernel_ct<TEST_DIVISOR, uint32_t>;
		static constexpr auto magic_div_kernel_rt_ptr = &magic_div_kernel_rt<uint32_t>;

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt", magic_div_kernel_rt_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, magic_div, N_ELEMENTS,
			d_iteration_counter);
		Bench::print_results();
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-ct", magic_div_kernel_ct_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, N_ELEMENTS, d_iteration_counter);
		Bench::print_results();
		cleanup(d_input, d_output_native, d_output_magic, d_iteration_counter);
	}

	std::cout << "\nBenchmark finished.\n";
}

int main() {
	test_function<32>();
	test_function<64>();
	test_function<256>();
	test_function<512>();
	test_function<1024>();
	test_function<2048>();
	test_function<4096>();
	test_function<100>();
	test_function<1337>();
	test_function<768>();
	test_function<2048>();
	test_function<11008>();
	test_function<14336>();
	test_function<997>();
	test_function<102039132>();
	test_function<10000000000000000000>();
	return 0;
}