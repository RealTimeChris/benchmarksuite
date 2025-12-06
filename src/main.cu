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
template<typename value_type> BNCH_SWT_HOST_DEVICE constexpr value_type lzcnt_device(value_type value) noexcept {
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
		return lzcnt_device(value);
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
	return lzcnt_device(value);
#endif
}

template<uint64_types value_type> BNCH_SWT_HOST constexpr value_type lzcnt(const value_type value) noexcept {
	if (std::is_constant_evaluated()) {
		return lzcnt_device(value);
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
	return lzcnt_device(value);
#endif
}

template<uint_types value_type> struct BNCH_SWT_ALIGN(16) uint_pair {
	template<uint_types value_type_new> friend struct div_mod_logic_new;
	bnch_swt::aligned_const<value_type, (sizeof(value_type) * 2) % 16> multiplicand;
	bnch_swt::aligned_const<value_type, (sizeof(value_type) * 2) % 16> shift;

	using signed_type = std::make_signed_t<value_type>;
	static constexpr signed_type single_bits{ static_cast<signed_type>(sizeof(value_type) * 8) };
	static constexpr value_type double_bits{ static_cast<value_type>(sizeof(value_type) * 2 * 8) };
	static constexpr value_type double_bits_sub_1{ static_cast<value_type>((sizeof(value_type) * 2 * 8) - 1) };

	BNCH_SWT_HOST_DEVICE constexpr uint_pair() {
	}

	BNCH_SWT_HOST_DEVICE friend constexpr uint_pair operator+(const uint_pair& a, const uint_pair& b) noexcept {
		value_type n_smultiplicandft = a.shift + b.shift;
		value_type n_multiplicand	 = a.multiplicand + b.multiplicand + (n_smultiplicandft < a.shift ? 1ULL : 0ULL);
		return { { n_multiplicand }, { n_smultiplicandft } };
	}

	BNCH_SWT_HOST_DEVICE friend constexpr uint_pair operator-(const uint_pair& a, const uint_pair& b) noexcept {
		value_type n_multiplicand = a.multiplicand - b.multiplicand - (a.shift < b.shift ? 1ULL : 0ULL);
		return { { n_multiplicand }, { a.shift - b.shift } };
	}

	BNCH_SWT_HOST_DEVICE friend constexpr uint_pair operator<<(const uint_pair& v, value_type s) noexcept {
		if (s == 0ULL) {
			return v;
		}
		if (s >= double_bits) {
			return {};
		}
		if (s >= single_bits) {
			return { { v.shift << (s - single_bits) }, { 0ULL } };
		}
		return { { (v.multiplicand << s) | (v.shift >> (single_bits - s)) }, { v.shift << s } };
	}

	BNCH_SWT_HOST_DEVICE friend constexpr bool operator>=(const uint_pair& a, const uint_pair& b) noexcept {
		return (a.multiplicand > b.multiplicand) || (a.multiplicand == b.multiplicand && a.shift >= b.shift);
	}

  public:
	BNCH_SWT_HOST_DEVICE constexpr uint_pair(value_type value_01, value_type value_02) : multiplicand{ value_01 }, shift{ value_02 } {
	}

	BNCH_SWT_HOST_DEVICE static constexpr uint_pair collect_values(value_type divisor_new) noexcept {
		if (divisor_new == 1ULL) {
			return { { 1ULL }, { 0ULL } };
		}

		value_type div_m1 = divisor_new - 1ULL;

		value_type lz = (div_m1 == 0ULL) ? uint_pair::double_bits : lzcnt_device(div_m1) + uint_pair::single_bits;

		if (lz > uint_pair::double_bits_sub_1) {
			return { { 1ULL }, { 0ULL } };
		}

		const value_type l{ uint_pair::double_bits_sub_1 - lz };
		const value_type mul_and_shift{ uint_pair::single_bits + l };

		uint_pair rem{};
		uint_pair quo{};
		const uint_pair num{ uint_pair{ { 0ULL }, { 1ULL } } << mul_and_shift };
		const uint_pair divisor{ { 0ULL }, { divisor_new } };

		const uint_pair target = num + uint_pair{ { static_cast<value_type>(0ULL) }, { static_cast<value_type>(divisor_new - 1ULL) } };

		for (signed_type i = uint_pair::double_bits_sub_1; i >= 0LL; --i) {
			rem = rem << 1ULL;
			if ((i >= uint_pair::single_bits && (target.multiplicand & (1ULL << (i - uint_pair::single_bits)))) || (i < uint_pair::single_bits && (target.shift & (1ULL << i)))) {
				rem.shift.value |= 1ULL;
			}
			if (rem >= divisor) {
				rem = rem - divisor;
				if (i >= uint_pair::single_bits) {
					quo.multiplicand.value |= (1ULL << (i - uint_pair::single_bits));
				} else {
					quo.shift.value |= (1ULL << i);
				}
			}
		}

		return { { static_cast<value_type>(quo.shift) }, { mul_and_shift } };
	}
};

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

template<uint_types value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) aligned_uint_new {
  protected:
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> value;
};

template<uint_types value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic_new : public aligned_uint_new<value_type>, public uint_pair<value_type> {
	BNCH_SWT_HOST constexpr value_type get_value() const noexcept {
		return aligned_uint_new<value_type>::value.value;
	}

	BNCH_SWT_HOST_DEVICE static constexpr auto collect_values(value_type d) noexcept {
		div_mod_logic_new return_value{};
		return_value.value.emplace(d);
		static_cast<uint_pair<value_type>&>(return_value) = uint_pair<value_type>::collect_values(d);
		return return_value;
	}

	template<uint_types other_type> BNCH_SWT_HOST_DEVICE friend value_type operator/(const other_type& lhs, const div_mod_logic_new& rhs) noexcept {
		return rhs.div(lhs);
	}

	template<uint_types other_type> BNCH_SWT_HOST_DEVICE friend value_type operator%(const other_type& lhs, const div_mod_logic_new& rhs) noexcept {
		return rhs.mod(lhs);
	}

  protected:
	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const noexcept {
		if (aligned_uint_new<value_type>::value.value == 1) {
			return val;
		}
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
		if constexpr (std::is_same_v<value_type, uint64_t>) {
			return __umul64hi(val, uint_pair<value_type>::multiplicand) >> (uint_pair<value_type>::shift - 64ULL);
		} else {
			return __umulhi(val, uint_pair<value_type>::multiplicand) >> (uint_pair<value_type>::shift - 32ULL);
		}
#else
		if constexpr (std::is_same_v<value_type, uint64_t>) {
			uint64_t high_part = host_umulhi(uint_pair<value_type>::multiplicand.value, val);
			return high_part >> (uint_pair<value_type>::shift - 64ULL);
		} else {
			uint64_t product = static_cast<uint64_t>(val) * uint_pair<value_type>::multiplicand;
			return static_cast<value_type>(product >> uint_pair<value_type>::shift);
		}
#endif
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const noexcept {
		return val - (div(val) * aligned_uint_new<value_type>::value.value);
	}
};

template<typename value_type, value_type static_divisor> struct division {
	BNCH_SWT_DEVICE static value_type div(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			static constexpr value_type shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			static constexpr div_mod_logic_new<value_type> mul_shift{ div_mod_logic_new<value_type>::collect_values(static_divisor) };
			return value / mul_shift;
		}
	}
};

template<typename value_type, value_type static_divisor> struct modulo {
	
	BNCH_SWT_DEVICE static value_type mod(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1ULL);
		} else {
			static constexpr div_mod_logic_new<value_type> mul_shift{ div_mod_logic_new<value_type>::collect_values(static_divisor) };
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

	for (uint32_t x = 0; x < 65536; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += input[offset + idx] / divisor;
	}
}

template<typename value_type> __global__ void magic_div_kernel_rt(const value_type* __restrict__ input, value_type* __restrict__ output,
	const div_mod_logic_new<value_type>& __restrict__ magic_new, size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	for (uint32_t x = 0; x < 65536; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += (input[offset + idx]) / magic_new;
	}
}

template<typename value_type> __constant__ div_mod_logic_new<value_type> magic_new;

template<typename value_type> __global__ void magic_div_kernel_rt_const(const value_type* __restrict__ input, value_type* __restrict__ output,
	size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	for (uint32_t x = 0; x < 65536; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += (input[offset + idx]) / magic_new<value_type>;
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

	for (uint32_t x = 0; x < 65536; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += division<value_type, divisor>::div(input[offset + idx]);
	}
}

template<uint64_t TEST_DIVISOR> void test_function() {
	cudaDeviceReset();
	
	{
		std::vector<uint64_t> host_input;
		uint64_t *d_input = nullptr, *d_output_native = nullptr, *d_output_magic = nullptr, *d_iteration_counter = nullptr;

		prepare_data(host_input, d_input, d_output_native, d_output_magic, d_iteration_counter, N_ELEMENTS, TOTAL_ITERATIONS);

		using MagicType = div_mod_logic_new<uint64_t>;
		MagicType magic_div{ div_mod_logic_new<uint64_t>::collect_values(TEST_DIVISOR) };

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
		static constexpr auto magic_div_kernel_rt_const_ptr = &magic_div_kernel_rt_const<uint64_t>;

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt", magic_div_kernel_rt_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, magic_div, N_ELEMENTS,
			d_iteration_counter);
		cudaMemcpyToSymbol(magic_new<uint64_t>, &magic_div, sizeof(magic_div));
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt-const", magic_div_kernel_rt_const_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, N_ELEMENTS,
			d_iteration_counter);
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

		using MagicType = div_mod_logic_new<uint32_t>;
		MagicType magic_div{ MagicType::collect_values(TEST_DIVISOR) };

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
		static constexpr auto magic_div_kernel_rt_ptr		= &magic_div_kernel_rt<uint32_t>;
		static constexpr auto magic_div_kernel_rt_const_ptr = &magic_div_kernel_rt_const<uint32_t>;

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt", magic_div_kernel_rt_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, magic_div, N_ELEMENTS,
			d_iteration_counter);
		cudaMemcpyToSymbol(magic_new<uint32_t>, &magic_div, sizeof(magic_div));
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division-rt-const", magic_div_kernel_rt_const_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, N_ELEMENTS,
			d_iteration_counter);
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