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

static constexpr uint64_t total_iterations{ 100 };
static constexpr uint64_t measured_iterations{ 10 };
static constexpr uint64_t floats_to_generate{ 1000000 };

#include <iostream>
#include <stdint.h>

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <intrin.h>
#endif

template<auto enum_error, typename... types> struct error_printer_impl;

template<bool value, auto enum_error, typename... value_to_test> struct static_assert_printer {
	static constexpr bool impl{ [] {
		if constexpr (!value) {
			error_printer_impl<enum_error, value_to_test...>::nonexistent_value;
			return false;
		} else {
			return true;
		}
	}() };
};

template<auto enum_error, auto... values> struct error_printer_impl_val;

template<bool value, auto enum_error, auto... values> struct static_assert_printer_val {
	static constexpr bool impl{ [] {
		if constexpr (!value) {
			error_printer_impl_val<enum_error, values...>::nonexistent_value;
			return false;
		} else {
			return true;
		}
	}() };
};

enum class kernel_types : uint8_t {
	weights,
	global_inputs,
	get_rows,
	rms_norm,
	mul,
	mul_mat,
	mul_mat_moe,
	reshape,
	transpose,
	permute,
	view,
	rope,
	softmax,
	silu,
	copy,
	cont,
	add,
	sub,
	div,
	top_k,
	weighted_sum,
	sample_tokens,
	count,
};

template<typename value_type>
concept uint16_types = sizeof(std::remove_cvref_t<value_type>) == 2;

template<typename value_type>
concept uint32_types = sizeof(std::remove_cvref_t<value_type>) == 4;

template<typename value_type>
concept uint64_types = sizeof(std::remove_cvref_t<value_type>) == 8;

template<typename value_type>
concept dimensions_types = requires() { std::remove_cvref_t<value_type>::dimension_identifiers; };

template<typename value_type>
concept kernel_types_types = requires() { std::remove_cvref_t<value_type>::kernel_type; };

template<typename value_type>
concept unary_kernel_types = kernel_types_types<value_type> &&
	(std::remove_cvref_t<value_type>::kernel_type == kernel_types::rms_norm || std::remove_cvref_t<value_type>::kernel_type == kernel_types::reshape ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::transpose || std::remove_cvref_t<value_type>::kernel_type == kernel_types::permute ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::view || std::remove_cvref_t<value_type>::kernel_type == kernel_types::silu ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::cont || std::remove_cvref_t<value_type>::kernel_type == kernel_types::top_k);

template<typename value_type>
concept binary_kernel_types = kernel_types_types<value_type> &&
	(std::remove_cvref_t<value_type>::kernel_type == kernel_types::mul || std::remove_cvref_t<value_type>::kernel_type == kernel_types::add ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::sub || std::remove_cvref_t<value_type>::kernel_type == kernel_types::div ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::mul_mat || std::remove_cvref_t<value_type>::kernel_type == kernel_types::get_rows ||
		std::remove_cvref_t<value_type>::kernel_type == kernel_types::softmax || std::remove_cvref_t<value_type>::kernel_type == kernel_types::copy);

template<typename value_type>
concept ternary_kernel_types = kernel_types_types<value_type> && (std::remove_cvref_t<value_type>::kernel_type == kernel_types::rope);

template<typename value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) uint_pair {
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) value_type multiplicand {};
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) value_type shift {};
};

template<typename value_type, value_type...> struct uint_type;

template<uint64_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	value_type lo{};
	value_type hi{};

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type h, value_type l = 0) : lo{ l }, hi{ h } {}

	constexpr explicit operator value_type() const {
		return lo;
	}

	constexpr bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	constexpr bool operator>(const uint_type& other) const {
		if (hi != other.hi) {
			return hi > other.hi;
		}
		return lo > other.lo;
	}

	constexpr bool operator<(const uint_type& other) const {
		return other > *this;
	}

	constexpr bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	constexpr uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	constexpr uint_type operator-(const uint_type& other) const {
		const value_type new_lo = lo - other.lo;
		const value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	constexpr uint_type operator<<(int32_t shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ lo << (shift - 64), 0 };
		}
		return uint_type{ (hi << shift) | (lo >> (64 - shift)), lo << shift };
	}

	constexpr uint_type operator/(const uint_type& other) const {
		if (other.hi == 0 && other.lo == 0) {
			return uint_type{ 0, 0 };
		}

		if (other > *this) {
			return uint_type{ 0, 0 };
		}

		if (other == *this) {
			return uint_type{ 0, 1 };
		}

		uint_type quotient{ 0, 0 };
		uint_type remainder{ 0, 0 };
		const uint_type divisor = other;

		for (int32_t i = 127; i >= 0; --i) {
			remainder = remainder << 1;

			if ((i >= 64 && (hi & (1ULL << (i - 64)))) || (i < 64 && (lo & (1ULL << i)))) {
				remainder.lo |= 1;
			}

			if (remainder >= divisor) {
				remainder = remainder - divisor;
				if (i >= 64) {
					quotient.hi |= (1ULL << (i - 64));
				} else {
					quotient.lo |= (1ULL << i);
				}
			}
		}
		return quotient;
	}

	constexpr value_type lzcnt() const {
		if (hi != 0) {
			value_type x = hi;
			value_type n = 0;
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		} else {
			value_type x = lo;
			value_type n = 64;
			if (x == 0) {
				return 128;
			}
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		}
	}

	consteval static uint_pair<value_type> collect_values() {
		constexpr uint_type div_temp	= divisor_new;
		constexpr uint_type div_minus_1 = divisor_new - 1;
		constexpr value_type l			= 127 - div_minus_1.lzcnt();
		constexpr uint_type numerator	= uint_type{ 1 } << (64 + static_cast<value_type>(l));
		constexpr uint_type m_128		= (numerator + div_temp - 1) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64 + l };
	}
};

template<uint32_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	static constexpr uint64_t shift(uint64_t value, int32_t shift) {
		if (shift == 0ULL) {
			return value;
		}
		if (shift >= 64ULL) {
			return 0ULL;
		}
		return value << shift;
	}

	static constexpr uint64_t div(const uint64_t& lhs, const uint64_t& rhs) {
		if (rhs == 0ULL) {
			return 0ULL;
		}
		if (rhs > lhs) {
			return 0ULL;
		}
		if (rhs == lhs) {
			return 1ULL;
		}
		return static_cast<value_type>(static_cast<uint64_t>(lhs) / static_cast<uint64_t>(rhs));
	}

	static constexpr value_type lzcnt(uint64_t value) {
		if (value == 0ULL) {
			return 64ULL;
		}
		uint64_t x	 = value;
		value_type n = 0ULL;
		if (x <= 0x00000000FFFFFFFFULL) {
			n += 32ULL;
			x <<= 32ULL;
		}
		if (x <= 0x0000FFFFFFFFFFFFULL) {
			n += 16ULL;
			x <<= 16ULL;
		}
		if (x <= 0x00FFFFFFFFFFFFFFULL) {
			n += 8ULL;
			x <<= 8ULL;
		}
		if (x <= 0x0FFFFFFFFFFFFFFFULL) {
			n += 4ULL;
			x <<= 4ULL;
		}
		if (x <= 0x3FFFFFFFFFFFFFFFULL) {
			n += 2ULL;
			x <<= 2ULL;
		}
		if (x <= 0x7FFFFFFFFFFFFFFFULL) {
			n += 1ULL;
		}
		return n;
	}

	consteval static uint_pair<value_type> collect_values() {
		constexpr uint64_t div_temp	   = divisor_new;
		constexpr uint64_t div_minus_1 = divisor_new - 1ULL;
		constexpr value_type l		   = 63ULL - lzcnt(div_minus_1);
		constexpr uint64_t numerator   = shift(1ULL, static_cast<int32_t>(32ULL + static_cast<value_type>(l)));
		constexpr uint64_t m_128	   = div(numerator + divisor_new - 1, divisor_new);
		return uint_pair<value_type>{ static_cast<value_type>(m_128), static_cast<value_type>(32ULL + l) };
	}
};

template<uint64_types value_type> struct uint_type<value_type> {
	value_type lo{};
	value_type hi{};

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type h, value_type l = 0) : lo{ l }, hi{ h } {}

	BNCH_SWT_HOST_DEVICE explicit operator value_type() const {
		return lo;
	}

	BNCH_SWT_HOST_DEVICE bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	BNCH_SWT_HOST_DEVICE bool operator>(const uint_type& other) const {
		return other < *this;
	}

	BNCH_SWT_HOST_DEVICE bool operator<(const uint_type& other) const {
		if (hi != other.hi) {
			return hi < other.hi;
		}
		return lo < other.lo;
	}

	BNCH_SWT_HOST_DEVICE bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	BNCH_SWT_HOST_DEVICE uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator-(const uint_type& other) const {
		const value_type new_lo = lo - other.lo;
		const value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator<<(int32_t shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ lo << (shift - 64), 0 };
		}
		return uint_type{ (hi << shift) | (lo >> (64 - shift)), lo << shift };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator/(const uint_type& other) const {
		if (other.hi == 0 && other.lo == 0) {
			return uint_type{ 0, 0 };
		}

		if (other > *this) {
			return uint_type{ 0, 0 };
		}

		if (other == *this) {
			return uint_type{ 0, 1 };
		}

		uint_type quotient{ 0, 0 };
		uint_type remainder{ 0, 0 };
		const uint_type divisor_new = other;

		for (int32_t i = 127; i >= 0; --i) {
			remainder = remainder << 1;

			if ((i >= 64 && (hi & (1ULL << (i - 64)))) || (i < 64 && (lo & (1ULL << i)))) {
				remainder.lo |= 1;
			}

			if (remainder >= divisor_new) {
				remainder = remainder - divisor_new;
				if (i >= 64) {
					quotient.hi |= (1ULL << (i - 64));
				} else {
					quotient.lo |= (1ULL << i);
				}
			}
		}
		return quotient;
	}

	BNCH_SWT_HOST_DEVICE value_type lzcnt() const {
		if (hi != 0) {
			value_type x = hi;
			value_type n = 0;
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		} else {
			value_type x = lo;
			value_type n = 64;
			if (x == 0) {
				return 128;
			}
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		}
	}

	BNCH_SWT_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		uint_type div_temp	  = divisor_new;
		uint_type div_minus_1 = divisor_new - 1;
		value_type l		  = 127 - div_minus_1.lzcnt();
		uint_type numerator	  = uint_type{ 1 } << static_cast<int32_t>(64 + static_cast<value_type>(l));
		uint_type m_128		  = (numerator + div_temp - 1) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64 + l };
	}
};

template<uint32_types value_type> struct uint_type<value_type> {
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(uint64_t v) {
	}

	BNCH_SWT_HOST_DEVICE static value_type lzcnt(uint64_t value) {
		if (value == 0) {
			return 64;
		}
		uint64_t x	 = value;
		value_type n = 0;
		if (x <= 0x00000000FFFFFFFF) {
			n += 32;
			x <<= 32;
		}
		if (x <= 0x0000FFFFFFFFFFFF) {
			n += 16;
			x <<= 16;
		}
		if (x <= 0x00FFFFFFFFFFFFFF) {
			n += 8;
			x <<= 8;
		}
		if (x <= 0x0FFFFFFFFFFFFFFF) {
			n += 4;
			x <<= 4;
		}
		if (x <= 0x3FFFFFFFFFFFFFFF) {
			n += 2;
			x <<= 2;
		}
		if (x <= 0x7FFFFFFFFFFFFFFF) {
			n += 1;
		}
		return n;
	}

	BNCH_SWT_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		const uint64_t div_temp	   = divisor_new;
		const uint64_t div_minus_1 = divisor_new - 1ULL;
		const value_type l		   = 63ULL - lzcnt(div_minus_1);
		const uint64_t numerator   = 1ULL << static_cast<int32_t>(32ULL + static_cast<value_type>(l));
		const uint64_t m_128	   = (numerator + div_temp - 1) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), static_cast<value_type>(32ULL + l) };
	}
};

template<typename value_type, value_type const_value_new> struct const_aligned_uint {
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) static constexpr value_type const_value { const_value_new };
};

template<typename value_type, value_type const_value_new> struct aligned_uint : public const_aligned_uint<value_type, const_value_new> {
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) mutable value_type value {};
};

template<typename derived_type, typename value_type, value_type...> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic;

template<typename value_type> BNCH_SWT_HOST_DEVICE static value_type mul128Generic(value_type ab, value_type cd, value_type& hi) noexcept {
	value_type aHigh = ab >> 32;
	value_type aLow	 = ab & 0xFFFFFFFF;
	value_type bHigh = cd >> 32;
	value_type bLow	 = cd & 0xFFFFFFFF;
	value_type ad	 = aHigh * bLow;
	value_type bd	 = aHigh * bLow;
	value_type adbc	 = ad + aLow * bHigh;
	value_type lo	 = bd + (adbc << 32);
	value_type carry = (lo < bd);
	hi				 = aHigh * bHigh + (adbc >> 32) + carry;
	return lo;
}

BNCH_SWT_HOST_DEVICE static unsigned long long host_umulhi64(unsigned long long a, unsigned long long b) {
	unsigned long long high;
	mul128Generic(a, b, high);
	return high;
}

template<typename derived_type, typename value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic<derived_type, value_type> : public uint_type<value_type> {
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8) - 1 };
	static constexpr value_type bit_count{ sizeof(value_type) * 8 };
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) mutable value_type multiplicand {};
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) mutable value_type shift {};

	BNCH_SWT_HOST_DEVICE constexpr div_mod_logic() {
	}

	BNCH_SWT_HOST_DEVICE value_type& get_value() const {
		return static_cast<const derived_type*>(this)->value;
	}

	BNCH_SWT_HOST void collect_values(value_type d) const {
		get_value()	 = d;
		auto values	 = uint_type<value_type>::collect_values(d);
		multiplicand = values.multiplicand;
		shift		 = values.shift;
	}

	BNCH_SWT_HOST_DEVICE value_type div(value_type value) const {
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
		if constexpr (std::same_as<value_type, uint64_t>) {
			return __umulhi64(value, multiplicand) >> shift;
		} else {
			return __umulhi(value, multiplicand) >> shift;
		}
#else
		if constexpr (std::same_as<value_type, uint64_t>) {
			uint64_t high_part = host_umulhi64(multiplicand, value);

			uint64_t result;
			if (shift >= 64) {
				result = high_part >> (shift - 64);
			} else {
				uint64_t low_part = multiplicand * value;
				result			  = (high_part << (64 - shift)) | (low_part >> shift);
			}
			return result;
		} else {
			return static_cast<value_type>((static_cast<uint64_t>(value) * multiplicand) >> shift);
		}
#endif
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type value) const {
		return value - (div(value) * get_value());
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator<(value_type lhs, const div_mod_logic& rhs) {
		return lhs < rhs.get_value();
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>(value_type lhs, const div_mod_logic& rhs) {
		return lhs > rhs.get_value();
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(value_type lhs, const div_mod_logic& rhs) {
		return lhs >= rhs.get_value();
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(const div_mod_logic& lhs, value_type rhs) {
		return lhs.get_value() >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(value_type lhs, const div_mod_logic& rhs) {
		return lhs * rhs.get_value();
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(const div_mod_logic& lhs, value_type rhs) {
		return lhs.get_value() * rhs;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
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

template<typename derived_type, typename value_type, value_type divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment)
	div_mod_logic<derived_type, value_type, divisor> : public uint_type<value_type, divisor> {
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8) - 1 };
	static constexpr value_type bit_count{ sizeof(value_type) * 8 };

	BNCH_SWT_DEVICE static constexpr value_type get_value() {
		return derived_type::const_value;
	}

	BNCH_SWT_ALIGN(bnch_swt::device_alignment) static constexpr uint_pair multiplicand_and_shift { uint_type<value_type, divisor>::collect_values() };

	BNCH_SWT_HOST_DEVICE value_type div(value_type value) const {
		if constexpr (is_power_of_2(divisor)) {
			static constexpr value_type shift_amount{ log2_ct(divisor) };
			return value >> shift_amount;
		} else {
			if constexpr (divisor == 1) {
				return value;
			}
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
			if constexpr (std::same_as<value_type, uint64_t>) {
				return __umulhi64(value, multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift;
			} else {
				return __umulhi(value, multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift;
			}
#else
			if constexpr (std::same_as<value_type, uint64_t>) {
				uint64_t high_part = host_umulhi64(multiplicand_and_shift.multiplicand, value);

				uint64_t result;
				if constexpr (multiplicand_and_shift.shift >= 64) {
					result = high_part >> (multiplicand_and_shift.shift - 64);
				} else {
					uint64_t low_part = multiplicand_and_shift.multiplicand * value;
					result			  = (high_part << (64 - multiplicand_and_shift.shift)) | (low_part >> multiplicand_and_shift.shift);
				}
				return result;
			} else {
				return static_cast<value_type>((static_cast<uint64_t>(value) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
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

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator<(value_type lhs, const div_mod_logic&) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return lhs < value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>(value_type lhs, const div_mod_logic&) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return lhs > value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(value_type lhs, const div_mod_logic&) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return lhs >= value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(const div_mod_logic&, value_type rhs) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return value >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(value_type lhs, const div_mod_logic&) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return lhs * value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(const div_mod_logic&, value_type rhs) {
		BNCH_SWT_ALIGN(bnch_swt::device_alignment) constexpr value_type value{ div_mod_logic::get_value() };
		return value * rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<uint32_t static_divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) division {
	BNCH_SWT_DEVICE static uint32_t div(uint32_t value) {
		if constexpr (is_power_of_2(static_divisor)) {
			static constexpr uint32_t shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			static constexpr div_mod_logic<const_aligned_uint<uint32_t, static_divisor>, uint32_t, false, static_divisor> mul_shift{};
			return mul_shift.div(value);
		}
	}
};

template<uint32_t static_divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) modulo {
	BNCH_SWT_DEVICE static uint32_t mod(uint32_t value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1);
		} else {
			static constexpr div_mod_logic<const_aligned_uint<uint32_t, static_divisor>, uint32_t, false, static_divisor> mul_shift{};
			return mul_shift.mod(value);
		}
	}
};

enum class runtime_dimension_value_types {
	none			= 0x0,
	batch_size		= 0x1,
	sequence_length = 0x2,
	token_count		= 0x4,
	count			= token_count + 1,
};

struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) dimension_identifier {
	BNCH_SWT_HOST constexpr dimension_identifier(runtime_dimension_value_types runtime_dimension_value_type_new, uint32_t index_new = 0) {
		runtime_dimension_value_type = runtime_dimension_value_type_new;
		index						 = index_new;
	}
	BNCH_SWT_HOST explicit constexpr dimension_identifier(uint32_t index_new = 0) {
		index = index_new;
	}
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) runtime_dimension_value_types runtime_dimension_value_type {};
	BNCH_SWT_ALIGN(bnch_swt::device_alignment) uint32_t index {};
	BNCH_SWT_HOST_DEVICE constexpr bool operator==(const dimension_identifier& other) const {
		return (runtime_dimension_value_type == other.runtime_dimension_value_type) && (index == other.index);
	}
};

template<typename value_type, dimension_identifier dimension_identifier_val_new = dimension_identifier{}> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) dimension;

template<typename value_type, dimension_identifier dimension_identifier_val_new>
	requires(dimension_identifier_val_new.runtime_dimension_value_type == runtime_dimension_value_types::none)
struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) dimension<value_type, dimension_identifier_val_new>
	: public const_aligned_uint<value_type, dimension_identifier_val_new.index>,
	  public div_mod_logic<dimension<value_type, dimension_identifier_val_new>, value_type, dimension_identifier_val_new.index> {
	static constexpr dimension_identifier dimension_identifier_val{ dimension_identifier_val_new };
};

template<typename value_type, dimension_identifier dimension_identifier_val_new>
	requires(dimension_identifier_val_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) dimension<value_type, dimension_identifier_val_new>
	: public aligned_uint<value_type, dimension_identifier_val_new.index>, public div_mod_logic<dimension<value_type, dimension_identifier_val_new>, value_type> {
	static constexpr dimension_identifier dimension_identifier_val{ dimension_identifier_val_new };
};

static constexpr uint64_t compute_elements(const auto& elems) {
	uint64_t return_value{ 1 };
	for (uint32_t x = 0; x < elems.size(); ++x) {
		return_value *= elems[x];
	}
	return return_value;
}

enum class runtime_dimensions_errors {
	incorrect_dimensions,
	unequal_amount_of_runtime_dimensions_passed,
	incorrect_runtime_dim,
};

static consteval bool any_duplicates(auto values) {
	bool values_checked[4]{};
	for (uint32_t x = 0; x < 4; ++x) {
		if (values_checked[values[x]]) {
			return true;
		}
		values_checked[values[x]] = true;
	}
	return false;
}

consteval bool check_runtime_dimension_types(const auto& runtime_dimension_new) {
	std::array<int8_t, 5> values{};
	for (uint64_t x = 0; x < 4; ++x) {
		++values[static_cast<uint64_t>(runtime_dimension_new[x].runtime_dimension_value_type)];
	}
	for (uint64_t x = 0; x < 4; ++x) {
		if (values[x] > 1 && x != 0) {
			return false;
		}
	}
	return 1;
}

consteval bool valid_dimension_identifiers(const auto& values) {
	return check_runtime_dimension_types(values);
};

template<uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> using dimension_t =
	std::conditional_t<compute_elements(std::array<uint64_t, 4>{ dim_00, dim_01, dim_02, dim_03 }) >= std::numeric_limits<uint32_t>::max() ||
			dim_00 >= std::numeric_limits<uint32_t>::max() || dim_01 >= std::numeric_limits<uint32_t>::max() || dim_02 >= std::numeric_limits<uint32_t>::max() ||
			dim_03 >= std::numeric_limits<uint32_t>::max(),
		uint64_t, uint32_t>;

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new>
struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) dimensions {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	static constexpr dimension<dimension_type, dim_00_new> dim_00_ct{};
	static constexpr dimension<dimension_type, dim_01_new> dim_01_ct{};
	static constexpr dimension<dimension_type, dim_02_new> dim_02_ct{};
	static constexpr dimension<dimension_type, dim_03_new> dim_03_ct{};
	dimension<dimension_type, dim_00_new> d0{};
	dimension<dimension_type, dim_01_new> d1{};
	dimension<dimension_type, dim_02_new> d2{};
	dimension<dimension_type, dim_03_new> d3{};

	static constexpr std::array dimension_identifiers{ dim_00_new, dim_01_new, dim_02_new, dim_03_new };
	static constexpr std::array dims{ dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index };
	static_assert(static_assert_printer_val<valid_dimension_identifiers(dimension_identifiers), runtime_dimensions_errors::incorrect_dimensions>::impl);
	template<runtime_dimension_value_types runtime_dimension_value_type_new> static consteval dimension_type get_runtime_dimension() {
		for (uint32_t x = 0; x < 4; ++x) {
			if (dimension_identifiers[x].runtime_dimension_value_type == runtime_dimension_value_type_new) {
				return x;
			}
		}
		return std::numeric_limits<dimension_type>::max();
	}

	template<runtime_dimension_value_types runtime_dimension_value_type_new> static consteval bool runtime_dimension_presence() {
		constexpr dimension_type rt_dim = get_runtime_dimension<runtime_dimension_value_type_new>();
		return (rt_dim != std::numeric_limits<dimension_type>::max());
	}

	BNCH_SWT_HOST_DEVICE auto& get_dims() const {
		static constexpr std::array<dimension_type, 4ull> dims_new{ dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index };
		return dims_new;
	}

	BNCH_SWT_HOST_DEVICE auto& dim_00() const {
		static constexpr dimension<dimension_type, dim_00_new> dim_00_ct_new{};
		if constexpr (dim_00_ct_new.dimension_identifier_val.runtime_dimension_value_type == runtime_dimension_value_types::none) {
			return dim_00_ct_new;
		} else {
			return d0;
		}
	}

	BNCH_SWT_HOST_DEVICE auto& dim_01() const {
		static constexpr dimension<dimension_type, dim_00_new> dim_01_ct_new{};
		if constexpr (dim_01_ct_new.dimension_identifier_val.runtime_dimension_value_type == runtime_dimension_value_types::none) {
			return dim_01_ct_new;
		} else {
			return d1;
		}
	}

	BNCH_SWT_HOST_DEVICE auto& dim_02() const {
		static constexpr dimension<dimension_type, dim_02_new> dim_02_ct_new{};
		if constexpr (dim_02_ct_new.dimension_identifier_val.runtime_dimension_value_type == runtime_dimension_value_types::none) {
			return dim_02_ct_new;
		} else {
			return d2;
		}
	}

	BNCH_SWT_HOST_DEVICE auto& dim_03() const {
		static constexpr dimension<dimension_type, dim_03_new> dim_03_ct_new{};
		if constexpr (dim_03_ct_new.dimension_identifier_val.runtime_dimension_value_type == runtime_dimension_value_types::none) {
			return dim_03_ct_new;
		} else {
			return d3;
		}
	}

	template<uint32_t index> BNCH_SWT_HOST_DEVICE constexpr auto& get_dim() const {
		if constexpr (index == 0) {
			return dim_00();
		} else if constexpr (index == 1) {
			return dim_01();
		} else if constexpr (index == 2) {
			return dim_02();
		} else if constexpr (index == 3) {
			return dim_03();
		} else {
			static_assert(index < 4, "Sorry, but you tried to claim an OOB index.");
		}
	}
};

enum class dim_trait_static_assert_errors : uint8_t {
	binary_element_count_mismatch,
	ternary_element_count_mismatch,
	quaternary_element_count_mismatch,
	reshape_total_element_count_mismatch,
	copy_total_element_count_mismatch,
	view_total_element_count_mismatch,
	transpose_total_element_count_mismatch,
	permute_total_element_count_mismatch,
	cont_total_element_count_mismatch,
	softmax_mask_not_broadcastable,
};

template<typename... arg_types> struct get_first_type {};

template<typename arg_type, typename... arg_types> struct get_first_type<arg_type, arg_types...> {
	using type = arg_type;
};

template<typename... arg_types> using get_first_type_t = get_first_type<arg_types...>::type;

template<dimensions_types dimension_type, typename...> struct rt_dimensions;

template<dimensions_types runtime_dimensions_type_new> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment)
	rt_dimensions<runtime_dimensions_type_new> : public dimensions<runtime_dimensions_type_new::dimension_identifiers[0], runtime_dimensions_type_new::dimension_identifiers[1],
													 runtime_dimensions_type_new::dimension_identifiers[2], runtime_dimensions_type_new::dimension_identifiers[3]> {
	using base_type				  = dimensions<runtime_dimensions_type_new::dimension_identifiers[0], runtime_dimensions_type_new::dimension_identifiers[1],
					  runtime_dimensions_type_new::dimension_identifiers[2], runtime_dimensions_type_new::dimension_identifiers[3]>;
	using runtime_dimensions_type = runtime_dimensions_type_new;

	BNCH_SWT_HOST_DEVICE constexpr rt_dimensions() noexcept {
	}

	template<runtime_dimension_value_types dimension_type> BNCH_SWT_HOST void set_rt_dim(uint32_t runtime_value) {
		if constexpr (base_type::template runtime_dimension_presence<dimension_type>()) {
			constexpr typename base_type::dimension_type dim_index = base_type::template get_runtime_dimension<dimension_type>();
			this->template get_dim<dim_index>().collect_values(runtime_value);
		}
	}
};

template<dimensions_types runtime_dimensions_type, dimensions_types mod_mask_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment)
	rt_dimensions<runtime_dimensions_type, mod_mask_type> : public rt_dimensions<runtime_dimensions_type> {
	using base_type = rt_dimensions<runtime_dimensions_type>;
	static constexpr std::array<bnch_swt::aligned_const<uint32_t>, 4ULL> mod_mask{ mod_mask_type::dims[0], mod_mask_type::dims[1], mod_mask_type::dims[2], mod_mask_type::dims[3] };
};

template<auto value_new> struct make_static {
	static constexpr auto value{ value_new };
};

template<uint32_t dim_00 = 0, uint32_t dim_01 = 1, uint32_t dim_02 = 2, uint32_t dim_03 = 3> consteval auto generate_dimensions() {
	return dimensions<dimension_identifier{ runtime_dimension_value_types::none, dim_00 }, dimension_identifier{ runtime_dimension_value_types::none, dim_01 },
		dimension_identifier{ runtime_dimension_value_types::none, dim_02 }, dimension_identifier{ runtime_dimension_value_types::none, dim_03 }>{};
}

template<dimension_identifier dim_00, dimension_identifier dim_01, dimension_identifier dim_02, dimension_identifier dim_03> consteval auto generate_dimensions() {
	constexpr dimension_identifier corrected_dim_00 = dimension_identifier{ dim_00.runtime_dimension_value_type, dim_00.index };

	constexpr dimension_identifier corrected_dim_01 = dimension_identifier{ dim_01.runtime_dimension_value_type, dim_01.index };

	constexpr dimension_identifier corrected_dim_02 = dimension_identifier{ dim_02.runtime_dimension_value_type, dim_02.index };

	constexpr dimension_identifier corrected_dim_03 = dimension_identifier{ dim_03.runtime_dimension_value_type, dim_03.index };

	return dimensions<corrected_dim_00, corrected_dim_01, corrected_dim_02, corrected_dim_03>{};
}

template<auto... dims> struct get_dimensions_type {
	using type = decltype(generate_dimensions<dims...>());
};

template<auto... dims> using get_dimensions_type_t = get_dimensions_type<dims...>::type;

template<kernel_types kernel_type_new> struct kernel_types_type {
	static constexpr kernel_types kernel_type{ kernel_type_new };
};

template<typename kernel_type, typename... dims_types> struct kernel_traits;

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::weights>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<unary_kernel_types kernel_type, dimensions_types input_dims_01> struct kernel_traits<kernel_type, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<binary_kernel_types kernel_type, dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_type, input_dims_01, input_dims_02> {
	static constexpr auto dims_array = std::array<uint32_t, 4>{ std::max(input_dims_01::dims[0], input_dims_02::dims[0]), std::max(input_dims_01::dims[1], input_dims_02::dims[1]),
		std::max(input_dims_01::dims[2], input_dims_02::dims[2]), std::max(input_dims_01::dims[3], input_dims_02::dims[3]) };
	static constexpr bool dim0_ok	 = (input_dims_01::dims[0] == input_dims_02::dims[0]) || (input_dims_01::dims[0] == 1) || (input_dims_02::dims[0] == 1);
	static constexpr bool dim1_ok	 = (input_dims_01::dims[1] == input_dims_02::dims[1]) || (input_dims_01::dims[1] == 1) || (input_dims_02::dims[1] == 1);
	static constexpr bool dim2_ok	 = (input_dims_01::dims[2] == input_dims_02::dims[2]) || (input_dims_01::dims[2] == 1) || (input_dims_02::dims[2] == 1);
	static constexpr bool dim3_ok	 = (input_dims_01::dims[3] == input_dims_02::dims[3]) || (input_dims_01::dims[3] == 1) || (input_dims_02::dims[3] == 1);
	static_assert(static_assert_printer_val<(dim0_ok && dim1_ok && dim2_ok && dim3_ok), dim_trait_static_assert_errors::binary_element_count_mismatch, input_dims_01::dims[1],
		input_dims_02::dims[1], input_dims_01::dims[2], input_dims_02::dims[2], input_dims_01::dims[3], input_dims_02::dims[3]>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims_array[0], dims_array[1], dims_array[2], dims_array[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::softmax>, input_dims_01, input_dims_02> {
	static constexpr bool dim0_ok = (input_dims_02::dims[0] == input_dims_01::dims[0]);
	static constexpr bool dim2_ok = (input_dims_02::dims[2] == input_dims_01::dims[3]);
	static_assert(static_assert_printer_val<(dim0_ok && dim2_ok), dim_trait_static_assert_errors::softmax_mask_not_broadcastable, input_dims_01::dims[0], input_dims_02::dims[0],
		input_dims_01::dims[1], input_dims_02::dims[1], input_dims_01::dims[2], input_dims_02::dims[2], input_dims_01::dims[3], input_dims_02::dims[3]>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::copy>, input_dims_01, input_dims_02> {
	static constexpr auto dims01			  = input_dims_01::dims;
	static constexpr auto dims02			  = input_dims_02::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(
		static_assert_printer_val<(input_elements == output_elements), dim_trait_static_assert_errors::copy_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::softmax>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<dimensions_types input_dims, dimensions_types output_dims> struct kernel_traits<kernel_types_type<kernel_types::top_k>, input_dims, output_dims> {
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>>;
};

template<dimensions_types expert_weights, dimensions_types input_acts, dimensions_types expert_selection>
struct kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, expert_weights, input_acts, expert_selection> {
	static constexpr auto weights_dims = expert_weights::dims;
	static constexpr auto input_dims   = input_acts::dims;
	static constexpr auto select_dims  = expert_selection::dims;

	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims[0], input_dims[1], select_dims[2], weights_dims[2]>>;
};

template<dimensions_types expert_outputs, dimensions_types router_weights> struct kernel_traits<kernel_types_type<kernel_types::weighted_sum>, expert_outputs, router_weights> {
	static constexpr auto output_dims = expert_outputs::dims;

	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims[0], output_dims[1], output_dims[3], 1U>>;
};

template<ternary_kernel_types kernel_type, dimensions_types input_dims_01, dimensions_types input_dims_02, dimensions_types input_dims_03>
struct kernel_traits<kernel_type, input_dims_01, input_dims_02, input_dims_03> {
	static_assert(static_assert_printer_val<((input_dims_01::dims[0] == input_dims_02::dims[0]) || (input_dims_01::dims[0] == 1) || (input_dims_02::dims[0] == 1)),
		dim_trait_static_assert_errors::ternary_element_count_mismatch, input_dims_01::dims[0], input_dims_02::dims[0]>::impl);
	static_assert(static_assert_printer_val<((input_dims_01::dims[0] == input_dims_03::dims[0]) || (input_dims_01::dims[0] == 1) || (input_dims_03::dims[0] == 1)),
		dim_trait_static_assert_errors::ternary_element_count_mismatch, input_dims_01::dims[0], input_dims_03::dims[0]>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

template<dimensions_types input_dims, dimensions_types output_dims> struct kernel_traits<kernel_types_type<kernel_types::reshape>, input_dims, output_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_elements == output_elements), dim_trait_static_assert_errors::reshape_total_element_count_mismatch, input_elements,
		output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::view>, input_dims, mod_mask_type> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = mod_mask_type::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<true, dim_trait_static_assert_errors::view_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<mod_mask_type::dims[0], mod_mask_type::dims[1], mod_mask_type::dims[2], mod_mask_type::dims[3]>,
		dimensions<dimension_identifier{ dims01[0] - dims02[0] }, dimension_identifier{ dims01[1] - dims02[1] }, dimension_identifier{ dims01[2] - dims02[2] },
			dimension_identifier{ dims01[3] - dims02[3] }>>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::transpose>, input_dims, mod_mask_type> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = mod_mask_type::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<((static_cast<uint32_t>(dims02[0]) <= 3 && static_cast<uint32_t>(dims02[1]) <= 3 && static_cast<uint32_t>(dims02[2]) <= 3 &&
												 static_cast<uint32_t>(dims02[3]) <= 3) &&
												!any_duplicates(dims02)),
		dim_trait_static_assert_errors::transpose_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims01[static_cast<uint32_t>(dims02[0])], dims01[static_cast<uint32_t>(dims02[1])],
										dims01[static_cast<uint32_t>(dims02[2])], dims01[static_cast<uint32_t>(dims02[3])]>,
		mod_mask_type>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::permute>, input_dims, mod_mask_type> {
	static constexpr auto input_dims_array = input_dims::dims;
	static constexpr auto permutation	   = mod_mask_type::dims;
	static constexpr auto dims01		   = input_dims_array;
	static constexpr auto dims02 = std::array<uint32_t, 4>{ input_dims_array[static_cast<uint32_t>(permutation[0])], input_dims_array[static_cast<uint32_t>(permutation[1])],
		input_dims_array[static_cast<uint32_t>(permutation[2])], input_dims_array[static_cast<uint32_t>(permutation[3])] };
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_elements == output_elements), dim_trait_static_assert_errors::permute_total_element_count_mismatch, input_elements,
		output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims02[0], dims02[1], dims02[2], dims02[3]>, mod_mask_type>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::mul_mat>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = rt_dimensions<get_dimensions_type_t<dims02[0], dims01[2], dims02[2], dims02[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::get_rows>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, dims02[0] },
					 dimension_identifier{ runtime_dimension_value_types::sequence_length, dims01[1] }, dimension_identifier{ dims02[1] }, dimension_identifier{ dims01[3] }>>;
};

template<dimensions_types output_dims, dimensions_types input_dims> struct kernel_traits<kernel_types_type<kernel_types::cont>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(
		static_assert_printer_val<(input_elements == output_elements), dim_trait_static_assert_errors::cont_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>>;
};

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::sample_tokens>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>>;
};

enum class core_types {
	// Weights.
	attn_q,
	attn_k,
	attn_v,
	attn_output,
	attn_norm,
	ffn_gate,
	ffn_up,
	ffn_down,
	moe_gate,
	moe_experts_gate,
	moe_experts_up,
	moe_experts_down,
	ffn_norm,
	token_embd,
	rope_freqs,
	output_norm,
	output,
	end_of_weights,
	// Global Inputs.
	inp_tokens,
	inp_pos,
	inp_out_ids,
	cache_k,
	cache_v,
	kq_mask,
	benchmark_data,
	end_of_input_only,
	// Token-Embeddings Mega-Kernel.
	inp_embd_get_rows,
	end_of_global_inputs,
	// attn_prep_and_score Mega-Kernel.
	norm_rms_norm,
	attn_norm_mul,
	qcur_mul_mat,
	qcur_reshape,
	qcur_rope,
	kcur_mul_mat,
	kcur_reshape,
	kcur_rope,
	vcur_mul_mat,
	k_cache_view,
	k_cache_view_copy,
	vcur_transpose,
	v_cache_view,
	v_cache_view_copy,
	v_view,
	k_view,
	q_permute,
	kq_mul_mat,
	// attn_and_ffn_out Mega-Kernel (Dense FFN - Llama).
	kq_soft_max,
	kqv_mul_mat,
	kqv_merged_permute,
	kqv_merged_cont,
	kqv_out_mul_mat,
	ffn_inp_add,
	norm_pre_ffn_rms_norm,
	ffn_norm_mul,
	ffn_gate_mul_mat,
	ffn_silu,
	ffn_up_mul_mat,
	ffn_gate_par_mul,
	ffn_out_mul_mat,
	// attn_and_moe_out Mega-Kernel (MoE - Grok).
	moe_inp_add,
	norm_pre_moe_rms_norm,
	moe_norm_mul,
	moe_router_mul_mat,
	moe_router_softmax,
	moe_expert_select,
	moe_expert_gate_mul_mat,
	moe_expert_silu,
	moe_expert_up_mul_mat,
	moe_expert_gate_par_mul,
	moe_expert_down_mul_mat,
	moe_expert_weighted_sum,
	layer_out_add,
	end_of_per_block,
	// global_output_and_sampling Mega-Kernel (Dense FFN - Llama).
	node_1016_get_rows,
	node_1017_get_rows,
	final_ffn_inp_add,
	final_norm_pre_rms_norm,
	final_ffn_norm_mul,
	final_ffn_gate_mul_mat,
	final_ffn_silu,
	final_ffn_up_mul_mat,
	final_ffn_gate_par_mul,
	final_ffn_out_mul_mat,
	// global_output_and_sampling Mega-Kernel (MoE - Grok).
	final_moe_inp_add,
	final_norm_pre_moe_rms_norm,
	final_moe_norm_mul,
	final_moe_router_mul_mat,
	final_moe_router_softmax,
	final_moe_expert_select,
	final_moe_expert_gate_mul_mat,
	final_moe_expert_silu,
	final_moe_expert_up_mul_mat,
	final_moe_expert_gate_par_mul,
	final_moe_expert_down_mul_mat,
	final_moe_expert_weighted_sum,
	final_layer_out_add,
	final_norm_rms_norm,
	result_norm_mul,
	result_output_mul_mat,
	sample_tokens,
	count
};

enum class data_strategy_types : uint8_t {
	none,
	global,
	per_block,
};

enum class transformer_phases {
	prefill,
	decode,
};

enum class kernel_classes {
	global_input,
	per_block,
	global_output,
};

enum class alloc_classes {
	none,
	allocate_heap,
	allocate_cache,
	mmap,
};
enum class model_arches : uint8_t {
	llama,
	deci,
	falcon,
	baichuan,
	grok,
	gpt2,
	gptj,
	gptneox,
	mpt,
	starcoder,
	refact,
	bert,
	nomic_bert,
	jina_bert_v2,
	bloom,
	stablelm,
	qwen,
	qwen2,
	qwen2moe,
	qwen2vl,
	phi2,
	phi3,
	phimoe,
	plamo,
	codeshell,
	orion,
	internlm2,
	minicpm,
	minicpm3,
	gemma,
	gemma2,
	starcoder2,
	mamba,
	xverse,
	command_r,
	cohere2,
	dbrx,
	olmo,
	olmo2,
	olmoe,
	openelm,
	arctic,
	deepseek,
	deepseek2,
	chatglm,
	bitnet,
	t5,
	t5encoder,
	jais,
	nemotron,
	exaone,
	rwkv6,
	rwkv6qwen2,
	granite,
	granite_moe,
	chameleon,
	wavtokenizer_dec,
	unknown,
	count,
};

enum class kernel_type_profiles : uint8_t {
	fp16_mha,
	fp16_moe,
	bf16_mha,
	bf16_gqa,
	q4_mha,
	q4_gqa,
	q4_moe,
	q8_mha,
	q8_gqa,
	q8_moe,
	mixed_fp16_fp32,
	mixed_bf16_fp32,
	count,
};

enum class model_generations : uint8_t {
	v1,
	v1_v2,
	v1_5,
	v2,
	v3,
	v3_1,
	v3_2,
	count,
};

enum class model_sizes : uint8_t {
	llm_unknown,
	llm_14M,
	llm_17M,
	llm_22M,
	llm_33M,
	llm_60M,
	llm_70M,
	llm_80M,
	llm_109M,
	llm_137M,
	llm_160M,
	llm_220M,
	llm_250M,
	llm_270M,
	llm_335M,
	llm_410M,
	llm_450M,
	llm_770M,
	llm_780M,
	llm_0_5B,
	llm_1B,
	llm_1_3B,
	llm_1_4B,
	llm_1_5B,
	llm_1_6B,
	llm_2B,
	llm_2_8B,
	llm_3B,
	llm_4B,
	llm_6B,
	llm_6_9B,
	llm_7B,
	llm_8B,
	llm_9B,
	llm_11B,
	llm_12B,
	llm_13B,
	llm_14B,
	llm_15B,
	llm_16B,
	llm_20B,
	llm_30B,
	llm_32B,
	llm_34B,
	llm_35B,
	llm_40B,
	llm_46B,
	llm_65B,
	llm_70B,
	llm_314B,
	llm_405B,
	llm_SMALL,
	llm_MEDIUM,
	llm_LARGE,
	llm_XL,
	llm_A1_7B,
	llm_A2_7B,
	llm_8x7B,
	llm_8x22B,
	llm_16x12B,
	llm_16x3_8B,
	llm_10B_128x3_66B,
	llm_57B_A14B,
	llm_27B,
	count,
};

enum class device_types : uint8_t {
	cpu,
	gpu,
	numa,
};

enum class exceptions_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

enum class benchmark_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

enum class dev_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

enum class max_context_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class gpu_count_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_generation_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_prompt_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_batch_size_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

template<model_arches model_arch, model_sizes model_size, model_generations model_generation> struct model_traits;

template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

struct model_config {
	model_generations model_generation{ model_generations::v3_1 };
	model_sizes model_size{ model_sizes::llm_405B };
	kernel_type_profiles kernel_type_profile{};
	model_arches model_arch{};
	exceptions_type exceptions{};
	max_context_length_type max_context_length{ static_cast<max_context_length_type>(1024) };
	max_prompt_length_type max_prompt_length{ static_cast<max_prompt_length_type>(std::numeric_limits<uint64_t>::max()) };
	max_generation_length_type max_generation_length{ static_cast<max_generation_length_type>(std::numeric_limits<uint64_t>::max()) };
	max_batch_size_type max_batch_size{ static_cast<max_batch_size_type>(1) };
	device_types device_type{};
	gpu_count_type gpu_count{};
	benchmark_type benchmark{};
	dev_type dev{};

	template<std::same_as<model_generations> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_generation = value;
		return return_value;
	}

	template<std::same_as<model_sizes> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_size = value;
		return return_value;
	}

	template<std::same_as<kernel_type_profiles> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.kernel_type_profile = value;
		return return_value;
	}

	template<std::same_as<model_arches> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_arch = value;
		return return_value;
	}

	template<std::same_as<exceptions_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.exceptions = value;
		return return_value;
	}

	template<std::same_as<max_context_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_context_length = value;
		return return_value;
	}

	template<std::same_as<gpu_count_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.gpu_count = value;
		return return_value;
	}

	template<std::same_as<max_prompt_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_prompt_length = value;
		return return_value;
	}

	template<std::same_as<max_generation_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_generation_length = value;
		return return_value;
	}

	template<std::same_as<max_batch_size_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_batch_size = value;
		return return_value;
	}

	template<std::same_as<device_types> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.device_type = value;
		return return_value;
	}

	template<std::same_as<benchmark_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.benchmark = value;
		return return_value;
	}

	template<std::same_as<dev_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.dev = value;
		return return_value;
	}
};
constexpr uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
	return (a + b - 1) / b;
}

template<uint64_t value_01, uint64_t value_02> consteval uint64_t get_updated_value() {
	if constexpr (value_01 == std::numeric_limits<uint64_t>::max()) {
		return ceil_div(value_02, 2);
	} else {
		return value_01;
	}
}

enum class model_config_errors {
	context_length_too_large,
	context_length_too_short,
	prompt_length_or_generation_length_too_large,
};

template<const model_config& config> struct model_config_type {
	static constexpr model_generations model_generation		  = config.model_generation;
	static constexpr model_sizes model_size					  = config.model_size;
	static constexpr kernel_type_profiles kernel_type_profile = config.kernel_type_profile;
	static constexpr model_arches model_arch				  = config.model_arch;
	static constexpr bool exceptions						  = static_cast<bool>(config.exceptions);
	static constexpr uint64_t max_context_length			  = static_cast<uint64_t>(config.max_context_length);
	static constexpr uint64_t max_prompt_length				  = get_updated_value<static_cast<uint64_t>(config.max_prompt_length), max_context_length>();
	static constexpr uint64_t max_generation_length			  = get_updated_value<static_cast<uint64_t>(config.max_generation_length), max_context_length>();
	static constexpr uint64_t max_batch_size				  = static_cast<uint64_t>(config.max_batch_size);
	static constexpr device_types device_type				  = config.device_type;
	static constexpr bool benchmark							  = static_cast<bool>(config.benchmark);
	static constexpr bool dev								  = static_cast<bool>(config.dev);
	static_assert(static_assert_printer_val<(max_context_length <= model_traits_type<model_config_type>::context_length), model_config_errors::context_length_too_large,
		max_context_length, model_traits_type<model_config_type>::context_length>::impl);
	static_assert(static_assert_printer_val<(max_context_length > 1), model_config_errors::context_length_too_short, max_context_length>::impl);
	static_assert(static_assert_printer_val<(max_generation_length + max_prompt_length) <= max_context_length, model_config_errors::prompt_length_or_generation_length_too_large,
		max_context_length, max_generation_length, max_prompt_length>::impl);

	static constexpr const model_config& get_config() {
		return config;
	}
};

template<typename... arg_types> inline static consteval auto generate_model_config(arg_types... args) {
	model_config config_new{};
	((config_new = config_new.update(args)), ...);
	return config_new;
};

template<typename... arg_types> inline static consteval auto generate_model_config(model_config config_new, arg_types... args) {
	((config_new = config_new.update(args)), ...);
	return config_new;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_3B, model_generations::v3_2> {
	static constexpr const char name[]{ "llama-3.2-3B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_2 };
	static constexpr auto model_size{ model_sizes::llm_3B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 3072;
	static constexpr uint32_t block_count			  = 28;
	static constexpr uint32_t feed_forward_length	  = 8192;
	static constexpr uint32_t attention_head_count	  = 24;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-8B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_8B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 4096;
	static constexpr uint32_t block_count			  = 32;
	static constexpr uint32_t feed_forward_length	  = 14336;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-70B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_70B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 8192;
	static constexpr uint32_t block_count			  = 80;
	static constexpr uint32_t feed_forward_length	  = 28672;
	static constexpr uint32_t attention_head_count	  = 64;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-405B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_405B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 16384;
	static constexpr uint32_t block_count			  = 126;
	static constexpr uint32_t feed_forward_length	  = 53248;
	static constexpr uint32_t attention_head_count	  = 128;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v1> {
	static constexpr const char name[]{ "grok-1-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 8192;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v1_5> {
	static constexpr const char name[]{ "grok-1.5-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1_5 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v2> {
	static constexpr const char name[]{ "grok-2-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v2 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_46B, model_generations::v1> {
	static constexpr const char name[]{ "grok-mini-46B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1 };
	static constexpr auto model_size{ model_sizes::llm_46B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 4096;
	static constexpr uint32_t block_count			  = 32;
	static constexpr uint32_t feed_forward_length	  = 16384;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 8192;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<typename model_config_type> struct model_dimensions {
	using model_traits_type_new = model_traits_type<model_config_type>;
	enum : uint64_t {
		vocab_size				= model_traits_type_new::vocab_size,
		feed_forward_length		= model_traits_type_new::feed_forward_length,
		attention_head_count	= model_traits_type_new::attention_head_count,
		attention_head_count_kv = model_traits_type_new::attention_head_count_kv,
		rope_dimension_count	= model_traits_type_new::rope_dimension_count,
		n_embd_kv_gqa			= model_traits_type_new::n_embd_kv_gqa,
		block_count				= model_traits_type_new::block_count,
		embedding_length		= model_traits_type_new::embedding_length,
		max_context_length		= model_config_type::max_context_length,
		max_prompt_length		= model_config_type::max_prompt_length,
		max_generation_length	= model_config_type::max_generation_length,
		max_batch_size			= model_config_type::max_batch_size,
	};
};

template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename embedding_type_new, typename logit_type_new, typename token_type_new,
	typename attention_type_new, typename norm_type_new, typename scale_type_new, typename zero_point_type_new, typename kv_cache_type_new, typename mask_type_new,
	typename index_type_new>
struct kernel_type_profile_traits_impl {
	using weight_type	  = weight_type_new;
	using activation_type = activation_type_new;
	using compute_type	  = compute_type_new;
	using embedding_type  = embedding_type_new;
	using logit_type	  = logit_type_new;
	using token_type	  = token_type_new;
	using attention_type  = attention_type_new;
	using norm_type		  = norm_type_new;
	using scale_type	  = scale_type_new;
	using zero_point_type = zero_point_type_new;
	using kv_cache_type	  = kv_cache_type_new;
	using mask_type		  = mask_type_new;
	using index_type	  = index_type_new;
};

template<kernel_type_profiles kernel_type_profile> struct kernel_type_profile_traits;

template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_mha> : public kernel_type_profile_traits_impl<half,// weight_type
																				   half,// activation_type
																				   float,// compute_type
																				   half,// embedding_type
																				   half,// logit_type
																				   int32_t,// token_token_type
																				   half,// attention_type
																				   half,// norm_type
																				   half,// scale_type
																				   int8_t,// zero_point_type
																				   half,// kv_cache_type
																				   half,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_mha };
	static constexpr const char name[]{ "FP16-MHA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_moe> : public kernel_type_profile_traits_impl<half,// weight_type
																				   half,// activation_type
																				   half,// compute_type (Assuming hardware support)
																				   half,// embedding_type
																				   half,// logit_type
																				   int32_t,// token_token_type
																				   half,// attention_type
																				   half,// norm_type
																				   half,// scale_type
																				   int8_t,// zero_point_type
																				   half,// kv_cache_type
																				   half,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_moe };
	static constexpr const char name[]{ "FP16-MoE" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_mha> : public kernel_type_profile_traits_impl<bf16_t,// weight_type
																				   bf16_t,// activation_type
																				   bf16_t,// compute_type (Best for high dynamic range)
																				   bf16_t,// embedding_type
																				   bf16_t,// logit_type
																				   int32_t,// token_type
																				   bf16_t,// attention_type
																				   bf16_t,// norm_type
																				   bf16_t,// scale_type
																				   int8_t,// zero_point_type
																				   bf16_t,// kv_cache_type
																				   bf16_t,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_mha };
	static constexpr const char name[]{ "BF16-MHA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_gqa> : public kernel_type_profile_traits_impl<bf16_t,// weight_type
																				   bf16_t,// activation_type
																				   bf16_t,// compute_type
																				   bf16_t,// embedding_type
																				   bf16_t,// logit_type
																				   int32_t,// token_type
																				   bf16_t,// attention_type
																				   bf16_t,// norm_type
																				   bf16_t,// scale_type
																				   int8_t,// zero_point_type
																				   bf16_t,// kv_cache_type
																				   bf16_t,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_gqa };
	static constexpr const char name[]{ "BF16-GQA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_fp16_fp32> : public kernel_type_profile_traits_impl<half,// weight_type (Storage is FP16)
																						  half,// activation_type
																						  float,// compute_type (Compute is FP32)
																						  half,// embedding_type
																						  float,// logit_type
																						  int32_t,// token_type
																						  float,// attention_type
																						  float,// norm_type
																						  half,// scale_type
																						  int8_t,// zero_point_type
																						  half,// kv_cache_type
																						  float,// mask_type
																						  int32_t// index_type
																						  > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_fp16_fp32 };
	static constexpr const char name[]{ "Mixed-FP16/FP32" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_bf16_fp32> : public kernel_type_profile_traits_impl<bf16_t,// weight_type (Storage is BF16)
																						  bf16_t,// activation_type
																						  float,// compute_type (Compute is FP32)
																						  bf16_t,// embedding_type
																						  float,// logit_type
																						  int32_t,// token_type
																						  float,// attention_type
																						  float,// norm_type
																						  bf16_t,// scale_type
																						  int8_t,// zero_point_type
																						  bf16_t,// kv_cache_type
																						  float,// mask_type
																						  int32_t// index_type
																						  > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_bf16_fp32 };
	static constexpr const char name[]{ "Mixed-BF16/FP32" };
};


template<typename config_type, core_types> struct core_traits;

template<device_types device_type> constexpr alloc_classes alloc_class_weights{ [] {
	if constexpr (device_type == device_types::gpu) {
#if BNCH_SWT_COMPILER_CUDA
		return alloc_classes::allocate_heap;
#else
		static_assert(false, "Sorry, but it appears as though you have selected device_types::gpu, without enabling CUDA.");
#endif
	}
	return alloc_classes::mmap;
}() };

template<typename config_type> struct core_traits<config_type, core_types::attn_q>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::embedding_length, 1>> {
	static constexpr auto enum_value{ core_types::attn_q };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::attn_k>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::n_embd_kv_gqa, 1>> {
	static constexpr auto enum_value{ core_types::attn_k };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::attn_v>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::n_embd_kv_gqa, 1>> {
	static constexpr auto enum_value{ core_types::attn_v };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::attn_output>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::embedding_length, 1>> {
	static constexpr auto enum_value{ core_types::attn_output };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::attn_norm>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, 1, 1>> {
	static constexpr auto enum_value{ core_types::attn_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_gate>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::feed_forward_length, 1>> {
	static constexpr auto enum_value{ core_types::ffn_gate };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_up>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::feed_forward_length, 1>> {
	static constexpr auto enum_value{ core_types::ffn_up };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_down>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::feed_forward_length, model_dimensions<config_type>::embedding_length, 1>> {
	static constexpr auto enum_value{ core_types::ffn_down };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_norm>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, 1, 1>> {
	static constexpr auto enum_value{ core_types::ffn_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::token_embd>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::vocab_size, 1>> {
	static constexpr auto enum_value{ core_types::token_embd };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::rope_freqs>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::rope_dimension_count / 2, 1, 1>> {
	static constexpr auto enum_value{ core_types::rope_freqs };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::output_norm>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, 1, 1>> {
	static constexpr auto enum_value{ core_types::output_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::output>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::vocab_size, 1>> {
	static constexpr auto enum_value{ core_types::output };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

// MoE-specific weights (already done in previous message, but including for completeness)
template<typename config_type> struct core_traits<config_type, core_types::moe_gate>
	: public rt_dimensions<get_dimensions_type_t<1, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::num_experts, 1>> {
	static constexpr auto enum_value{ core_types::moe_gate };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_experts_gate>
	: public rt_dimensions<get_dimensions_type_t<model_dimensions<config_type>::num_experts, model_dimensions<config_type>::embedding_length,
		  model_dimensions<config_type>::feed_forward_length, 1>> {
	static constexpr auto enum_value{ core_types::moe_experts_gate };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_experts_up>
	: public rt_dimensions<get_dimensions_type_t<model_dimensions<config_type>::num_experts, model_dimensions<config_type>::embedding_length,
		  model_dimensions<config_type>::feed_forward_length, 1>> {
	static constexpr auto enum_value{ core_types::moe_experts_up };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_experts_down>
	: public rt_dimensions<get_dimensions_type_t<model_dimensions<config_type>::num_experts, model_dimensions<config_type>::feed_forward_length,
		  model_dimensions<config_type>::embedding_length, 1>> {
	static constexpr auto enum_value{ core_types::moe_experts_down };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

// Global inputs with runtime dimensions
template<typename config_type> struct core_traits<config_type, core_types::inp_tokens>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }, dimension_identifier{ 1 },
		  dimension_identifier{ 1 }>> {
	static constexpr auto enum_value{ core_types::inp_tokens };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::inp_pos>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }, dimension_identifier{ 1 },
		  dimension_identifier{ 1 }>> {
	static constexpr auto enum_value{ core_types::inp_pos };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::inp_out_ids>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ 1 }, dimension_identifier{ 1 }, dimension_identifier{ 1 }>> {
	static constexpr auto enum_value{ core_types::inp_out_ids };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::index_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::cache_k>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }, dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }>> {
	static constexpr auto enum_value{ core_types::cache_k };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::cache_v>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }, dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }>> {
	static constexpr auto enum_value{ core_types::cache_v };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::kq_mask>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ model_dimensions<config_type>::attention_head_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ 1 }>> {
	static constexpr auto enum_value{ core_types::kq_mask };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::mask_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::benchmark_data>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ model_dimensions<config_type>::attention_head_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ 1 }>> {
	static constexpr auto enum_value{ core_types::benchmark_data };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::mask_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, core_types::inp_embd_get_rows>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, core_types::token_embd>,
		  core_traits<config_type, core_types::inp_tokens>>::dims_type {
	static constexpr auto enum_value{ core_types::inp_embd_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::token_embd>;
	using input_type_02 = core_traits<config_type, core_types::inp_tokens>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_input };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ true };
};

template<typename config_type> struct core_traits<config_type, core_types::norm_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto enum_value{ core_types::norm_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::inp_embd_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::attn_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::norm_rms_norm>, core_traits<config_type, core_types::attn_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::attn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::norm_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::attn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::qcur_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::attn_q>, core_traits<config_type, core_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::qcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::attn_q>;
	using input_type_02 = core_traits<config_type, core_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::qcur_reshape>
	: public kernel_traits<kernel_types_type<kernel_types::reshape>, core_traits<config_type, core_types::qcur_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }>>>::dims_type {
	static constexpr auto enum_value{ core_types::qcur_reshape };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::qcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::reshape };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool active_math_mixin{ true };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::qcur_rope>
	: public kernel_traits<kernel_types_type<kernel_types::rope>, core_traits<config_type, core_types::qcur_reshape>, core_traits<config_type, core_types::inp_pos>,
		  core_traits<config_type, core_types::rope_freqs>>::dims_type {
	static constexpr auto enum_value{ core_types::qcur_rope };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::qcur_reshape>;
	using input_type_02 = core_traits<config_type, core_types::inp_pos>;
	using input_type_03 = core_traits<config_type, core_types::rope_freqs>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rope };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kcur_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::attn_k>, core_traits<config_type, core_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::kcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::attn_k>;
	using input_type_02 = core_traits<config_type, core_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kcur_reshape>
	: public kernel_traits<kernel_types_type<kernel_types::reshape>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }>>,
		  core_traits<config_type, core_types::kcur_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::kcur_reshape };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::reshape };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool active_math_mixin{ true };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kcur_rope>
	: public kernel_traits<kernel_types_type<kernel_types::rope>, core_traits<config_type, core_types::kcur_reshape>, core_traits<config_type, core_types::inp_pos>,
		  core_traits<config_type, core_types::rope_freqs>>::dims_type {
	static constexpr auto enum_value{ core_types::kcur_rope };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kcur_reshape>;
	using input_type_02 = core_traits<config_type, core_types::inp_pos>;
	using input_type_03 = core_traits<config_type, core_types::rope_freqs>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rope };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::vcur_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::attn_v>, core_traits<config_type, core_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::vcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::attn_v>;
	using input_type_02 = core_traits<config_type, core_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::k_cache_view>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, core_types::cache_k>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::n_embd_kv_gqa * model_dimensions<config_type>::max_context_length }, dimension_identifier{ 1 },
			  dimension_identifier{ 1 }>>>::dims_type {
	static constexpr auto enum_value{ core_types::k_cache_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::cache_k>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::k_cache_view_copy>
	: public kernel_traits<kernel_types_type<kernel_types::copy>, core_traits<config_type, core_types::kcur_rope>, core_traits<config_type, core_types::k_cache_view>>::dims_type {
	static constexpr auto enum_value{ core_types::k_cache_view_copy };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::kcur_rope>;
	using input_type_02 = core_traits<config_type, core_types::k_cache_view>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::copy };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::vcur_transpose>
	: public kernel_traits<kernel_types_type<kernel_types::transpose>, core_traits<config_type, core_types::vcur_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<0, 2, 1, 3>>>::dims_type {
	static constexpr auto enum_value{ core_types::vcur_transpose };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::vcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::transpose };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::v_cache_view>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, core_types::cache_v>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
			  dimension_identifier{ model_dimensions<config_type>::n_embd_kv_gqa }, dimension_identifier{ 1 }>>>::dims_type {
	static constexpr auto enum_value{ core_types::v_cache_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::cache_v>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::v_cache_view_copy>
	: public kernel_traits<kernel_types_type<kernel_types::copy>, core_traits<config_type, core_types::vcur_transpose>,
		  core_traits<config_type, core_types::v_cache_view>>::dims_type {
	static constexpr auto enum_value{ core_types::v_cache_view_copy };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::v_cache_view>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::copy };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::v_view>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, core_types::v_cache_view_copy>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count }, dimension_identifier{ model_dimensions<config_type>::rope_dimension_count },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }>>>::dims_type {
	static constexpr auto enum_value{ core_types::v_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::v_cache_view_copy>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::k_view>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, core_types::k_cache_view_copy>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }>>>::dims_type {
	static constexpr auto enum_value{ core_types::k_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, core_types::k_cache_view_copy>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::q_permute>
	: public kernel_traits<kernel_types_type<kernel_types::permute>, core_traits<config_type, core_types::qcur_rope>, rt_dimensions<get_dimensions_type_t<0, 1, 3, 2>>>::dims_type {
	static constexpr auto enum_value{ core_types::q_permute };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::qcur_rope>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::permute };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kq_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::k_view>, core_traits<config_type, core_types::q_permute>>::dims_type {
	static constexpr auto enum_value{ core_types::kq_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::k_view>;
	using input_type_02 = core_traits<config_type, core_types::q_permute>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ true };
};

template<typename config_type> struct core_traits<config_type, core_types::kq_soft_max>
	: public kernel_traits<kernel_types_type<kernel_types::softmax>, core_traits<config_type, core_types::kq_mul_mat>, core_traits<config_type, core_types::kq_mask>>::dims_type {
	static constexpr auto enum_value{ core_types::kq_soft_max };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kq_mask>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::softmax };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kqv_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::v_view>, core_traits<config_type, core_types::kq_soft_max>>::dims_type {
	static constexpr auto enum_value{ core_types::kqv_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::v_view>;
	using input_type_02 = core_traits<config_type, core_types::kq_soft_max>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kqv_merged_permute>
	: public kernel_traits<kernel_types_type<kernel_types::permute>, core_traits<config_type, core_types::kqv_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<0, 1, 3, 2>>>::dims_type {
	static constexpr auto enum_value{ core_types::kqv_merged_permute };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kqv_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::permute };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kqv_merged_cont>
	: public kernel_traits<kernel_types_type<kernel_types::cont>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::embedding_length },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }, dimension_identifier{ 1 }>>,
		  core_traits<config_type, core_types::kqv_merged_permute>>::dims_type {
	static constexpr auto enum_value{ core_types::kqv_merged_cont };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kqv_merged_permute>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::cont };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool active_math_mixin{ true };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::kqv_out_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::attn_output>,
		  core_traits<config_type, core_types::kqv_merged_cont>>::dims_type {
	static constexpr auto enum_value{ core_types::kqv_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::attn_output>;
	using input_type_02 = core_traits<config_type, core_types::kqv_merged_cont>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

// FFN PIPELINE (continuing with remaining ~100+ more...)

template<typename config_type> struct core_traits<config_type, core_types::ffn_inp_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::kqv_out_mul_mat>,
		  core_traits<config_type, core_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kqv_out_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::inp_embd_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::norm_pre_ffn_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ core_types::norm_pre_ffn_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::norm_pre_ffn_rms_norm>,
		  core_traits<config_type, core_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::norm_pre_ffn_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_gate_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_gate>,
		  core_traits<config_type, core_types::ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_gate>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_silu>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, core_types::ffn_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_up_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_up>, core_traits<config_type, core_types::ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_up>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_gate_par_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::ffn_silu>, core_traits<config_type, core_types::ffn_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_silu>;
	using input_type_02 = core_traits<config_type, core_types::ffn_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::ffn_out_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_down>,
		  core_traits<config_type, core_types::ffn_gate_par_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::ffn_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_down>;
	using input_type_02 = core_traits<config_type, core_types::ffn_gate_par_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::layer_out_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::ffn_out_mul_mat>,
		  core_traits<config_type, core_types::ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ core_types::layer_out_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_out_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ true };
};

template<typename config_type> struct core_traits<config_type, core_types::node_1016_get_rows>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, core_types::kqv_out_mul_mat>,
		  core_traits<config_type, core_types::inp_out_ids>>::dims_type {
	static constexpr auto enum_value{ core_types::node_1016_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kqv_out_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::inp_out_ids>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::node_1017_get_rows>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, core_types::layer_out_add>,
		  core_traits<config_type, core_types::inp_out_ids>>::dims_type {
	static constexpr auto enum_value{ core_types::node_1017_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::layer_out_add>;
	using input_type_02 = core_traits<config_type, core_types::inp_out_ids>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_inp_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::node_1016_get_rows>,
		  core_traits<config_type, core_types::node_1017_get_rows>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::node_1016_get_rows>;
	using input_type_02 = core_traits<config_type, core_types::node_1017_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_norm_pre_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::layer_out_add>>::dims_type {
	static constexpr auto enum_value{ core_types::final_norm_pre_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::layer_out_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::final_norm_pre_rms_norm>,
		  core_traits<config_type, core_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_norm_pre_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_gate_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_gate>,
		  core_traits<config_type, core_types::final_ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_gate>;
	using input_type_02 = core_traits<config_type, core_types::final_ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_silu>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, core_types::final_ffn_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_ffn_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_up_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_up>,
		  core_traits<config_type, core_types::final_ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_up>;
	using input_type_02 = core_traits<config_type, core_types::final_ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_gate_par_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::final_ffn_silu>,
		  core_traits<config_type, core_types::final_ffn_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_ffn_silu>;
	using input_type_02 = core_traits<config_type, core_types::final_ffn_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_ffn_out_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::ffn_down>,
		  core_traits<config_type, core_types::final_ffn_gate_par_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::final_ffn_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::ffn_down>;
	using input_type_02 = core_traits<config_type, core_types::final_ffn_gate_par_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_layer_out_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::final_ffn_out_mul_mat>,
		  core_traits<config_type, core_types::final_ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ core_types::final_layer_out_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_ffn_out_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::final_ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_norm_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::final_layer_out_add>>::dims_type {
	static constexpr auto enum_value{ core_types::final_norm_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_layer_out_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::result_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::final_norm_pre_rms_norm>,
		  core_traits<config_type, core_types::output_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::result_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_norm_pre_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::output_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::result_output_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::output>,
		  core_traits<config_type, core_types::result_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::result_output_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::output>;
	using input_type_02 = core_traits<config_type, core_types::result_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::sample_tokens>
	: public kernel_traits<kernel_types_type<kernel_types::sample_tokens>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::max_generation_length }, dimension_identifier{ 1 },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, 1 }>>>::dims_type {
	static constexpr auto enum_value{ core_types::sample_tokens };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::result_output_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::sample_tokens };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_inp_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::kqv_out_mul_mat>,
		  core_traits<config_type, core_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::kqv_out_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::inp_embd_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::norm_pre_moe_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::moe_inp_add>>::dims_type {
	static constexpr auto enum_value{ core_types::norm_pre_moe_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::norm_pre_moe_rms_norm>,
		  core_traits<config_type, core_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::norm_pre_moe_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_router_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::moe_gate>,
		  core_traits<config_type, core_types::moe_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_router_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_gate>;
	using input_type_02 = core_traits<config_type, core_types::moe_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_router_softmax>
	: public kernel_traits<kernel_types_type<kernel_types::softmax>, core_traits<config_type, core_types::moe_router_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_router_softmax };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_router_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::softmax };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_select>
	: public kernel_traits<kernel_types_type<kernel_types::top_k>, core_traits<config_type, core_types::moe_router_softmax>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
			  dimension_identifier{ model_dimensions<config_type>::num_experts_per_tok }, dimension_identifier{ 2 }>>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_select };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_router_softmax>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::top_k };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_gate_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_gate>, core_traits<config_type, core_types::moe_norm_mul>,
		  core_traits<config_type, core_types::moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_gate>;
	using input_type_02 = core_traits<config_type, core_types::moe_norm_mul>;
	using input_type_03 = core_traits<config_type, core_types::moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_silu>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, core_types::moe_expert_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_expert_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_up_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_up>, core_traits<config_type, core_types::moe_norm_mul>,
		  core_traits<config_type, core_types::moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_up>;
	using input_type_02 = core_traits<config_type, core_types::moe_norm_mul>;
	using input_type_03 = core_traits<config_type, core_types::moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_gate_par_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::moe_expert_silu>,
		  core_traits<config_type, core_types::moe_expert_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_expert_silu>;
	using input_type_02 = core_traits<config_type, core_types::moe_expert_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_down_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_down>,
		  core_traits<config_type, core_types::moe_expert_gate_par_mul>, core_traits<config_type, core_types::moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_down_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_down>;
	using input_type_02 = core_traits<config_type, core_types::moe_expert_gate_par_mul>;
	using input_type_03 = core_traits<config_type, core_types::moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::moe_expert_weighted_sum>
	: public kernel_traits<kernel_types_type<kernel_types::weighted_sum>, core_traits<config_type, core_types::moe_expert_down_mul_mat>,
		  core_traits<config_type, core_types::moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::moe_expert_weighted_sum };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_expert_down_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weighted_sum };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

// FINAL MOE PIPELINE
template<typename config_type> struct core_traits<config_type, core_types::final_moe_inp_add>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, core_types::node_1016_get_rows>,
		  core_traits<config_type, core_types::node_1017_get_rows>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::node_1016_get_rows>;
	using input_type_02 = core_traits<config_type, core_types::node_1017_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_norm_pre_moe_rms_norm>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, core_types::layer_out_add>>::dims_type {
	static constexpr auto enum_value{ core_types::final_norm_pre_moe_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::layer_out_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_norm_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::final_norm_pre_moe_rms_norm>,
		  core_traits<config_type, core_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_norm_pre_moe_rms_norm>;
	using input_type_02 = core_traits<config_type, core_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_router_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, core_types::moe_gate>,
		  core_traits<config_type, core_types::final_moe_norm_mul>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_router_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_gate>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_router_softmax>
	: public kernel_traits<kernel_types_type<kernel_types::softmax>, core_traits<config_type, core_types::final_moe_router_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_router_softmax };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_moe_router_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::softmax };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_select>
	: public kernel_traits<kernel_types_type<kernel_types::top_k>, core_traits<config_type, core_types::final_moe_router_softmax>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
			  dimension_identifier{ model_dimensions<config_type>::num_experts_per_tok }, dimension_identifier{ 2 }>>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_select };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_moe_router_softmax>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::top_k };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_gate_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_gate>,
		  core_traits<config_type, core_types::final_moe_norm_mul>, core_traits<config_type, core_types::final_moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_gate>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_norm_mul>;
	using input_type_03 = core_traits<config_type, core_types::final_moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_silu>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, core_types::final_moe_expert_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_moe_expert_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_up_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_up>,
		  core_traits<config_type, core_types::final_moe_norm_mul>, core_traits<config_type, core_types::final_moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_up>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_norm_mul>;
	using input_type_03 = core_traits<config_type, core_types::final_moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_gate_par_mul>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, core_types::final_moe_expert_silu>,
		  core_traits<config_type, core_types::final_moe_expert_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_moe_expert_silu>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_expert_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_down_mul_mat>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>, core_traits<config_type, core_types::moe_experts_down>,
		  core_traits<config_type, core_types::final_moe_expert_gate_par_mul>, core_traits<config_type, core_types::final_moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_down_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::moe_experts_down>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_expert_gate_par_mul>;
	using input_type_03 = core_traits<config_type, core_types::final_moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat_moe };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename config_type> struct core_traits<config_type, core_types::final_moe_expert_weighted_sum>
	: public kernel_traits<kernel_types_type<kernel_types::weighted_sum>, core_traits<config_type, core_types::final_moe_expert_down_mul_mat>,
		  core_traits<config_type, core_types::final_moe_expert_select>>::dims_type {
	static constexpr auto enum_value{ core_types::final_moe_expert_weighted_sum };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, core_types::final_moe_expert_down_mul_mat>;
	using input_type_02 = core_traits<config_type, core_types::final_moe_expert_select>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weighted_sum };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
	static constexpr bool sync_point{ false };
};

template<typename core_traits_type> struct math_mixin {};

template<typename core_traits_type>
	requires(core_traits_type::kernel_type == kernel_types::reshape || core_traits_type::kernel_type == kernel_types::cont)
struct math_mixin<core_traits_type> {
	using dims_type								   = core_traits_type;
	using pre_input_type_01						   = typename core_traits_type::input_type_01;
	static constexpr auto output_dims			   = dims_type::dims;
	static constexpr auto input_dims			   = pre_input_type_01::dims;
	static constexpr uint64_t output_stride_2	   = output_dims[3];
	static constexpr uint64_t output_stride_1_base = output_dims[2] * output_dims[3];
	static constexpr uint64_t input_stride_2	   = input_dims[3];
	using dimension_type						   = dimension_t<input_dims[0], input_dims[1], input_dims[2], input_dims[3]>;
	div_mod_logic<dimension<dimension_type>, dimension_type> rt_math_ops_01{};
	div_mod_logic<dimension<dimension_type>, dimension_type> rt_math_ops_02{};
	uint32_t output_stride_0{};
	template<typename output_type, typename input_type_01> void update_math_values(output_type& output_rt_dims, input_type_01& input_rt_dims) {
		static constexpr uint64_t input_stride_1_base = input_dims[2] * input_dims[3];
		const uint32_t input_stride_0				  = input_rt_dims.template get_dim<1>() * input_stride_1_base;
		output_stride_0								  = output_rt_dims.template get_dim<1>() * output_stride_1_base;
		const uint32_t input_stride_1				  = input_stride_1_base;
		rt_math_ops_01.collect_values(input_stride_0);
		rt_math_ops_02.collect_values(input_stride_1);
	}
};

template<typename config_type, typename derived_type> struct data_mixin;

template<typename config_type, typename derived_type> struct data_mixin : public derived_type {
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	using output_type = derived_type::output_type;
	using pointer	  = output_type*;

	pointer get_data([[maybe_unused]] uint64_t index = 0) {
		return data;
	}

	pointer* get_data_ptr() {
		return &data;
	}

	void set_data(pointer data_new) {
		data = data_new;
	}

  protected:
	pointer data{};
};

template<uint64_t index> struct tag : public std::integral_constant<uint64_t, index> {};

template<auto index, typename derived_type_new> struct core_elem_base {
	using derived_type = derived_type_new;

	constexpr decltype(auto) operator[](tag<index>) & noexcept {
		return *static_cast<derived_type*>(this);
	}

	constexpr decltype(auto) operator[](tag<index>) const& noexcept {
		return *static_cast<const derived_type*>(this);
	}
};

template<auto enum_value_new, typename config_type_new> struct core_interface : public data_mixin<config_type_new, core_traits<config_type_new, enum_value_new>>,
																				public core_elem_base<enum_value_new, core_interface<enum_value_new, config_type_new>>,
																				public math_mixin<core_traits<config_type_new, enum_value_new>> {
	using config_type = config_type_new;
	uint64_t kernel_iteration_count{};
};

template<typename config_type> struct core_aggregator;

template<typename config_type> struct core_aggregator {
	static constexpr std::array values{ core_types::attn_q, core_types::attn_k, core_types::attn_v, core_types::attn_output, core_types::attn_norm, core_types::ffn_gate,
		core_types::ffn_up, core_types::ffn_down, core_types::ffn_norm, core_types::token_embd, core_types::rope_freqs, core_types::output_norm, core_types::output,
		core_types::inp_tokens, core_types::inp_pos, core_types::inp_out_ids, core_types::cache_k, core_types::cache_v, core_types::kq_mask, core_types::inp_embd_get_rows,
		core_types::norm_rms_norm, core_types::attn_norm_mul, core_types::qcur_mul_mat, core_types::qcur_reshape, core_types::qcur_rope, core_types::kcur_mul_mat,
		core_types::kcur_reshape, core_types::kcur_rope, core_types::vcur_mul_mat, core_types::k_cache_view, core_types::k_cache_view_copy, core_types::vcur_transpose,
		core_types::v_cache_view, core_types::v_cache_view_copy, core_types::v_view, core_types::k_view, core_types::q_permute, core_types::kq_mul_mat, core_types::kq_soft_max,
		core_types::kqv_mul_mat, core_types::kqv_merged_permute, core_types::kqv_merged_cont, core_types::kqv_out_mul_mat, core_types::ffn_inp_add,
		core_types::norm_pre_ffn_rms_norm, core_types::ffn_norm_mul, core_types::ffn_gate_mul_mat, core_types::ffn_silu, core_types::ffn_up_mul_mat, core_types::ffn_gate_par_mul,
		core_types::ffn_out_mul_mat, core_types::layer_out_add, core_types::node_1016_get_rows, core_types::node_1017_get_rows, core_types::final_ffn_inp_add,
		core_types::final_norm_pre_rms_norm, core_types::final_ffn_norm_mul, core_types::final_ffn_gate_mul_mat, core_types::final_ffn_silu, core_types::final_ffn_up_mul_mat,
		core_types::final_ffn_gate_par_mul, core_types::final_ffn_out_mul_mat, core_types::final_layer_out_add, core_types::final_norm_rms_norm, core_types::result_norm_mul,
		core_types::result_output_mul_mat, core_types::sample_tokens };
};

constexpr void update_index_impl(uint64_t& current_max_index, uint64_t& current_max_value, uint64_t current_new_index, uint64_t current_new_value) {
	if (current_new_value > current_max_value) {
		current_max_value = current_new_value;
		current_max_index = current_new_index;
	}
}
template<typename value_type>
concept ephemeral_kernel_types = std::remove_cvref_t<value_type>::kernel_type == kernel_types::view || std::remove_cvref_t<value_type>::kernel_type == kernel_types::copy ||
	std::remove_cvref_t<value_type>::kernel_type == kernel_types::reshape || std::remove_cvref_t<value_type>::kernel_type == kernel_types::permute ||
	std::remove_cvref_t<value_type>::kernel_type == kernel_types::transpose || std::remove_cvref_t<value_type>::kernel_type == kernel_types::cont;

template<typename value_type_01, typename value_type_02>
concept static_castable_types = requires { static_cast<value_type_02>(std::declval<value_type_01>()); };

template<typename value_type_01, typename value_type_02>
concept bit_castable_types = !static_castable_types<value_type_01, value_type_02> && std::is_trivially_copyable_v<std::remove_cvref_t<value_type_01>> &&
	std::is_trivially_copyable_v<std::remove_cvref_t<value_type_02>> && sizeof(std::remove_cvref_t<value_type_01>) == sizeof(std::remove_cvref_t<value_type_02>);

template<typename value_type>
concept weight_types = static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value) < static_cast<uint64_t>(core_types::end_of_weights);

template<typename value_type>
concept input_only_types = static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value) < static_cast<uint64_t>(core_types::end_of_input_only);

template<typename value_type>
concept global_input_types = static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value)<static_cast<uint64_t>(core_types::end_of_global_inputs) &&
	static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value)> static_cast<uint64_t>(core_types::end_of_input_only);

template<typename value_type>
concept per_block_types = static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value)<static_cast<uint64_t>(core_types::end_of_per_block) &&
	static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value)> static_cast<uint64_t>(core_types::end_of_global_inputs);

template<typename value_type>
concept global_output_types = static_cast<uint64_t>(std::remove_cvref_t<value_type>::enum_value) > static_cast<uint64_t>(core_types::end_of_per_block);

template<typename value_type>
concept active_kernel_types = global_output_types<value_type> || per_block_types<value_type> || global_input_types<value_type>;

template<typename core_traits_type> constexpr void update_index(uint64_t& current_max_index, uint64_t& current_max_value, uint64_t current_new_index, uint64_t current_new_value) {
	if constexpr (active_kernel_types<core_traits_type>) {
		update_index_impl(current_max_index, current_max_value, current_new_index, current_new_value);
	}
}

template<typename config_type> static constexpr uint64_t max_tensor_element_index{ []() {
	constexpr auto values = core_aggregator<config_type>::values;
	return []<size_t... indices>(std::index_sequence<indices...>) constexpr {
		uint64_t max_val{};
		uint64_t index{};
		(update_index<core_traits<config_type, values[indices]>>(index, max_val, indices, compute_elements(core_traits<config_type, values[indices]>::dims)), ...);
		return index;
	}(std::make_index_sequence<values.size()>{});
}() };

template<typename config_type> static constexpr uint64_t max_tensor_io_bytes_index{ []() {
	constexpr auto values = core_aggregator<config_type>::values;
	return []<size_t... indices>(std::index_sequence<indices...>) constexpr {
		uint64_t max_val{};
		uint64_t index{};
		(get_io_bytes<core_traits<config_type, values[indices]>>(index, max_val, indices), ...);
		return index;
	}.template operator()(std::make_index_sequence<values.size()>{});
}() };

enum class nihilus_cathedral_errors {
	get_core_by_index_oob,
	invalid_base_cast,
	empty_cathedral_bases_pack,
};

template<typename config_type_new, typename... bases> struct nihilus_cathedral : public bases... {
	static_assert(static_assert_printer_val<(sizeof...(bases) > 0), nihilus_cathedral_errors::empty_cathedral_bases_pack>::impl);
	using bases::operator[]...;
	using first_type  = get_first_type_t<bases...>;
	using config_type = config_type_new;
	using enum_type	  = decltype(first_type::enum_value);
	constexpr nihilus_cathedral() {
	}

	static constexpr uint64_t size{ sizeof...(bases) };

	template<template<typename, typename> typename mixin_type, typename... arg_types> constexpr void impl(arg_types&&... args) noexcept {
		(impl_internal_filtered<mixin_type, bases>(args...), ...);
	}

	template<template<typename, typename, auto...> typename mixin_type, auto... values, typename... arg_types> void impl_thread(arg_types&&... args) noexcept {
		(impl_internal_filtered_thread<mixin_type, bases, values...>(args...), ...);
	}

	template<enum_type enum_value> decltype(auto) get_core_by_enum() noexcept {
		return (*this)[tag<static_cast<uint64_t>(enum_value)>()];
	}

	template<uint64_t index_new> decltype(auto) get_core_by_index() const noexcept {
		static_assert(static_assert_printer_val<(index_new < size), nihilus_cathedral_errors::get_core_by_index_oob, index_new>::impl);
		static constexpr uint64_t index{ static_cast<uint64_t>(index_transform_values[static_cast<uint64_t>(index_new)]) };
		return (*this)[tag<index>()];
	}

	template<enum_type enum_value> static consteval uint64_t get_index_by_enum() noexcept {
		for (uint64_t x = 0; x < size; ++x) {
			if (static_cast<enum_type>(index_transform_values[x]) == enum_value) {
				return x;
			}
		}
		return std::numeric_limits<uint64_t>::max();
	}

	void* intermediate_buffer{};

  protected:
	template<template<typename, typename> typename mixin_type, typename base_type, typename... arg_types>
	constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) noexcept {
		if constexpr (mixin_type<config_type, base_type>::filter()) {
			static_assert(static_assert_printer_val<std::is_base_of_v<base_type, nihilus_cathedral>, nihilus_cathedral_errors::invalid_base_cast>::impl);
			mixin_type<config_type, base_type>::impl(*this, std::forward<arg_types>(args)...);
		}
	}

	template<template<typename, typename, auto...> typename mixin_type, typename base_type, auto... values, typename... arg_types>
	void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) noexcept {
		if constexpr (mixin_type<config_type, base_type, values...>::filter()) {
			static_assert(static_assert_printer_val<std::is_base_of_v<base_type, nihilus_cathedral>, nihilus_cathedral_errors::invalid_base_cast>::impl);
			mixin_type<config_type, base_type, values...>::impl(*this, std::forward<arg_types>(args)...);
		}
	}

	static constexpr uint64_t index_transform_values[sizeof...(bases)]{ static_cast<uint64_t>(bases::enum_value)... };
};

template<typename nihilus_cathedral_type, auto index> using get_nihilus_cathedral_type_at_enum =
	std::remove_cvref_t<decltype(std::declval<nihilus_cathedral_type>().template get_core_by_enum<index>())>;

template<typename config_type, typename enum_type, template<typename> typename aggregator_type, template<enum_type, typename...> typename base_type, typename... value_type>
struct get_nihilus_cathedral_array;

template<typename config_type, typename enum_type, template<typename> typename aggregator_type, template<enum_type, typename...> typename base_type, uint64_t... indices>
struct get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type, std::index_sequence<indices...>> {
	using type = nihilus_cathedral<config_type, base_type<static_cast<enum_type>(aggregator_type<config_type>::values[indices]), config_type>...>;
};

template<typename config_type, typename enum_type, template<typename> typename aggregator_type, template<enum_type, typename...> typename base_type>
using get_nihilus_cathedral_array_t = std::remove_cvref_t<typename get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type,
	std::make_index_sequence<static_cast<uint64_t>(aggregator_type<config_type>::values.size())>>::type>;

template<typename config_type> struct core {
	using type = get_nihilus_cathedral_array_t<config_type, core_types, core_aggregator, core_interface>;
};

template<typename config_type> using core_t = core<config_type>::type;

int main() {
	using dims_4096_1_1_1	= get_dimensions_type_t<131072, 131072, 131072, 131072>;
	using dims_1_4096_1_1	= get_dimensions_type_t<1, 4096, 1, 1>;
	using dims_1024_512_1_1 = get_dimensions_type_t<1024, 512, 1, 1>;
	using dims_8_4096_1_1	= get_dimensions_type_t<8, 4096, 1, 1>;
	using dims_8_1024_1_1	= get_dimensions_type_t<8, 1024, 1, 1>;
	using dims_32_128_1_1	= get_dimensions_type_t<32, 128, 1, 1>;
	using dims_permute_0213 = get_dimensions_type_t<0, 2, 1, 3>;
	using dims_32_1024_1_1	= get_dimensions_type_t<32, 1024, 1, 1>;
	using dims_permute_0312 = get_dimensions_type_t<0, 3, 1, 2>;

	// Runtime dimensions
	using dims_batch_seq_4096_1 = get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, 1 },
		dimension_identifier{ runtime_dimension_value_types::sequence_length, 1 }, dimension_identifier{ 4096 }, dimension_identifier{ 1 }>;

	using dims_batch_seq_128_64 = get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, 1 },
		dimension_identifier{ runtime_dimension_value_types::sequence_length, 1 }, dimension_identifier{ 128 }, dimension_identifier{ 64 }>;

	// NOW THE INSTANTIATIONS:

	// 1. softmax (unary - single input)
	using softmax_trait = kernel_traits<kernel_types_type<kernel_types::softmax>, dims_batch_seq_4096_1>;
	softmax_trait softmax{};

	// 2. softmax (binary - with mask)
	using softmax_masked_trait = kernel_traits<kernel_types_type<kernel_types::softmax>, dims_batch_seq_4096_1, dims_batch_seq_128_64>;
	//softmax_masked_trait softmax_binary{};

	// 3. top_k
	using top_k_trait = kernel_traits<kernel_types_type<kernel_types::top_k>, dims_8_4096_1_1, dims_8_1024_1_1>;
	top_k_trait top_k{};

	// 4. mul_mat_moe
	using moe_trait = kernel_traits<kernel_types_type<kernel_types::mul_mat_moe>,
		dims_4096_1_1_1,// expert_weights
		dims_batch_seq_4096_1,// input_acts
		dims_8_1024_1_1// expert_selection
		>;
	moe_trait moe{};

	// 5. weighted_sum
	using weighted_sum_trait = kernel_traits<kernel_types_type<kernel_types::weighted_sum>,
		dims_batch_seq_128_64,// expert_outputs
		dims_32_128_1_1// router_weights
		>;
	weighted_sum_trait weighted_sum{};

	// 6. rope (ternary)
	using rope_trait = kernel_traits<kernel_types_type<kernel_types::rope>, dims_batch_seq_4096_1, dims_batch_seq_4096_1, dims_batch_seq_4096_1>;
	rope_trait rope{};
	// 7. reshape

	using reshape_trait = kernel_traits<kernel_types_type<kernel_types::reshape>, dims_8_4096_1_1, dims_32_1024_1_1>;

	reshape_trait reshape{};

	// 8. view
	using view_trait = kernel_traits<kernel_types_type<kernel_types::view>,
		dims_8_4096_1_1,// input
		dims_4096_1_1_1// mod_mask
		>;
	view_trait view{};

	// 9. transpose
	using transpose_trait = kernel_traits<kernel_types_type<kernel_types::transpose>,
		dims_batch_seq_128_64,// input
		dims_permute_0213// transpose axes
		>;
	transpose_trait transpose{};

	// 10. permute
	using permute_trait = kernel_traits<kernel_types_type<kernel_types::permute>,
		dims_batch_seq_128_64,// input
		dims_permute_0312// permutation
		>;
	permute_trait permute{};

	// 11. mul_mat
	using mul_mat_trait = kernel_traits<kernel_types_type<kernel_types::mul_mat>,
		dims_4096_1_1_1,// weights (transposed)
		dims_batch_seq_4096_1// activations
		>;
	mul_mat_trait mul_mat{};

	// 12. get_rows
	using get_rows_trait = kernel_traits<kernel_types_type<kernel_types::get_rows>,
		dims_4096_1_1_1,// embedding table
		dims_batch_seq_128_64// token indices
		>;
	get_rows_trait get_rows{};

	// 13. cont
	using cont_trait = kernel_traits<kernel_types_type<kernel_types::cont>,
		dims_batch_seq_4096_1,// output
		dims_batch_seq_4096_1// input (same shape, making contiguous)
		>;
	cont_trait cont{};

	// 14. sample_tokens
	using sample_trait = kernel_traits<kernel_types_type<kernel_types::sample_tokens>, dims_batch_seq_4096_1>;
	sample_trait sample{};
	static constexpr auto config = generate_model_config(max_context_length_type{ 131072 }, model_sizes::llm_8B, kernel_type_profiles::fp16_mha, model_arches::llama,
		device_types::cpu, benchmark_type::enabled, model_generations::v3_1, exceptions_type::enabled, max_generation_length_type{ 131072 / 2 }, max_batch_size_type{ 1 },
		max_prompt_length_type{ 131072 / 2 });
	using config_type			 = model_config_type<config>;
	config_type config_type_new{};
	core_traits<config_type, core_types::output> output{};
	core_traits<config_type, core_types::qcur_rope> qcur_mul_mat{};
	core_traits<config_type, core_types::sample_tokens> sample_tokens{};
	// 15. binary operations (add, mul, sub, div, copy)
	using add_trait	 = kernel_traits<kernel_types_type<kernel_types::add>, dims_8_4096_1_1, dims_8_4096_1_1>;
	using mul_trait	 = kernel_traits<kernel_types_type<kernel_types::mul>, dims_8_4096_1_1, dims_1_4096_1_1>;// broadcasting
	using copy_trait = kernel_traits<kernel_types_type<kernel_types::copy>, dims_8_4096_1_1, dims_8_4096_1_1>;
	add_trait add{};
	mul_trait mul{};
	copy_trait copy{};

	// USAGE EXAMPLE:
	// Access the resulting dimensions type:
	using softmax_output_dims = softmax_trait::dims_type;
	{
		static constexpr dimension_identifier value_01{ runtime_dimension_value_types::sequence_length, 2048 };
		static constexpr dimension_identifier value_02{ runtime_dimension_value_types::none, 2048 };
		static constexpr dimension_identifier value_03{ runtime_dimension_value_types::none, 2048 };
		static constexpr dimension_identifier value_04{ runtime_dimension_value_types::none, 2048 };
		rt_dimensions<decltype(generate_dimensions<value_01, value_02, value_03, value_04>())> dimensions_new{};
		dimensions_new.template set_rt_dim<runtime_dimension_value_types::sequence_length>(10);
		std::cout << "VALUE-01 VALUE: " << dimensions_new.get_dim<0>().value << std::endl;
		std::cout << "VALUE-01 DIV-VALUE: " << 8192 / dimensions_new.get_dim<0>() << std::endl;
		std::cout << "VALUE-01 TYPE: " << typeid(dimensions_new.get_dim<0>().const_value).name() << std::endl;
		std::cout << "VALUE-02 VALUE: " << dimensions_new.get_dim<1>().const_value << std::endl;
		std::cout << "VALUE-02 DIV-VALUE: " << 8192 / dimensions_new.get_dim<1>() << std::endl;
		std::cout << "VALUE-02 TYPE: " << typeid(dimensions_new.get_dim<1>().const_value).name() << std::endl;
		std::cout << "VALUE-03 VALUE: " << dimensions_new.get_dim<2>().const_value << std::endl;
		std::cout << "VALUE-03 DIV-VALUE: " << 8192 / dimensions_new.get_dim<2>() << std::endl;
		std::cout << "VALUE-04 TYPE: " << typeid(dimensions_new.get_dim<3>().const_value).name() << std::endl;
		std::cout << "VALUE-04 VALUE: " << dimensions_new.get_dim<3>().const_value << std::endl;
		std::cout << "VALUE-04 DIV-VALUE: " << 8192 / dimensions_new.get_dim<3>() << std::endl;
	}

	{
		static constexpr dimension_identifier value_01{ runtime_dimension_value_types::sequence_length, 32 };
		static constexpr dimension_identifier value_02{ runtime_dimension_value_types::none, 32 };
		static constexpr dimension_identifier value_03{ runtime_dimension_value_types::none, 32 };
		static constexpr dimension_identifier value_04{ runtime_dimension_value_types::none, 32 };
		rt_dimensions<decltype(generate_dimensions<value_01, value_02, value_03, value_04>())> dimensions_new{};
		dimensions_new.template set_rt_dim<runtime_dimension_value_types::sequence_length>(10);
		std::cout << "VALUE-01 VALUE: " << dimensions_new.get_dim<0>().value << std::endl;
		std::cout << "VALUE-01 DIV-VALUE: " << 8192 / dimensions_new.get_dim<0>() << std::endl;
		std::cout << "VALUE-01 TYPE: " << typeid(dimensions_new.get_dim<0>().const_value).name() << std::endl;
		std::cout << "VALUE-02 VALUE: " << dimensions_new.get_dim<1>().const_value << std::endl;
		std::cout << "VALUE-02 DIV-VALUE: " << 8192 / dimensions_new.get_dim<1>() << std::endl;
		std::cout << "VALUE-02 TYPE: " << typeid(dimensions_new.get_dim<1>().const_value).name() << std::endl;
		std::cout << "VALUE-03 VALUE: " << dimensions_new.get_dim<2>().const_value << std::endl;
		std::cout << "VALUE-03 DIV-VALUE: " << 8192 / dimensions_new.get_dim<2>() << std::endl;
		std::cout << "VALUE-04 TYPE: " << typeid(dimensions_new.get_dim<3>().const_value).name() << std::endl;
		std::cout << "VALUE-04 VALUE: " << dimensions_new.get_dim<3>().const_value << std::endl;
		std::cout << "VALUE-04 DIV-VALUE: " << 8192 / dimensions_new.get_dim<3>() << std::endl;
	}


	static thread_local std::mt19937 engine{ static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) };

	static thread_local std::uniform_real_distribution<double> dist(-1.0f, 1.0f);

	std::vector<std::vector<double>> floats_01{};
	floats_01.resize(total_iterations);
	std::vector<std::vector<double>> floats_02{};
	floats_02.resize(total_iterations);
	for (uint64_t x = 0; x < total_iterations; ++x) {
		floats_01[x].resize(floats_to_generate);
		floats_02[x].resize(floats_to_generate);
	}
	uint64_t current_index{};
	struct test_std_random_generation {
		BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<double>>& floats, uint64_t& current_index) {
			auto& out = floats[current_index];

			for (uint64_t x = 0; x < floats_to_generate; ++x) {
				out[x] = dist(engine);
				bnch_swt::do_not_optimize_away(out[x]);
			}

			++current_index;
			return out.size() * sizeof(double);
		}
	};


	struct test_bnch_swt_randomizer {
		BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<double>>& floats, uint64_t& current_index) {
			for (uint64_t x = 0; x < floats_to_generate; ++x) {
				floats[current_index][x] = bnch_swt::random_generator<double>::impl();
				bnch_swt::do_not_optimize_away(floats[current_index][x]);
			}
			++current_index;
			return floats[current_index].size() * sizeof(double);
		}
	};

	bnch_swt::benchmark_stage<"generate_random_values", total_iterations, measured_iterations>::run_benchmark<"test_std_random_generation", test_std_random_generation>(floats_01,
		current_index);
	current_index = 0;
	bnch_swt::benchmark_stage<"generate_random_values", total_iterations, measured_iterations>::run_benchmark<"test_bnch_swt_randomizer", test_bnch_swt_randomizer>(floats_02,
		current_index);
	bnch_swt::benchmark_stage<"generate_random_values", total_iterations, measured_iterations>::print_results();

	return 0;
}