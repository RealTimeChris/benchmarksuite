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
#ifndef NIHILUS_ALIGN
	#define NIHILUS_ALIGN(alignment) alignas(alignment)
#endif

#ifndef NIHILUS_HOST
	#define NIHILUS_HOST __forceinline__ __host__
#endif

#ifndef NIHILUS_DEVICE
	#define NIHILUS_DEVICE __forceinline__ __device__
#endif

#ifndef NIHILUS_HOST_DEVICE
	#define NIHILUS_HOST_DEVICE __forceinline__ __host__ __device__
#endif

template<typename value_type, value_type...> struct uint_type;

template<typename value_type>
concept uint64_types = std::is_integral_v<value_type> && sizeof(value_type) == 8;

template<typename value_type>
concept uint32_types = std::is_integral_v<value_type> && sizeof(value_type) == 4;

template<typename value_type> struct NIHILUS_ALIGN(bnch_swt::device_alignment) uint_pair {
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> multiplicand{};
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> shift{};
};

template<typename value_type, value_type...> struct uint_type;

template<uint64_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	value_type lo{};
	value_type hi{};

	NIHILUS_HOST_DEVICE constexpr uint_type() {
	}

	NIHILUS_HOST_DEVICE constexpr uint_type(value_type l) : lo{ l } {
	}

	constexpr uint_type(value_type h, value_type l) : lo{ l }, hi{ h } {
	}

	constexpr explicit operator value_type() const {
		return lo;
	}

	constexpr bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	constexpr bool operator!=(const uint_type& other) const {
		return !(*this == other);
	}

	constexpr bool operator<(const uint_type& other) const {
		if (hi != other.hi) {
			return hi < other.hi;
		}
		return lo < other.lo;
	}

	constexpr bool operator>(const uint_type& other) const {
		return other < *this;
	}

	constexpr bool operator<=(const uint_type& other) const {
		return !(*this > other);
	}

	constexpr bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	constexpr uint_type operator~() const {
		return uint_type{ ~hi, ~lo };
	}

	constexpr uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	friend constexpr uint_type operator+(value_type lhs, const uint_type& other) {
		return other + lhs;
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

	constexpr uint_type operator>>(int32_t shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ 0, hi >> (shift - 64) };
		}
		return uint_type{ hi >> shift, (lo >> shift) | (hi << (64 - shift)) };
	}

	constexpr uint_type operator*(const uint_type& other) const {
		value_type u1 = lo >> 32;
		value_type u0 = lo & 0xFFFFFFFF;
		value_type v1 = other.lo >> 32;
		value_type v0 = other.lo & 0xFFFFFFFF;

		value_type t  = u0 * v0;
		value_type w0 = t & 0xFFFFFFFF;
		value_type k  = t >> 32;

		t			  = (u1 * v0) + k;
		value_type w1 = t & 0xFFFFFFFF;
		value_type w2 = t >> 32;

		t = (u0 * v1) + w1;
		k = t >> 32;

		value_type split_hi = (u1 * v1) + w2 + k;
		value_type split_lo = (t << 32) + w0;

		value_type cross_1 = lo * other.hi;
		value_type cross_2 = hi * other.lo;

		return uint_type{ split_hi + cross_1 + cross_2, split_lo };
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

	constexpr uint_type& operator+=(const uint_type& other) {
		*this = *this + other;
		return *this;
	}

	constexpr uint_type& operator-=(const uint_type& other) {
		*this = *this - other;
		return *this;
	}

	constexpr uint_type& operator*=(const uint_type& other) {
		*this = *this * other;
		return *this;
	}

	constexpr uint_type& operator/=(const uint_type& other) {
		*this = *this / other;
		return *this;
	}

	constexpr uint_type& operator<<=(int32_t shift) {
		*this = *this << shift;
		return *this;
	}

	constexpr uint_type& operator>>=(int32_t shift) {
		*this = *this >> shift;
		return *this;
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
		constexpr uint_type div_minus_1 = divisor_new - 1ULL;
		constexpr value_type l			= 127ULL - div_minus_1.lzcnt();
		constexpr uint_type numerator	= uint_type{ 1ULL } << (64ULL + static_cast<value_type>(l));
		constexpr uint_type m_128		= (numerator + div_temp - 1ULL) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64ULL + l };
	}
};

template<uint32_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	NIHILUS_HOST_DEVICE constexpr uint_type() {
	}

	static constexpr uint64_t shift(uint64_t value, uint64_t shift) {
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
			return 64U;
		}
		uint64_t x	 = value;
		value_type n = 0U;
		if (x <= 0x00000000FFFFFFFFULL) {
			n += 32U;
			x <<= 32ULL;
		}
		if (x <= 0x0000FFFFFFFFFFFFULL) {
			n += 16U;
			x <<= 16ULL;
		}
		if (x <= 0x00FFFFFFFFFFFFFFULL) {
			n += 8U;
			x <<= 8ULL;
		}
		if (x <= 0x0FFFFFFFFFFFFFFFULL) {
			n += 4U;
			x <<= 4ULL;
		}
		if (x <= 0x3FFFFFFFFFFFFFFFULL) {
			n += 2U;
			x <<= 2ULL;
		}
		if (x <= 0x7FFFFFFFFFFFFFFFULL) {
			n += 1U;
		}
		return n;
	}

	consteval static uint_pair<value_type> collect_values() {
		if constexpr (divisor_new == 1U) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		constexpr uint64_t div_minus_1{ divisor_new - 1ULL };
		constexpr uint64_t lz{ lzcnt(div_minus_1) };

		if constexpr (lz > 63ULL) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		constexpr uint64_t l{ 63ULL - lz };
		constexpr uint64_t numerator{ shift(1ULL, static_cast<value_type>(32ULL + l)) };
		constexpr uint64_t m_128{ div(numerator + divisor_new - 1ULL, divisor_new) };
		return uint_pair<value_type>{ { static_cast<value_type>(m_128) }, { static_cast<value_type>(32ULL + l) } };
	}
};

template<uint64_types value_type> struct uint_type<value_type> {
	value_type lo{};
	value_type hi{};

	NIHILUS_HOST_DEVICE constexpr uint_type() {
	}

	NIHILUS_HOST_DEVICE constexpr uint_type(value_type h, value_type l = 0) : lo{ l }, hi{ h } {
	}

	NIHILUS_HOST_DEVICE explicit operator value_type() const {
		return lo;
	}

	NIHILUS_HOST_DEVICE bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	NIHILUS_HOST_DEVICE bool operator>(const uint_type& other) const {
		return other < *this;
	}

	NIHILUS_HOST_DEVICE bool operator<(const uint_type& other) const {
		if (hi != other.hi) {
			return hi < other.hi;
		}
		return lo < other.lo;
	}

	NIHILUS_HOST_DEVICE bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	NIHILUS_HOST_DEVICE uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	NIHILUS_HOST_DEVICE uint_type operator-(const uint_type& other) const {
		const value_type new_lo = lo - other.lo;
		const value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	NIHILUS_HOST_DEVICE uint_type operator<<(value_type shift) const {
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

	NIHILUS_HOST_DEVICE uint_type operator/(const uint_type& other) const {
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

		for (std::make_signed_t<value_type> i = 127; i >= 0; --i) {
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

	NIHILUS_HOST_DEVICE value_type lzcnt() const {
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

	NIHILUS_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		if (divisor_new == 1ULL) {
			return uint_pair<value_type>{ 1ULL, 0ULL };
		}

		uint_type div_temp{ 0ULL, divisor_new };
		uint_type div_minus_1 = div_temp - uint_type{ 0ULL, 1ULL };
		value_type lz		  = div_minus_1.lzcnt();
		if (lz > 127ULL) {
			return uint_pair<value_type>{ 1ULL, 0ULL };
		}
		value_type l		= 127ULL - lz;
		uint_type numerator = uint_type{ 0ULL, 1ULL } << static_cast<value_type>(64ULL + l);
		uint_type m_128		= (numerator + div_temp - uint_type{ 0ULL, 1ULL }) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64ULL + l };
	}
};

template<uint32_types value_type> struct uint_type<value_type> {
	NIHILUS_HOST_DEVICE constexpr uint_type() {
	}

	NIHILUS_HOST_DEVICE static value_type lzcnt(uint64_t value) {
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

	NIHILUS_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		if (divisor_new == 1U) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		uint64_t div_minus_1 = divisor_new - 1ULL;
		uint64_t lz			 = lzcnt(div_minus_1);

		if (lz > 63ULL) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		uint64_t l			 = 63ULL - lz;
		uint64_t numerator	 = 1ULL << static_cast<value_type>(32ULL + l);
		const uint64_t m_128 = (numerator + divisor_new - 1) / divisor_new;
		return uint_pair<value_type>{ { static_cast<value_type>(m_128) }, { static_cast<value_type>(32ULL + l) } };
	}
};

template<typename value_type, value_type const_value_new> struct NIHILUS_ALIGN(bnch_swt::device_alignment) const_aligned_uint {
	static constexpr bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> const_value{ const_value_new };
};

template<typename value_type, value_type const_value_new> struct NIHILUS_ALIGN(bnch_swt::device_alignment) aligned_uint : public const_aligned_uint<value_type, const_value_new> {
	mutable bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> value{};
};

template<typename, typename value_type, bool, value_type> struct div_mod_logic;

template<typename value_type> NIHILUS_HOST_DEVICE static value_type mul128Generic(value_type u, value_type v, value_type& hi) noexcept {
	value_type u1 = u >> 32;
	value_type u0 = u & 0xFFFFFFFF;
	value_type v1 = v >> 32;
	value_type v0 = v & 0xFFFFFFFF;

	value_type t  = (u0 * v0);
	value_type w0 = t & 0xFFFFFFFF;
	value_type k  = t >> 32;

	t			  = (u1 * v0) + k;
	value_type w1 = t & 0xFFFFFFFF;
	value_type w2 = t >> 32;

	t = (u0 * v1) + w1;
	k = t >> 32;

	hi = (u1 * v1) + w2 + k;
	return (t << 32) + w0;
}

NIHILUS_HOST_DEVICE static uint64_t host_umulhi64(uint64_t a, uint64_t b) {
	uint64_t high;
	mul128Generic(a, b, high);
	return high;
}

template<typename derived_type, typename value_type, value_type divisor> struct NIHILUS_ALIGN(bnch_swt::device_alignment) div_mod_logic<derived_type, value_type, true, divisor>
	: public aligned_uint<value_type, divisor>, public uint_type<value_type> {
	uint_pair<value_type> multiplicand_and_shift{};
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	NIHILUS_HOST_DEVICE constexpr div_mod_logic() {
	}

	NIHILUS_DEVICE constexpr value_type& get_value() const {
		return static_cast<const aligned_uint<value_type, divisor>*>(this)->value.value;
	}

	NIHILUS_HOST void collect_values(value_type d) {
		aligned_uint<value_type, divisor>::value.emplace(d);
		multiplicand_and_shift = uint_type<value_type>::collect_values(d);
	}

	NIHILUS_HOST_DEVICE value_type div(value_type val) const {
		if (get_value() == 1) {
			return val;
		}
#if NIHILUS_COMPILER_CUDA && defined(__CUDA_ARCH__)
		if constexpr (std::same_as<value_type, uint64_t>) {
			return __umul64hi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 64ULL);
		} else {
			return __umulhi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 32ULL);
		}
#else
		if constexpr (std::same_as<value_type, uint64_t>) {
			uint64_t high_part = host_umulhi64(multiplicand_and_shift.multiplicand, val);
			return high_part >> (multiplicand_and_shift.shift - 64ULL);
		} else {
			return static_cast<value_type>((static_cast<uint64_t>(val) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
		}
#endif
	}

	NIHILUS_HOST_DEVICE value_type mod(value_type val) const {
		return val - (div(val) * get_value());
	}

	NIHILUS_HOST_DEVICE friend value_type operator<(value_type lhs, const div_mod_logic& rhs) {
		return lhs < rhs.value.value;
	}

	NIHILUS_HOST_DEVICE friend value_type operator>(value_type lhs, const div_mod_logic& rhs) {
		return lhs > rhs.value.value;
	}

	NIHILUS_HOST_DEVICE friend value_type operator>=(value_type lhs, const div_mod_logic& rhs) {
		return lhs >= rhs.value.value;
	}

	NIHILUS_HOST_DEVICE friend value_type operator>=(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value >= rhs;
	}

	NIHILUS_HOST_DEVICE friend value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	NIHILUS_HOST_DEVICE friend value_type operator*(value_type lhs, const div_mod_logic& rhs) {
		return lhs * rhs.value.value;
	}

	NIHILUS_HOST_DEVICE friend value_type operator*(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value * rhs;
	}

	NIHILUS_HOST_DEVICE friend value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<typename value_type> NIHILUS_HOST_DEVICE consteval bool is_power_of_2(value_type N) {
	return N > 0 && (N & (N - 1)) == 0;
}

template<typename value_type> NIHILUS_HOST_DEVICE consteval value_type log2_ct(value_type N) {
	value_type result = 0;
	value_type value  = N;
	while (value >>= 1) {
		++result;
	}
	return result;
}

template<typename derived_type, typename value_type, value_type divisor> struct NIHILUS_ALIGN(bnch_swt::device_alignment) div_mod_logic<derived_type, value_type, false, divisor>
	: public uint_type<value_type, divisor>, public const_aligned_uint<value_type, divisor> {
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	NIHILUS_DEVICE static constexpr value_type get_value() {
		return derived_type::const_value;
	}

	static constexpr uint_pair<value_type> multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };

	NIHILUS_HOST_DEVICE value_type div(value_type val) const {
		if constexpr (divisor == 1ULL) {
			return val;
		}
		if constexpr (is_power_of_2(divisor)) {
			static constexpr value_type shift_amount{ log2_ct(divisor) };
			return val >> shift_amount;
		} else {
#if NIHILUS_COMPILER_CUDA && defined(__CUDA_ARCH__)
			if constexpr (std::same_as<value_type, uint64_t>) {
				return __umul64hi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 64ULL);
			} else {
				return __umulhi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 32ULL);
			}
#else
			if constexpr (std::same_as<value_type, uint64_t>) {
				uint64_t high_part = host_umulhi64(multiplicand_and_shift.multiplicand, val);
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

	NIHILUS_HOST_DEVICE value_type mod(value_type val) const {
		if constexpr (is_power_of_2(divisor)) {
			return val & (divisor - 1);
		} else {
			return val - (div(val) * divisor);
		}
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator<(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs < value;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator>(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs > value;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator>=(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs >= value;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator>=(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value >= rhs;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator*(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs * value;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator*(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value * rhs;
	}

	NIHILUS_HOST_DEVICE friend constexpr value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<typename value_type, value_type static_divisor> struct division {
	NIHILUS_DEVICE static value_type div(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			static constexpr value_type shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			static constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.div(value);
		}
	}
};

template<typename value_type, value_type static_divisor> struct modulo {
	NIHILUS_DEVICE static value_type mod(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1ULL);
		} else {
			static constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.mod(value);
		}
	}
};

constexpr uint64_t TOTAL_ITERATIONS	   = 10;
constexpr uint64_t MEASURED_ITERATIONS = 5;
constexpr size_t N_ELEMENTS			   = 4096ULL * 256ULL;
constexpr uint64_t TEST_DIVISOR		   = 2048ULL;

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
	const div_mod_logic<aligned_uint<value_type, 0>, value_type, true, 0> magic_new,
	size_t n, value_type* __restrict__ iteration_counter) {
	int idx					   = blockIdx.x * blockDim.x + threadIdx.x;
	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	if (idx >= n) {
		return;
	}

	size_t offset = current_iteration * n;

	recursive_div_bomb(input, output, magic_new, offset, idx, 0, 16384);
}

template<typename value_type> __global__ void magic_div_kernel(const value_type* __restrict__ input, value_type* __restrict__ output,
	const div_mod_logic<aligned_uint<value_type, 0>, value_type, true, 0>* __restrict__ magic_new, size_t n, value_type* __restrict__ iteration_counter) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	value_type current_iteration = *iteration_counter;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*iteration_counter = current_iteration + 1;
	}

	if (idx >= n) {
		return;
	}
	static constexpr div_mod_logic<aligned_uint<value_type, 2048>, value_type, false, 2048> magic{};

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += magic.div(input[offset + idx]);
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += magic.div(input[offset + idx]);
	}

	for (uint32_t x = 0; x < 16384; ++x) {
		size_t offset = current_iteration * n;
		output[idx] += magic.div(input[offset + idx]);
	}
}

int main() {
	cudaDeviceReset(); 
	{
		std::vector<uint64_t> host_input;
		uint64_t *d_input = nullptr, *d_output_native = nullptr, *d_output_magic = nullptr, *d_iteration_counter = nullptr;

		prepare_data(host_input, d_input, d_output_native, d_output_magic, d_iteration_counter, N_ELEMENTS, TOTAL_ITERATIONS);

		using MagicType = div_mod_logic<aligned_uint<uint64_t, 0>, uint64_t, true, 0>;
		MagicType magic_div;
		magic_div.collect_values(TEST_DIVISOR);

		MagicType* d_magic = nullptr;
		cudaMalloc(&d_magic, sizeof(MagicType));
		cudaMemcpy(d_magic, &magic_div, sizeof(MagicType), cudaMemcpyHostToDevice);

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

		using Bench = bnch_swt::benchmark_stage<"division-benchmark-64-bit", TOTAL_ITERATIONS, MEASURED_ITERATIONS, bnch_swt::benchmark_types::cuda, false, "Operations">;

		uint64_t reset_counter = 0;

		std::cout << "=== BASELINE BENCHMARKS ===\n\n";

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		native_div_kernel<<<grid, block>>>(d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS, d_iteration_counter);
		if (auto error = cudaGetLastError(); error != cudaSuccess) {
			std::cout << "Warmup ERROR: " << cudaGetErrorString(error) << "\n";
		}
		cudaDeviceSynchronize();
		static constexpr auto native_div_kernel_ptr = &native_div_kernel<uint64_t>;
		static constexpr auto magic_div_kernel_ptr	= &magic_div_kernel<uint64_t>;
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint64_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division", magic_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, d_magic, N_ELEMENTS, d_iteration_counter);
		Bench::print_results();
		cudaFree(d_magic);
		cleanup(d_input, d_output_native, d_output_magic, d_iteration_counter);
	}

	{
		std::vector<uint32_t> host_input;
		uint32_t *d_input = nullptr, *d_output_native = nullptr, *d_output_magic = nullptr, *d_iteration_counter = nullptr;

		prepare_data(host_input, d_input, d_output_native, d_output_magic, d_iteration_counter, N_ELEMENTS, TOTAL_ITERATIONS);

		using MagicType = div_mod_logic<aligned_uint<uint32_t, 0>, uint32_t, true, 0>;
		MagicType magic_div;
		magic_div.collect_values(TEST_DIVISOR);

		MagicType* d_magic = nullptr;
		cudaMalloc(&d_magic, sizeof(MagicType));
		cudaMemcpy(d_magic, &magic_div, sizeof(MagicType), cudaMemcpyHostToDevice);

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

		using Bench = bnch_swt::benchmark_stage<"division-benchmark-32-bit", TOTAL_ITERATIONS, MEASURED_ITERATIONS, bnch_swt::benchmark_types::cuda, false, "Operations">;

		uint32_t reset_counter = 0;

		std::cout << "=== BASELINE BENCHMARKS ===\n\n";

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		native_div_kernel<<<grid, block>>>(d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS, d_iteration_counter);
		if (auto error = cudaGetLastError(); error != cudaSuccess) {
			std::cout << "Warmup ERROR: " << cudaGetErrorString(error) << "\n";
		}
		cudaDeviceSynchronize();
		static constexpr auto native_div_kernel_ptr = &native_div_kernel<uint32_t>;
		static constexpr auto magic_div_kernel_ptr	= &magic_div_kernel<uint32_t>;
		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"native-division", native_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_native, TEST_DIVISOR, N_ELEMENTS,
			d_iteration_counter);

		cudaMemcpy(d_iteration_counter, &reset_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
		Bench::run_benchmark<"magic-division", magic_div_kernel_ptr>(grid, block, shared_mem, bytes_transferred, d_input, d_output_magic, d_magic, N_ELEMENTS, d_iteration_counter);
		Bench::print_results();
		cudaFree(d_magic);
		cleanup(d_input, d_output_native, d_output_magic, d_iteration_counter);
	}

	std::cout << "\nBenchmark finished.\n";
	return 0;
}