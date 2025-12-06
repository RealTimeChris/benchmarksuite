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
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstddef>

template<typename value_type, value_type...> struct uint_type;

template<typename value_type>
concept uint64_types = std::is_integral_v<value_type> && sizeof(value_type) == 8;

template<typename value_type>
concept uint32_types = std::is_integral_v<value_type> && sizeof(value_type) == 4;

template<typename value_type> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) uint_pair {
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> multiplicand{};
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> shift{};
};

template<typename value_type, value_type...> struct uint_type;

template<uint64_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	value_type lo{};
	value_type hi{};

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type l) : lo{ l } {
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
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
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

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type h, value_type l = 0) : lo{ l }, hi{ h } {
	}

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

	BNCH_SWT_HOST_DEVICE uint_type operator<<(value_type shift) const {
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
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
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

template<typename value_type, value_type const_value_new> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) const_aligned_uint {
	static constexpr bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> const_value{ const_value_new };
};

template<typename value_type, value_type const_value_new> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) aligned_uint : public const_aligned_uint<value_type, const_value_new> {
	mutable bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> value{};
};

template<typename, typename value_type, bool, value_type> struct div_mod_logic;

template<typename value_type> BNCH_SWT_HOST_DEVICE static value_type mul128Generic(value_type u, value_type v, value_type& hi) noexcept {
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

BNCH_SWT_HOST_DEVICE static uint64_t host_umulhi64(uint64_t a, uint64_t b) {
	uint64_t high;
	mul128Generic(a, b, high);
	return high;
}

template<typename derived_type, typename value_type, value_type divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic<derived_type, value_type, true, divisor>
	: public aligned_uint<value_type, divisor>, public uint_type<value_type> {
	uint_pair<value_type> multiplicand_and_shift{};
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	BNCH_SWT_HOST_DEVICE constexpr div_mod_logic() {
	}

	BNCH_SWT_DEVICE constexpr value_type& get_value() const {
		return static_cast<const aligned_uint<value_type, divisor>*>(this)->value.value;
	}

	BNCH_SWT_HOST void collect_values(value_type d) {
		aligned_uint<value_type, divisor>::value.emplace(d);
		multiplicand_and_shift = uint_type<value_type>::collect_values(d);
	}

	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const {
		if (get_value() == 1) {
			return val;
		}
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
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

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		return val - (div(val) * get_value());
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator<(value_type lhs, const div_mod_logic& rhs) {
		return lhs < rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>(value_type lhs, const div_mod_logic& rhs) {
		return lhs > rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(value_type lhs, const div_mod_logic& rhs) {
		return lhs >= rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(value_type lhs, const div_mod_logic& rhs) {
		return lhs * rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value * rhs;
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

template<typename derived_type, typename value_type, value_type divisor> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) div_mod_logic<derived_type, value_type, false, divisor>
	: public uint_type<value_type, divisor>, public const_aligned_uint<value_type, divisor> {
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	BNCH_SWT_DEVICE static constexpr value_type get_value() {
		return derived_type::const_value;
	}

	static constexpr uint_pair<value_type> multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };

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

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		if constexpr (is_power_of_2(divisor)) {
			return val & (divisor - 1);
		} else {
			return val - (div(val) * divisor);
		}
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator<(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs < value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs > value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs >= value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs * value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value * rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<typename value_type, value_type static_divisor> struct division {
	BNCH_SWT_DEVICE static constexpr value_type div(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			constexpr value_type shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.div(value);
		}
	}
};

template<typename value_type, value_type static_divisor> struct modulo {
	BNCH_SWT_DEVICE static constexpr value_type mod(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1ULL);
		} else {
			constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.mod(value);
		}
	}
};

enum class kernel_types : uint8_t {
	weights,
	global_inputs,
	get_rows,
	rms_norm,
	mul,
	mul_mat,
	mul_mat_add,
	mul_mat_add_fused,
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

static constexpr const std::string_view kernel_types_names[static_cast<uint64_t>(kernel_types::count)]{
	"weights",
	"global_inputs",
	"get_rows",
	"rms_norm",
	"mul",
	"mul_mat",
	"mul_mat_add",
	"mul_mat_add_fused",
	"mul_mat_moe",
	"reshape",
	"transpose",
	"permute",
	"view",
	"rope",
	"softmax",
	"silu",
	"copy",
	"cont",
	"add",
	"sub",
	"div",
	"top_k",
	"weighted_sum",
	"sample_tokens",
};


struct model_traits {
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 8192;
	static constexpr uint32_t block_count			  = 126;
	static constexpr uint32_t feed_forward_length	  = 53248;
	static constexpr uint32_t attention_head_count	  = 128;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
	static constexpr uint32_t sequence_length		  = 2;
};

template<auto multiple, typename value_type_01 = decltype(multiple)> BNCH_SWT_HOST constexpr value_type_01 round_up_to_multiple(value_type_01 value) noexcept {
	if constexpr ((multiple > 0) && ((multiple & (multiple - 1)) == 0)) {
		constexpr value_type_01 mulSub1{ multiple - 1 };
		return (value + mulSub1) & ~mulSub1;
	} else {
		return ((value + multiple - 1) / multiple) * multiple;
	}
}

template<typename value_type> BNCH_SWT_HOST constexpr decltype(auto) move(value_type&& arg) noexcept {
	return static_cast<std::remove_reference_t<value_type>&&>(arg);
}

template<class value_type_01> BNCH_SWT_HOST constexpr void swap(value_type_01& left, value_type_01& right) noexcept(
	std::is_nothrow_move_constructible_v<value_type_01> && std::is_nothrow_move_assignable_v<value_type_01>) {
	value_type_01 tmp = ::move(left);
	left			  = ::move(right);
	right			  = ::move(tmp);
}

struct cuda_buffer {
	using size_type	 = uint64_t;
	using value_type = std::byte;
	using pointer	 = value_type*;
	BNCH_SWT_HOST cuda_buffer() noexcept {
	}
	BNCH_SWT_HOST cuda_buffer& operator=(const cuda_buffer&) noexcept = delete;
	BNCH_SWT_HOST cuda_buffer(const cuda_buffer&) noexcept			  = delete;

	BNCH_SWT_HOST cuda_buffer& operator=(cuda_buffer&& other) noexcept {
		if (this != &other) {
			::swap(data_val, other.data_val);
			::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_HOST cuda_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_HOST void init(uint64_t size) {
		if (data_val) {
			clear();
		}

		cudaError_t result = cudaMalloc(&data_val, size);
		if (result != cudaSuccess) {
			data_val = nullptr;
			throw std::exception{ "Failed to allocate!" };
		}

		size_val = size;
	}

	BNCH_SWT_HOST void deinit() noexcept {
		clear();
	}

	BNCH_SWT_HOST size_type size() noexcept {
		return size_val;
	}

	template<typename value_type> BNCH_SWT_HOST value_type* data() noexcept {
		return std::bit_cast<value_type*>(data_val);
	}

	template<typename value_type> BNCH_SWT_HOST value_type* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<512ULL>(offset_to_claim);
		pointer return_value	= data_val + aligned_amount;
		return std::bit_cast<value_type*>(return_value);
	}

	BNCH_SWT_HOST ~cuda_buffer() noexcept {
		clear();
	}

  protected:
	size_type size_val{};
	pointer data_val{};

	BNCH_SWT_HOST void clear() noexcept {
		if (data_val) {
			cudaFree(data_val);
			data_val = nullptr;
			size_val = 0;
		}
	}
};

template<typename value_type>
concept integral_types = std::is_integral_v<std::remove_cvref_t<value_type>>;

enum class get_value_type_errors {
	invalid_type,
};

template<typename value_type> using base_t = std::remove_cvref_t<value_type>;

template<typename value_type>
concept r_value_reference_types = std::is_rvalue_reference_v<value_type>;

template<typename value_type>
concept uint_types = std::is_unsigned_v<base_t<value_type>> && integral_types<value_type>;

template<typename value_type>
concept int_types = std::is_signed_v<base_t<value_type>> && integral_types<value_type>;

template<typename value_type>
concept integral8_types = integral_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept integral16_types = integral_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept integral32_types = integral_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept integral64_types = integral_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type>
concept int8_types = int_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept int16_types = int_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept int32_types = int_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept int64_types = int_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type>
concept uint8_types = uint_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept uint16_types = uint_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept float_types = std::floating_point<base_t<value_type>>;

template<typename value_type>
concept float16_types = std::is_same_v<base_t<value_type>, half> || std::is_same_v<base_t<value_type>, bf16_t>;

template<typename value_type>
concept float32_types = float_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept float64_types = float_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type> using x_type = decltype(base_t<value_type>::x);

template<typename value_type>
concept half_cuda_types = std::is_same_v<__half, base_t<value_type>>;

template<typename value_type>
concept cuda_integral8_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept cuda_integral16_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept cuda_integral32_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept cuda_integral64_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept int8_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept int16_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept int32_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept int64_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept uint8_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept uint16_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept uint32_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept uint64_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept float32_cuda_types = float32_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept float64_cuda_types = float64_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept dim04_types = requires() { base_t<value_type>::w; };

template<typename value_type>
concept dim03_types = requires() { base_t<value_type>::z; } && !dim04_types<value_type>;

template<typename value_type>
concept dim02_types = requires() { base_t<value_type>::y; } && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim01_types = requires() { base_t<value_type>::x; } && !dim02_types<value_type> && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim_types = requires() { base_t<value_type>::x; };

template<typename value_type> struct get_value {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		static_assert(false);
	}
};

struct block_q8_0 {};

template<typename derived_type> struct type_traits_base {
	BNCH_SWT_HOST static constexpr uint64_t count_elements(const auto& dims) {
		return cast<uint64_t>(dims[0]) * cast<uint64_t>(dims[1]) * cast<uint64_t>(dims[2]) * cast<uint64_t>(dims[3]);
	}

	BNCH_SWT_HOST static constexpr uint64_t total_byte_size(const auto& dims_new) {
		uint64_t element_count{ count_elements(dims_new) };
		if constexpr (derived_type::block_size == 1) {
			return element_count * derived_type::type_size;
		} else {
			return (element_count + derived_type::block_size - 1) / derived_type::block_size * derived_type::type_size;
		}
	}
};

struct type_traits_dynamic {
	uint64_t block_size{};
	uint64_t type_size{};
	bool is_quantized{};
};

template<typename data_types> struct type_traits;

template<typename derived_type> struct get_dynamic_type_traits {
	BNCH_SWT_HOST_DEVICE consteval static type_traits_dynamic get_dynamic_type_traits_impl() {
		type_traits_dynamic return_values{};
		return_values.is_quantized = derived_type::is_quantized;
		return_values.block_size   = derived_type::block_size;
		return_values.type_size	   = derived_type::type_size;
		return return_values;
	}
};

template<typename value_type_new> struct type_traits;

template<integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			  public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<> struct type_traits<bf16_t> : public type_traits_base<type_traits<bf16_t>>, public get_dynamic_type_traits<type_traits<bf16_t>> {
	using value_type = bf16_t;
	using quant_type = bf16_t;
	inline static constexpr uint64_t type_size{ sizeof(bf16_t) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

#if BNCH_SWT_COMPILER_CUDA
template<> struct type_traits<half> : public type_traits_base<type_traits<half>>, public get_dynamic_type_traits<type_traits<half>> {
	using value_type = half;
	using quant_type = half;
	inline static constexpr uint64_t type_size{ sizeof(half) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<float32_cuda_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				 public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<float64_cuda_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				 public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

#endif

template<> struct type_traits<float> : public type_traits_base<type_traits<float>>, public get_dynamic_type_traits<type_traits<float>> {
	using value_type = float;
	using quant_type = float;
	inline static constexpr uint64_t type_size{ sizeof(float) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<> struct type_traits<double> : public type_traits_base<type_traits<double>>, public get_dynamic_type_traits<type_traits<double>> {
	using value_type = double;
	using quant_type = double;
	inline static constexpr uint64_t type_size{ sizeof(double) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};


template<typename value_type> struct get_value_type;

template<> struct get_value_type<float> {
	using type = float4;
};

template<> struct get_value_type<double> {
	using type = double4;
};

template<> struct get_value_type<half> {
	using type = ushort4;
};

template<> struct get_value_type<bf16_t> {
	using type = ushort4;
};

template<> struct get_value_type<uint8_t> {
	using type = uchar4;
};

template<> struct get_value_type<uint16_t> {
	using type = ushort4;
};

template<> struct get_value_type<uint32_t> {
	using type = uint4;
};

template<> struct get_value_type<uint64_t> {
	using type = ulong4;
};

template<> struct get_value_type<int8_t> {
	using type = char4;
};

template<> struct get_value_type<int16_t> {
	using type = short4;
};

template<> struct get_value_type<int32_t> {
	using type = int4;
};

template<> struct get_value_type<int64_t> {
	using type = long4;
};

template<typename value_type> using get_value_type_t = get_value_type<value_type>::type;

template<int8_cuda_types value_type> struct get_value<value_type> {
	template<typename... value_types>
		requires(sizeof...(value_types) == 1)
	BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char1(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char1(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char2(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char3(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char4(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short1(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short2(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short3(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short4(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int1(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int2(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int3(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int4(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long1(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long2(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long3(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long4(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar1(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar2(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar3(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar4(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort1(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort2(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort3(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort4(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint1(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint2(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint3(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint4(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong1(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong2(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong3(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong4(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float1(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float2(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float3(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float4(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double1(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double2(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double3(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double4(std::forward<value_types>(args)...);
	}
};

enum class binary_op_types {
	add,
	mul,
	sub,
	div,
};

template<binary_op_types> struct binary_op_core;

template<> struct binary_op_core<binary_op_types::add> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) + static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 += static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::mul> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) * static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 *= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::sub> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) - static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 -= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::div> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) / static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 /= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<typename value_type, binary_op_types binary_op_type> struct binary_op_base;

template<dim01_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type	 = binary_op_core<binary_op_type>;
		using get_value_type = get_value<value_type>;
		return get_value_type::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
	}
};

template<dim02_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type	 = binary_op_core<binary_op_type>;
		using get_value_type = get_value<value_type>;
		return get_value_type::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
	}
};

template<dim03_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type	 = binary_op_core<binary_op_type>;
		using get_value_type = get_value<value_type>;
		return get_value_type::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y),
			op_core_type::impl(std::forward<value_type01>(val01).z, std::forward<value_type02>(val02).z));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, std::forward<value_type02>(val02).z);
	}
};

template<dim04_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type	 = binary_op_core<binary_op_type>;
		using get_value_type = get_value<value_type>;
		return get_value_type::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y),
			op_core_type::impl(std::forward<value_type01>(val01).z, std::forward<value_type02>(val02).z),
			op_core_type::impl(std::forward<value_type01>(val01).w, std::forward<value_type02>(val02).w));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, std::forward<value_type02>(val02).z);
		op_core_type::impl_in_place(val01.w, std::forward<value_type02>(val02).w);
	}
};

template<binary_op_types binary_op_type> struct binary_op {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return binary_op_base<value_type01, binary_op_type>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		return binary_op_base<value_type01, binary_op_type>::impl_in_place(val01, std::forward<value_type02>(val02));
	}
};

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE void operator+=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator+(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE void operator*=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator*(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE void operator-=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator-(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE void operator/=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator/(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

struct cpu_buffer {
	using size_type	 = uint64_t;
	using value_type = std::byte;
	using pointer	 = value_type*;
	BNCH_SWT_HOST cpu_buffer() noexcept {
	}
	BNCH_SWT_HOST cpu_buffer& operator=(const cpu_buffer&) noexcept = delete;
	BNCH_SWT_HOST cpu_buffer(const cpu_buffer&) noexcept			= delete;

	BNCH_SWT_HOST cpu_buffer& operator=(cpu_buffer&& other) noexcept {
		if (this != &other) {
			::swap(data_val, other.data_val);
			::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_HOST cpu_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_HOST void init(uint64_t size) noexcept {
		if (data_val.size()) {
			clear();
		}
		data_val.resize(size);

		size_val = size;
	}

	BNCH_SWT_HOST void deinit() noexcept {
		clear();
	}

	BNCH_SWT_HOST size_type size() noexcept {
		return size_val;
	}

	BNCH_SWT_HOST pointer data() noexcept {
		return data_val.data();
	}

	BNCH_SWT_HOST void* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<512ULL>(offset_to_claim);
		pointer return_value	= data_val.data() + aligned_amount;
		return return_value;
	}

	BNCH_SWT_HOST ~cpu_buffer() noexcept {
		clear();
	}

  protected:
	std::vector<value_type> data_val{};
	size_type size_val{};

	BNCH_SWT_HOST void clear() noexcept {
		data_val.clear();
	}
};

template<uint32_types value_type> BNCH_SWT_DEVICE value_type fast_min(value_type value_01, value_type value_02) {
	uint32_t res;
	asm("min.u32 %0, %1, %2;" : "=r"(res) : "r"(value_01), "r"(value_02));
	return res;
}

template<uint64_types value_type> BNCH_SWT_DEVICE value_type fast_min(value_type value_01, value_type value_02) {
	uint64_t res;
	asm("min.u64 %0, %1, %2;" : "=l"(res) : "l"(value_01), "l"(value_02));
	return res;
}

template<typename value_type, kernel_types> struct cuda_kernel_traits_impl {};

template<typename value_type, typename, kernel_types...> struct cuda_sub_kernel;

template<typename value_type, kernel_types kernel_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new, uint32_t max_values>
struct cpu_baseline;

template<typename value_type, kernel_types kernel_type> struct cuda_kernel_traits : public cuda_kernel_traits_impl<value_type, kernel_type> {
	static constexpr uint64_t register_byte_count{ 255 * 4 };
	static constexpr uint64_t register_value_count{ register_byte_count / sizeof(value_type) };
	static constexpr uint64_t bytes_per_execution{ cuda_kernel_traits_impl<value_type, kernel_type>::values_per_execution * sizeof(value_type) };
	static constexpr uint64_t total_executions{ register_byte_count / bytes_per_execution };
};

template<typename value_type, kernel_types kernel_type>
	requires(kernel_type == kernel_types::add || kernel_type == kernel_types::mul)
struct cuda_kernel_traits_impl<value_type, kernel_type> {
	static constexpr uint64_t flops_per_byte_moved{ bnch_swt::gpu_properties::flops / bnch_swt::gpu_properties::memory_bw };
	static constexpr uint64_t values_per_execution{ 28 };
};

template<typename value_type, size_t... indices> struct cuda_sub_kernel<value_type, std::index_sequence<indices...>, kernel_types::add> {
	BNCH_SWT_DEVICE static void impl(value_type* __restrict __grid_constant__ input_01, value_type* __restrict __grid_constant__ input_02,
		value_type* __restrict __grid_constant__ output) {
		((*(output + indices) = *(input_01 + indices) + *(input_02 + indices)), ...);
	};
};

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new, kernel_types... kernel_types>
	requires(((kernel_types == kernel_types::add) || ...))
BNCH_SWT_GLOBAL void test_function(value_type* __restrict __grid_constant__ input_01, value_type* __restrict __grid_constant__ input_02,
	value_type* __restrict __grid_constant__ output) {
	constexpr uint32_t vectors_per_thread			= cuda_kernel_traits<value_type, kernel_types::add>::total_executions;
	const uint32_t global_thread_id					= blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t total_threads					= gridDim.x * blockDim.x;
	static constexpr uint32_t total_scalar_elements = dim_00_new * dim_01_new * dim_02_new * dim_03_new;
	static constexpr uint32_t total_vector_elements = (total_scalar_elements + 3) / 4;
	static constexpr uint32_t total_chunks			= (total_vector_elements + vectors_per_thread - 1) / vectors_per_thread;
	for (uint32_t chunk_id = global_thread_id; chunk_id < total_chunks; chunk_id += total_threads) {
		uint32_t base_offset = fast_min(chunk_id * vectors_per_thread, total_vector_elements);
		cuda_sub_kernel<value_type, std::make_index_sequence<vectors_per_thread>, kernel_types...>::impl(input_01 + base_offset, input_02 + base_offset, output + base_offset);
	}
};

template<typename value_type> struct cuda_sub_kernel_runtime_add {
	BNCH_SWT_DEVICE static void impl(value_type* __restrict __grid_constant__ input_01, value_type* __restrict __grid_constant__ input_02,
		value_type* __restrict __grid_constant__ output, uint32_t base_offset, uint32_t max_elements, uint32_t vectors_per_thread) {
		for (uint32_t i = 0; i < vectors_per_thread; ++i) {
			uint32_t idx = base_offset + i;
			if (idx < max_elements) {
				output[idx] = input_01[idx] + input_02[idx];
			}
		}
	}
};

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new> BNCH_SWT_GLOBAL void test_function_branched(
	value_type* __restrict __grid_constant__ input_01, value_type* __restrict __grid_constant__ input_02, value_type* __restrict __grid_constant__ output) {
	constexpr uint32_t vectors_per_thread = cuda_kernel_traits<value_type, kernel_types::add>::total_executions;

	const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t total_threads	= gridDim.x * blockDim.x;

	static constexpr uint32_t total_scalar_elements = dim_00_new * dim_01_new * dim_02_new * dim_03_new;

	static constexpr uint32_t total_vector_elements = (total_scalar_elements + 3) / 4;

	static constexpr uint32_t total_chunks = (total_vector_elements + vectors_per_thread - 1) / vectors_per_thread;

	for (uint32_t chunk_id = global_thread_id; chunk_id < total_chunks; chunk_id += total_threads) {
		uint32_t base_offset = chunk_id * vectors_per_thread;
		if (base_offset >= total_vector_elements) {
			return;
		}
		cuda_sub_kernel_runtime_add<value_type>::impl(input_01, input_02, output, base_offset, total_vector_elements, vectors_per_thread);
	}
}

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new, uint32_t max_values>
struct cpu_baseline<value_type, kernel_types::add, dim_00_new, dim_01_new, dim_02_new, dim_03_new, max_values> {
	BNCH_SWT_HOST static uint64_t impl(value_type* cpu_input_01, value_type* cpu_input_02, value_type* cpu_output, uint64_t output_element_count) {
		for (uint64_t i = 0; i < output_element_count; ++i) {
			cpu_output[i] = cpu_input_01[i] + cpu_input_02[i];
		}
		return output_element_count * sizeof(value_type) * 3;
	};
};

static constexpr uint64_t total_iteration_count{ 12 };
static constexpr uint64_t measured_iterations{ 4 };

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t previous_offset, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new,
	uint64_t dim_03_new>
struct tensor {
	static constexpr uint64_t dim_00{ dim_00_new };
	static constexpr uint64_t dim_01{ dim_01_new };
	static constexpr uint64_t dim_02{ dim_02_new };
	static constexpr uint64_t dim_03{ dim_03_new };
	static constexpr uint64_t element_count{ dim_00 * dim_01 * dim_02 * dim_03 };
	static constexpr uint64_t padding_elements{ cuda_kernel_traits<value_type, kernel_type>::values_per_execution };
	static constexpr uint64_t byte_count{ round_up_to_multiple<512ULL>(sizeof(value_type) * ((element_count + padding_elements) + vectors_per_thread + 1)) };
	static constexpr uint64_t offset{ previous_offset };
	value_type* data{};
};

struct memory_footprint {
	uint64_t byte_count{};
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new>
struct cuda_tensors {
	tensor<value_type, kernel_type, vectors_per_thread, 0, dim_00_new, dim_01_new, dim_02_new, dim_03_new> input_01{};

	static constexpr uint64_t offset_01{ tensor<value_type, kernel_type, vectors_per_thread, 0, dim_00_new, dim_01_new, dim_02_new, dim_03_new>::byte_count };
	tensor<value_type, kernel_type, vectors_per_thread, offset_01, dim_00_new, dim_01_new, dim_02_new, dim_03_new> input_02{};

	static constexpr uint64_t offset_02{ offset_01 + tensor<value_type, kernel_type, vectors_per_thread, offset_01, dim_00_new, dim_01_new, dim_02_new, dim_03_new>::byte_count };

	tensor<value_type, kernel_type, vectors_per_thread, offset_02, dim_00_new, dim_01_new, dim_02_new, dim_03_new> output{};
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new>
constexpr std::array footprints{
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>::input_01)::byte_count },
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>::input_02)::byte_count },
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>::output)::byte_count },
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new>
uint64_t byte_count{ [] {
	uint64_t return_value{};
	for (uint64_t x = 0; x < footprints<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>.size(); ++x) {
		return_value += footprints<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>[x].byte_count;
	}
	return return_value;
}() };

template<typename value_type, uint64_t value_count> BNCH_SWT_HOST void generate_values(void* cuda_memory, void* cpu_memory, float min = -1.0f, float max = 1.0f) {
	static std::vector<value_type> host_values;
	if (host_values.size() < value_count)
		host_values.resize(value_count);

	for (uint64_t x = 0; x < value_count; ++x) {
		float val = bnch_swt::random_generator<float>::impl(min, max);

		if constexpr (std::is_same_v<value_type, __half>) {
			host_values[x] = __float2half(val);
		} else {
			host_values[x] = static_cast<value_type>(val);
		}
	}

	std::memcpy(cpu_memory, host_values.data(), value_count * sizeof(value_type));
	cudaMemcpy(cuda_memory, host_values.data(), value_count * sizeof(value_type), cudaMemcpyHostToDevice);
}

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new>
BNCH_SWT_HOST cuda_tensors<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new> generate_cuda_data(cuda_buffer& buffer,
	cpu_buffer& cpu_buffer) {
	cuda_tensors<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new> return_values{};
	return_values.input_01.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.input_01.offset);
	auto* cpu_buffer_ptr		= cpu_buffer.data() + return_values.input_01.offset;

	return_values.input_02.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.input_02.offset);
	cpu_buffer_ptr				= cpu_buffer.data() + return_values.input_02.offset;

	return_values.output.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.output.offset);
	cpu_buffer_ptr			  = cpu_buffer.data() + return_values.output.offset;
	return return_values;
}

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t dim_00_new, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new>
cuda_buffer buffer{ [] {
	cuda_buffer return_values{};
	return_values.init(byte_count<value_type, kernel_type, vectors_per_thread, dim_00_new, dim_01_new, dim_02_new, dim_03_new>);
	return return_values;
}() };

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new, uint32_t max_values>
struct cpu_baseline<value_type, kernel_types::mul_mat_add, dim_00_new, dim_01_new, dim_02_new, dim_03_new, max_values> {
	BNCH_SWT_HOST static uint64_t impl(value_type* cpu_input_A, value_type* cpu_input_B, value_type* cpu_input_C, value_type* cpu_output, uint32_t M, uint32_t K, uint32_t N) {
		uint32_t current_count{};
		for (uint32_t m = 0; m < M; ++m) {
			for (uint32_t n = 0; n < N; ++n) {
				value_type accumulator = 0;
				for (uint32_t k = 0; k < K; ++k) {
					accumulator += cpu_input_A[m * K + k] * cpu_input_B[k * N + n];
				}
				if (current_count < max_values) {
					cpu_output[m * N + n] = accumulator + cpu_input_C[m * N + n];
					++current_count;
				} else {
					return (M * K + K * N + M * N + M * N) * sizeof(value_type);
				}
			}
		}

		return (M * K + K * N + M * N + M * N) * sizeof(value_type);
	};
};

template<typename value_type> struct cuda_kernel_traits_impl<value_type, kernel_types::mul_mat_add> {
	static constexpr int BM					= 128;
	static constexpr int BN					= 128;
	static constexpr int BK					= 16;
	static constexpr uint32_t TM			= 16;
	static constexpr uint32_t TN			= 16;
	static constexpr uint32_t THREADS_X		= 1024;
	static constexpr uint32_t THREADS_Y		= 1;
	static constexpr uint32_t THREADS_TOTAL = THREADS_X * THREADS_Y;

	static constexpr uint64_t flops_per_element = 2;
};
#include <cutlass/gemm/device/gemm.h>

struct cutlass_gemm_host_wrapper {
	static cudaStream_t stream_global;
	using Gemm = cutlass::gemm::device::Gemm<half, cutlass::layout::RowMajor, half, cutlass::layout::RowMajor, half, cutlass::layout::RowMajor, half>;

	template<typename GemmType, typename ArgsType> static void impl(half* A, half* B, half* C, half* D, uint32_t M, uint32_t K, uint32_t N, GemmType gemm_op, ArgsType args) {
		cutlass::Status status = gemm_op(args, nullptr, stream_global);
	}
};

cudaStream_t cutlass_gemm_host_wrapper::stream_global = nullptr;
#include <mma.h>
using namespace nvcuda;

template<typename value_type, size_t... indices> struct cuda_sub_kernel<value_type, std::index_sequence<indices...>, kernel_types::mul_mat_add> {
	static constexpr int WMMA_M = 16;
	static constexpr int WMMA_N = 16;
	static constexpr int WMMA_K = 16;

	BNCH_SWT_DEVICE static void impl(value_type* __restrict input_A, value_type* __restrict input_B, value_type* __restrict input_C, value_type* __restrict output_D, uint32_t M,
		uint32_t K, uint32_t N, void* intermediate_buffer) {
		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

		const uint32_t tid			= threadIdx.x;
		const uint32_t warp_id		= division<uint32_t, 32>::div(tid);
		const uint32_t block_warps	= division<uint32_t, 32>::div(blockDim.x);
		const uint32_t grid_warp_id = (blockIdx.x * block_warps) + warp_id;

		const uint32_t warps_n = division<uint32_t, WMMA_N>::div(N + WMMA_N - 1);

		const uint32_t warp_row = (grid_warp_id / warps_n) * WMMA_M;
		const uint32_t warp_col = (grid_warp_id % warps_n) * WMMA_N;

		if (warp_row < M && warp_col < N) {
			wmma::load_matrix_sync(acc_frag, reinterpret_cast<float*>(input_C) + (warp_row * N + warp_col), N, wmma::mem_row_major);
		} else {
			wmma::fill_fragment(acc_frag, 0.0f);
		}

		__shared__ half sA_h[16][17];
		__shared__ half sB_h[16][17];

		ushort4* sA_v	   = reinterpret_cast<ushort4*>(sA_h);
		ushort4* sB_v	   = reinterpret_cast<ushort4*>(sB_h);
		const ushort4* A_v = reinterpret_cast<const ushort4*>(input_A);
		const ushort4* B_v = reinterpret_cast<const ushort4*>(input_B);

		const uint32_t lane_id = modulo<uint32_t, 32>::mod(tid);

		const uint32_t k_vec_width	  = K >> 2;
		const uint32_t n_vec_width	  = N >> 2;
		const uint32_t col_vec_offset = warp_col >> 2;

		for (uint32_t i = 0; i < K; i += WMMA_K) {
			const uint32_t i_vec_offset = i >> 2;

#pragma unroll
			for (uint32_t load_step = 0; load_step < 2; ++load_step) {
				uint32_t vec_idx = lane_id + (load_step * 32);

				uint32_t r	   = vec_idx >> 2;
				uint32_t c_vec = vec_idx & 3;
				if (warp_row + r < M && (i + (c_vec << 2)) < K) {
					sA_v[vec_idx] = A_v[((warp_row + r) * k_vec_width) + (i_vec_offset + c_vec)];
					sB_v[vec_idx] = B_v[((i + r) * n_vec_width) + (col_vec_offset + c_vec)];
				}
			}

			__syncwarp();
			wmma::load_matrix_sync(a_frag, reinterpret_cast<half*>(sA_h), 16);
			wmma::load_matrix_sync(b_frag, reinterpret_cast<half*>(sB_h), 16);
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
			__syncwarp();
		}

		wmma::store_matrix_sync(reinterpret_cast<float*>(output_D) + (warp_row * N + warp_col), acc_frag, N, wmma::mem_row_major);
	}
};

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new>
BNCH_SWT_GLOBAL void test_function_matmul(value_type* __restrict input_A, value_type* __restrict input_B, value_type* __restrict input_C, value_type* __restrict output_D,
	uint32_t M, uint32_t K, uint32_t N, void* __restrict intermediate_buffer) {
	cuda_sub_kernel<value_type, std::index_sequence<>, kernel_types::mul_mat_add>::impl(input_A, input_B, input_C, output_D, M, K, N, intermediate_buffer);
}

BNCH_SWT_HOST bool check_cuda_result() {
	if (auto result = cudaGetLastError(); result) {
		std::cerr << "Cuda Error of Type: " << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
		return false;
	} else {
		return true;
	}
}

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<uint32_t M, uint32_t K, uint32_t N, uint32_t P> BNCH_SWT_GLOBAL void test_function_fused_dual_matmul(half* __restrict__ input_A, half* __restrict__ input_B,
	half* __restrict__ input_E, half* __restrict__ bias_C1, half* __restrict__ bias_D, half* __restrict__ output_D, void* __restrict__ l2_c1_storage) {
	static constexpr uint32_t WMMA_TILE = 16;
	static constexpr uint32_t WARP_SIZE = 32;

	static constexpr uint32_t M_tiles = (M + WMMA_TILE - 1) / WMMA_TILE;
	static constexpr uint32_t N_tiles = (N + WMMA_TILE - 1) / WMMA_TILE;
	static constexpr uint32_t P_tiles = (P + WMMA_TILE - 1) / WMMA_TILE;
	static constexpr uint32_t K_tiles = (K + WMMA_TILE - 1) / WMMA_TILE;

	static constexpr uint32_t total_c1_tiles = M_tiles * N_tiles;
	static constexpr uint32_t total_d_tiles	 = M_tiles * P_tiles;

	const uint32_t tid			= threadIdx.x;
	const uint32_t warp_id		= division<uint32_t, WARP_SIZE>::div(tid);
	const uint32_t block_warps	= division<uint32_t, WARP_SIZE>::div(blockDim.x);
	const uint32_t grid_warp_id = (blockIdx.x * block_warps) + warp_id;

	__shared__ half sA[WMMA_TILE][WMMA_TILE + 1];
	__shared__ half sB_or_E[WMMA_TILE][WMMA_TILE + 1];

	half* l2_c1 = reinterpret_cast<half*>(l2_c1_storage);

	wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, half> acc_frag;

	if (grid_warp_id < total_c1_tiles) {
		const uint32_t tile_row = division<uint32_t, N_tiles>::div(grid_warp_id);
		const uint32_t tile_col = modulo<uint32_t, N_tiles>::mod(grid_warp_id);
		const uint32_t warp_row = tile_row * WMMA_TILE;
		const uint32_t warp_col = tile_col * WMMA_TILE;

		wmma::fill_fragment(acc_frag, __float2half(0.0f));

		if (warp_row < M && warp_col < N) {
			wmma::load_matrix_sync(acc_frag, bias_C1 + (warp_row * N + warp_col), N, wmma::mem_row_major);
		}

		for (uint32_t k_tile = 0; k_tile < K_tiles; ++k_tile) {
			const uint32_t k = k_tile * WMMA_TILE;
			wmma::load_matrix_sync(a_frag, input_A + (warp_row * K + k), K);
			wmma::load_matrix_sync(b_frag, input_B + (k * N + warp_col), N);
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}

		if (warp_row < M && warp_col < N) {
			wmma::store_matrix_sync(l2_c1 + (warp_row * N + warp_col), acc_frag, N, wmma::mem_row_major);
		}
	}

	cg::grid_group grid = cg::this_grid();
	grid.sync();

	if (grid_warp_id < total_d_tiles) {
		const uint32_t tile_row	  = division<uint32_t, P_tiles>::div(grid_warp_id);
		const uint32_t tile_col_p = modulo<uint32_t, P_tiles>::mod(grid_warp_id);
		const uint32_t warp_row	  = tile_row * WMMA_TILE;
		const uint32_t warp_col_p = tile_col_p * WMMA_TILE;

		wmma::fill_fragment(acc_frag, __float2half(0.0f));

		if (warp_row < M && warp_col_p < P) {
			wmma::load_matrix_sync(acc_frag, bias_D + (warp_row * P + warp_col_p), P, wmma::mem_row_major);
		}

		for (uint32_t n_tile = 0; n_tile < N_tiles; ++n_tile) {
			const uint32_t n = n_tile * WMMA_TILE;
			wmma::load_matrix_sync(a_frag, l2_c1 + (warp_row * N + n), N);
			wmma::load_matrix_sync(b_frag, input_E + (n * P + warp_col_p), P);
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}

		if (warp_row < M && warp_col_p < P) {
			wmma::store_matrix_sync(output_D + (warp_row * P + warp_col_p), acc_frag, P, wmma::mem_row_major);
		}
	}
}

template<uint32_t M, uint32_t K, uint32_t N, uint32_t P> BNCH_SWT_GLOBAL void nihilus_fused_striped_tiled(half* __restrict__ input_A, half* __restrict__ input_B,
	half* __restrict__ input_E, half* __restrict__ bias_C1, half* __restrict__ bias_D, half* __restrict__ output_D, void* __restrict__ l2_c1_storage) {
	static constexpr uint32_t WMMA_TILE = 16;
	static constexpr uint32_t WARP_SIZE = 32;

	// Calculate max rows that fit in L2 cache
	static constexpr uint32_t max_l2_bytes		  = bnch_swt::gpu_properties::max_persisting_l2_bytes;
	static constexpr uint32_t bytes_per_row		  = N * sizeof(half);
	static constexpr uint32_t max_rows_per_stripe = max_l2_bytes / bytes_per_row;

	// Round down to WMMA_TILE multiple, ensure at least one tile height
	static constexpr uint32_t rows_per_stripe = max_rows_per_stripe >= WMMA_TILE ? (max_rows_per_stripe / WMMA_TILE) * WMMA_TILE : WMMA_TILE;

	static constexpr uint32_t num_stripes = (M + rows_per_stripe - 1) / rows_per_stripe;

	// Tile counts for full dimensions
	static constexpr uint32_t N_tiles = (N + WMMA_TILE - 1) / WMMA_TILE;
	static constexpr uint32_t P_tiles = (P + WMMA_TILE - 1) / WMMA_TILE;
	static constexpr uint32_t K_tiles = (K + WMMA_TILE - 1) / WMMA_TILE;

	const uint32_t tid			= threadIdx.x;
	const uint32_t warp_id		= division<uint32_t, WARP_SIZE>::div(tid);
	const uint32_t block_warps	= division<uint32_t, WARP_SIZE>::div(blockDim.x);
	const uint32_t grid_warp_id = (blockIdx.x * block_warps) + warp_id;

	half* l2_c1 = reinterpret_cast<half*>(l2_c1_storage);

	wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, half> acc_frag;

	cg::grid_group grid = cg::this_grid();

	// Process each M-stripe sequentially
	for (uint32_t stripe = 0; stripe < num_stripes; ++stripe) {
		const uint32_t stripe_start_row = stripe * rows_per_stripe;
		const uint32_t stripe_end_row	= min(stripe_start_row + rows_per_stripe, M);
		const uint32_t stripe_height	= stripe_end_row - stripe_start_row;
		const uint32_t stripe_m_tiles	= (stripe_height + WMMA_TILE - 1) / WMMA_TILE;

		const uint32_t total_c1_tiles_in_stripe = stripe_m_tiles * N_tiles;

		// PHASE 1: Compute C1 stripe (A × B)
		if (grid_warp_id < total_c1_tiles_in_stripe) {
			const uint32_t tile_row_in_stripe = division<uint32_t, N_tiles>::div(grid_warp_id);
			const uint32_t tile_col			  = modulo<uint32_t, N_tiles>::mod(grid_warp_id);

			const uint32_t global_row = stripe_start_row + (tile_row_in_stripe * WMMA_TILE);
			const uint32_t warp_col	  = tile_col * WMMA_TILE;

			wmma::fill_fragment(acc_frag, __float2half(0.0f));

			if (global_row < M && warp_col < N) {
				wmma::load_matrix_sync(acc_frag, bias_C1 + (global_row * N + warp_col), N, wmma::mem_row_major);
			}

			for (uint32_t k_tile = 0; k_tile < K_tiles; ++k_tile) {
				const uint32_t k = k_tile * WMMA_TILE;
				wmma::load_matrix_sync(a_frag, input_A + (global_row * K + k), K);
				wmma::load_matrix_sync(b_frag, input_B + (k * N + warp_col), N);
				wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
			}

			if (global_row < M && warp_col < N) {
				// Store to L2 tile workspace (relative to stripe start)
				const uint32_t local_row = global_row - stripe_start_row;
				wmma::store_matrix_sync(l2_c1 + (local_row * N + warp_col), acc_frag, N, wmma::mem_row_major);
			}
		}

		// Sync: C1 stripe complete
		grid.sync();

		// PHASE 2: Compute D stripe (C1_stripe × E)
		const uint32_t total_d_tiles_in_stripe = stripe_m_tiles * P_tiles;

		if (grid_warp_id < total_d_tiles_in_stripe) {
			const uint32_t tile_row_in_stripe = division<uint32_t, P_tiles>::div(grid_warp_id);
			const uint32_t tile_col_p		  = modulo<uint32_t, P_tiles>::mod(grid_warp_id);

			const uint32_t global_row = stripe_start_row + (tile_row_in_stripe * WMMA_TILE);
			const uint32_t warp_col_p = tile_col_p * WMMA_TILE;

			wmma::fill_fragment(acc_frag, __float2half(0.0f));

			if (global_row < M && warp_col_p < P) {
				wmma::load_matrix_sync(acc_frag, bias_D + (global_row * P + warp_col_p), P, wmma::mem_row_major);
			}

			for (uint32_t n_tile = 0; n_tile < N_tiles; ++n_tile) {
				const uint32_t n = n_tile * WMMA_TILE;
				// Load from L2 tile workspace (relative to stripe start)
				const uint32_t local_row = global_row - stripe_start_row;
				wmma::load_matrix_sync(a_frag, l2_c1 + (local_row * N + n), N);
				wmma::load_matrix_sync(b_frag, input_E + (n * P + warp_col_p), P);
				wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
			}

			if (global_row < M && warp_col_p < P) {
				wmma::store_matrix_sync(output_D + (global_row * P + warp_col_p), acc_frag, P, wmma::mem_row_major);
			}
		}

		// Sync: D stripe complete, ready for next stripe
		grid.sync();
	}
}

template<typename value_type> struct min_max_vals;

template<typename value_type> struct min_max_vals {
	static constexpr uint32_t min{ 0 };
	static constexpr uint32_t max{ 255 };
};

template<typename value_type>
	requires(std::is_floating_point_v<std::remove_cvref_t<value_type>>)
struct min_max_vals<value_type> {
	static constexpr float min{ -1.0f };
	static constexpr float max{ 1.0f };
};

template<typename value_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new, uint32_t max_values>
struct cpu_baseline<value_type, kernel_types::mul_mat_add_fused, dim_00_new, dim_01_new, dim_02_new, dim_03_new, max_values> {
	BNCH_SWT_HOST static uint64_t impl(value_type* cpu_input_A, value_type* cpu_input_B, value_type* cpu_input_E, value_type* cpu_bias_C1, value_type* cpu_bias_D,
		value_type* cpu_output_D, uint32_t M, uint32_t K, uint32_t N, uint32_t P) {
		if (max_values == 0) {
			return (M * K + K * N + M * N + N * P + M * P + M * P) * sizeof(value_type);
		}
		std::vector<float> intermediate_C1(M * N);

		for (uint32_t m = 0; m < M; ++m) {
			for (uint32_t n = 0; n < N; ++n) {
				float accumulator = 0.0f;
				for (uint32_t k = 0; k < K; ++k) {
					accumulator += static_cast<float>(cpu_input_A[m * K + k]) * static_cast<float>(cpu_input_B[k * N + n]);
				}
				intermediate_C1[m * N + n] = accumulator + static_cast<float>(cpu_bias_C1[m * N + n]);
			}
		}

		uint32_t elements_computed = 0;
		for (uint32_t m = 0; m < M && elements_computed < max_values; ++m) {
			for (uint32_t p = 0; p < P && elements_computed < max_values; ++p) {
				float accumulator = 0.0f;
				for (uint32_t n = 0; n < N; ++n) {
					accumulator += intermediate_C1[m * N + n] * static_cast<float>(cpu_input_E[n * P + p]);
				}

				if constexpr (std::is_same_v<value_type, half>) {
					cpu_output_D[m * P + p] = __float2half(accumulator + static_cast<float>(cpu_bias_D[m * P + p]));
				} else {
					cpu_output_D[m * P + p] = static_cast<value_type>(accumulator + static_cast<float>(cpu_bias_D[m * P + p]));
				}
				++elements_computed;
			}
		}

		return (M * K + K * N + M * N + N * P + M * P + M * P) * sizeof(value_type);
	}
};

struct cutlass_dual_gemm_host_wrapper {
	static cudaStream_t stream_global;
	using Gemm = cutlass::gemm::device::Gemm<half, cutlass::layout::RowMajor, half, cutlass::layout::RowMajor, half, cutlass::layout::RowMajor, half>;

	template<typename GemmType, typename ArgsType1, typename ArgsType2> static void impl(half* A, half* B, half* E, half* bias_C1, half* bias_D, half* C1, half* D, uint32_t M,
		uint32_t K, uint32_t N, uint32_t P, GemmType gemm_op1, GemmType gemm_op2, ArgsType1 args1, ArgsType2 args2) {
		cutlass::Status status1 = gemm_op1(args1, nullptr, stream_global);
		cudaStreamSynchronize(stream_global);

		cutlass::Status status2 = gemm_op2(args2, nullptr, stream_global);
		cudaStreamSynchronize(stream_global);
	}
};

cudaStream_t cutlass_dual_gemm_host_wrapper::stream_global = nullptr;

template<typename value_type, kernel_types kernel_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new>
	requires(kernel_type == kernel_types::mul_mat_add_fused)
void test_function_fused() {
	static constexpr bnch_swt::string_literal stage_name{ bnch_swt::string_literal{ "fused-dual-gemm-" } + bnch_swt::internal::to_string_literal<dim_00_new>() + "-" +
		bnch_swt::internal::to_string_literal<dim_01_new>() + "-" + bnch_swt::internal::to_string_literal<dim_02_new>() + "-" +
		bnch_swt::internal::to_string_literal<dim_03_new>() };

	static constexpr uint32_t dim_00{ round_up_to_multiple<16>(dim_00_new) };
	static constexpr uint32_t dim_01{ round_up_to_multiple<16>(dim_01_new) };
	static constexpr uint32_t dim_02{ round_up_to_multiple<16>(dim_02_new) };
	static constexpr uint32_t dim_03{ round_up_to_multiple<16>(dim_03_new) };

	using benchmark = bnch_swt::benchmark_stage<stage_name, total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cuda>;

	cpu_buffer cpu_buffer_A{};
	cpu_buffer cpu_buffer_B{};
	cpu_buffer cpu_buffer_E{};
	cpu_buffer cpu_buffer_bias_C1{};
	cpu_buffer cpu_buffer_bias_D{};
	cpu_buffer cutlass_reference_D{};
	cpu_buffer nihilus_output_D{};
	cpu_buffer nihilus_tiled_output_D{};

	cuda_buffer cuda_buffer_A{};
	cuda_buffer cuda_buffer_B{};
	cuda_buffer cuda_buffer_E{};
	cuda_buffer cuda_buffer_bias_C1{};
	cuda_buffer cuda_buffer_bias_D{};
	cuda_buffer cuda_buffer_D{};
	cuda_buffer cuda_buffer_C1_intermediate{};

	static constexpr uint32_t M = dim_00;
	static constexpr uint32_t K = dim_01;
	static constexpr uint32_t N = dim_02;
	static constexpr uint32_t P = dim_03;

	uint32_t M_new = dim_00;
	uint32_t K_new = dim_01;
	uint32_t N_new = dim_02;
	uint32_t P_new = dim_03;

	const uint64_t size_A  = M * K * sizeof(value_type);
	const uint64_t size_B  = K * N * sizeof(value_type);
	const uint64_t size_E  = N * P * sizeof(value_type);
	const uint64_t size_C1 = M * N * sizeof(value_type);
	const uint64_t size_D  = M * P * sizeof(value_type);

	static constexpr uint64_t total_output_elements{ dim_00 * dim_03 };

	cpu_buffer_A.init(size_A);
	cpu_buffer_B.init(size_B);
	cpu_buffer_E.init(size_E);
	cpu_buffer_bias_C1.init(size_C1);
	cpu_buffer_bias_D.init(size_D);
	cutlass_reference_D.init(size_D);
	nihilus_output_D.init(size_D);
	nihilus_tiled_output_D.init(size_D);

	cuda_buffer_A.init(size_A);
	cuda_buffer_B.init(size_B);
	cuda_buffer_E.init(size_E);
	cuda_buffer_bias_C1.init(size_C1);
	cuda_buffer_bias_D.init(size_D);
	cuda_buffer_D.init(size_D);
	cuda_buffer_C1_intermediate.init(size_C1);

	auto* cpu_A		  = std::bit_cast<value_type*>(cpu_buffer_A.data());
	auto* cpu_B		  = std::bit_cast<value_type*>(cpu_buffer_B.data());
	auto* cpu_E		  = std::bit_cast<value_type*>(cpu_buffer_E.data());
	auto* cpu_bias_C1 = std::bit_cast<value_type*>(cpu_buffer_bias_C1.data());
	auto* cpu_bias_D  = std::bit_cast<value_type*>(cpu_buffer_bias_D.data());

	generate_values<value_type, M * K>(cuda_buffer_A.data<value_type>(), cpu_A, -0.01f, 0.01f);
	generate_values<value_type, K * N>(cuda_buffer_B.data<value_type>(), cpu_B, -0.01f, 0.01f);
	generate_values<value_type, N * P>(cuda_buffer_E.data<value_type>(), cpu_E, -0.01f, 0.01f);
	generate_values<value_type, M * N>(cuda_buffer_bias_C1.data<value_type>(), cpu_bias_C1, -0.01f, 0.01f);
	generate_values<value_type, M * P>(cuda_buffer_bias_D.data<value_type>(), cpu_bias_D, -0.01f, 0.01f);

	auto* A_ptr		  = cuda_buffer_A.data<get_value_type_t<value_type>>();
	auto* B_ptr		  = cuda_buffer_B.data<get_value_type_t<value_type>>();
	auto* E_ptr		  = cuda_buffer_E.data<get_value_type_t<value_type>>();
	auto* bias_C1_ptr = cuda_buffer_bias_C1.data<get_value_type_t<value_type>>();
	auto* bias_D_ptr  = cuda_buffer_bias_D.data<get_value_type_t<value_type>>();
	auto* D_ptr		  = cuda_buffer_D.data<get_value_type_t<value_type>>();

	std::cout << "\n=== FUSED DUAL GEMM CONFIG ===" << std::endl;
	std::cout << "First GEMM: " << M << " x " << K << " @ " << K << " x " << N << " = " << M << " x " << N << std::endl;
	std::cout << "Second GEMM: " << M << " x " << N << " @ " << N << " x " << P << " = " << M << " x " << P << std::endl;

	std::cout << "\n=== CUTLASS BASELINE (Two Separate GEMMs) ===" << std::endl;

	using RowMajor = cutlass::layout::RowMajor;
	using Gemm	   = cutlass::gemm::device::Gemm<half, RowMajor, half, RowMajor, half, RowMajor, half>;

	typename Gemm::EpilogueOutputOp::Params epilogue_params(1.0f, 1.0f);

	typename Gemm::Arguments args1({ static_cast<int>(M), static_cast<int>(N), static_cast<int>(K) }, { reinterpret_cast<half*>(A_ptr), static_cast<int>(K) },
		{ reinterpret_cast<half*>(B_ptr), static_cast<int>(N) }, { reinterpret_cast<half*>(bias_C1_ptr), static_cast<int>(N) },
		{ reinterpret_cast<half*>(cuda_buffer_C1_intermediate.data<half>()), static_cast<int>(N) }, epilogue_params);

	typename Gemm::Arguments args2({ static_cast<int>(M), static_cast<int>(P), static_cast<int>(N) },
		{ reinterpret_cast<half*>(cuda_buffer_C1_intermediate.data<half>()), static_cast<int>(N) }, { reinterpret_cast<half*>(E_ptr), static_cast<int>(P) },
		{ reinterpret_cast<half*>(bias_D_ptr), static_cast<int>(P) }, { reinterpret_cast<half*>(D_ptr), static_cast<int>(P) }, epilogue_params);

	Gemm gemm_op1;
	Gemm gemm_op2;

	cutlass::Status status1 = gemm_op1.initialize(args1);
	cutlass::Status status2 = gemm_op2.initialize(args2);

	if (status1 != cutlass::Status::kSuccess || status2 != cutlass::Status::kSuccess) {
		std::cout << "✗ CUTLASS initialization failed!" << std::endl;
		return;
	}

	status1 = gemm_op1();
	status2 = gemm_op2();

	if (status1 != cutlass::Status::kSuccess || status2 != cutlass::Status::kSuccess) {
		std::cout << "✗ CUTLASS execution failed!" << std::endl;
		return;
	}

	cudaDeviceSynchronize();

	if (!check_cuda_result()) {
		return;
	}

	cudaMemcpy(cutlass_reference_D.data(), D_ptr, size_D, cudaMemcpyDeviceToHost);

	if (!check_cuda_result()) {
		return;
	}

	std::cout << "✓ CUTLASS reference computed successfully!" << std::endl;

	static constexpr auto kernel_ptr	   = &test_function_fused_dual_matmul<M, K, N, P>;
	static constexpr auto kernel_ptr_tiled = &nihilus_fused_striped_tiled<M, K, N, P>;

	static constexpr dim3 block(256, 1, 1);
	int32_t threads_per_block = block.x * block.y * block.z;
	int32_t num_blocks_per_sm = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel_ptr, threads_per_block, 0);

	if (!check_cuda_result()) {
		return;
	}

	int32_t max_cooperative_blocks = num_blocks_per_sm * static_cast<uint32_t>(bnch_swt::gpu_properties::sm_count);

	dim3 grid(max_cooperative_blocks, 1, 1);

	std::cout << "Max cooperative blocks: " << max_cooperative_blocks << std::endl;
	std::cout << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
	std::cout << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;

	cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bnch_swt::gpu_properties::max_persisting_l2_bytes);

	if (!check_cuda_result()) {
		return;
	}

	uint64_t actual_data_size = M * N * sizeof(half);
	void* l2_buffer{};
	cudaMalloc(&l2_buffer, actual_data_size);

	cudaStream_t stream = 0;
	cudaStreamAttrValue stream_attribute;
	stream_attribute.accessPolicyWindow.base_ptr  = l2_buffer;
	stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
	stream_attribute.accessPolicyWindow.hitProp	  = cudaAccessPropertyPersisting;
	stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
	stream_attribute.accessPolicyWindow.num_bytes = bnch_swt::gpu_properties::max_persisting_l2_bytes;

	cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

	if (!check_cuda_result()) {
		return;
	}

	void* kernel_args[] = { &A_ptr, &B_ptr, &E_ptr, &bias_C1_ptr, &bias_D_ptr, &D_ptr, &l2_buffer };

	std::cout << "\n=== VALIDATING NIHILUS FUSED ===" << std::endl;

	cudaMemset(D_ptr, 0, size_D);
	cudaLaunchCooperativeKernel(kernel_ptr, grid, block, kernel_args, 0, stream);
	cudaDeviceSynchronize();

	if (!check_cuda_result()) {
		return;
	}

	cudaMemcpy(nihilus_output_D.data(), D_ptr, size_D, cudaMemcpyDeviceToHost);

	if (!check_cuda_result()) {
		return;
	}

	auto* cutlass_D		 = std::bit_cast<value_type*>(cutlass_reference_D.data());
	auto* nihilus_D		 = std::bit_cast<value_type*>(nihilus_output_D.data());
	uint64_t mismatches	 = 0;
	uint64_t nan_count	 = 0;
	uint64_t inf_count	 = 0;
	value_type max_error = 0;

	static constexpr uint64_t values_to_check{ std::min(total_output_elements, 100000ULL) };

	for (uint64_t i = 0; i < values_to_check; ++i) {
		float cutlass_val = static_cast<float>(cutlass_D[i]);
		float nihilus_val = static_cast<float>(nihilus_D[i]);

		if (std::isnan(nihilus_val) || std::isinf(nihilus_val)) {
			if (std::isnan(nihilus_val)) {
				++nan_count;
				if (nan_count <= 5) {
					std::cout << "NaN at index " << i << " (CUTLASS=" << cutlass_val << ")" << std::endl;
				}
			}
			if (std::isinf(nihilus_val)) {
				++inf_count;
				if (inf_count <= 5) {
					std::cout << "Inf at index " << i << " (CUTLASS=" << cutlass_val << ", Nihilus=" << nihilus_val << ")" << std::endl;
				}
			}
			++mismatches;
			continue;
		}

		value_type diff = std::abs(cutlass_val - nihilus_val);
		if (!std::isnan(diff) && !std::isinf(diff)) {
			max_error = std::max(max_error, diff);
		}

		const float relative_tol = 1e-1f;
		const float absolute_tol = 1e-1f;
		value_type tolerance	 = std::max(relative_tol * std::abs(cutlass_val), absolute_tol);
		if (diff > tolerance) {
			if (mismatches < 10) {
				std::cout << "Mismatch at index " << i << ": CUTLASS=" << cutlass_val << " Nihilus=" << nihilus_val << " diff=" << static_cast<float>(diff) << std::endl;
			}
			++mismatches;
		}
	}

	std::cout << "Max absolute error: " << static_cast<float>(max_error) << std::endl;
	std::cout << "NaN count: " << nan_count << std::endl;
	std::cout << "Inf count: " << inf_count << std::endl;

	if (mismatches == 0) {
		std::cout << "✓ NIHILUS FUSED VALIDATION PASSED! All " << values_to_check << " elements match!" << std::endl;
	} else {
		std::cout << "✗ NIHILUS FUSED VALIDATION FAILED! " << mismatches << " mismatches out of " << values_to_check << " elements" << std::endl;
	}

	std::cout << "\n=== VALIDATING NIHILUS FUSED TILED ===" << std::endl;

	cudaMemset(D_ptr, 0, size_D);
	cudaLaunchCooperativeKernel(kernel_ptr_tiled, grid, block, kernel_args, 0, stream);
	cudaDeviceSynchronize();

	if (!check_cuda_result()) {
		return;
	}

	cudaMemcpy(nihilus_tiled_output_D.data(), D_ptr, size_D, cudaMemcpyDeviceToHost);

	if (!check_cuda_result()) {
		return;
	}

	auto* nihilus_tiled_D = std::bit_cast<value_type*>(nihilus_tiled_output_D.data());
	mismatches			  = 0;
	nan_count			  = 0;
	inf_count			  = 0;
	max_error			  = 0;

	for (uint64_t i = 0; i < values_to_check; ++i) {
		float cutlass_val = static_cast<float>(cutlass_D[i]);
		float nihilus_val = static_cast<float>(nihilus_tiled_D[i]);

		if (std::isnan(nihilus_val) || std::isinf(nihilus_val)) {
			if (std::isnan(nihilus_val)) {
				++nan_count;
				if (nan_count <= 5) {
					std::cout << "NaN at index " << i << " (CUTLASS=" << cutlass_val << ")" << std::endl;
				}
			}
			if (std::isinf(nihilus_val)) {
				++inf_count;
				if (inf_count <= 5) {
					std::cout << "Inf at index " << i << " (CUTLASS=" << cutlass_val << ", Nihilus=" << nihilus_val << ")" << std::endl;
				}
			}
			++mismatches;
			continue;
		}

		value_type diff = std::abs(cutlass_val - nihilus_val);
		if (!std::isnan(diff) && !std::isinf(diff)) {
			max_error = std::max(max_error, diff);
		}

		const float relative_tol = 1e-1f;
		const float absolute_tol = 1e-1f;
		value_type tolerance	 = std::max(relative_tol * std::abs(cutlass_val), absolute_tol);
		if (diff > tolerance) {
			if (mismatches < 10) {
				std::cout << "Mismatch at index " << i << ": CUTLASS=" << cutlass_val << " Nihilus=" << nihilus_val << " diff=" << static_cast<float>(diff) << std::endl;
			}
			++mismatches;
		}
	}

	std::cout << "Max absolute error: " << static_cast<float>(max_error) << std::endl;
	std::cout << "NaN count: " << nan_count << std::endl;
	std::cout << "Inf count: " << inf_count << std::endl;

	if (mismatches == 0) {
		std::cout << "✓ NIHILUS FUSED TILED VALIDATION PASSED! All " << values_to_check << " elements match!" << std::endl;
	} else {
		std::cout << "✗ NIHILUS FUSED TILED VALIDATION FAILED! " << mismatches << " mismatches out of " << values_to_check << " elements" << std::endl;
	}

	const uint64_t flops_gemm1 = 2ULL * M * N * K + (M * N);
	const uint64_t flops_gemm2 = 2ULL * M * P * N + (M * P);
	const uint64_t total_flops = flops_gemm1 + flops_gemm2;

	const uint64_t bytes_fused	  = size_A + size_B + size_E + size_C1 + size_D + size_D;
	const uint64_t bytes_separate = size_A + size_B + size_C1 + size_C1 + size_E + size_D + size_D;

	std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
	std::cout << "Total FLOPs: " << total_flops << std::endl;
	std::cout << "Fused bytes: " << bytes_fused << " (saves " << (bytes_separate - bytes_fused) << " bytes vs separate!)" << std::endl;
	std::cout << "Arithmetic Intensity (fused): " << (static_cast<double>(total_flops) / bytes_fused) << " FLOP/byte" << std::endl;
	std::cout << "Arithmetic Intensity (separate): " << (static_cast<double>(total_flops) / bytes_separate) << " FLOP/byte" << std::endl;

	cutlass_dual_gemm_host_wrapper::stream_global = stream;

	benchmark::template run_benchmark_cooperative<"fused_dual_gemm", kernel_ptr>(grid, block, 0, stream, bytes_fused, kernel_args);

	benchmark::template run_benchmark_cooperative<"fused_dual_gemm_tiled", kernel_ptr_tiled>(grid, block, 0, stream, bytes_fused, kernel_args);

	benchmark::template run_from_host<"cutlass_dual_gemm_separate", cutlass_dual_gemm_host_wrapper>(static_cast<uint64_t>(bytes_separate), reinterpret_cast<half*>(A_ptr),
		reinterpret_cast<half*>(B_ptr), reinterpret_cast<half*>(E_ptr), reinterpret_cast<half*>(bias_C1_ptr), reinterpret_cast<half*>(bias_D_ptr),
		reinterpret_cast<half*>(cuda_buffer_C1_intermediate.data<half>()), reinterpret_cast<half*>(D_ptr), M, K, N, P, gemm_op1, gemm_op2, args1, args2);

	benchmark::print_results(true, false);

	cudaFree(l2_buffer);
}

template<uint32_t... Dims> struct test_runner;

template<uint32_t M, uint32_t K, uint32_t N> struct test_runner<M, K, N> {
	static void run() {
		std::cout << "\n========================================" << std::endl;
		std::cout << "Testing: M=" << M << " K=" << K << " N=" << N << " P=16" << std::endl;
		std::cout << "========================================" << std::endl;
		test_function_fused<half, kernel_types::mul_mat_add_fused, M, K, N, 1>();
	}
};

template<uint32_t M, uint32_t... Rest> struct dimension_sweep;

template<uint32_t M, uint32_t K> struct dimension_sweep<M, K> {
	static void run() {
		test_runner<M, K, 8192>::run();
	}
};

template<uint32_t M> struct dimension_sweep<M> {
	static void run() {
		dimension_sweep<M, 128>::run();
		dimension_sweep<M, 256>::run();
		dimension_sweep<M, 512>::run();
		dimension_sweep<M, 1024>::run();
		dimension_sweep<M, 2048>::run();
		dimension_sweep<M, 4096>::run();
		dimension_sweep<M, 8192>::run();
		dimension_sweep<M, 16384>::run();
	}
};

struct full_sweep {
	static void run() {
		dimension_sweep<128>::run();
		dimension_sweep<256>::run();
		dimension_sweep<512>::run();
		dimension_sweep<1024>::run();
		dimension_sweep<2048>::run();
		dimension_sweep<4096>::run();
		dimension_sweep<8192>::run();
		dimension_sweep<16384>::run();
	}
};

int main() {
	std::cout << "Starting EXHAUSTIVE fused dual GEMM regime discovery..." << std::endl;
	std::cout << "Testing all permutations of {128, 256, 256, 1024, 2048, 4096, 8192}³" << std::endl;
	std::cout << "Total test cases: 7³ = 343 configurations" << std::endl;
	std::cout << "P dimension fixed at 16 (rounded from 1)" << std::endl;

	full_sweep::run();

	std::cout << "\n========================================" << std::endl;
	std::cout << "REGIME DISCOVERY COMPLETE!" << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}