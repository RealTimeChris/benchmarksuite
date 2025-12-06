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

template<typename value_type, kernel_types kernel_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new>
	requires(kernel_type == kernel_types::mul_mat_add)
void test_function() {
	static constexpr bnch_swt::string_literal stage_name{ bnch_swt::string_literal{ "kernel-gegen-kernel-" } +
		bnch_swt::internal::string_literal_from_view<kernel_types_names[static_cast<uint64_t>(kernel_type)].size()>(kernel_types_names[static_cast<uint64_t>(kernel_type)]) + "-" +
		bnch_swt::internal::to_string_literal<dim_00_new>() + "-" + bnch_swt::internal::to_string_literal<dim_01_new>() + "-" +
		bnch_swt::internal::to_string_literal<dim_02_new>() + "-" + bnch_swt::internal::to_string_literal<dim_03_new>() };
	static constexpr uint32_t dim_00{ round_up_to_multiple<16>(dim_00_new) };
	static constexpr uint32_t dim_01{ round_up_to_multiple<16>(dim_01_new) };
	static constexpr uint32_t dim_02{ round_up_to_multiple<16>(dim_02_new) };
	static constexpr uint32_t dim_03{ round_up_to_multiple<16>(dim_03_new) };

	using benchmark		 = bnch_swt::benchmark_stage<stage_name, total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cuda>;
	using test_benchmark = bnch_swt::benchmark_stage<stage_name, 1, 1, bnch_swt::benchmark_types::cpu>;

	cpu_buffer cpu_buffer_A{};
	cpu_buffer cpu_buffer_B{};
	cpu_buffer cpu_buffer_C{};
	cpu_buffer cpu_reference_D{};
	cpu_buffer gpu_output_D{};

	cuda_buffer cuda_buffer_A{};
	cuda_buffer cuda_buffer_B{};
	cuda_buffer cuda_buffer_C{};
	cuda_buffer cuda_buffer_D{};

	const uint64_t size_A = dim_00 * dim_01 * sizeof(value_type);
	const uint64_t size_B = dim_01 * dim_02 * sizeof(value_type);
	const uint64_t size_C = dim_00 * dim_02 * sizeof(value_type);
	const uint64_t size_D = dim_00 * dim_02 * sizeof(value_type);
	static constexpr uint64_t total_output_elements{ dim_00 * dim_02 };

	cpu_buffer_A.init(size_A);
	cpu_buffer_B.init(size_B);
	cpu_buffer_C.init(size_C);
	cpu_reference_D.init(size_D);
	gpu_output_D.init(size_D);
	cuda_buffer_A.init(size_A);
	cuda_buffer_B.init(size_B);
	cuda_buffer_C.init(size_C);
	cuda_buffer_D.init(size_D);

	auto* cpu_A		= std::bit_cast<value_type*>(cpu_buffer_A.data());
	auto* cpu_B		= std::bit_cast<value_type*>(cpu_buffer_B.data());
	auto* cpu_C		= std::bit_cast<value_type*>(cpu_buffer_C.data());
	auto* cpu_D_ref = std::bit_cast<value_type*>(cpu_reference_D.data());

	generate_values<value_type, dim_00 * dim_01>(cuda_buffer_A.data<value_type>(), cpu_A, min_max_vals<value_type>::min, min_max_vals<value_type>::max);
	generate_values<value_type, dim_01 * dim_02>(cuda_buffer_B.data<value_type>(), cpu_B, min_max_vals<value_type>::min, min_max_vals<value_type>::max);
	generate_values<value_type, dim_00 * dim_02>(cuda_buffer_C.data<value_type>(), cpu_C, min_max_vals<value_type>::min, min_max_vals<value_type>::max);

	static constexpr uint64_t values_to_check{ total_output_elements / 1024 };

	std::cout << "Running CPU baseline (D = A @ B + C)..." << std::endl;
	test_benchmark::template run_benchmark<"cpu_baseline", cpu_baseline<value_type, kernel_type, dim_00, dim_02, dim_01, dim_03, values_to_check>>(cpu_A, cpu_B, cpu_C, cpu_D_ref,
		dim_00, dim_01, dim_02);
	test_benchmark::print_results();

	using traits = cuda_kernel_traits_impl<value_type, kernel_types::mul_mat_add>;
	static constexpr dim3 block(32, 1, 1);

	static constexpr auto kernel_ptr = &test_function_matmul<get_value_type_t<value_type>, dim_00, dim_01, 1, 1>;

	int32_t threads_per_block = block.x * block.y * block.z;
	int32_t num_blocks_per_sm = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel_ptr, threads_per_block, 0);
	if (!check_cuda_result()) {
		return;
	}

	int32_t max_cooperative_blocks = num_blocks_per_sm * static_cast<uint32_t>(bnch_swt::gpu_properties::sm_count);

	uint32_t M							  = dim_00;
	uint32_t N							  = dim_02;
	uint32_t K							  = dim_01;
	static constexpr uint32_t tiles_x	  = (dim_02 + traits::BN - 1) / traits::BN;
	static constexpr uint32_t tiles_y	  = (dim_00 + traits::BM - 1) / traits::BM;
	static constexpr uint32_t total_tiles = tiles_x * tiles_y;
	dim3 grid(max_cooperative_blocks, 1, 1);

	std::cout << "Total tiles needed: " << total_tiles << std::endl;
	std::cout << "Max cooperative blocks: " << max_cooperative_blocks << std::endl;
	std::cout << "Tiles per block: " << ((total_tiles + max_cooperative_blocks - 1) / max_cooperative_blocks) << std::endl;
	std::cout << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
	std::cout << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;

	cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bnch_swt::gpu_properties::max_persisting_l2_bytes);

	if (!check_cuda_result()) {
		return;
	}

	void* intermediate_buffer{};
	cudaMalloc(&intermediate_buffer, bnch_swt::gpu_properties::max_persisting_l2_bytes);
	cudaStream_t stream = 0;
	cudaStreamAttrValue stream_attribute;
	stream_attribute.accessPolicyWindow.base_ptr  = intermediate_buffer;
	stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
	stream_attribute.accessPolicyWindow.hitProp	  = cudaAccessPropertyPersisting;
	stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
	stream_attribute.accessPolicyWindow.num_bytes = bnch_swt::gpu_properties::max_persisting_l2_bytes;

	cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

	if (!check_cuda_result()) {
		return;
	}

	auto* A_ptr = cuda_buffer_A.data<get_value_type_t<value_type>>();
	auto* B_ptr = cuda_buffer_B.data<get_value_type_t<value_type>>();
	auto* C_ptr = cuda_buffer_C.data<get_value_type_t<value_type>>();
	auto* D_ptr = cuda_buffer_D.data<get_value_type_t<value_type>>();

	void* kernel_args[] = { &A_ptr, &B_ptr, &C_ptr, &D_ptr, &M, &K, &N, &intermediate_buffer };
	cudaLaunchCooperativeKernel(kernel_ptr, grid, block, kernel_args, 0, stream);

	cudaDeviceSynchronize();

	if (!check_cuda_result()) {
		return;
	}

	cudaMemcpy(gpu_output_D.data(), cuda_buffer_D.data<get_value_type_t<value_type>>(), size_D, cudaMemcpyDeviceToHost);

	if (!check_cuda_result()) {
		return;
	}

	auto* gpu_D			 = std::bit_cast<value_type*>(gpu_output_D.data());
	uint64_t mismatches	 = 0;
	value_type max_error = 0;

	for (uint64_t i = 0; i < values_to_check; ++i) {
		value_type diff = std::abs(static_cast<float>(cpu_D_ref[i]) - static_cast<float>(gpu_D[i]));
		max_error		= std::max(max_error, diff);

		const float relative_tol = 1e-2f;
		const float absolute_tol = 1e-2f;
		value_type tolerance	 = std::max(relative_tol * std::abs(static_cast<float>(cpu_D_ref[i])), absolute_tol);
		if (diff > tolerance) {
			if (mismatches < 10) {
				std::cout << "Mismatch at index " << i << ": CPU=" << static_cast<float>(cpu_D_ref[i]) << " GPU=" << static_cast<float>(gpu_D[i])
						  << " diff=" << static_cast<float>(diff) << std::endl;
			}
			++mismatches;
		}
	}

	std::cout << "Max absolute error: " << static_cast<float>(max_error) << std::endl;

	if (mismatches == 0) {
		std::cout << "✓ VALIDATION PASSED! All " << values_to_check << " elements match!" << std::endl;
	} else {
		std::cout << "✗ VALIDATION FAILED! " << mismatches << " mismatches out of " << values_to_check << " elements" << std::endl;
	}

	const uint64_t flops = 2ULL * dim_00 * dim_02 * dim_01 + (dim_00 * dim_02);
	const uint64_t bytes = (size_A + size_B + size_C + size_D);

	std::cout << "\nFLOPs: " << flops << std::endl;
	std::cout << "Bytes: " << bytes << std::endl;
	std::cout << "Arithmetic Intensity: " << (static_cast<double>(flops) / bytes) << " FLOP/byte" << std::endl;
	cutlass_gemm_host_wrapper::stream_global = stream;

	benchmark::template run_benchmark_cooperative<"tiled_matmul", kernel_ptr>(grid, block, 0, stream, bytes, kernel_args);
	std::cout << "\nRunning CUTLASS Comparison..." << std::endl;

	using RowMajor = cutlass::layout::RowMajor;
	using Gemm	   = cutlass::gemm::device::Gemm<half, RowMajor, half, RowMajor, half, RowMajor, half>;

	std::cout << "\nRunning CUTLASS Comparison..." << std::endl;
	typename Gemm::EpilogueOutputOp::Params epilogue_params(1.0f, 1.0f);
	typename Gemm::Arguments args({ static_cast<int>(M), static_cast<int>(N), static_cast<int>(K) }, { reinterpret_cast<half*>(A_ptr), static_cast<int>(K) },
		{ reinterpret_cast<half*>(B_ptr), static_cast<int>(N) }, { reinterpret_cast<half*>(C_ptr), static_cast<int>(N) }, { reinterpret_cast<half*>(D_ptr), static_cast<int>(N) },
		epilogue_params);

	Gemm gemm_op;

	benchmark::template run_from_host<"cutlass_tensor_core", cutlass_gemm_host_wrapper>(static_cast<uint64_t>(bytes), reinterpret_cast<half*>(A_ptr),
		reinterpret_cast<half*>(B_ptr), reinterpret_cast<half*>(C_ptr), reinterpret_cast<half*>(D_ptr), static_cast<uint32_t>(M), static_cast<uint32_t>(K),
		static_cast<uint32_t>(N), gemm_op, args);

	benchmark::print_results();
}

template<typename value_type, kernel_types kernel_type, uint32_t dim_00_new, uint32_t dim_01_new, uint32_t dim_02_new, uint32_t dim_03_new> void test_function() {
	static constexpr bnch_swt::string_literal stage_name{ bnch_swt::string_literal{ "kernel-gegen-kernel-" } +
		bnch_swt::internal::string_literal_from_view<kernel_types_names[static_cast<uint64_t>(kernel_type)].size()>(kernel_types_names[static_cast<uint64_t>(kernel_type)]) + "-" +
		bnch_swt::internal::to_string_literal<dim_00_new>() + "-" + bnch_swt::internal::to_string_literal<dim_01_new>() + "-" +
		bnch_swt::internal::to_string_literal<dim_02_new>() + "-" + bnch_swt::internal::to_string_literal<dim_03_new>() };
	static constexpr uint64_t total_vector_elements{ dim_00_new * dim_01_new * dim_02_new * dim_03_new };
	static constexpr auto test_function_ptr			 = &test_function<get_value_type_t<value_type>, dim_00_new, dim_01_new, dim_02_new, dim_03_new, kernel_type>;
	static constexpr auto test_function_branched_ptr = &test_function_branched<get_value_type_t<value_type>, dim_00_new, dim_01_new, dim_02_new, dim_03_new>;
	static constexpr uint32_t dim_00{ round_up_to_multiple<4>(dim_00_new) };
	static constexpr uint32_t dim_01{ round_up_to_multiple<4>(dim_01_new) };
	static constexpr uint32_t dim_02{ round_up_to_multiple<4>(dim_02_new) };
	static constexpr uint32_t dim_03{ round_up_to_multiple<4>(dim_03_new) };
	using benchmark		 = bnch_swt::benchmark_stage<stage_name, total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cuda>;
	using test_benchmark = bnch_swt::benchmark_stage<stage_name, total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cpu>;
	cpu_buffer cpu_buffer_val{};
	cpu_buffer finished_cpu_buffer_val{};
	cpu_buffer cpu_reference_buffer{};

	finished_cpu_buffer_val.init(
		byte_count<value_type, kernel_type, cuda_kernel_traits<get_value_type<value_type>, kernel_type>::total_executions, dim_00, dim_01, dim_02, dim_03>);
	cpu_buffer_val.init(byte_count<value_type, kernel_type, cuda_kernel_traits<get_value_type<value_type>, kernel_type>::total_executions, dim_00, dim_01, dim_02, dim_03>);
	cpu_reference_buffer.init(byte_count<value_type, kernel_type, cuda_kernel_traits<get_value_type<value_type>, kernel_type>::total_executions, dim_00, dim_01, dim_02, dim_03>);

	auto tensors = generate_cuda_data<value_type, kernel_type, cuda_kernel_traits<get_value_type<value_type>, kernel_type>::total_executions, dim_00, dim_01, dim_02, dim_03>(
		buffer<value_type, kernel_type, cuda_kernel_traits<get_value_type<value_type>, kernel_type>::total_executions, dim_00, dim_01, dim_02, dim_03>, cpu_buffer_val);

	auto* cpu_input_01 = std::bit_cast<value_type*>(cpu_buffer_val.data() + tensors.input_01.offset);
	auto* cpu_input_02 = std::bit_cast<value_type*>(cpu_buffer_val.data() + tensors.input_02.offset);
	auto* cpu_output   = std::bit_cast<value_type*>(cpu_reference_buffer.data() + tensors.output.offset);

	test_benchmark::template run_benchmark<"cpu_baseline", cpu_baseline<value_type, kernel_type, dim_00, dim_01, dim_02, dim_03, 1024 * 1024>>(cpu_input_01, cpu_input_02,
		cpu_output, tensors.output.element_count);
	test_benchmark::print_results();

	static constexpr uint64_t total_uint4s = (total_vector_elements + 3) / 4;
	static constexpr uint64_t total_chunks = (total_uint4s + 6 - 1) / 6;
	dim3 block{ 1024, 1, 1 };
	dim3 grid{ (total_chunks + 1023) / 1024, 1, 1 };

	test_function_ptr<<<grid, block>>>(std::bit_cast<get_value_type_t<value_type>*>(tensors.input_01.data), std::bit_cast<get_value_type_t<value_type>*>(tensors.input_02.data),
		std::bit_cast<get_value_type_t<value_type>*>(tensors.output.data));
	cudaDeviceSynchronize();

	if (auto error = cudaGetLastError(); error) {
		std::cout << "Error: " << cudaGetErrorString(error) << std::endl;
	}

	if (auto result = cudaMemcpy(finished_cpu_buffer_val.data() + tensors.output.offset, tensors.output.data, tensors.output.element_count * sizeof(value_type),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		result) {
		std::cout << "cudaMemcpy Error: " << cudaGetErrorString(result) << std::endl;
	}

	auto* gpu_output	= std::bit_cast<value_type*>(finished_cpu_buffer_val.data() + tensors.output.offset);
	uint64_t mismatches = 0;
	for (uint64_t i = 0; i < tensors.output.element_count; ++i) {
		if (cpu_output[i] != gpu_output[i]) {
			if (mismatches < 10) {
				std::cout << "Mismatch at index " << i << ": CPU=" << cpu_output[i] << " GPU=" << gpu_output[i] << std::endl;
			}
			++mismatches;
		}
	}

	if (mismatches == 0) {
		std::cout << "✓ VALIDATION PASSED! All " << tensors.output.element_count << " elements match!" << std::endl;
	} else {
		std::cout << "✗ VALIDATION FAILED! " << mismatches << " mismatches out of " << tensors.output.element_count << " elements" << std::endl;
	}
	uint64_t bytes_transferred{ dim_00 * dim_00 * sizeof(value_type) * 3 };

	benchmark::template run_benchmark<"branched", test_function_branched_ptr>(grid, block, 0, bytes_transferred,
		std::bit_cast<get_value_type_t<value_type>*>(tensors.input_01.data), std::bit_cast<get_value_type_t<value_type>*>(tensors.input_02.data),
		std::bit_cast<get_value_type_t<value_type>*>(tensors.output.data));

	benchmark::template run_benchmark<"index_unrolled", test_function_ptr>(grid, block, 0, bytes_transferred, std::bit_cast<get_value_type_t<value_type>*>(tensors.input_01.data),
		std::bit_cast<get_value_type_t<value_type>*>(tensors.input_02.data), std::bit_cast<get_value_type_t<value_type>*>(tensors.output.data));

	benchmark::print_results();
}

int main() {
	test_function<float, kernel_types::mul_mat_add, 8192, 1024, 1, 1>();
	test_function<uint32_t, kernel_types::add, model_traits::embedding_length, model_traits::embedding_length, 1ULL, 1ULL>();
	return 0;
}