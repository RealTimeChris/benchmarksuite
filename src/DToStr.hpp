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
/// https://github.com/RealTimeChris/jsonifier
/// Nov 13, 2023
#pragma once

#include "DragonBox.hpp"

#include <concepts>
#include <cstdint>
#include <cstring>
#include <array>

namespace concepts {

	template<typename value_type>
	concept signed_t = std::signed_integral<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept unsigned_t = std::unsigned_integral<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept uns8_t = sizeof(std::remove_cvref_t<value_type>) == 1 && unsigned_t<value_type>;

	template<typename value_type>
	concept uns16_t = sizeof(std::remove_cvref_t<value_type>) == 2 && unsigned_t<value_type>;

	template<typename value_type>
	concept uns32_t = sizeof(std::remove_cvref_t<value_type>) == 4 && unsigned_t<value_type>;

	template<typename value_type>
	concept uns64_t = sizeof(std::remove_cvref_t<value_type>) == 8 && unsigned_t<value_type>;

	template<typename value_type>
	concept sig8_t = sizeof(std::remove_cvref_t<value_type>) == 1 && signed_t<value_type>;

	template<typename value_type>
	concept sig16_t = sizeof(std::remove_cvref_t<value_type>) == 2 && signed_t<value_type>;

	template<typename value_type>
	concept sig32_t = sizeof(std::remove_cvref_t<value_type>) == 4 && signed_t<value_type>;

	template<typename value_type>
	concept sig64_t = sizeof(std::remove_cvref_t<value_type>) == 8 && signed_t<value_type>;

}

namespace bnch_swt {
	
	template<typename value_type> struct uint_pair {
		value_type multiplicand;
		value_type shift;
	};

	template<typename value_type, value_type divisor_new> struct uint_type;

	template<concepts::uns64_t value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
		value_type lo{};
		value_type hi{};

		constexpr uint_type() {
		}

		constexpr uint_type(value_type l) : lo{ l } {
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
			value_type new_lo = lo + other.lo;
			value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
			return uint_type{ new_hi, new_lo };
		}

		friend constexpr uint_type operator+(value_type lhs, const uint_type& other) {
			return other + lhs;
		}

		constexpr uint_type operator-(const uint_type& other) const {
			value_type new_lo = lo - other.lo;
			value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
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
			uint_type divisor = other;

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
			uint_pair<value_type> return_value{};
			uint_type div_temp{ divisor_new };
			uint_type div_minus_1{ divisor_new - 1 };
			value_type l			  = 127 - div_minus_1.lzcnt();
			uint_type numerator		  = uint_type{ 1 } << (64 + static_cast<int32_t>(l));
			uint_type m_128			  = (numerator + div_temp - 1) / div_temp;
			return_value.multiplicand = static_cast<value_type>(m_128);
			return_value.shift		  = 64 + l;
			return return_value;
		}
	};

	template<concepts::uns32_t value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
		uint64_t value{};

		constexpr uint_type() {
		}

		constexpr uint_type(uint64_t v) : value{ v } {
		}

		constexpr explicit operator value_type() const {
			return static_cast<value_type>(value);
		}

		constexpr bool operator==(const uint_type& other) const {
			return value == other.value;
		}

		constexpr bool operator!=(const uint_type& other) const {
			return !(*this == other);
		}

		constexpr bool operator<(const uint_type& other) const {
			return value < other.value;
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
			return uint_type{ ~value };
		}

		constexpr uint_type operator+(const uint_type& other) const {
			return uint_type{ value + other.value };
		}

		friend constexpr uint_type operator+(uint64_t lhs, const uint_type& other) {
			return other + lhs;
		}

		constexpr uint_type operator-(const uint_type& other) const {
			return uint_type{ value - other.value };
		}

		constexpr uint_type operator<<(int32_t shift) const {
			if (shift == 0) {
				return *this;
			}
			if (shift >= 64) {
				return uint_type{ 0 };
			}
			return uint_type{ value << shift };
		}

		constexpr uint_type operator>>(int32_t shift) const {
			if (shift == 0) {
				return *this;
			}
			if (shift >= 64) {
				return uint_type{ 0 };
			}
			return uint_type{ value >> shift };
		}

		constexpr uint_type operator*(const uint_type& other) const {
			return uint_type{ value * other.value };
		}

		constexpr uint_type operator/(const uint_type& other) const {
			if (other.value == 0) {
				return uint_type{ 0 };
			}
			if (other > *this) {
				return uint_type{ 0 };
			}
			if (other == *this) {
				return uint_type{ 1 };
			}
			uint64_t quotient  = 0;
			uint64_t remainder = 0;
			uint64_t divisor   = other.value;

			for (int32_t i = 63; i >= 0; --i) {
				remainder = remainder << 1;

				if (value & (1ULL << i)) {
					remainder |= 1;
				}

				if (remainder >= divisor) {
					remainder = remainder - divisor;
					quotient |= (1ULL << i);
				}
			}
			return uint_type{ quotient };
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

		consteval static uint_pair<value_type> collect_values() {
			uint_pair<value_type> return_value{};
			uint_type div_temp{ divisor_new };
			uint_type div_minus_1{ divisor_new - 1 };
			value_type l			  = 63 - div_minus_1.lzcnt();
			uint_type numerator		  = uint_type{ 1 } << (32 + static_cast<int32_t>(l));
			uint_type m_128			  = (numerator + div_temp - 1) / div_temp;
			return_value.multiplicand = static_cast<value_type>(m_128);
			return_value.shift		  = 32 + l;
			return return_value;
		}
	};

	template<typename typeName> struct fiwb {
		BNCH_SWT_ALIGN(64ULL)
		inline static constexpr char charTable01[]{ 0x30, 0x30, 0x30, 0x31, 0x30, 0x32, 0x30, 0x33, 0x30, 0x34, 0x30, 0x35, 0x30, 0x36, 0x30, 0x37, 0x30, 0x38, 0x30, 0x39, 0x31,
			0x30, 0x31, 0x31, 0x31, 0x32, 0x31, 0x33, 0x31, 0x34, 0x31, 0x35, 0x31, 0x36, 0x31, 0x37, 0x31, 0x38, 0x31, 0x39, 0x32, 0x30, 0x32, 0x31, 0x32, 0x32, 0x32, 0x33, 0x32,
			0x34, 0x32, 0x35, 0x32, 0x36, 0x32, 0x37, 0x32, 0x38, 0x32, 0x39, 0x33, 0x30, 0x33, 0x31, 0x33, 0x32, 0x33, 0x33, 0x33, 0x34, 0x33, 0x35, 0x33, 0x36, 0x33, 0x37, 0x33,
			0x38, 0x33, 0x39, 0x34, 0x30, 0x34, 0x31, 0x34, 0x32, 0x34, 0x33, 0x34, 0x34, 0x34, 0x35, 0x34, 0x36, 0x34, 0x37, 0x34, 0x38, 0x34, 0x39, 0x35, 0x30, 0x35, 0x31, 0x35,
			0x32, 0x35, 0x33, 0x35, 0x34, 0x35, 0x35, 0x35, 0x36, 0x35, 0x37, 0x35, 0x38, 0x35, 0x39, 0x36, 0x30, 0x36, 0x31, 0x36, 0x32, 0x36, 0x33, 0x36, 0x34, 0x36, 0x35, 0x36,
			0x36, 0x36, 0x37, 0x36, 0x38, 0x36, 0x39, 0x37, 0x30, 0x37, 0x31, 0x37, 0x32, 0x37, 0x33, 0x37, 0x34, 0x37, 0x35, 0x37, 0x36, 0x37, 0x37, 0x37, 0x38, 0x37, 0x39, 0x38,
			0x30, 0x38, 0x31, 0x38, 0x32, 0x38, 0x33, 0x38, 0x34, 0x38, 0x35, 0x38, 0x36, 0x38, 0x37, 0x38, 0x38, 0x38, 0x39, 0x39, 0x30, 0x39, 0x31, 0x39, 0x32, 0x39, 0x33, 0x39,
			0x34, 0x39, 0x35, 0x39, 0x36, 0x39, 0x37, 0x39, 0x38, 0x39, 0x39 };
		BNCH_SWT_ALIGN(64ULL)
		inline static constexpr uint16_t charTable02[]{ 0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930, 0x3031, 0x3131, 0x3231, 0x3331, 0x3431,
			0x3531, 0x3631, 0x3731, 0x3831, 0x3931, 0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932, 0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533,
			0x3633, 0x3733, 0x3833, 0x3933, 0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934, 0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635,
			0x3735, 0x3835, 0x3935, 0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936, 0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737,
			0x3837, 0x3937, 0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938, 0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839,
			0x3939 };
		BNCH_SWT_ALIGN(64ULL)
		inline static constexpr auto charTable04{ [] {
			std::array<uint32_t, 10000> return_values{};
			for (uint32_t i = 0; i < 10000; ++i) {
				return_values[i] = (0x30 + (i / 1000)) | ((0x30 + ((i / 100) % 10)) << 8) | ((0x30 + ((i / 10) % 10)) << 16) | ((0x30 + (i % 10)) << 24);
			}
			return return_values;
		}() };
	};

	template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type operator<<(const value_type arg, std::integral_constant<uint64_t, shift>) noexcept {
		constexpr uint64_t shift_amount{ shift };
		return arg << shift_amount;
	}

	template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type& operator<<=(value_type& arg, std::integral_constant<uint64_t, shift>) noexcept {
		return arg = arg << std::integral_constant<uint64_t, shift>{};
	}

	template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type operator>>(const value_type arg, std::integral_constant<uint64_t, shift>) noexcept {
		constexpr uint64_t shift_amount{ shift };
		return arg >> shift_amount;
	}

	template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type& operator>>=(value_type& arg, std::integral_constant<uint64_t, shift>) noexcept {
		return arg = arg >> std::integral_constant<uint64_t, shift>{};
	}

	template<typename value_type, value_type divisor> struct multiply_and_shift;

	template<concepts::uns64_t value_type, value_type divisor> struct multiply_and_shift<value_type, divisor> {
		static constexpr uint_pair multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };
		BNCH_SWT_HOST static value_type impl(value_type value) noexcept {
#if BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
			const __uint128_t product = static_cast<__uint128_t>(value) * multiplicand_and_shift.multiplicand;
			return static_cast<value_type>(product >> std::integral_constant<value_type, multiplicand_and_shift.shift>{});
#elif BNCH_SWT_COMPILER_MSVC
			value_type high_part;
			value_type low_part = _umul128(multiplicand_and_shift.multiplicand, value, &high_part);
			if constexpr (multiplicand_and_shift.shift < 64ULL) {
				return static_cast<value_type>((low_part >> std::integral_constant<value_type, multiplicand_and_shift.shift>{}) |
					(high_part << std::integral_constant<value_type, 64ULL - multiplicand_and_shift.shift>{}));
			} else {
				return static_cast<value_type>(high_part >> std::integral_constant<value_type, multiplicand_and_shift.shift - 64ULL>{});
			}
#else
			value_type high_part;
			const value_type low_part = mul128Generic(value, multiplicand_and_shift.multiplicand, &high_part);
			if constexpr (multiplicand_and_shift.shift < 64ULL) {
				return static_cast<value_type>((low_part >> std::integral_constant<value_type, multiplicand_and_shift.shift>{}) |
					(high_part << std::integral_constant<value_type, 64ULL - multiplicand_and_shift.shift>{}));
			} else {
				return static_cast<value_type>(high_part >> std::integral_constant<value_type, multiplicand_and_shift.shift - 64ULL>{});
			}
#endif
		}
	};

	template<concepts::uns32_t value_type, value_type divisor> struct multiply_and_shift<value_type, divisor> {
		static constexpr uint_pair multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };
		BNCH_SWT_HOST static value_type impl(value_type value) noexcept {
			return static_cast<uint32_t>((static_cast<uint64_t>(value) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
		}
	};

	inline constexpr uint8_t decTrailingZeroTable[] = { 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0 };

	BNCH_SWT_HOST auto* writeu64Len15To17Trim(auto* buf, uint64_t sig) noexcept {
		uint32_t tz1, tz2, tz;
		const uint64_t abbccddee = multiply_and_shift<uint64_t, 100000000>::impl(sig);
		const uint64_t ffgghhii	 = sig - abbccddee * 100000000;
		uint32_t abbcc			 = multiply_and_shift<uint64_t, 10000>::impl(abbccddee);
		uint32_t ddee			 = abbccddee - abbcc * 10000;
		uint32_t abb			 = uint32_t((uint64_t(abbcc) * 167773) >> 24);
		uint32_t a				 = (abb * 41) >> 12;
		uint32_t bb				 = abb - a * 100;
		uint32_t cc				 = abbcc - abb * 100;
		buf[0]					 = uint8_t(a + '0');
		buf += a > 0;
		bool lz = bb < 10 && a == 0;
		std::memcpy(buf, fiwb<void>::charTable01 + (bb * 2 + lz), 2);
		buf -= lz;
		std::memcpy(buf + 2, fiwb<void>::charTable01 + 2 * cc, 2);

		if (ffgghhii) {
			uint32_t dd	  = (ddee * 5243) >> 19;
			uint32_t ee	  = ddee - dd * 100;
			uint32_t ffgg = uint32_t((uint64_t(ffgghhii) * 109951163) >> 40);
			uint32_t hhii = ffgghhii - ffgg * 10000;
			uint32_t ff	  = (ffgg * 5243) >> 19;
			uint32_t gg	  = ffgg - ff * 100;
			std::memcpy(buf + 4, fiwb<void>::charTable01 + 2 * dd, 2);
			std::memcpy(buf + 6, fiwb<void>::charTable01 + 2 * ee, 2);
			std::memcpy(buf + 8, fiwb<void>::charTable01 + 2 * ff, 2);
			std::memcpy(buf + 10, fiwb<void>::charTable01 + 2 * gg, 2);
			if (hhii) {
				uint32_t hh = (hhii * 5243) >> 19;
				uint32_t ii = hhii - hh * 100;
				std::memcpy(buf + 12, fiwb<void>::charTable01 + 2 * hh, 2);
				std::memcpy(buf + 14, fiwb<void>::charTable01 + 2 * ii, 2);
				tz1 = decTrailingZeroTable[hh];
				tz2 = decTrailingZeroTable[ii];
				tz	= ii ? tz2 : (tz1 + 2);
				buf += 16 - tz;
				return buf;
			} else {
				tz1 = decTrailingZeroTable[ff];
				tz2 = decTrailingZeroTable[gg];
				tz	= gg ? tz2 : (tz1 + 2);
				buf += 12 - tz;
				return buf;
			}
		} else {
			if (ddee) {
				uint32_t dd = (ddee * 5243) >> 19;
				uint32_t ee = ddee - dd * 100;
				std::memcpy(buf + 4, fiwb<void>::charTable01 + 2 * dd, 2);
				std::memcpy(buf + 6, fiwb<void>::charTable01 + 2 * ee, 2);
				tz1 = decTrailingZeroTable[dd];
				tz2 = decTrailingZeroTable[ee];
				tz	= ee ? tz2 : (tz1 + 2);
				buf += 8 - tz;
				return buf;
			} else {
				tz1 = decTrailingZeroTable[bb];
				tz2 = decTrailingZeroTable[cc];
				tz	= cc ? tz2 : (tz1 + tz2);
				buf += 4 - tz;
				return buf;
			}
		}
	}

	consteval uint32_t numbits(uint32_t x) noexcept {
		return x < 2 ? x : 1 + numbits(x >> 1);
	}

	BNCH_SWT_HOST static int64_t abs(int64_t value) noexcept {
		const uint64_t temp = static_cast<uint64_t>(value >> 63);
		value ^= temp;
		value += temp & 1;
		return value;
	}

	template<typename value_type> struct to_chars;

	template<std::floating_point value_type> struct to_chars<value_type> {
		BNCH_SWT_HOST static char* impl(char* buf, value_type val) noexcept {
			static_assert(std::numeric_limits<value_type>::is_iec559);
			static_assert(std::numeric_limits<value_type>::radix == 2);
			static_assert(std::is_same_v<float, value_type> || std::is_same_v<double, value_type>);
			static_assert(sizeof(float) == 4 && sizeof(double) == 8);
			constexpr bool is_float = std::is_same_v<float, value_type>;
			using Raw				= std::conditional_t<std::is_same_v<float, value_type>, uint32_t, uint64_t>;

			if (val == 0.0) {
				*buf = '-';
				buf += (std::bit_cast<Raw>(val) >> (sizeof(value_type) * 8 - 1));
				*buf = '0';
				return buf + 1;
			}

			using Conversion						 = jsonifier_jkj::dragonbox::default_float_bit_carrier_conversion_traits<value_type>;
			using FormatTraits						 = jsonifier_jkj::dragonbox::ieee754_binary_traits<typename Conversion::format, typename Conversion::carrier_uint>;
			static constexpr uint32_t exp_bits_count = numbits(std::numeric_limits<value_type>::max_exponent - std::numeric_limits<value_type>::min_exponent + 1);
			const auto float_bits					 = jsonifier_jkj::dragonbox::make_float_bits<value_type, Conversion, FormatTraits>(val);
			const auto exp_bits						 = float_bits.extract_exponent_bits();
			const auto s							 = float_bits.remove_exponent_bits();

			if (exp_bits == (uint32_t(1) << exp_bits_count) - 1) [[unlikely]] {
				if (s.u == 0) {
					if (float_bits.is_negative()) {
						std::memcpy(buf, "-inf", 4);
						return buf + 4;
					}
					std::memcpy(buf, "inf", 3);
					return buf + 3;
				} else {
					std::memcpy(buf, "nan", 3);
					return buf + 3;
				}
			}

			*buf				= '-';
			constexpr auto zero = value_type(0.0);
			buf += (val < zero);

			const auto v =
				jsonifier_jkj::dragonbox::to_decimal_ex(s, exp_bits, jsonifier_jkj::dragonbox::policy::sign::ignore, jsonifier_jkj::dragonbox::policy::trailing_zero::ignore);

			uint64_t sig_dec = v.significand;
			int32_t exp_dec	 = v.exponent;

			int32_t sig_len = 17;
			sig_len -= (sig_dec < 100000000ull * 100000000ull);
			sig_len -= (sig_dec < 100000000ull * 10000000ull);
			int32_t dot_pos = sig_len + exp_dec;

			if (-6 < dot_pos && dot_pos <= 21) {
				if (dot_pos <= 0) {
					auto num_hdr = buf + (2 - dot_pos);
					auto num_end = writeu64Len15To17Trim(num_hdr, sig_dec);
					buf[0]		 = '0';
					buf[1]		 = '.';
					buf += 2;
					std::memset(buf, '0', size_t(num_hdr - buf));
					return num_end;
				} else {
					std::memset(buf, '0', 24);
					auto num_hdr = buf + 1;
					auto num_end = writeu64Len15To17Trim(num_hdr, sig_dec);
					std::memmove(buf, buf + 1, size_t(dot_pos));
					buf[dot_pos] = '.';
					return ((num_end - num_hdr) <= dot_pos) ? buf + dot_pos : num_end;
				}
			} else {
				auto end = writeu64Len15To17Trim(buf + 1, sig_dec);
				end -= (end == buf + 2);
				exp_dec += sig_len - 1;
				buf[0] = buf[1];
				buf[1] = '.';
				end[0] = 'E';
				buf	   = end + 1;
				buf[0] = '-';
				buf += exp_dec < 0;
				exp_dec = abs(exp_dec);
				if (exp_dec < 100) {
					uint32_t lz = exp_dec < 10;
					std::memcpy(buf, fiwb<void>::charTable01 + (exp_dec * 2 + lz), 2);
					return buf + 2 - lz;
				} else {
					const uint32_t hi = (uint32_t(exp_dec) * 656) >> 16;
					const uint32_t lo = uint32_t(exp_dec) - hi * 100;
					buf[0]			  = uint8_t(hi) + '0';
					std::memcpy(&buf[1], fiwb<void>::charTable01 + (lo * 2), 2);
					return buf + 3;
				}
			}
		}
	};
}