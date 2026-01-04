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
#include <cstring>
#include <random>

constexpr char char_table[200] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1',
	'4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0', '3', '1',
	'3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4',
	'9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6',
	'6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3', '8',
	'4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

template<class value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, uint32_t>
auto* to_chars_glz(auto* buf, value_type value) noexcept {
	/* The maximum value of uint32_t is 4294967295 (10 digits), */
	/* these digits are named as 'aabbccddee' here.             */
	uint32_t aa, bb, cc, dd, ee, aabb, bbcc, ccdd, ddee, aabbcc;

	/* Leading zero count in the first pair.                    */
	uint32_t lz;

	/* Although most compilers may convert the "division by     */
	/* constant value" into "multiply and shift", manual        */
	/* conversion can still help some compilers generate        */
	/* fewer and better instructions.                           */

	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + (value * 2 + lz), 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) { /* 3-4 digits: aabb */
		aa = (value * 5243) >> 19; /* (value / 100) */
		bb = value - aa * 100; /* (value % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + (aa * 2 + lz), 2);
		buf -= lz;
		std::memcpy(&buf[2], char_table + (2 * bb), 2);

		return buf + 4;
	} else if (value < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		bbcc = value - aa * 10000; /* (value % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else if (value < 100000000) { /* 7~8 digits: aabbccdd */
		/* (value / 10000) */
		aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		ccdd = value - aabb * 10000; /* (value % 10000) */
		aa	 = (aabb * 5243) >> 19; /* (aabb / 100) */
		cc	 = (ccdd * 5243) >> 19; /* (ccdd / 100) */
		bb	 = aabb - aa * 100; /* (aabb % 100) */
		dd	 = ccdd - cc * 100; /* (ccdd % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		return buf + 8;
	} else { /* 9~10 digits: aabbccddee */
		/* (value / 10000) */
		aabbcc = uint32_t((uint64_t(value) * 3518437209ul) >> 45);
		/* (aabbcc / 10000) */
		aa	 = uint32_t((uint64_t(aabbcc) * 429497) >> 32);
		ddee = value - aabbcc * 10000; /* (value % 10000) */
		bbcc = aabbcc - aa * 10000; /* (aabbcc % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		dd	 = (ddee * 5243) >> 19; /* (ddee / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		ee	 = ddee - dd * 100; /* (ddee % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		std::memcpy(buf + 8, char_table + ee * 2, 2);
		return buf + 10;
	}
}

template<class value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, int32_t>
auto* to_chars_glz(auto* buf, value_type x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int32_t>::min case
	return to_chars_glz(buf + (x < 0), uint32_t(x ^ (x >> 31)) - (x >> 31));
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
BNCH_SWT_HOST auto* to_chars_u64_len_8(auto* buf, value_type value) noexcept {
	/* 8 digits: aabbccdd */
	const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40); /* (value / 10000) */
	const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
	const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
	const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
	const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
	const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	std::memcpy(buf + 4, char_table + cc * 2, 2);
	std::memcpy(buf + 6, char_table + dd * 2, 2);
	return buf + 8;
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
BNCH_SWT_HOST auto* to_chars_u64_len_4(auto* buf, value_type value) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (value * 5243) >> 19; /* (value / 100) */
	const uint32_t bb = value - aa * 100; /* (value % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	return buf + 4;
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
inline auto* to_chars_u64_len_1_8(auto* buf, value_type value) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + value * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) { /* 3-4 digits: aabb */
		aa = (value * 5243) >> 19; /* (value / 100) */
		bb = value - aa * 100; /* (value % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		return buf + 4;
	} else if (value < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		bbcc = value - aa * 10000; /* (value % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		ccdd = value - aabb * 10000; /* (value % 10000) */
		aa	 = (aabb * 5243) >> 19; /* (aabb / 100) */
		cc	 = (ccdd * 5243) >> 19; /* (ccdd / 100) */
		bb	 = aabb - aa * 100; /* (aabb % 100) */
		dd	 = ccdd - cc * 100; /* (ccdd % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		return buf + 8;
	}
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
auto* to_chars_u64_len_5_8(auto* buf, value_type value) noexcept {
	if (value < 1000000) { /* 5-6 digits: aabbcc */
		const uint32_t aa	= uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		const uint32_t bbcc = value - aa * 10000; /* (value % 10000) */
		const uint32_t bb	= (bbcc * 5243) >> 19; /* (bbcc / 100) */
		const uint32_t cc	= bbcc - bb * 100; /* (bbcc % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
		const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
		const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
		const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
		const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		return buf + 8;
	}
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint64_t>)
auto* to_chars_glz(auto* buf, value_type value) noexcept {
	if (value < 100000000) { /* 1-8 digits */
		buf = to_chars_u64_len_1_8(buf, uint32_t(value));
		return buf;
	} else if (value < 100000000ULL * 100000000ULL) { /* 9-16 digits */
		const uint64_t hgh = value / 100000000;
		const auto low	   = uint32_t(value - hgh * 100000000); /* (value % 100000000) */
		buf				   = to_chars_u64_len_1_8(buf, uint32_t(hgh));
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = value / 100000000;
		const auto low	   = uint32_t(value - tmp * 100000000); /* (value % 100000000) */
		const auto hgh	   = uint32_t(tmp / 10000);
		const auto mid	   = uint32_t(tmp - hgh * 10000); /* (tmp % 10000) */
		buf				   = to_chars_u64_len_5_8(buf, hgh);
		buf				   = to_chars_u64_len_4(buf, mid);
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	}
}

template<class value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, int64_t>
auto* to_chars_glz(auto* buf, value_type x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars_glz(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

namespace concepts {

	template<typename value_type>
	concept uns64_t = std::unsigned_integral<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept sig64_t = std::signed_integral<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept uns32_t = std::unsigned_integral<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept sig32_t = std::signed_integral<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 4;

}

template<class value_type_new, value_type_new valueNew> struct integral_constant {
	BNCH_SWT_ALIGN(64ULL) static constexpr value_type_new value = valueNew;

	using value_type = value_type_new;
	using type		 = integral_constant;

	BNCH_SWT_HOST constexpr operator value_type() const noexcept {
		return value;
	}

	BNCH_SWT_HOST constexpr value_type operator()() const noexcept {
		return value;
	}
};

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
		constexpr uint_type div_minus_1 = divisor_new - 1;
		constexpr value_type l			= 127 - div_minus_1.lzcnt();
		constexpr uint_type numerator	= uint_type{ 1 } << (64 + static_cast<value_type>(l));
		constexpr uint_type m_128		= (numerator + div_temp - 1) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64 + l };
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
		constexpr uint_type div_temp	= divisor_new;
		constexpr uint_type div_minus_1 = divisor_new - 1;
		constexpr value_type l			= 63 - div_minus_1.lzcnt();
		constexpr uint_type numerator	= uint_type{ 1 } << (32 + static_cast<value_type>(l));
		constexpr uint_type m_128		= (numerator + div_temp - 1) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 32 + l };
	}
};

template<typename typeName> struct fiwb {
	BNCH_SWT_ALIGN(64ULL)
	inline static constexpr char charTable01[]{ 0x30, 0x30, 0x30, 0x31, 0x30, 0x32, 0x30, 0x33, 0x30, 0x34, 0x30, 0x35, 0x30, 0x36, 0x30, 0x37, 0x30, 0x38, 0x30, 0x39, 0x31, 0x30,
		0x31, 0x31, 0x31, 0x32, 0x31, 0x33, 0x31, 0x34, 0x31, 0x35, 0x31, 0x36, 0x31, 0x37, 0x31, 0x38, 0x31, 0x39, 0x32, 0x30, 0x32, 0x31, 0x32, 0x32, 0x32, 0x33, 0x32, 0x34,
		0x32, 0x35, 0x32, 0x36, 0x32, 0x37, 0x32, 0x38, 0x32, 0x39, 0x33, 0x30, 0x33, 0x31, 0x33, 0x32, 0x33, 0x33, 0x33, 0x34, 0x33, 0x35, 0x33, 0x36, 0x33, 0x37, 0x33, 0x38,
		0x33, 0x39, 0x34, 0x30, 0x34, 0x31, 0x34, 0x32, 0x34, 0x33, 0x34, 0x34, 0x34, 0x35, 0x34, 0x36, 0x34, 0x37, 0x34, 0x38, 0x34, 0x39, 0x35, 0x30, 0x35, 0x31, 0x35, 0x32,
		0x35, 0x33, 0x35, 0x34, 0x35, 0x35, 0x35, 0x36, 0x35, 0x37, 0x35, 0x38, 0x35, 0x39, 0x36, 0x30, 0x36, 0x31, 0x36, 0x32, 0x36, 0x33, 0x36, 0x34, 0x36, 0x35, 0x36, 0x36,
		0x36, 0x37, 0x36, 0x38, 0x36, 0x39, 0x37, 0x30, 0x37, 0x31, 0x37, 0x32, 0x37, 0x33, 0x37, 0x34, 0x37, 0x35, 0x37, 0x36, 0x37, 0x37, 0x37, 0x38, 0x37, 0x39, 0x38, 0x30,
		0x38, 0x31, 0x38, 0x32, 0x38, 0x33, 0x38, 0x34, 0x38, 0x35, 0x38, 0x36, 0x38, 0x37, 0x38, 0x38, 0x38, 0x39, 0x39, 0x30, 0x39, 0x31, 0x39, 0x32, 0x39, 0x33, 0x39, 0x34,
		0x39, 0x35, 0x39, 0x36, 0x39, 0x37, 0x39, 0x38, 0x39, 0x39 };
	BNCH_SWT_ALIGN(64ULL)
	inline static constexpr uint16_t charTable02[]{ 0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930, 0x3031, 0x3131, 0x3231, 0x3331, 0x3431, 0x3531,
		0x3631, 0x3731, 0x3831, 0x3931, 0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932, 0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533, 0x3633,
		0x3733, 0x3833, 0x3933, 0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934, 0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635, 0x3735,
		0x3835, 0x3935, 0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936, 0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737, 0x3837,
		0x3937, 0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938, 0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839, 0x3939 };
	BNCH_SWT_ALIGN(64ULL)
	inline static constexpr auto charTable04{ [] {
		std::array<uint32_t, 10000> return_values{};
		for (uint32_t i = 0; i < 10000; ++i) {
			return_values[i] = (0x30 + (i / 1000)) | ((0x30 + ((i / 100) % 10)) << 8) | ((0x30 + ((i / 10) % 10)) << 16) | ((0x30 + (i % 10)) << 24);
		}
		return return_values;
	}() };
};

template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type operator<<(const value_type arg, integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg << shift_amount;
}

template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type& operator<<=(value_type& arg, integral_constant<uint64_t, shift>) noexcept {
	return arg = arg << integral_constant<uint64_t, shift>{};
}

template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type operator>>(const value_type arg, integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg >> shift_amount;
}

template<uint64_t shift, std::integral value_type> BNCH_SWT_HOST constexpr value_type& operator>>=(value_type& arg, integral_constant<uint64_t, shift>) noexcept {
	return arg = arg >> integral_constant<uint64_t, shift>{};
}

template<typename value_type, value_type divisor> struct multiply_and_shift;

template<concepts::uns64_t value_type, value_type divisor> struct multiply_and_shift<value_type, divisor> {
	static constexpr uint_pair multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };
	BNCH_SWT_HOST static value_type impl(value_type value) noexcept {
#if BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
		const __uint128_t product = static_cast<__uint128_t>(value) * multiplicand_and_shift.multiplicand;
		return static_cast<value_type>(product >> integral_constant<value_type, multiplicand_and_shift.shift>{});
#elif BNCH_SWT_COMPILER_MSVC
		value_type high_part;
		value_type low_part = _umul128(multiplicand_and_shift.multiplicand, value, &high_part);
		if constexpr (multiplicand_and_shift.shift < 64ULL) {
			return static_cast<value_type>(
				(low_part >> integral_constant<value_type, multiplicand_and_shift.shift>{}) | (high_part << integral_constant<value_type, 64ULL - multiplicand_and_shift.shift>{}));
		} else {
			return static_cast<value_type>(high_part >> integral_constant<value_type, multiplicand_and_shift.shift - 64ULL>{});
		}
#else
		value_type high_part;
		const value_type low_part = mul128Generic(value, multiplicand_and_shift.multiplicand, &high_part);
		if constexpr (multiplicand_and_shift.shift < 64ULL) {
			return static_cast<value_type>(
				(low_part >> integral_constant<value_type, multiplicand_and_shift.shift>{}) | (high_part << integral_constant<value_type, 64ULL - multiplicand_and_shift.shift>{}));
		} else {
			return static_cast<value_type>(high_part >> integral_constant<value_type, multiplicand_and_shift.shift - 64ULL>{});
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

template<typename value_type, uint64_t digit_length> struct to_chars_impl;

template<concepts::uns32_t value_type> struct to_chars_impl<value_type, 2ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		const uint32_t lz = value < 10U;
		std::memcpy(buf, char_table_ptr + (value * 2U + lz), 2ULL);
		buf -= lz;
		return buf + 2ULL;
	}
};

template<concepts::uns32_t value_type> struct to_chars_impl<value_type, 4ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint32_t aa = multiply_and_shift<value_type, 100U>::impl(value);
		const uint32_t lz = value < 1000U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (value - aa * 100U), 2ULL);
		return buf + 4ULL;
	}
};

template<concepts::uns32_t value_type> struct to_chars_impl<value_type, 6ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
#if !BNCH_SWT_COMPILER_MSVC
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint32_t aa = multiply_and_shift<value_type, 10000U>::impl(value);
		const uint32_t lz = value < 100000U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		const uint32_t remainder = value - aa * 10000U;
		std::memcpy(buf + 2ULL, int32_table + remainder, 4ULL);
#else
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint32_t aa	= multiply_and_shift<value_type, 10000U>::impl(value);
		const uint32_t bbcc = value - aa * 10000U;
		const uint32_t bb	= multiply_and_shift<value_type, 100U>::impl(bbcc);
		const uint32_t cc	= bbcc - bb * 100U;
		const uint32_t lz	= aa < 10U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + bb, 2ULL);
		std::memcpy(buf + 4ULL, int16_table + cc, 2ULL);
#endif
		return buf + 6ULL;
	}
};

template<concepts::uns32_t value_type> struct to_chars_impl<value_type, 8ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
#if !BNCH_SWT_COMPILER_MSVC
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint32_t aabb = multiply_and_shift<value_type, 10000U>::impl(value);
		const uint32_t aa	= multiply_and_shift<value_type, 100U>::impl(aabb);
		const uint32_t lz	= value < 10000000U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (aabb - aa * 100U), 2ULL);
		const uint32_t ccdd = value - aabb * 10000U;
		std::memcpy(buf + 4ULL, int32_table + ccdd, 4ULL);
#else
		const uint32_t aabb = multiply_and_shift<value_type, 10000U>::impl(value);
		const uint32_t ccdd = value - aabb * 10000U;
		const uint32_t aa	= multiply_and_shift<value_type, 100U>::impl(aabb);
		const uint32_t cc	= multiply_and_shift<value_type, 100U>::impl(ccdd);
		const uint32_t bb	= aabb - aa * 100U;
		const uint32_t dd	= ccdd - cc * 100U;
		const uint32_t lz	= aa < 10U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + bb, 2ULL);
		std::memcpy(buf + 4ULL, int16_table + cc, 2ULL);
		std::memcpy(buf + 6ULL, int16_table + dd, 2ULL);
#endif
		return buf + 8ULL;
	}
};

template<concepts::uns32_t value_type> struct to_chars_impl<value_type, 10ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
#if !BNCH_SWT_COMPILER_MSVC
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint32_t high = multiply_and_shift<value_type, 100000000U>::impl(value);
		const uint32_t low	= value - high * 100000000U;
		const uint32_t lz	= high < 10U;
		std::memcpy(buf, char_table_ptr + (high * 2U + lz), 2ULL);
		buf -= lz;
		const uint32_t aabb = multiply_and_shift<value_type, 10000U>::impl(low);
		const uint32_t ccdd = low - aabb * 10000U;
		std::memcpy(buf + 2ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 6ULL, int32_table + ccdd, 4ULL);
#else
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint32_t aabbcc = multiply_and_shift<value_type, 10000U>::impl(value);
		const uint32_t aa	  = multiply_and_shift<value_type, 10000U>::impl(aabbcc);
		const uint32_t ddee	  = value - aabbcc * 10000U;
		const uint32_t bbcc	  = aabbcc - aa * 10000U;
		const uint32_t bb	  = multiply_and_shift<value_type, 100U>::impl(bbcc);
		const uint32_t dd	  = multiply_and_shift<value_type, 100U>::impl(ddee);
		const uint32_t cc	  = bbcc - bb * 100U;
		const uint32_t ee	  = ddee - dd * 100U;
		const uint32_t lz	  = aa < 10U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + bb, 2ULL);
		std::memcpy(buf + 4ULL, int16_table + cc, 2ULL);
		std::memcpy(buf + 6ULL, int16_table + dd, 2ULL);
		std::memcpy(buf + 8ULL, int16_table + ee, 2ULL);
#endif
		return buf + 10ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 2ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		const uint64_t lz = value < 10ULL;
		std::memcpy(buf, char_table_ptr + (value * 2ULL + lz), 2ULL);
		buf -= lz;
		return buf + 2ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 4ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint64_t aa = multiply_and_shift<value_type, 100ULL>::impl(value);
		const uint64_t lz = value < 1000ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (value - aa * 100ULL), 2ULL);
		return buf + 4ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 6ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t aa = multiply_and_shift<value_type, 10000ULL>::impl(value);
		const uint64_t lz = value < 100000ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t remainder = value - aa * 10000ULL;
		std::memcpy(buf + 2ULL, int32_table + remainder, 4ULL);
		return buf + 6ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 8ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(value);
		const uint64_t aa	= multiply_and_shift<value_type, 100ULL>::impl(aabb);
		const uint64_t lz	= value < 10000000ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (aabb - aa * 100ULL), 2ULL);
		const uint64_t ccdd = value - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, int32_table + ccdd, 4ULL);
		return buf + 8ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 10ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		const uint64_t lz	= high < 10ULL;
		std::memcpy(buf, char_table_ptr + (high * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 2ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 6ULL, int32_table + ccdd, 4ULL);
		return buf + 10ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 12ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		const uint64_t aa	= multiply_and_shift<value_type, 100ULL>::impl(high);
		const uint64_t lz	= aa < 10ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (high - aa * 100ULL), 2ULL);
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 8ULL, int32_table + ccdd, 4ULL);
		return buf + 12ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 14ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		const uint64_t aa	= multiply_and_shift<value_type, 10000ULL>::impl(high);
		const uint64_t lz	= aa < 10ULL;
		const uint64_t bbcc = high - aa * 10000ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int32_table + bbcc, 4ULL);
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 6ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 10ULL, int32_table + ccdd, 4ULL);
		return buf + 14ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 16ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(high);
		const uint64_t ccdd = high - aabb * 10000ULL;
		const uint64_t aa	= multiply_and_shift<value_type, 100ULL>::impl(aabb);
		const uint64_t lz	= aa < 10ULL;
		const uint64_t bb	= aabb - aa * 100ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + bb, 2ULL);
		std::memcpy(buf + 4ULL, int32_table + ccdd, 4ULL);
		const uint64_t eeff = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t gghh = low - eeff * 10000ULL;
		std::memcpy(buf + 8ULL, int32_table + eeff, 4ULL);
		std::memcpy(buf + 12ULL, int32_table + gghh, 4ULL);
		return buf + 16ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 18ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high	  = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	  = value - high * 100000000ULL;
		const uint64_t high10 = multiply_and_shift<value_type, 100000000ULL>::impl(high);
		const uint64_t low10  = high - high10 * 100000000ULL;
		const uint64_t lz	  = high10 < 10ULL;
		std::memcpy(buf, char_table_ptr + (high10 * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(low10);
		const uint64_t ccdd = low10 - aabb * 10000ULL;
		std::memcpy(buf + 2ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 6ULL, int32_table + ccdd, 4ULL);
		const uint64_t eeff = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t gghh = low - eeff * 10000ULL;
		std::memcpy(buf + 10ULL, int32_table + eeff, 4ULL);
		std::memcpy(buf + 14ULL, int32_table + gghh, 4ULL);
		return buf + 18ULL;
	}
};

template<concepts::uns64_t value_type> struct to_chars_impl<value_type, 20ULL> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		BNCH_SWT_ALIGN(64ULL) static constexpr const char* char_table_ptr{ fiwb<void>::charTable01 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint32_t* int32_table{ fiwb<void>::charTable04.data() };
		const uint64_t high	  = multiply_and_shift<value_type, 100000000ULL>::impl(value);
		const uint64_t low	  = value - high * 100000000ULL;
		const uint64_t high12 = multiply_and_shift<value_type, 100000000ULL>::impl(high);
		const uint64_t low12  = high - high12 * 100000000ULL;
		const uint64_t aa	  = multiply_and_shift<value_type, 100ULL>::impl(high12);
		const uint64_t lz	  = aa < 10ULL;
		std::memcpy(buf, char_table_ptr + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (high12 - aa * 100ULL), 2ULL);
		const uint64_t aabb = multiply_and_shift<value_type, 10000ULL>::impl(low12);
		const uint64_t ccdd = low12 - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 8ULL, int32_table + ccdd, 4ULL);
		const uint64_t eeff = multiply_and_shift<value_type, 10000ULL>::impl(low);
		const uint64_t gghh = low - eeff * 10000ULL;
		std::memcpy(buf + 12ULL, int32_table + eeff, 4ULL);
		std::memcpy(buf + 16ULL, int32_table + gghh, 4ULL);
		return buf + 20ULL;
	}
};

template<typename value_type> struct to_chars;

template<concepts::uns64_t value_type> struct to_chars<value_type> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		if (value < 10000ULL) {
			if (value < 100ULL) {
				return to_chars_impl<value_type, 2ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 4ULL>::impl(buf, value);
			}
		} else if (value < 100000000ULL) {
			if (value < 1000000ULL) {
				return to_chars_impl<value_type, 6ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 8ULL>::impl(buf, value);
			}
		} else if (value < 1000000000000ULL) {
			if (value < 10000000000ULL) {
				return to_chars_impl<value_type, 10ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 12ULL>::impl(buf, value);
			}
		} else if (value < 10000000000000000ULL) {
			if (value < 100000000000000ULL) {
				return to_chars_impl<value_type, 14ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 16ULL>::impl(buf, value);
			}
		} else if (value < 1000000000000000000ULL) {
			return to_chars_impl<value_type, 18ULL>::impl(buf, value);
		} else {
			return to_chars_impl<value_type, 20ULL>::impl(buf, value);
		}
	}
};

template<concepts::sig64_t value_type> struct to_chars<value_type> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		using unsigned_type					 = std::make_unsigned_t<value_type>;
		constexpr unsigned_type shift_amount = sizeof(value_type) * 8ULL - 1ULL;
		*buf								 = '-';
		return to_chars<unsigned_type>::impl(buf + (value < 0), (static_cast<unsigned_type>(value) ^ (value >> shift_amount)) - (value >> shift_amount));
	}
};

template<concepts::uns32_t value_type> struct to_chars<value_type> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		if (value < 10000U) {
			if (value < 100U) {
				return to_chars_impl<value_type, 2ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 4ULL>::impl(buf, value);
			}
		} else if (value < 100000000U) {
			if (value < 1000000U) {
				return to_chars_impl<value_type, 6ULL>::impl(buf, value);
			} else {
				return to_chars_impl<value_type, 8ULL>::impl(buf, value);
			}
		} else {
			return to_chars_impl<value_type, 10ULL>::impl(buf, value);
		}
	}
};

template<concepts::sig32_t value_type> struct to_chars<value_type> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		using unsigned_type					 = std::make_unsigned_t<value_type>;
		constexpr unsigned_type shift_amount = static_cast<unsigned_type>(sizeof(value_type) * 8ULL - 1ULL);
		*buf								 = '-';
		return to_chars<unsigned_type>::impl(buf + (value < 0), (static_cast<unsigned_type>(value) ^ (value >> shift_amount)) - (value >> shift_amount));
	}
};

template<typename value_type> BNCH_SWT_HOST value_type max_value_for_digits(uint64_t num_digits) noexcept {
	if (num_digits <= 0) {
		return value_type{ 0 };
	}
	long double power_of_10		= std::pow(10.0L, num_digits);
	long double theoretical_max = power_of_10 - 1.0L;
	const value_type type_max	= std::numeric_limits<value_type>::max();

	if constexpr (std::is_unsigned_v<value_type>) {
		if (theoretical_max >= type_max) {
			return type_max;
		}
		return static_cast<value_type>(theoretical_max);

	} else {
		if (theoretical_max >= type_max) {
			return type_max;
		}
		return static_cast<value_type>(theoretical_max);
	}
}

template<typename value_type> value_type get_min_value_requiring_digits(uint64_t num_digits) noexcept {
	if (num_digits <= 0)
		return value_type{ 0 };
	long double min_val = std::pow(10.0L, num_digits - 1);

	const value_type type_max = std::numeric_limits<value_type>::max();
	if (min_val >= type_max)
		return type_max;

	return static_cast<value_type>(std::min<long double>(static_cast<long double>(min_val), static_cast<long double>(type_max)));
}

template<typename value_type> value_type get_max_value_for_digits(uint64_t num_digits) noexcept {
	if (num_digits <= 0)
		return value_type{ 0 };
	long double max_val = std::pow(10.0L, num_digits) - 1.0L;

	const value_type type_max = std::numeric_limits<value_type>::max();
	if (max_val >= type_max)
		return type_max;

	return static_cast<value_type>(max_val);
}

template<typename value_type> std::vector<value_type> generate_digit_vector(size_t vector_size, uint64_t min_digits, uint64_t max_digits) {
	if (max_digits < min_digits) {
		std::swap(max_digits, min_digits);
	}

	if (min_digits < 1) {
		min_digits = 1;
	}

	value_type absolute_min = get_min_value_requiring_digits<value_type>(min_digits);

	value_type absolute_max = get_max_value_for_digits<value_type>(max_digits);

	if constexpr (std::is_signed_v<value_type>) {
		std::vector<value_type> result;
		result.reserve(vector_size);

		value_type negative_min = static_cast<value_type>(-absolute_max);
		value_type negative_max = static_cast<value_type>(-absolute_min);

		for (size_t i = 0; i < vector_size; ++i) {
			if (i % 2 == 0) {
				result.push_back(bnch_swt::random_generator<value_type>::impl(absolute_min, absolute_max));
			} else {
				result.push_back(bnch_swt::random_generator<value_type>::impl(negative_min, negative_max));
			}
		}
		return result;

	} else {
		std::vector<value_type> result;
		result.reserve(vector_size);

		for (size_t i = 0; i < vector_size; ++i) {
			result.push_back(bnch_swt::random_generator<value_type>::impl(absolute_min, absolute_max));
		}
		return result;
	}
}

static constexpr auto max_iterations{ 20 };
static constexpr auto measured_iterations{ max_iterations / 5 };

template<typename value_type> struct benchmark_glz_to_chars {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomIntegers, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			to_chars_glz(resultsTest[current_index][x].data(), randomIntegers[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += resultsTest[current_index][x].size();
		}
		return currentCount;
	}
};

template<typename value_type> struct benchmark_jsonifier_to_chars {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomIntegers, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			to_chars<value_type>::impl(resultsTest[current_index][x].data(), randomIntegers[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += resultsTest[current_index][x].size();
		}
		return currentCount;
	}
};

template<typename value_type> struct benchmark_std_to_chars {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomIntegers, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			std::to_chars(resultsTest[current_index][x].data(), resultsTest[current_index][x].data() + resultsTest[current_index][x].size(), randomIntegers[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += resultsTest[current_index][x].size();
		}
		return currentCount;
	}
};

template<typename value_type> struct benchmark_std_to_string {
	BNCH_SWT_HOST static uint64_t impl(std::vector<std::vector<std::string>>& resultsTest, const std::vector<std::vector<value_type>>& randomIntegers, uint64_t count,
		uint64_t current_index) {
		uint64_t currentCount{};
		for (uint64_t x = 0; x < count; ++x) {
			resultsTest[current_index][x] = std::to_string(randomIntegers[current_index][x]);
			bnch_swt::do_not_optimize_away(resultsTest[current_index][x]);
			currentCount += resultsTest[current_index][x].size();
		}
		return currentCount;
	}
};

template<bnch_swt::string_literal stage_name, typename benchmark_type, bnch_swt::string_literal benchmark_name>
BNCH_SWT_HOST void run_and_validate(auto& resultsTest, const auto& resultsReal, const auto& randomIntegers, uint64_t count, uint64_t& current_index) {
	using benchmark = bnch_swt::benchmark_stage<stage_name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, false>;

	benchmark::template run_benchmark<benchmark_name, benchmark_type>(resultsTest, randomIntegers, count, current_index);

	for (uint64_t y = 0; y < count; ++y) {
		if (resultsReal[current_index][y] != resultsTest[current_index][y]) {
			std::cout << benchmark_name.operator std::string_view() << " failed to serialize an integer of value: " << resultsReal[current_index][y]
					  << ", instead it serialized: " << resultsTest[current_index][y] << std::endl;
			return;
		}
	}
}

template<typename value_type, bnch_swt::string_literal name, uint64_t min_length, uint64_t max_length, uint64_t count> inline void testFunction() {
	std::vector<std::vector<value_type>> randomIntegers{};
	randomIntegers.resize(max_iterations);
	for (uint64_t x = 0; x < max_iterations; ++x) {
		randomIntegers[x] = generate_digit_vector<value_type>(count, min_length, max_length);
	}
	using benchmark = bnch_swt::benchmark_stage<name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, false>;
	std::vector<std::vector<std::string>> resultsReal{};
	std::vector<std::vector<std::string>> resultsTest01{};
	std::vector<std::vector<std::string>> resultsTest02{};
	std::vector<std::vector<std::string>> resultsTest03{};
	std::vector<std::vector<std::string>> resultsTest04{};
	resultsReal.resize(max_iterations);
	resultsTest01.resize(max_iterations);
	resultsTest02.resize(max_iterations);
	resultsTest03.resize(max_iterations);
	resultsTest04.resize(max_iterations);
	for (uint64_t x = 0; x < max_iterations; ++x) {
		resultsTest01[x].resize(count);
		resultsTest02[x].resize(count);
		resultsTest03[x].resize(count);
		resultsTest04[x].resize(count);
		resultsReal[x].resize(count);

		for (uint64_t y = 0; y < count; ++y) {
			resultsReal[x][y] = std::to_string(randomIntegers[x][y]);
			resultsTest01[x][y].resize(resultsReal[x][y].size());
			resultsTest02[x][y].resize(resultsReal[x][y].size());
			resultsTest03[x][y].resize(resultsReal[x][y].size());
			resultsTest04[x][y].resize(resultsReal[x][y].size());
		}
	}
	uint64_t currentIndex{};
	bnch_swt::internal::cache_clearer<bnch_swt::benchmark_types::cpu> cache_clearer{};
	cache_clearer.evict_caches();
	run_and_validate<name, benchmark_std_to_string<value_type>, "std::to_string">(resultsTest01, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	cache_clearer.evict_caches();
	run_and_validate<name, benchmark_glz_to_chars<value_type>, "glz::to_chars">(resultsTest02, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	cache_clearer.evict_caches();
	run_and_validate<name, benchmark_std_to_chars<value_type>, "std::to_chars">(resultsTest03, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	cache_clearer.evict_caches();
	run_and_validate<name, benchmark_jsonifier_to_chars<value_type>, "jsonifier::to_chars">(resultsTest04, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	static constexpr bnch_swt::performance_metrics_presence<bnch_swt::benchmark_types::cpu> presences{ .throughput_mb_per_sec = true,
		.bytes_processed																									  = true,
		.cycles_per_byte																									  = true,
		.name																												  = true };
	benchmark::template print_results<presences>(true, true);
}

int32_t main() {
	testFunction<int64_t, "int64-test-0-to-5", 0, 5, 10000>();
	testFunction<int64_t, "int64-test-0-to-10", 0, 10, 10000>();
	testFunction<int64_t, "int64-test-0-to-15", 0, 15, 10000>();
	testFunction<int64_t, "int64-test-0-to-20", 0, 20, 10000>();
	testFunction<int64_t, "int64-test-5-to-10", 5, 10, 10000>();
	testFunction<int64_t, "int64-test-5-to-15", 5, 15, 10000>();
	testFunction<int64_t, "int64-test-5-to-20", 5, 20, 10000>();
	testFunction<int64_t, "int64-test-10-to-15", 10, 15, 10000>();
	testFunction<int64_t, "int64-test-10-to-20", 10, 19, 10000>();
	testFunction<int64_t, "int64-test-15-to-20", 15, 19, 10000>();
	testFunction<int64_t, "int64-test-20", 20, 19, 10000>();
	testFunction<uint64_t, "uint64-test-0-to-5", 0, 5, 10000>();
	testFunction<uint64_t, "uint64-test-0-to-10", 0, 10, 10000>();
	testFunction<uint64_t, "uint64-test-0-to-15", 0, 15, 10000>();
	testFunction<uint64_t, "uint64-test-0-to-20", 0, 20, 10000>();
	testFunction<uint64_t, "uint64-test-5-to-10", 5, 10, 10000>();
	testFunction<uint64_t, "uint64-test-5-to-15", 5, 15, 10000>();
	testFunction<uint64_t, "uint64-test-5-to-20", 5, 20, 10000>();
	testFunction<uint64_t, "uint64-test-10-to-15", 10, 15, 10000>();
	testFunction<uint64_t, "uint64-test-10-to-20", 10, 20, 10000>();
	testFunction<uint64_t, "uint64-test-15-to-20", 15, 20, 10000>();
	testFunction<uint64_t, "uint64-test-20", 20, 20, 10000>();
	testFunction<int32_t, "int32-test-0-to-5", 0, 5, 10000>();
	testFunction<int32_t, "int32-test-0-to-10", 0, 10, 10000>();
	testFunction<int32_t, "int32-test-5-to-10", 5, 10, 10000>();
	testFunction<uint32_t, "uint32-test-0-to-5", 0, 5, 10000>();
	testFunction<uint32_t, "uint32-test-0-to-10", 0, 10, 10000>();
	testFunction<uint32_t, "uint32-test-5-to-10", 5, 10, 10000>();
	return 0;
}