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
		std::array<uint32_t, 10000ULL> return_values{};
		for (uint32_t i = 0; i < 10000; ++i) {
			return_values[i] = (0x30 + (i / 1000)) | ((0x30 + ((i / 100) % 10)) << 8) | ((0x30 + ((i / 10) % 10)) << 16) | ((0x30 + (i % 10)) << 24);
		}
		return return_values;
	}() };
};

template<typename value_type, value_type divisor> struct mul_and_shift;

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 10ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 14757395258967641293ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 67ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 100ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 11805916207174113035ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 70ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 1000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 9444732965739290428ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 73ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 10000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 15111572745182864684ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 77ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 100000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 12089258196146291748ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 80ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 1000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 9671406556917033398ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 83ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 10000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 15474250491067253437ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 87ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 100000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 12379400392853802749ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 90ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 1000000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 9903520314283042200ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 93ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 10000000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 15845632502852867519ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 97ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 100000000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 12676506002282294015ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 100ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint64_t, 1000000000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 10141204801825835212ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 103ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 10ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 3435973837ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 35ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 100ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2748779070ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 38ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 1000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2199023256ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 41ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 10000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 3518437209ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 45ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 100000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2814749768ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 48ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 1000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2251799814ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 51ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 10000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 3602879702ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 55ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 100000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2882303762ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 58ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<> struct BNCH_SWT_ALIGN(bnch_swt::device_alignment) mul_and_shift<uint32_t, 1000000000ULL> {
	static constexpr bnch_swt::aligned_const multiplicand_raw{ 2305843010ULL };
	static constexpr bnch_swt::aligned_const shift_raw{ 61ULL };
	static constexpr const uint64_t& multiplicand{ *multiplicand_raw };
	static constexpr const uint64_t& shift{ *shift_raw };
};

template<typename value_type>
concept uns32_t = std::unsigned_integral<value_type> && sizeof(value_type) == 4;

template<typename value_type>
concept uns64_t = std::unsigned_integral<value_type> && sizeof(value_type) == 8;

template<uns32_t value_type> BNCH_SWT_HOST value_type lzcnt(const value_type value) noexcept {
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
	return static_cast<value_type>(__clz(static_cast<int>(value)));
#elif BNCH_SWT_COMPILER_MSVC
	#if BNCH_SWT_ARCH_ARM64
	unsigned int leading_zero = 0;
	if (_BitScanReverse32(&leading_zero, value)) {
		return 31U - static_cast<value_type>(leading_zero);
	} else {
		return 32U;
	}
	#else
	return _lzcnt_u32(value);
	#endif
#else
	return (value == 0) ? 32 : static_cast<value_type>(__builtin_clz(static_cast<unsigned int>(value)));
#endif
}

template<uns64_t value_type> BNCH_SWT_HOST value_type lzcnt(const value_type value) noexcept {
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
	return static_cast<value_type>(__clzll(static_cast<long long>(value)));
#elif BNCH_SWT_COMPILER_MSVC
	#if BNCH_SWT_ARCH_ARM64
	unsigned long leading_zero = 0;
	if (_BitScanReverse64(&leading_zero, value)) {
		return 63ULL - static_cast<value_type>(leading_zero);
	} else {
		return 64ULL;
	}
	#else
	return _lzcnt_u64(value);
	#endif
#else
	return (value == 0) ? 64 : static_cast<value_type>(__builtin_clzll(static_cast<unsigned long long>(value)));
#endif
}

template<uint64_t digits> consteval uint64_t max_value_for_digits() noexcept {
	uint64_t power = 1;
	for (uint64_t i = 0; i < digits; ++i) {
		power *= 10;
	}
	return power - 1;
}

template<typename value_type, value_type maxLength> std::vector<value_type> generate_random_integers(uint64_t count) {
	std::vector<value_type> randomNumbers;
	for (uint64_t i = 0; i < count; ++i) {
		randomNumbers.emplace_back(bnch_swt::random_generator<value_type>::impl(static_cast<value_type>(max_value_for_digits<maxLength>())));
	}
	return randomNumbers;
}

BNCH_SWT_ALIGN(64ULL) static constexpr uint8_t digitCounts_32[]{ 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
BNCH_SWT_ALIGN(64ULL) static constexpr uint32_t digitCountThresholds_32[]{ 0u, 9u, 99u, 999u, 9999u, 99999u, 999999u, 9999999u, 99999999u, 999999999u, 4294967295u };

BNCH_SWT_HOST uint32_t fastDigitCount(const uint32_t inputValue) {
	const uint32_t originalDigitCount{ digitCounts_32[lzcnt(inputValue)] };
	return originalDigitCount + static_cast<uint32_t>(inputValue > digitCountThresholds_32[originalDigitCount]);
}

BNCH_SWT_ALIGN(64ULL)
static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 9,
	9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
BNCH_SWT_ALIGN(64ULL)
static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull, 99999999999ull,
	999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull, 9999999999999999999ull };

BNCH_SWT_HOST uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ digitCounts[lzcnt(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

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
	BNCH_SWT_HOST static value_type impl(value_type value) noexcept {
#if BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
		const __uint128_t product = static_cast<__uint128_t>(value) * mul_and_shift<value_type, divisor>::multiplicand;
		return static_cast<value_type>(product >> integral_constant<value_type, mul_and_shift<value_type, divisor>::shift>{});
#elif BNCH_SWT_COMPILER_MSVC
		value_type high_part;
		_umul128(mul_and_shift<value_type, divisor>::multiplicand, value, &high_part);
		return static_cast<value_type>(high_part >> integral_constant<value_type, mul_and_shift<value_type, divisor>::shift - 64ULL>{});
#else
		value_type high_part;
		const value_type low_part = mul128Generic(value, mul_and_shift<value_type, divisor>::multiplicand, &high_part);
		return static_cast<value_type>(high_part >> integral_constant<value_type, mul_and_shift<value_type, divisor>::shift - 64ULL>{});
#endif
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
		const uint32_t aa = static_cast<uint32_t>((value * mul_and_shift<uint32_t, 100>::multiplicand) >> mul_and_shift<value_type, 100>::shift);
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
		const uint32_t aa = (value * 3518437209ULL) >> 45ULL;
		const uint32_t lz = value < 100000U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		const uint32_t remainder = value - aa * 10000U;
		std::memcpy(buf + 2ULL, int32_table + remainder, 4ULL);
#else
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint32_t aa	= (value * 3518437209ULL) >> 45ULL;
		const uint32_t bbcc = value - aa * 10000U;
		const uint32_t bb	= (bbcc * 2748779070ULL) >> 38ULL;
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
		const uint32_t aabb = (value * 3518437209ULL) >> 45ULL;
		const uint32_t aa	= (aabb * 2748779070ULL) >> 38ULL;
		const uint32_t lz	= value < 10000000U;
		std::memcpy(buf, char_table_ptr + (aa * 2U + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, int16_table + (aabb - aa * 100U), 2ULL);
		const uint32_t ccdd = value - aabb * 10000U;
		std::memcpy(buf + 4ULL, int32_table + ccdd, 4ULL);
#else
		const uint32_t aabb = (value * 3518437209ULL) >> 45ULL;
		const uint32_t ccdd = value - aabb * 10000U;
		const uint32_t aa	= (aabb * 2748779070ULL) >> 38ULL;
		const uint32_t cc	= (ccdd * 2748779070ULL) >> 38ULL;
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
		const uint32_t high = (value * 2882303762ULL) >> 58ULL;
		const uint32_t low	= value - high * 100000000U;
		const uint32_t lz	= high < 10U;
		std::memcpy(buf, char_table_ptr + (high * 2U + lz), 2ULL);
		buf -= lz;
		const uint32_t aabb = (low * 3518437209ULL) >> 45ULL;
		const uint32_t ccdd = low - aabb * 10000U;
		std::memcpy(buf + 2ULL, int32_table + aabb, 4ULL);
		std::memcpy(buf + 6ULL, int32_table + ccdd, 4ULL);
#else
		BNCH_SWT_ALIGN(64ULL) static constexpr const uint16_t* int16_table{ fiwb<void>::charTable02 };
		const uint32_t aabbcc = (value * 3518437209ULL) >> 45ULL;
		const uint32_t aa	  = (aabbcc * 3518437209ULL) >> 45ULL;
		const uint32_t ddee	  = value - aabbcc * 10000U;
		const uint32_t bbcc	  = aabbcc - aa * 10000U;
		const uint32_t bb	  = (bbcc * 2748779070ULL) >> 38ULL;
		const uint32_t dd	  = (ddee * 2748779070ULL) >> 38ULL;
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

template<typename value_type> using function_ptr_type = decltype(&to_chars_impl<value_type, 2>::impl);

template<typename value_type> static constexpr auto function_ptrs{ [] {
	std::array<function_ptr_type<value_type>, 21> return_values{};
	return_values[1]  = &to_chars_impl<value_type, 2>::impl;
	return_values[2]  = &to_chars_impl<value_type, 2>::impl;
	return_values[3] = &to_chars_impl<value_type, 4>::impl;
	return_values[4] = &to_chars_impl<value_type, 4>::impl;
	return_values[5] = &to_chars_impl<value_type, 6>::impl;
	return_values[6] = &to_chars_impl<value_type, 6>::impl;
	return_values[7] = &to_chars_impl<value_type, 8>::impl;
	return_values[8] = &to_chars_impl<value_type, 8>::impl;
	return_values[9] = &to_chars_impl<value_type, 10>::impl;
	return_values[10] = &to_chars_impl<value_type, 10>::impl;
	return_values[11] = &to_chars_impl<value_type, 12>::impl;
	return_values[12] = &to_chars_impl<value_type, 12>::impl;
	return_values[13] = &to_chars_impl<value_type, 14>::impl;
	return_values[14] = &to_chars_impl<value_type, 14>::impl;
	return_values[15] = &to_chars_impl<value_type, 16>::impl;
	return_values[16] = &to_chars_impl<value_type, 16>::impl;
	return_values[17] = &to_chars_impl<value_type, 18>::impl;
	return_values[18] = &to_chars_impl<value_type, 18>::impl;
	return_values[19] = &to_chars_impl<value_type, 20>::impl;
	return_values[20] = &to_chars_impl<value_type, 20>::impl;
	return return_values;
}() };

template<typename value_type> struct to_chars;

template<concepts::uns64_t value_type> struct to_chars<value_type> {
	BNCH_SWT_HOST static char* impl(char* buf, const value_type value) noexcept {
		uint64_t digit_count{ fastDigitCount(value) };
		return function_ptrs<value_type>[digit_count](buf, value);
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
		uint64_t digit_count{ fastDigitCount(value) };
		switch ((digit_count + 1) & ~1) {
			case 2:
				return to_chars_impl<value_type, 2ULL>::impl(buf, value);
			case 4:
				return to_chars_impl<value_type, 4ULL>::impl(buf, value);
			case 6:
				return to_chars_impl<value_type, 6ULL>::impl(buf, value);
			case 8:
				return to_chars_impl<value_type, 8ULL>::impl(buf, value);
			default:
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
	testFunction<int64_t, "int64-test-0-to-5", 0, 5, 10000ULL>();
	testFunction<int64_t, "int64-test-0-to-10", 0, 10, 10000ULL>();
	testFunction<int64_t, "int64-test-0-to-15", 0, 15, 10000ULL>();
	testFunction<int64_t, "int64-test-0-to-20", 0, 20, 10000ULL>();
	testFunction<int64_t, "int64-test-5-to-10", 5, 10, 10000ULL>();
	testFunction<int64_t, "int64-test-5-to-15", 5, 15, 10000ULL>();
	testFunction<int64_t, "int64-test-5-to-20", 5, 20, 10000ULL>();
	testFunction<int64_t, "int64-test-10-to-15", 10, 15, 10000ULL>();
	testFunction<int64_t, "int64-test-10-to-20", 10, 19, 10000ULL>();
	testFunction<int64_t, "int64-test-15-to-20", 15, 19, 10000ULL>();
	testFunction<int64_t, "int64-test-20", 20, 19, 10000ULL>();
	testFunction<uint64_t, "uint64-test-0-to-5", 0, 5, 10000ULL>();
	testFunction<uint64_t, "uint64-test-0-to-10", 0, 10, 10000ULL>();
	testFunction<uint64_t, "uint64-test-0-to-15", 0, 15, 10000ULL>();
	testFunction<uint64_t, "uint64-test-0-to-20", 0, 20, 10000ULL>();
	testFunction<uint64_t, "uint64-test-5-to-10", 5, 10, 10000ULL>();
	testFunction<uint64_t, "uint64-test-5-to-15", 5, 15, 10000ULL>();
	testFunction<uint64_t, "uint64-test-5-to-20", 5, 20, 10000ULL>();
	testFunction<uint64_t, "uint64-test-10-to-15", 10, 15, 10000ULL>();
	testFunction<uint64_t, "uint64-test-10-to-20", 10, 20, 10000ULL>();
	testFunction<uint64_t, "uint64-test-15-to-20", 15, 20, 10000ULL>();
	testFunction<uint64_t, "uint64-test-20", 20, 20, 10000ULL>();
	testFunction<int32_t, "int32-test-0-to-5", 0, 5, 10000ULL>();
	testFunction<int32_t, "int32-test-0-to-10", 0, 10, 10000ULL>();
	testFunction<int32_t, "int32-test-5-to-10", 5, 10, 10000ULL>();
	testFunction<uint32_t, "uint32-test-0-to-5", 0, 5, 10000ULL>();
	testFunction<uint32_t, "uint32-test-0-to-10", 0, 10, 10000ULL>();
	testFunction<uint32_t, "uint32-test-5-to-10", 5, 10, 10000ULL>();
	return 0;
}