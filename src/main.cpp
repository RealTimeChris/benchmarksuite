#include "DragonBox.hpp"
#include <bnch_swt/index.hpp>
#include <cstring>
#include <random>

constexpr char char_table[200] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1',
	'4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0', '3', '1',
	'3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4',
	'9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6',
	'6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3', '8',
	'4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

template<typename value_type>
concept uns64_t = std::is_integral_v<value_type> && std::unsigned_integral<value_type>;

template<typename value_type>
concept sig64_t = std::is_integral_v<value_type> && std::signed_integral<value_type>;

template<class value_type_new, value_type_new valueNew> struct integral_constant {
	static constexpr value_type_new value = valueNew;

	using value_type = value_type_new;
	using type		 = integral_constant;

	BNCH_SWT_HOST constexpr operator value_type() const noexcept {
		return value;
	}

	BNCH_SWT_HOST constexpr value_type operator()() const noexcept {
		return value;
	}
};

template<typename value_type_new, uint64_t alignment = 8> struct BNCH_SWT_ALIGN(alignment) aligned_const {
	using value_type = value_type_new;
	value_type value{};

	BNCH_SWT_HOST constexpr aligned_const() {
	}
	BNCH_SWT_HOST constexpr aligned_const(const value_type& v) : value(v) {
	}
	BNCH_SWT_HOST constexpr aligned_const(value_type&& v) : value(std::move(v)) {
	}

	BNCH_SWT_HOST constexpr operator const value_type&() const& {
		return value;
	}

	BNCH_SWT_HOST explicit constexpr operator value_type&() & {
		return value;
	}

	BNCH_SWT_HOST explicit constexpr operator value_type&&() && {
		return std::move(value);
	}

	BNCH_SWT_HOST constexpr const value_type* get() const {
		return &value;
	}

	BNCH_SWT_HOST constexpr value_type* get() {
		return &value;
	}

	BNCH_SWT_HOST constexpr const value_type& operator*() const {
		return value;
	}

	BNCH_SWT_HOST constexpr value_type& operator*() {
		return value;
	}

	template<typename value_type_newer> BNCH_SWT_HOST constexpr void emplace(value_type_newer&& value_new) {
		value = std::forward<value_type_newer>(value_new);
	}

	BNCH_SWT_HOST constexpr value_type multiply(const aligned_const& other) const {
		return value * other.value;
	}

	BNCH_SWT_HOST constexpr bool operator==(const aligned_const& other) const {
		return value == other.value;
	}

	BNCH_SWT_HOST constexpr bool operator!=(const aligned_const& other) const {
		return value != other.value;
	}

	BNCH_SWT_HOST constexpr bool operator<(const aligned_const& other) const {
		return value < other.value;
	}

	BNCH_SWT_HOST constexpr bool operator>(const aligned_const& other) const {
		return value > other.value;
	}
};

template<typename value_type> aligned_const(value_type) -> aligned_const<value_type>;

template<typename typeName> struct fiwb {
	inline static constexpr char charTable01[]{ 0x30, 0x30, 0x30, 0x31, 0x30, 0x32, 0x30, 0x33, 0x30, 0x34, 0x30, 0x35, 0x30, 0x36, 0x30, 0x37, 0x30, 0x38, 0x30, 0x39, 0x31, 0x30,
		0x31, 0x31, 0x31, 0x32, 0x31, 0x33, 0x31, 0x34, 0x31, 0x35, 0x31, 0x36, 0x31, 0x37, 0x31, 0x38, 0x31, 0x39, 0x32, 0x30, 0x32, 0x31, 0x32, 0x32, 0x32, 0x33, 0x32, 0x34,
		0x32, 0x35, 0x32, 0x36, 0x32, 0x37, 0x32, 0x38, 0x32, 0x39, 0x33, 0x30, 0x33, 0x31, 0x33, 0x32, 0x33, 0x33, 0x33, 0x34, 0x33, 0x35, 0x33, 0x36, 0x33, 0x37, 0x33, 0x38,
		0x33, 0x39, 0x34, 0x30, 0x34, 0x31, 0x34, 0x32, 0x34, 0x33, 0x34, 0x34, 0x34, 0x35, 0x34, 0x36, 0x34, 0x37, 0x34, 0x38, 0x34, 0x39, 0x35, 0x30, 0x35, 0x31, 0x35, 0x32,
		0x35, 0x33, 0x35, 0x34, 0x35, 0x35, 0x35, 0x36, 0x35, 0x37, 0x35, 0x38, 0x35, 0x39, 0x36, 0x30, 0x36, 0x31, 0x36, 0x32, 0x36, 0x33, 0x36, 0x34, 0x36, 0x35, 0x36, 0x36,
		0x36, 0x37, 0x36, 0x38, 0x36, 0x39, 0x37, 0x30, 0x37, 0x31, 0x37, 0x32, 0x37, 0x33, 0x37, 0x34, 0x37, 0x35, 0x37, 0x36, 0x37, 0x37, 0x37, 0x38, 0x37, 0x39, 0x38, 0x30,
		0x38, 0x31, 0x38, 0x32, 0x38, 0x33, 0x38, 0x34, 0x38, 0x35, 0x38, 0x36, 0x38, 0x37, 0x38, 0x38, 0x38, 0x39, 0x39, 0x30, 0x39, 0x31, 0x39, 0x32, 0x39, 0x33, 0x39, 0x34,
		0x39, 0x35, 0x39, 0x36, 0x39, 0x37, 0x39, 0x38, 0x39, 0x39 };
	inline static constexpr uint16_t charTable02[]{ 0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930, 0x3031, 0x3131, 0x3231, 0x3331, 0x3431, 0x3531,
		0x3631, 0x3731, 0x3831, 0x3931, 0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932, 0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533, 0x3633,
		0x3733, 0x3833, 0x3933, 0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934, 0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635, 0x3735,
		0x3835, 0x3935, 0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936, 0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737, 0x3837,
		0x3937, 0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938, 0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839, 0x3939 };
	inline static constexpr auto charTable04{ [] {
		std::array<uint32_t, 10000> return_values{};
		for (uint32_t i = 0; i < 10000; ++i) {
			uint32_t d0	   = i / 1000;
			uint32_t d1	   = (i / 100) % 10;
			uint32_t d2	   = (i / 10) % 10;
			uint32_t d3	   = i % 10;
			return_values[i] = (0x30 + d0) | ((0x30 + d1) << 8) | ((0x30 + d2) << 16) | ((0x30 + d3) << 24);
		}
		return return_values;
	}() };
};

template<uint64_t shift, std::integral value_type>
BNCH_SWT_HOST constexpr value_type operator<<(const value_type arg, integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg << shift_amount;
}

template<uint64_t shift, std::integral value_type>
BNCH_SWT_HOST constexpr value_type& operator<<=(value_type& arg, integral_constant<uint64_t, shift>) noexcept {
	return arg = arg << integral_constant<uint64_t, shift>{};
}

template<uint64_t shift, std::integral value_type>
BNCH_SWT_HOST constexpr value_type operator>>(const value_type arg, integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg >> shift_amount;
}

template<uint64_t shift, std::integral value_type>
BNCH_SWT_HOST constexpr value_type& operator>>=(value_type& arg, integral_constant<uint64_t, shift>) noexcept {
	return arg = arg >> integral_constant<uint64_t, shift>{};
}

template<uint64_t multiplier, uint64_t shift> struct multiply_and_shift {
	template<typename value_type> BNCH_SWT_HOST static uint64_t impl(value_type value) noexcept {
#if BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
		const __uint128_t product = static_cast<__uint128_t>(value) * multiplier;
		return static_cast<uint64_t>(product >> integral_constant<uint64_t, shift>{});
#elif BNCH_SWT_COMPILER_MSVC
		uint64_t high_part;
		uint64_t low_part = _umul128(multiplier, value, &high_part);
		if constexpr (shift < 64ULL) {
			return static_cast<uint64_t>(
				(low_part >> integral_constant<uint64_t, shift>{}) | (high_part << integral_constant<uint64_t, 64ULL - shift>{}));
		} else {
			return static_cast<uint64_t>(high_part >> integral_constant<uint64_t, shift - 64ULL>{});
		}
#else
		uint64_t high_part;
		const uint64_t low_part = mul128Generic(value, multiplier, &high_part);
		if constexpr (shift < 64ULL) {
			return static_cast<uint64_t>(
				(low_part >> integral_constant<uint64_t, shift>{}) | (high_part << integral_constant<uint64_t, 64ULL - shift>{}));
		} else {
			return static_cast<uint64_t>(high_part >> integral_constant<uint64_t, shift - 64ULL>{});
		}
#endif
	}
};

template<uint64_t digit_length> struct to_chars_impl;

template<> struct to_chars_impl<2> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table  = fiwb<void>;
		const uint64_t lz = value < 10ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (value * 2ULL + lz), 2ULL);
		buf -= lz;
		return buf + 2ULL;
	}
};
template<> struct to_chars_impl<4> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		const uint64_t aa = (value * 5243ULL) >> integral_constant<uint64_t, 19ULL>{};
		const uint64_t lz = value < 1000ULL;
		std::memcpy(buf, fiwb<void>::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, fiwb<void>::charTable02 + (value - aa * 100ULL), 2ULL);
		return buf + 4ULL;
	}
};
template<> struct to_chars_impl<6> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table  = fiwb<void>;
		uint64_t aa		  = (value * 429497ULL) >> integral_constant<uint64_t, 32ULL>{};
		const uint64_t lz = value < 100000ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t remainder = value - aa * 10000ULL;
		std::memcpy(buf + 2ULL, &fiwb_table::charTable04[remainder], 4ULL);
		return buf + 6ULL;
	}
};
template<> struct to_chars_impl<8> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table  = fiwb<void>;
		uint64_t aabb	  = (value * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		uint64_t aa		  = (aabb * 5243ULL) >> integral_constant<uint64_t, 19ULL>{};
		const uint64_t lz = value < 10000000ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, fiwb_table::charTable02 + (aabb - aa * 100ULL), 2ULL);
		const uint64_t ccdd = value - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		return buf + 8ULL;
	}
};
template<> struct to_chars_impl<10> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	= fiwb<void>;
		const uint64_t high = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		const uint64_t lz	= high < 10ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (high * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t aabb = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 2ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 6ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		return buf + 10ULL;
	}
};
template<> struct to_chars_impl<12> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	= fiwb<void>;
		const uint64_t high = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		uint64_t aa			= (high * 5243ULL) >> integral_constant<uint64_t, 19ULL>{};
		const uint64_t lz	= aa < 10ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, fiwb_table::charTable02 + (high - aa * 100ULL), 2ULL);
		const uint64_t aabb = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 8ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		return buf + 12ULL;
	}
};
template<> struct to_chars_impl<14> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	= fiwb<void>;
		const uint64_t high = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		uint64_t aa			= (high * 429497ULL) >> integral_constant<uint64_t, 32ULL>{};
		const uint64_t lz	= aa < 10ULL;
		const uint64_t bbcc = high - aa * 10000ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, &fiwb_table::charTable04[bbcc], 4ULL);
		const uint64_t aabb = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 6ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 10ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		return buf + 14ULL;
	}
};
template<> struct to_chars_impl<16> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	= fiwb<void>;
		const uint64_t high = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	= value - high * 100000000ULL;
		uint64_t aabb		= (high * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		uint64_t ccdd		= high - aabb * 10000ULL;
		uint64_t aa			= (aabb * 5243ULL) >> integral_constant<uint64_t, 19ULL>{};
		const uint64_t lz	= aa < 10ULL;
		const uint64_t bb	= aabb - aa * 100ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, fiwb_table::charTable02 + bb, 2ULL);
		std::memcpy(buf + 4ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		aabb = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		ccdd = low - aabb * 10000ULL;
		std::memcpy(buf + 8ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 12ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		return buf + 16ULL;
	}
};
template<> struct to_chars_impl<18> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	  = fiwb<void>;
		const uint64_t high	  = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	  = value - high * 100000000ULL;
		const uint64_t high10 = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(high);
		const uint64_t low10  = high - high10 * 100000000ULL;
		const uint64_t lz	  = high10 < 10ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (high10 * 2ULL + lz), 2ULL);
		buf -= lz;
		const uint64_t aabb = (low10 * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t ccdd = low10 - aabb * 10000ULL;
		std::memcpy(buf + 2ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 6ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		const uint64_t eeff = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t gghh = low - eeff * 10000ULL;
		std::memcpy(buf + 10ULL, &fiwb_table::charTable04[eeff], 4ULL);
		std::memcpy(buf + 14ULL, &fiwb_table::charTable04[gghh], 4ULL);
		return buf + 18ULL;
	}
};
template<> struct to_chars_impl<20> {
	BNCH_SWT_HOST static char* impl(char* buf, const uint64_t value) noexcept {
		using fiwb_table	  = fiwb<void>;
		const uint64_t high	  = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(value);
		const uint64_t low	  = value - high * 100000000ULL;
		const uint64_t high12 = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(high);
		const uint64_t low12  = high - high12 * 100000000ULL;
		uint64_t aa			  = (high12 * 5243ULL) >> integral_constant<uint64_t, 19ULL>{};
		const uint64_t lz	  = aa < 10ULL;
		std::memcpy(buf, fiwb_table::charTable01 + (aa * 2ULL + lz), 2ULL);
		buf -= lz;
		std::memcpy(buf + 2ULL, fiwb_table::charTable02 + (high12 - aa * 100ULL), 2ULL);
		const uint64_t aabb = (low12 * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t ccdd = low12 - aabb * 10000ULL;
		std::memcpy(buf + 4ULL, &fiwb_table::charTable04[aabb], 4ULL);
		std::memcpy(buf + 8ULL, &fiwb_table::charTable04[ccdd], 4ULL);
		const uint64_t eeff = (low * 109951163ULL) >> integral_constant<uint64_t, 40ULL>{};
		const uint64_t gghh = low - eeff * 10000ULL;
		std::memcpy(buf + 12ULL, &fiwb_table::charTable04[eeff], 4ULL);
		std::memcpy(buf + 16ULL, &fiwb_table::charTable04[gghh], 4ULL);
		return buf + 20ULL;
	}
};


template<typename value_type> struct to_chars;

template<std::integral value_type> struct to_chars<value_type> {
	template<uns64_t value_type_new> BNCH_SWT_HOST static char* impl(char* buf, const value_type_new value) noexcept {
		if (value < 10000ULL) {
			if (value < 100ULL) {
				return to_chars_impl<2>::impl(buf, value);
			} else {
				return to_chars_impl<4>::impl(buf, value);
			}
		} else if (value < 100000000ULL) {
			if (value < 1000000ULL) {
				return to_chars_impl<6>::impl(buf, value);
			} else {
				return to_chars_impl<8>::impl(buf, value);
			}
		} else if (value < 1000000000000ULL) {
			if (value < 10000000000ULL) {
				return to_chars_impl<10>::impl(buf, value);
			} else {
				return to_chars_impl<12>::impl(buf, value);
			}
		} else if (value < 10000000000000000ULL) {
			if (value < 100000000000000ULL) {
				return to_chars_impl<14>::impl(buf, value);
			} else {
				return to_chars_impl<16>::impl(buf, value);
			}
		} else if (value < 1000000000000000000ULL) {
			return to_chars_impl<18>::impl(buf, value);
		} else {
			return to_chars_impl<20>::impl(buf, value);
		}
	}

	template<sig64_t value_type_new> BNCH_SWT_HOST static char* impl(char* buf, const value_type_new value) noexcept {
		constexpr auto shift_amount = sizeof(value_type_new) * 8 - 1;
		using unsigned_type			= std::make_unsigned_t<value_type_new>;
		*buf						= '-';
		return to_chars::impl(buf + (value < 0), static_cast<uint64_t>((static_cast<unsigned_type>(value) ^ (value >> shift_amount)) - (value >> shift_amount)));
	}
};

inline constexpr uint8_t decTrailingZeroTable[] = { 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

BNCH_SWT_HOST auto* writeu64Len15To17Trim(auto* buf, uint64_t sig) noexcept {
	uint32_t tz1, tz2, tz;
	const uint64_t abbccddee = multiply_and_shift<6189700196426901375ULL, 89ULL>::impl(sig);
	const uint64_t ffgghhii	 = sig - abbccddee * 100000000;
	uint32_t abbcc			 = abbccddee / 10000;
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

BNCH_SWT_HOST static int64_t abs_new(int64_t value) noexcept {
	const uint64_t temp = static_cast<uint64_t>(value >> 63);
	value ^= temp;
	value += temp & 1;
	return value;
}

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
			std::memcpy(buf, "null", 4);
			return buf + 4;
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
			exp_dec = abs_new(exp_dec);
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

template<class value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, uint32_t>
auto* to_chars_glz(auto* buf, value_type val) noexcept {
	/* The maximum value of uint32_t is 4294967295 (10 digits), */
	/* these digits are named as 'aabbccddee' here.             */
	uint32_t aa, bb, cc, dd, ee, aabb, bbcc, ccdd, ddee, aabbcc;

	/* Leading zero count in the first pair.                    */
	uint32_t lz;

	/* Although most compilers may convert the "division by     */
	/* constant value" into "multiply and shift", manual        */
	/* conversion can still help some compilers generate        */
	/* fewer and better instructions.                           */

	if (val < 100) { /* 1-2 digits: aa */
		lz = val < 10;
		std::memcpy(buf, char_table + (val * 2 + lz), 2);
		buf -= lz;
		return buf + 2;
	} else if (val < 10000) { /* 3-4 digits: aabb */
		aa = (val * 5243) >> 19; /* (val / 100) */
		bb = val - aa * 100; /* (val % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + (aa * 2 + lz), 2);
		buf -= lz;
		std::memcpy(&buf[2], char_table + (2 * bb), 2);

		return buf + 4;
	} else if (val < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		bbcc = val - aa * 10000; /* (val % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else if (val < 100000000) { /* 7~8 digits: aabbccdd */
		/* (val / 10000) */
		aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		ccdd = val - aabb * 10000; /* (val % 10000) */
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
		/* (val / 10000) */
		aabbcc = uint32_t((uint64_t(val) * 3518437209ul) >> 45);
		/* (aabbcc / 10000) */
		aa	 = uint32_t((uint64_t(aabbcc) * 429497) >> 32);
		ddee = val - aabbcc * 10000; /* (val % 10000) */
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
BNCH_SWT_HOST auto* to_chars_u64_len_8(auto* buf, value_type val) noexcept {
	/* 8 digits: aabbccdd */
	const uint32_t aabb = uint32_t((uint64_t(val) * 109951163) >> 40); /* (val / 10000) */
	const uint32_t ccdd = val - aabb * 10000; /* (val % 10000) */
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
BNCH_SWT_HOST auto* to_chars_u64_len_4(auto* buf, value_type val) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (val * 5243) >> 19; /* (val / 100) */
	const uint32_t bb = val - aa * 100; /* (val % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	return buf + 4;
}

template<class value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
inline auto* to_chars_u64_len_1_8(auto* buf, value_type val) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (val < 100) { /* 1-2 digits: aa */
		lz = val < 10;
		std::memcpy(buf, char_table + val * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (val < 10000) { /* 3-4 digits: aabb */
		aa = (val * 5243) >> 19; /* (val / 100) */
		bb = val - aa * 100; /* (val % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		return buf + 4;
	} else if (val < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		bbcc = val - aa * 10000; /* (val % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (val / 10000) */
		aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		ccdd = val - aabb * 10000; /* (val % 10000) */
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
auto* to_chars_u64_len_5_8(auto* buf, value_type val) noexcept {
	if (val < 1000000) { /* 5-6 digits: aabbcc */
		const uint32_t aa	= uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		const uint32_t bbcc = val - aa * 10000; /* (val % 10000) */
		const uint32_t bb	= (bbcc * 5243) >> 19; /* (bbcc / 100) */
		const uint32_t cc	= bbcc - bb * 100; /* (bbcc % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (val / 10000) */
		const uint32_t aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		const uint32_t ccdd = val - aabb * 10000; /* (val % 10000) */
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
auto* to_chars_glz(auto* buf, value_type val) noexcept {
	if (val < 100000000) { /* 1-8 digits */
		buf = to_chars_u64_len_1_8(buf, uint32_t(val));
		return buf;
	} else if (val < 100000000ULL * 100000000ULL) { /* 9-16 digits */
		const uint64_t hgh = val / 100000000;
		const auto low	   = uint32_t(val - hgh * 100000000); /* (val % 100000000) */
		buf				   = to_chars_u64_len_1_8(buf, uint32_t(hgh));
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = val / 100000000;
		const auto low	   = uint32_t(val - tmp * 100000000); /* (val % 100000000) */
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

BNCH_SWT_HOST std::string generate_integer_part(uint64_t min_length = 1, uint64_t max_length = 15) {
	uint64_t length = bnch_swt::random_generator<uint64_t>::impl(min_length, max_length);

	if (length == 1 && bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0 && bnch_swt::random_generator<uint64_t>::impl(0, 9) == 0)
		return "0";

	std::string s;
	s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(1, 9));

	for (uint64_t i = 1; i < length; ++i) {
		s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(0, 9));
	}
	return s;
}

BNCH_SWT_HOST std::string maybe_add_sign(const std::string& s) {
	return (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 1) ? ("-" + s) : s;
}

BNCH_SWT_HOST std::string generate_1_simple_integer() {
	return maybe_add_sign(generate_integer_part(1, 10));
}

BNCH_SWT_HOST std::string generate_2_simple_float() {
	std::string s = generate_integer_part(1, 5);
	s += ".";

	uint64_t fractional_length = bnch_swt::random_generator<uint64_t>::impl(1, 10);
	for (uint64_t i = 0; i < fractional_length; ++i) {
		s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(0, 9));
	}
	return maybe_add_sign(s);
}

BNCH_SWT_HOST std::string generate_3_scientific() {
	std::string s;

	if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0) {
		s = generate_integer_part(1, 3) + ".";
		s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(0, 9));
	} else {
		s = generate_integer_part(1, 5);
	}

	s += (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0 ? 'e' : 'E');
	s += (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0 ? '+' : '-');
	uint64_t exponent = bnch_swt::random_generator<uint64_t>::impl(1, 100);
	s += std::to_string(exponent);

	return maybe_add_sign(s);
}

BNCH_SWT_HOST std::string generate_4_min_max_boundary() {
	if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0) {
		double mantissa	  = bnch_swt::random_generator<double>::impl(1.0, 9.9);
		uint64_t exponent = bnch_swt::random_generator<uint64_t>::impl(300, 308);
		double val		  = mantissa * std::pow(10.0, exponent);
		if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 1)
			val = -val;

		std::stringstream ss;
		ss << std::scientific << std::setprecision(16) << val;
		return ss.str();
	} else {
		double mantissa	  = bnch_swt::random_generator<double>::impl(1.0, 9.9);
		uint64_t exponent = bnch_swt::random_generator<uint64_t>::impl(300, 308);
		double val		  = mantissa * std::pow(10.0, static_cast<double>(-static_cast<int64_t>(exponent)));
		if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 1)
			val = -val;

		std::stringstream ss;
		ss << std::scientific << std::setprecision(16) << val;
		return ss.str();
	}
}

BNCH_SWT_HOST std::string generate_5_precision_boundary() {
	std::string s;
	s += maybe_add_sign(std::to_string(bnch_swt::random_generator<uint64_t>::impl(1, 9)));
	s += ".";

	for (uint64_t i = 0; i < 18; ++i) {
		s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(0, 9));
	}

	if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0) {
		s += 'e';
		s += std::to_string(bnch_swt::random_generator<uint64_t>::impl(1, 100));
	}
	return s;
}

BNCH_SWT_HOST std::string generate_6_zero_subnormal() {
	if (bnch_swt::random_generator<uint64_t>::impl(0, 1) == 0) {
		static constexpr std::array zero_forms = { "0", "0.0", "-0.0", "0e0", "-0e5", "0.0e-10" };
		uint64_t index						   = bnch_swt::random_generator<uint64_t>::impl(0, zero_forms.size() - 1);
		return zero_forms[index];
	} else {
		double mantissa = bnch_swt::random_generator<double>::impl(1.0, 9.9);
		double val		= mantissa * std::pow(10.0, -315);

		std::stringstream ss;
		ss << std::scientific << std::setprecision(16) << val;
		return maybe_add_sign(ss.str());
	}
}

BNCH_SWT_HOST std::string generate_7_structural_edge() {
	static constexpr std::array edge_forms = { "1e10", "-9e-10", "0.1", "-9.0", "1.2e0", "-3e+0", "123.000000", "0.0000001" };
	uint64_t index						   = bnch_swt::random_generator<uint64_t>::impl(0, edge_forms.size() - 1);
	return edge_forms[index];
}

BNCH_SWT_HOST std::string generate_random_double_string() {
	static constexpr std::array weights = { 40.0, 30.0, 10.0, 5.0, 5.0, 5.0, 5.0 };

	static constexpr std::array generators = { generate_1_simple_integer, generate_2_simple_float, generate_3_scientific, generate_4_min_max_boundary,
		generate_5_precision_boundary, generate_6_zero_subnormal, generate_7_structural_edge };

	uint64_t roll = bnch_swt::random_generator<uint64_t>::impl(0, 99);

	uint64_t cumulative_weight = 0;
	for (size_t i = 0; i < weights.size(); ++i) {
		cumulative_weight += static_cast<uint64_t>(weights[i]);
		if (roll < cumulative_weight) {
			return generators[i]();
		}
	}
	return generators.back()();
}

BNCH_SWT_HOST double generate_random_double() {
	double test_double;
	do {
		std::string test_string = generate_random_double_string();
		auto new_ptr			= test_string.data() + test_string.size();
		test_double				= strtod(test_string.data(), &new_ptr);
	} while (test_double == std::numeric_limits<double>::infinity() || test_double == std::numeric_limits<double>::quiet_NaN() ||
		test_double == -std::numeric_limits<double>::infinity());
	return test_double;
}

template<typename value_type> std::vector<value_type> generate_digit_vector(size_t vector_size) {
	std::vector<value_type> return_values{};
	return_values.resize(vector_size);
	for (size_t i = 0; i < vector_size; ++i) {
		return_values[i] = bnch_swt::random_generator<value_type>::impl(std::numeric_limits<value_type>::min(), std::numeric_limits<value_type>::max());
	}
	return return_values;
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
	using benchmark = bnch_swt::benchmark_stage<stage_name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, true>;

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
		randomIntegers[x] = generate_digit_vector<value_type>(count);
	}
	std::vector<std::vector<std::string>> resultsReal{};
	std::vector<std::vector<std::string>> resultsTest{};
	resultsReal.resize(max_iterations);
	resultsTest.resize(max_iterations);
	for (uint64_t x = 0; x < max_iterations; ++x) {
		resultsTest[x].resize(count);
		resultsReal[x].resize(count);

		for (uint64_t y = 0; y < count; ++y) {
			resultsReal[x][y] = std::to_string(randomIntegers[x][y]);
			resultsTest[x][y].resize(resultsReal[x][y].size());
		}
	}
	uint64_t currentIndex{};
	run_and_validate<name, benchmark_std_to_string<value_type>, "std::to_string">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	run_and_validate<name, benchmark_std_to_chars<value_type>, "std::to_chars">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	run_and_validate<name, benchmark_jsonifier_to_chars<value_type>, "toChars">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	bnch_swt::benchmark_stage<name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, true>::print_results(true, true);
}

int32_t main() {
	testFunction<float, "float", 0, 5, 10000>();
	testFunction<double, "double", 0, 10, 10000>();
	return 0;
}
