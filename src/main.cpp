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

template<typename value_type> BNCH_SWT_HOST value_type max_value_for_digits(uint64_t num_digits) noexcept {
	if (num_digits <= 0) {
		return value_type(0);
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
		return value_type(0);
	long double min_val = std::pow(10.0L, num_digits - 1);

	const value_type type_max = std::numeric_limits<value_type>::max();
	if (min_val >= type_max)
		return type_max;

	return static_cast<value_type>(std::min<long double>(static_cast<long double>(min_val), static_cast<long double>(type_max)));
}

template<typename value_type> value_type get_max_value_for_digits(uint64_t num_digits) noexcept {
	if (num_digits <= 0)
		return value_type(0);
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
			to_chars<uint64_t>::impl(resultsTest[current_index][x].data(), randomIntegers[current_index][x]);
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
	run_and_validate<name, benchmark_glz_to_chars<value_type>, "glz::to_chars">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	run_and_validate<name, benchmark_std_to_chars<value_type>, "std::to_chars">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	run_and_validate<name, benchmark_jsonifier_to_chars<value_type>, "toChars">(resultsTest, resultsReal, randomIntegers, count, currentIndex);
	currentIndex = 0;
	bnch_swt::benchmark_stage<name, max_iterations, measured_iterations, bnch_swt::benchmark_types::cpu, false>::print_results(true, true);
}

int32_t main() {
	testFunction<int32_t, "int32-test-0-to-5", 0, 5, 1000000>();
	testFunction<int32_t, "int32-test-0-to-10", 0, 10, 1000000>();
	testFunction<int32_t, "int32-test-0-to-15", 0, 15, 1000000>();
	testFunction<int32_t, "int32-test-0-to-20", 0, 20, 1000000>();
	testFunction<int32_t, "int32-test-5-to-10", 5, 10, 1000000>();
	testFunction<int32_t, "int32-test-5-to-15", 5, 15, 1000000>();
	testFunction<int32_t, "int32-test-5-to-20", 5, 20, 1000000>();
	testFunction<int32_t, "int32-test-10-to-15", 10, 15, 1000000>();
	testFunction<int32_t, "int32-test-10-to-20", 10, 19, 1000000>();
	testFunction<int32_t, "int32-test-15-to-20", 15, 19, 1000000>();
	testFunction<int32_t, "int32-test-20", 20, 19, 1000000>();
	testFunction<uint32_t, "uint32-test-0-to-5", 0, 5, 1000000>();
	testFunction<uint32_t, "uint32-test-0-to-10", 0, 10, 1000000>();
	testFunction<uint32_t, "uint32-test-0-to-15", 0, 15, 1000000>();
	testFunction<uint32_t, "uint32-test-0-to-20", 0, 20, 1000000>();
	testFunction<uint32_t, "uint32-test-5-to-10", 5, 10, 1000000>();
	testFunction<uint32_t, "uint32-test-5-to-15", 5, 15, 1000000>();
	testFunction<uint32_t, "uint32-test-5-to-20", 5, 20, 1000000>();
	testFunction<uint32_t, "uint32-test-10-to-15", 10, 15, 1000000>();
	testFunction<uint32_t, "uint32-test-10-to-20", 10, 20, 1000000>();
	testFunction<uint32_t, "uint32-test-15-to-20", 15, 20, 1000000>();
	testFunction<uint32_t, "uint32-test-20", 20, 20, 1000000>();
	testFunction<uint64_t, "uint64-test-0-to-5", 0, 5, 1000000>();
	testFunction<uint64_t, "uint64-test-0-to-10", 0, 10, 1000000>();
	testFunction<uint64_t, "uint64-test-0-to-15", 0, 15, 1000000>();
	testFunction<uint64_t, "uint64-test-0-to-20", 0, 20, 1000000>();
	testFunction<uint64_t, "uint64-test-5-to-10", 5, 10, 1000000>();
	testFunction<uint64_t, "uint64-test-5-to-15", 5, 15, 1000000>();
	testFunction<uint64_t, "uint64-test-5-to-20", 5, 20, 1000000>();
	testFunction<uint64_t, "uint64-test-10-to-15", 10, 15, 1000000>();
	testFunction<uint64_t, "uint64-test-10-to-20", 10, 20, 1000000>();
	testFunction<uint64_t, "uint64-test-15-to-20", 15, 20, 1000000>();
	testFunction<uint64_t, "uint64-test-20", 20, 20, 1000000>();
	testFunction<int64_t, "int64-test-0-to-5", 0, 5, 1000000>();
	testFunction<int64_t, "int64-test-0-to-10", 0, 10, 1000000>();
	testFunction<int64_t, "int64-test-0-to-15", 0, 15, 1000000>();
	testFunction<int64_t, "int64-test-0-to-20", 0, 20, 1000000>();
	testFunction<int64_t, "int64-test-5-to-10", 5, 10, 1000000>();
	testFunction<int64_t, "int64-test-5-to-15", 5, 15, 1000000>();
	testFunction<int64_t, "int64-test-5-to-20", 5, 20, 1000000>();
	testFunction<int64_t, "int64-test-10-to-15", 10, 15, 1000000>();
	testFunction<int64_t, "int64-test-10-to-20", 10, 19, 1000000>();
	testFunction<int64_t, "int64-test-15-to-20", 15, 19, 1000000>();
	testFunction<int64_t, "int64-test-20", 20, 19, 1000000>();
	return 0;
}
