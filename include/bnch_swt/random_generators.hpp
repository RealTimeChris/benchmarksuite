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
/// https://github.com/RealTimeChris/benchmarksuite
/// Feb 3, 2023
#pragma once

#include <bnch_swt/concepts.hpp>
#include <type_traits>
#include <cstddef>
#include <utility>
#include <random>
#include <array>

namespace bnch_swt {

	using clock_type	  = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;
	using duration_type	  = std::chrono::duration<double, std::nano>;
	using time_point_type = std::chrono::time_point<clock_type, duration_type>;

	BNCH_SWT_HOST static uint64_t get_time_based_seed() noexcept {
		if constexpr (std::is_same_v<std::chrono::duration<uint64_t, std::nano>, clock_type::duration>) {
			return static_cast<uint64_t>(clock_type::now().time_since_epoch().count());
		} else {
			return static_cast<uint64_t>(std::chrono::duration_cast<duration_type>(clock_type::now().time_since_epoch()).count());
		}
	}

	struct xoshiro_256_base {
		BNCH_SWT_HOST xoshiro_256_base() {
			uint64_t s = get_time_based_seed();
			for (uint64_t y = 0; y < 4; ++y) {
				state[y] = splitmix64(s);
			}
		}

		BNCH_SWT_HOST uint64_t operator()() noexcept {
			const uint64_t result = rotl(state[1ull] * 5ull, 7ull) * 9ull;
			const uint64_t t	  = state[1ull] << 17ull;

			state[2ull] ^= state[0ull];
			state[3ull] ^= state[1ull];
			state[1ull] ^= state[2ull];
			state[0ull] ^= state[3ull];

			state[2ull] ^= t;
			state[3ull] = rotl(state[3ull], 45ull);

			return result;
		}

	  protected:
		std::array<uint64_t, 4ull> state{};

		BNCH_SWT_HOST uint64_t rotl(const uint64_t x, const uint64_t k) const noexcept {
			return (x << k) | (x >> (64ull - k));
		}

		BNCH_SWT_HOST uint64_t splitmix64(uint64_t& seed64) const noexcept {
			uint64_t result = seed64 += 0x9E3779B97F4A7C15ull;
			result			= (result ^ (result >> 30ull)) * 0xBF58476D1CE4E5B9ull;
			result			= (result ^ (result >> 27ull)) * 0x94D049BB133111EBull;
			return result ^ (result >> 31ull);
		}
	};

	template<typename value_type> struct xoshiro_256 : public xoshiro_256_base {
		BNCH_SWT_HOST value_type operator()(value_type min, value_type max) {
			if (min >= max) {
				return min;
			}

			uint64_t range = static_cast<uint64_t>(max - min);

			if (range == std::numeric_limits<uint64_t>::max()) {
				return static_cast<value_type>(min + xoshiro_256_base::operator()());
			}

			uint64_t threshold = (std::numeric_limits<uint64_t>::max() / (range + 1)) * (range + 1);

			uint64_t result;
			do {
				result = xoshiro_256_base::operator()();
			} while (result >= threshold);

			return static_cast<value_type>(min + (result % (range + 1)));
		}
	};

	template<typename value_type> struct xoshiro_256_traits;

	template<typename value_type>
		requires(sizeof(value_type) == 4)
	struct xoshiro_256_traits<value_type> {
		static constexpr value_type multiplicand{ 0x1.0p-24 };
		static constexpr uint64_t shift{ 40 };
	};

	template<typename value_type>
		requires(sizeof(value_type) == 8)
	struct xoshiro_256_traits<value_type> {
		static constexpr value_type multiplicand{ 0x1.0p-53 };
		static constexpr uint64_t shift{ 11 };
	};

	template<internal::floating_point_t value_type> struct xoshiro_256<value_type> : public xoshiro_256_base {
		BNCH_SWT_HOST value_type operator()(value_type min, value_type max) {
			return min + (max - min) * next();
		}

	  protected:
		BNCH_SWT_HOST value_type next() {
			return static_cast<value_type>((xoshiro_256_base::operator()() >> xoshiro_256_traits<value_type>::shift) * xoshiro_256_traits<value_type>::multiplicand);
		}
	};

	template<typename value_type> struct random_generator;

	template<bnch_swt::internal::string_t value_type> struct random_generator<value_type> {
		static constexpr std::string_view charset{ "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\"\\\r\b\f\t\n" };
		BNCH_SWT_HOST static value_type impl(uint64_t length) {
			static thread_local xoshiro_256<uint64_t> random_engine{};
			value_type result{};
			result.resize(length);
			for (auto& c: result) {
				c = charset[random_engine(0, charset.size() - 1)];
			}
			return result;
		}
	};

	template<bnch_swt::internal::floating_point_t value_type> struct random_generator<value_type> {
		BNCH_SWT_HOST static value_type impl(value_type min = std::numeric_limits<value_type>::lowest(), value_type max = std::numeric_limits<value_type>::max()) {
			static thread_local xoshiro_256<value_type> random_engine{};
			return static_cast<value_type>(random_engine(min, max));
		}
	};
	
	template<bnch_swt::internal::bool_t value_type> struct random_generator<value_type> {
		BNCH_SWT_HOST static value_type impl() {
			static thread_local xoshiro_256<uint64_t> random_engine{};
			return static_cast<value_type>(random_engine(0, 1));
		}
	};

	template<bnch_swt::internal::integer_t value_type>
		requires(std::is_unsigned_v<value_type>)
	struct random_generator<value_type> {
		BNCH_SWT_HOST static value_type impl(value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()) {
			static thread_local xoshiro_256<value_type> random_engine{};
			return static_cast<value_type>(random_engine(min, max));
		}
	};

	template<bnch_swt::internal::integer_t value_type>
		requires(std::is_signed_v<value_type>)
	struct random_generator<value_type> {
		BNCH_SWT_HOST static value_type impl(value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()) {
			static thread_local xoshiro_256<value_type> random_engine{};

			using unsigned_type	 = std::make_unsigned_t<value_type>;
			unsigned_type umin	 = static_cast<unsigned_type>(min);
			unsigned_type umax	 = static_cast<unsigned_type>(max);
			unsigned_type urange = umax - umin;

			return static_cast<value_type>(random_engine(static_cast<value_type>(0), static_cast<value_type>(urange)) + min);
		}
	};	

}
