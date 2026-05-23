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

#include <bnch_swt-incl/concepts.hpp>
#include <type_traits>
#include <cstddef>
#include <utility>
#include <random>
#include <array>

namespace bnch_swt {

	[[maybe_unused]] BNCH_SWT_HOST static uint64_t get_time_based_seed() noexcept {
		return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::nano>>(clock_type::now().time_since_epoch()).count();
	}

	enum class xoshiro_256_seeds : uint64_t {
		deterministic,
		time_based = std::numeric_limits<uint64_t>::max(),
	};

	template<xoshiro_256_seeds xoshiro_256_seed> struct xoshiro_256_base {
		std::array<uint64_t, 4> state{};

		BNCH_SWT_HOST constexpr xoshiro_256_base() {
			uint64_t s = 0;
			if constexpr (xoshiro_256_seed == xoshiro_256_seeds::time_based) {
				s = get_time_based_seed();
			} else {
				s = static_cast<uint64_t>(xoshiro_256_seed);
			}

			for (uint64_t y = 0; y < 4; ++y) {
				state[y] = splitmix64(s);
			}

			this->operator()();
			this->operator()();
		}

		BNCH_SWT_HOST constexpr uint64_t operator()() noexcept {
			const uint64_t result = rotl(state[0] + state[3], 23) + state[0];
			const uint64_t t	  = state[1] << 17;

			state[2] ^= state[0];
			state[3] ^= state[1];
			state[1] ^= state[2];
			state[0] ^= state[3];

			state[2] ^= t;
			state[3] = rotl(state[3], 45);

			return result;
		}

		using result_type = uint64_t;
		static constexpr uint64_t min() {
			return 0;
		}
		static constexpr uint64_t max() {
			return UINT64_MAX;
		}

	  protected:
		BNCH_SWT_HOST constexpr uint64_t rotl(const uint64_t x, const uint64_t k) const noexcept {
			return (x << k) | (x >> (64 - k));
		}

		BNCH_SWT_HOST constexpr uint64_t splitmix64(uint64_t& seed64) const noexcept {
			seed64 += 0x9E3779B97F4A7C15ULL;
			uint64_t z = seed64;
			z		   = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
			z		   = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
			return z ^ (z >> 31);
		}
	};

	template<typename value_type_new, xoshiro_256_seeds xoshiro_256_seed> struct xoshiro_256 : public xoshiro_256_base<xoshiro_256_seed> {
		using value_type = std::make_unsigned_t<value_type_new>;

		BNCH_SWT_HOST value_type_new operator()(value_type_new min, value_type_new max) {
			if (min >= max) {
				return min;
			}

			value_type range = static_cast<value_type>(max) - static_cast<value_type>(min);

			if (range == std::numeric_limits<value_type>::max()) {
				return static_cast<value_type_new>(xoshiro_256_base<xoshiro_256_seed>::operator()());
			}

			constexpr uint64_t max_val = std::numeric_limits<uint64_t>::max();
			const uint64_t bucket_size = range + 1;
			const uint64_t threshold   = (max_val / bucket_size) * bucket_size;

			uint64_t result;
			do {
				result = xoshiro_256_base<xoshiro_256_seed>::operator()();
			} while (result >= threshold);

			return static_cast<value_type_new>(static_cast<value_type>(min) + (result % bucket_size));
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

	template<internal::floating_point_t value_type, xoshiro_256_seeds xoshiro_256_seed> struct xoshiro_256<value_type, xoshiro_256_seed>
		: public xoshiro_256_base<xoshiro_256_seed> {
		BNCH_SWT_HOST value_type operator()(value_type min, value_type max) {
			return min + (max - min) * next();
		}

	  protected:
		BNCH_SWT_HOST value_type next() {
			return static_cast<value_type>(
				(xoshiro_256_base<xoshiro_256_seed>::operator()() >> xoshiro_256_traits<value_type>::shift) * xoshiro_256_traits<value_type>::multiplicand);
		}
	};

	template<typename value_type, xoshiro_256_seeds xoshiro_256_seeds = xoshiro_256_seeds::time_based> struct random_generator;

	template<bnch_swt::internal::string_t value_type, xoshiro_256_seeds xoshiro_256_seed> struct random_generator<value_type, xoshiro_256_seed>
		: public xoshiro_256<uint64_t, xoshiro_256_seed> {
		BNCH_SWT_HOST value_type impl(uint64_t length) {
			value_type result{};
			result.resize(length);
			for (uint64_t x = 0; x < length; ++x) {
				result[x] = static_cast<char>(xoshiro_256<uint64_t, xoshiro_256_seed>::operator()(32, 127));
			}
			return result;
		}
	};

	template<bnch_swt::internal::bool_t value_type, xoshiro_256_seeds xoshiro_256_seed> struct random_generator<value_type, xoshiro_256_seed>
		: public xoshiro_256<uint64_t, xoshiro_256_seed> {
		BNCH_SWT_HOST value_type impl() {
			return static_cast<value_type>(xoshiro_256<uint64_t, xoshiro_256_seed>::operator()(0, 1));
		}
	};

	template<bnch_swt::internal::floating_point_t value_type, xoshiro_256_seeds xoshiro_256_seed> struct random_generator<value_type, xoshiro_256_seed>
		: public xoshiro_256<value_type, xoshiro_256_seed> {
		BNCH_SWT_HOST value_type impl(value_type min = static_cast<value_type>(-1.0), value_type max = static_cast<value_type>(1.0)) {
			return xoshiro_256<value_type, xoshiro_256_seed>::operator()(min, max);
		}
	};

	template<bnch_swt::internal::integer_t value_type, xoshiro_256_seeds xoshiro_256_seed>
		requires(std::is_unsigned_v<value_type>)
	struct random_generator<value_type, xoshiro_256_seed> : public xoshiro_256<value_type, xoshiro_256_seed> {
		BNCH_SWT_HOST value_type impl(value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()) {
			return static_cast<value_type>(xoshiro_256<value_type, xoshiro_256_seed>::operator()(min, max));
		}
	};

	template<bnch_swt::internal::integer_t value_type, xoshiro_256_seeds xoshiro_256_seed>
		requires(std::is_signed_v<value_type>)
	struct random_generator<value_type, xoshiro_256_seed> : public xoshiro_256<value_type, xoshiro_256_seed> {
		BNCH_SWT_HOST value_type impl(value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()) {
			return xoshiro_256<value_type, xoshiro_256_seed>::operator()(min, max);
		}
	};

}
