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
#include <bnch_swt>
#include <bnch_swt-incl/concepts.hpp>
#include <bit>
#include <cstdint>
#include <vector>

#if defined(__ARM_NEON)
	#include <arm_neon.h>
#endif

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
	#include <emmintrin.h>
	#define BITMASK_HAS_SSE2
#endif

using namespace bnch_swt;

struct simd_int_128 {
	union {
		uint64_t xUint64[2];
		uint8_t xUint8[16];
#if defined(__ARM_NEON)
		uint8x16_t neon;
#endif
#if defined(BITMASK_HAS_SSE2)
		__m128i sse;
#endif
	} values{};
};

namespace bitmask_internal {
	BNCH_SWT_HOST uint64_t byteswap(uint64_t v) noexcept {
#if defined(_MSC_VER)
		return _byteswap_uint64(v);
#else
		return __builtin_bswap64(v);
#endif
	}
}

static constexpr uint64_t benchmark_iterations{ 1000 };

struct test_bitmask_swar {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<simd_int_128>& inputs) {
		uint16_t sink{};
		for (uint64_t i = 0; i < benchmark_iterations; ++i) {
			uint64_t rawLo = inputs[i].values.xUint64[0];
			uint64_t rawHi = inputs[i].values.xUint64[1];
			if constexpr (std::endian::native == std::endian::big) {
				rawLo = bitmask_internal::byteswap(rawLo);
				rawHi = bitmask_internal::byteswap(rawHi);
			}
			uint64_t highBits0 = rawLo & 0x8080808080808080ull;
			uint64_t highBits1 = rawHi & 0x8080808080808080ull;
			uint16_t mask0	   = static_cast<uint16_t>((highBits0 * 0x0002040810204081ull) >> 56);
			uint16_t mask1	   = static_cast<uint16_t>((highBits1 * 0x0002040810204081ull) >> 56);
			sink			   = static_cast<uint16_t>(mask0 | (mask1 << 8));
			bnch_swt::do_not_optimize_away(sink);
		}
		return benchmark_iterations;
	}
};

#if defined(BITMASK_HAS_SSE2)
struct test_bitmask_sse_movemask {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<simd_int_128>& inputs) {
		uint16_t sink{};
		for (uint64_t i = 0; i < benchmark_iterations; ++i) {
			sink = static_cast<uint16_t>(_mm_movemask_epi8(inputs[i].values.sse));
			bnch_swt::do_not_optimize_away(sink);
		}
		return benchmark_iterations;
	}
};
#endif

#if defined(__ARM_NEON)
struct test_bitmask_neon {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<simd_int_128>& inputs) {
		uint16_t sink{};
		for (uint64_t i = 0; i < benchmark_iterations; ++i) {
			constexpr uint8x16_t bit_mask{ 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 };
			uint8x16_t minput = vandq_u8(inputs[i].values.neon, bit_mask);
			uint8x16_t tmp	  = vpaddq_u8(minput, minput);
			tmp				  = vpaddq_u8(tmp, tmp);
			tmp				  = vpaddq_u8(tmp, tmp);
			sink			  = vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0);
			bnch_swt::do_not_optimize_away(sink);
		}
		return benchmark_iterations;
	}
};
#endif

int main() {
	bnch_swt::random_generator<uint64_t> rng{};
	std::vector<simd_int_128> inputs(benchmark_iterations);
	for (auto& v: inputs) {
		v.values.xUint64[0] = rng.impl();
		v.values.xUint64[1] = rng.impl();
	}

	using stage_type = benchmark_stage<"bitmask_stage", stage_config_data{}>;
	bnch_swt::pin_for_benchmark();

	stage_type::run_benchmark<"bitmask-comparison", "swar_bitmask", test_bitmask_swar>(inputs);

#if defined(BITMASK_HAS_SSE2)
	stage_type::run_benchmark<"bitmask-comparison", "sse_movemask", test_bitmask_sse_movemask>(inputs);
#endif

#if defined(__ARM_NEON)
	stage_type::run_benchmark<"bitmask-comparison", "neon_bitmask", test_bitmask_neon>(inputs);
#endif

	auto test_rankings = stage_type::get_test_results("bitmask-comparison");
	std::cout << test_rankings.to_csv() << std::endl;

	auto all_rankings = stage_type::get_all_results();
	std::cout << all_rankings.to_csv() << std::endl;

	return 0;
}
