#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <bit>

#if BNCH_SWT_AVX2
inline static constexpr uint64_t bitsPerStep{ 256 };
#elif BNCH_SWT_AVX512
inline static constexpr uint64_t bitsPerStep{ 512 };
#elif BNCH_SWT_NEON
inline static constexpr uint64_t bitsPerStep{ 128 };
#else
inline constexpr uint64_t bitsPerStep{ 128 };
#endif

inline constexpr uint64_t bytesPerRegister{ bitsPerStep / 8 };
inline constexpr uint64_t registersPerBlock{ 64 / bytesPerRegister };
inline constexpr uint64_t bytesPerBlock{ bytesPerRegister * registersPerBlock };

#if BNCH_SWT_COMPILER_MSVC
static constexpr uint64_t jsonifierTapeMax		 = 8;
static constexpr uint64_t jsonifierBlocksPerStep = 8;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
#elif BNCH_SWT_COMPILER_CLANG
	#if BNCH_SWT_ARCH_ARM64
static constexpr uint64_t jsonifierTapeMax		 = 16;
static constexpr uint64_t jsonifierBlocksPerStep = 2;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
	 #else
static constexpr uint64_t jsonifierTapeMax		 = 8;
static constexpr uint64_t jsonifierBlocksPerStep = 4;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
	 #endif
#elif BNCH_SWT_COMPILER_GCC
	#if BNCH_SWT_ARCH_ARM64
static constexpr uint64_t jsonifierTapeMax		 = 16;
static constexpr uint64_t jsonifierBlocksPerStep = 2;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
	 #else
static constexpr uint64_t jsonifierTapeMax		 = 8;
static constexpr uint64_t jsonifierBlocksPerStep = 4;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
	 #endif
#else
static constexpr uint64_t jsonifierTapeMax		 = 8;
static constexpr uint64_t jsonifierBlocksPerStep = 8;
static constexpr uint64_t jsonifierBytesPerStep	 = jsonifierBlocksPerStep * bytesPerBlock;
#endif

#if BNCH_SWT_AVX2
	#include <immintrin.h>
using jsonifier_simd_int_t = __m256i;
#elif BNCH_SWT_AVX512
	#include <immintrin.h>
using jsonifier_simd_int_t = __m512i;
#elif BNCH_SWT_NEON
	#include <arm_neon.h>
using jsonifier_simd_int_t = uint8x16_t;
#else
	#include <emmintrin.h>
	#include <tmmintrin.h>
	#include <smmintrin.h>
using jsonifier_simd_int_t = __m128i;
#endif

namespace jsonifier_internal {

	inline jsonifier_simd_int_t opSetzero() noexcept {
#if BNCH_SWT_AVX2
		return _mm256_setzero_si256();
#elif BNCH_SWT_AVX512
		return _mm512_setzero_si512();
#elif BNCH_SWT_NEON
		return vdupq_n_u8(0);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_setzero_si128();
#endif
	}

	template<typename jsonifier_simd_int_t> inline jsonifier_simd_int_t gatherValue(uint8_t value) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_set1_epi8(static_cast<char>(value));
#elif BNCH_SWT_AVX512
		return _mm512_set1_epi8(static_cast<char>(value));
#elif BNCH_SWT_NEON
		return vdupq_n_u8(value);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_set1_epi8(static_cast<char>(value));
#endif
	}

	template<typename jsonifier_simd_int_t> inline jsonifier_simd_int_t gatherValues(const void* src) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_load_si256(reinterpret_cast<const __m256i*>(src));
#elif BNCH_SWT_AVX512
		return _mm512_load_si512(reinterpret_cast<const void*>(src));
#elif BNCH_SWT_NEON
		return vld1q_u8(reinterpret_cast<const uint8_t*>(src));
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_load_si128(reinterpret_cast<const __m128i*>(src));
#endif
	}

	template<typename jsonifier_simd_int_t> inline jsonifier_simd_int_t gatherValuesU(const void* src) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
#elif BNCH_SWT_AVX512
		return _mm512_loadu_si512(reinterpret_cast<const void*>(src));
#elif BNCH_SWT_NEON
		return vld1q_u8(reinterpret_cast<const uint8_t*>(src));
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
#endif
	}

	inline jsonifier_simd_int_t opOr(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_or_si256(a, b);
#elif BNCH_SWT_AVX512
		return _mm512_or_si512(a, b);
#elif BNCH_SWT_NEON
		return vorrq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_or_si128(a, b);
#endif
	}

	inline jsonifier_simd_int_t opAnd(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_and_si256(a, b);
#elif BNCH_SWT_AVX512
		return _mm512_and_si512(a, b);
#elif BNCH_SWT_NEON
		return vandq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_and_si128(a, b);
#endif
	}

	inline jsonifier_simd_int_t opAndNot(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_andnot_si256(b, a);
#elif BNCH_SWT_AVX512
		return _mm512_andnot_si512(b, a);
#elif BNCH_SWT_NEON
		return vbicq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_andnot_si128(b, a);
#endif
	}

	inline jsonifier_simd_int_t opNot(jsonifier_simd_int_t a) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_xor_si256(a, _mm256_set1_epi8(static_cast<char>(0xFF)));
#elif BNCH_SWT_AVX512
		return _mm512_xor_si512(a, _mm512_set1_epi8(static_cast<char>(0xFF)));
#elif BNCH_SWT_NEON
		return vmvnq_u8(a);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_xor_si128(a, _mm_set1_epi8(static_cast<char>(0xFF)));
#endif
	}

	inline jsonifier_simd_int_t opXor(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_xor_si256(a, b);
#elif BNCH_SWT_AVX512
		return _mm512_xor_si512(a, b);
#elif BNCH_SWT_NEON
		return veorq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_xor_si128(a, b);
#endif
	}

	inline jsonifier_simd_int_t opShuffle(jsonifier_simd_int_t table, jsonifier_simd_int_t indices) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_shuffle_epi8(table, indices);
#elif BNCH_SWT_AVX512
		return _mm512_shuffle_epi8(table, indices);
#elif BNCH_SWT_NEON
		const jsonifier_simd_int_t bitMask{ vdupq_n_u8(0x0F) };
		return vqtbl1q_u8(table, vandq_u8(indices, bitMask));
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_shuffle_epi8(table, indices);
#endif
	}

	template<int shift> inline jsonifier_simd_int_t opSrLi(jsonifier_simd_int_t a) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_srli_epi16(a, shift);
#elif BNCH_SWT_AVX512
		return _mm512_srli_epi16(a, shift);
#elif BNCH_SWT_NEON
		return vshrq_n_u8(a, shift);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_srli_epi16(a, shift);
#endif
	}

	inline jsonifier_simd_int_t opSubs(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_subs_epu8(a, b);
#elif BNCH_SWT_AVX512
		return _mm512_subs_epu8(a, b);
#elif BNCH_SWT_NEON
		return vqsubq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_subs_epu8(a, b);
#endif
	}

	template<int imm> inline jsonifier_simd_int_t opPermute2x128(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_permute2x128_si256(a, b, imm);
#elif BNCH_SWT_AVX512
		return _mm512_shuffle_i32x4(a, b, imm);
#elif BNCH_SWT_NEON || NIHILUS_SIMD_BACKEND_SSE
		( void )a;
		( void )b;
		( void )imm;
		return a;
#endif
	}

	template<int nbytes> inline jsonifier_simd_int_t opAlignr(jsonifier_simd_int_t hi, jsonifier_simd_int_t lo) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_alignr_epi8(hi, lo, nbytes);
#elif BNCH_SWT_AVX512
		return _mm512_alignr_epi8(hi, lo, nbytes);
#elif BNCH_SWT_NEON
		return vextq_u8(lo, hi, nbytes);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_alignr_epi8(hi, lo, nbytes);
#endif
	}

	inline uint64_t opBitMask(jsonifier_simd_int_t a) noexcept {
#if BNCH_SWT_AVX2
		return static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(a)));
#elif BNCH_SWT_AVX512
		return static_cast<uint64_t>(_mm512_movepi8_mask(a));
#elif BNCH_SWT_NEON
		constexpr uint8x16_t bitMask{ 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80 };
		const uint8x16_t minput = vandq_u8(a, bitMask);
		uint8x16_t tmp			= vpaddq_u8(minput, minput);
		tmp						= vpaddq_u8(tmp, tmp);
		tmp						= vpaddq_u8(tmp, tmp);
		return static_cast<uint64_t>(vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0));
#elif NIHILUS_SIMD_BACKEND_SSE
		return static_cast<uint64_t>(static_cast<uint32_t>(_mm_movemask_epi8(a)));
#endif
	}

	inline bool opTest(jsonifier_simd_int_t a) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_testz_si256(a, a) != 0;
#elif BNCH_SWT_AVX512
		return _mm512_test_epi64_mask(a, a) == 0;
#elif BNCH_SWT_NEON
		return vmaxvq_u8(a) == 0;
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_testz_si128(a, a) != 0;
#endif
	}

	inline jsonifier_simd_int_t opCmpEqRaw(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_cmpeq_epi8(a, b);
#elif BNCH_SWT_AVX512
		return _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(a, b));
#elif BNCH_SWT_NEON
		return vceqq_u8(a, b);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_cmpeq_epi8(a, b);
#endif
	}

	inline jsonifier_simd_int_t opCmpLtRaw(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
#if BNCH_SWT_AVX2
		return _mm256_cmpgt_epi8(b, a);
#elif BNCH_SWT_AVX512
		return _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(b, a));
#elif BNCH_SWT_NEON
		return vcgtq_u8(b, a);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_cmpgt_epi8(b, a);
#endif
	}

	inline uint64_t opCmpEq(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
		return opBitMask(opCmpEqRaw(a, b));
	}

	inline uint64_t opCmpLt(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
		return opBitMask(opCmpLtRaw(a, b));
	}

	inline uint64_t opCmpEqBitMask(jsonifier_simd_int_t a, jsonifier_simd_int_t b) noexcept {
		return opBitMask(opCmpEqRaw(a, b));
	}

	inline jsonifier_simd_int_t opSetLSB(jsonifier_simd_int_t a, bool valueNew) noexcept {
#if BNCH_SWT_AVX2
		alignas(32) static constexpr uint8_t maskRaw[32]{ 0x01 };
		const jsonifier_simd_int_t mask = gatherValues<jsonifier_simd_int_t>(maskRaw);
		return valueNew ? opOr(a, mask) : opAndNot(a, mask);
#elif BNCH_SWT_AVX512
		alignas(64) static constexpr uint8_t maskRaw[64]{ 0x01 };
		const jsonifier_simd_int_t mask = gatherValues<jsonifier_simd_int_t>(maskRaw);
		return valueNew ? opOr(a, mask) : opAndNot(a, mask);
#elif BNCH_SWT_NEON
		constexpr uint8x16_t mask{ 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
		return valueNew ? vorrq_u8(a, mask) : vbicq_u8(a, mask);
#elif NIHILUS_SIMD_BACKEND_SSE
		alignas(16) static constexpr uint8_t maskRaw[16]{ 0x01 };
		const jsonifier_simd_int_t mask = gatherValues<jsonifier_simd_int_t>(maskRaw);
		return valueNew ? opOr(a, mask) : opAndNot(a, mask);
#endif
	}

	inline bool opGetMSB(jsonifier_simd_int_t a) noexcept {
#if BNCH_SWT_AVX2
		return (static_cast<uint32_t>(_mm256_movemask_epi8(a)) & (1u << 31)) != 0;
#elif BNCH_SWT_AVX512
		return (_mm512_movepi8_mask(a) & (1ull << 63)) != 0;
#elif BNCH_SWT_NEON
		return (vgetq_lane_u8(a, 15) & 0x80) != 0;
#elif NIHILUS_SIMD_BACKEND_SSE
		return (static_cast<uint32_t>(_mm_movemask_epi8(a)) & (1u << 15)) != 0;
#endif
	}

	template<typename value_type> inline value_type postCmpTzcnt(value_type value) noexcept {
		return tzcnt(value) >> 2;
	}

	template<int alignment> inline jsonifier_simd_int_t opPrev(jsonifier_simd_int_t cur, jsonifier_simd_int_t prev) {
#if BNCH_SWT_AVX2
		return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), alignment);
#elif BNCH_SWT_AVX512
		return _mm512_alignr_epi8(cur, _mm512_permutex2var_epi64(prev, _mm512_set_epi64(13, 12, 11, 10, 9, 8, 7, 6), cur), alignment);
#elif BNCH_SWT_NEON
		return vextq_u8(prev, cur, alignment);
#elif NIHILUS_SIMD_BACKEND_SSE
		return _mm_alignr_epi8(cur, prev, alignment);
#endif
	}

	template<typename simd_type> BNCH_SWT_HOST bool isAscii(simd_type value) {
#if BNCH_SWT_AVX2 || BNCH_SWT_AVX512
		return opBitMask(value) == 0;
#else
		return vmaxvq_u8(value) < 0x80u;
#endif
	}

	inline bool anyBitsSetAnywhere(jsonifier_simd_int_t a) noexcept {
		return !opTest(a);
	}

	template<uint64_t size> struct simd_array {
		jsonifier_simd_int_t values[size];

		template<uint64_t index_new> BNCH_SWT_HOST jsonifier_simd_int_t get() const noexcept {
			constexpr uint64_t index{ index_new % size };
			return values[index];
		}

		template<uint64_t index_new> BNCH_SWT_HOST void set(jsonifier_simd_int_t v) noexcept {
			constexpr uint64_t index{ index_new % size };
			values[index] = v;
		}
	};

	static constexpr uint8_t TOO_SHORT		= 1 << 0;
	static constexpr uint8_t TOO_LONG		= 1 << 1;
	static constexpr uint8_t OVERLONG_3		= 1 << 2;
	static constexpr uint8_t TOO_LARGE		= 1 << 3;
	static constexpr uint8_t SURROGATE		= 1 << 4;
	static constexpr uint8_t OVERLONG_2		= 1 << 5;
	static constexpr uint8_t TWO_CONTS		= 1 << 7;
	static constexpr uint8_t TOO_LARGE_1000 = 1 << 6;
	static constexpr uint8_t OVERLONG_4		= 1 << 6;
	static constexpr uint8_t CARRY			= TOO_SHORT | TOO_LONG | TWO_CONTS;

	static constexpr uint8_t byte_1_high_table_raw[]{ TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
		TOO_SHORT | OVERLONG_2, TOO_SHORT, TOO_SHORT | OVERLONG_3 | SURROGATE, TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4 };

	alignas(bytesPerRegister) static constexpr std::array<uint8_t, bytesPerRegister> byte_1_high_table{ [] {
		std::array<uint8_t, bytesPerRegister> return_value{};
		for (uint64_t x = 0; x < bytesPerRegister; ++x) {
			return_value[x] = byte_1_high_table_raw[x % std::size(byte_1_high_table_raw)];
		}
		return return_value;
	}() };

	static constexpr uint8_t byte_1_low_table_raw[]{ CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4, CARRY | OVERLONG_2, CARRY, CARRY, CARRY | TOO_LARGE,
		CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000,
		CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000,
		CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000 };

	alignas(bytesPerRegister) static constexpr std::array<uint8_t, bytesPerRegister> byte_1_low_table{ [] {
		std::array<uint8_t, bytesPerRegister> return_value{};
		for (uint64_t x = 0; x < bytesPerRegister; ++x) {
			return_value[x] = byte_1_low_table_raw[x % std::size(byte_1_low_table_raw)];
		}
		return return_value;
	}() };

	static constexpr uint8_t byte_2_high_table_raw[]{ TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
		TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4, TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
		TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE, TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT };

	alignas(bytesPerRegister) static constexpr std::array<uint8_t, bytesPerRegister> byte_2_high_table{ [] {
		std::array<uint8_t, bytesPerRegister> return_value{};
		for (uint64_t x = 0; x < bytesPerRegister; ++x) {
			return_value[x] = byte_2_high_table_raw[x % std::size(byte_2_high_table_raw)];
		}
		return return_value;
	}() };

	alignas(bytesPerRegister) static constexpr uint8_t is_incomplete_max[64] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, static_cast<uint8_t>(0xF0u - 1), static_cast<uint8_t>(0xE0u - 1), static_cast<uint8_t>(0xC0u - 1) };

	template<uint64_t total_chunks> struct chunk_loader {
		template<size_t... i> BNCH_SWT_HOST static simd_array<total_chunks> load_impl(const uint8_t* src, std::index_sequence<i...>) noexcept {
			simd_array<total_chunks> result;
			((result.template set<i>(gatherValuesU<jsonifier_simd_int_t>(reinterpret_cast<const jsonifier_simd_int_t*>(src + i * bytesPerRegister)))), ...);
			return result;
		}

		BNCH_SWT_HOST static simd_array<total_chunks> load(const uint8_t* src) noexcept {
			return load_impl(src, std::make_index_sequence<total_chunks>{});
		}
	};

	template<uint64_t total_chunks> struct ascii_checker {
		template<size_t... i> BNCH_SWT_HOST static jsonifier_simd_int_t or_impl(const simd_array<total_chunks>& chunks, std::index_sequence<i...>) noexcept {
			jsonifier_simd_int_t result = chunks.template get<0>();
			((result = opOr(result, chunks.template get<i + 1>())), ...);
			return result;
		}

		BNCH_SWT_HOST static jsonifier_simd_int_t or_all(const simd_array<total_chunks>& chunks) noexcept {
			return or_impl(chunks, std::make_index_sequence<total_chunks - 1>{});
		}
	};

	struct utf8_checker_new {
		jsonifier_simd_int_t prev_incomplete;
		jsonifier_simd_int_t prev_input;
		jsonifier_simd_int_t lookup_h;
		jsonifier_simd_int_t lookup_l;
		jsonifier_simd_int_t lookup_2;
		jsonifier_simd_int_t error;

		BNCH_SWT_HOST void reset() {
			prev_incomplete = jsonifier_simd_int_t{};
			prev_input		= jsonifier_simd_int_t{};
			lookup_h		= gatherValues<jsonifier_simd_int_t>(std::bit_cast<const jsonifier_simd_int_t*>(byte_1_high_table.data()));
			lookup_l		= gatherValues<jsonifier_simd_int_t>(std::bit_cast<const jsonifier_simd_int_t*>(byte_1_low_table.data()));
			lookup_2		= gatherValues<jsonifier_simd_int_t>(std::bit_cast<const jsonifier_simd_int_t*>(byte_2_high_table.data()));
			error			= jsonifier_simd_int_t{};
		}

		BNCH_SWT_HOST jsonifier_simd_int_t check_special_cases(jsonifier_simd_int_t input, jsonifier_simd_int_t p1) {
			const jsonifier_simd_int_t lo_nibble_mask = gatherValue<jsonifier_simd_int_t>(static_cast<char>(0x0Fu));
			const jsonifier_simd_int_t b1_hi		  = opShuffle(lookup_h, opAnd(opSrLi<4>(p1), lo_nibble_mask));
			const jsonifier_simd_int_t b1_lo		  = opShuffle(lookup_l, opAnd(p1, lo_nibble_mask));
			const jsonifier_simd_int_t b2_hi		  = opShuffle(lookup_2, opAnd(opSrLi<4>(input), lo_nibble_mask));
			return opAnd(opAnd(b1_hi, b1_lo), b2_hi);
		}

		BNCH_SWT_HOST jsonifier_simd_int_t must_be_2_3_continuation(jsonifier_simd_int_t p2, jsonifier_simd_int_t p3) {
			const jsonifier_simd_int_t is_3_or_4 = opSubs(p2, gatherValue<jsonifier_simd_int_t>(static_cast<char>(0xE0u - 0x80u)));
			const jsonifier_simd_int_t is_4		 = opSubs(p3, gatherValue<jsonifier_simd_int_t>(static_cast<char>(0xF0u - 0x80u)));
			return opOr(is_3_or_4, is_4);
		}

		BNCH_SWT_HOST jsonifier_simd_int_t check_multibyte_lengths(jsonifier_simd_int_t input, jsonifier_simd_int_t prev, jsonifier_simd_int_t sc) {
			const jsonifier_simd_int_t p2	  = opPrev<14>(input, prev);
			const jsonifier_simd_int_t p3	  = opPrev<13>(input, prev);
			const jsonifier_simd_int_t must23 = must_be_2_3_continuation(p2, p3);
			return opXor(opAnd(must23, gatherValue<jsonifier_simd_int_t>(static_cast<char>(0x80u))), sc);
		}

		BNCH_SWT_HOST jsonifier_simd_int_t check_incomplete(jsonifier_simd_int_t input) {
			const jsonifier_simd_int_t max_val = gatherValues<jsonifier_simd_int_t>(is_incomplete_max + (64 - bytesPerRegister));
			return opSubs(input, max_val);
		}

		BNCH_SWT_HOST void check_chunk(jsonifier_simd_int_t input, jsonifier_simd_int_t prev) {
			const jsonifier_simd_int_t p1 = opPrev<15>(input, prev);
			const jsonifier_simd_int_t sc = check_special_cases(input, p1);
			error						  = opOr(error, check_multibyte_lengths(input, prev, sc));
		}

		template<uint64_t total_chunks> struct chunk_processor {
			template<size_t... i>
			BNCH_SWT_HOST static void process_impl(utf8_checker_new& c, const simd_array<total_chunks>& chunks, jsonifier_simd_int_t& prev, std::index_sequence<i...>) noexcept {
				((c.check_chunk(chunks.template get<i>(), prev), prev = chunks.template get<i>()), ...);
			}

			BNCH_SWT_HOST static void process(utf8_checker_new& c, const simd_array<total_chunks>& chunks, jsonifier_simd_int_t& prev) noexcept {
				process_impl(c, chunks, prev, std::make_index_sequence<total_chunks>{});
			}
		};

		BNCH_SWT_HOST void checkBlock(simd_array<registersPerBlock> chunks) {
			jsonifier_simd_int_t ascii_or = ascii_checker<registersPerBlock>::or_all(chunks);

			if (isAscii(ascii_or)) {
				error			= opOr(error, prev_incomplete);
				prev_input		= chunks.template get<registersPerBlock - 1>();
				prev_incomplete = jsonifier_simd_int_t{};
				return;
			}

			jsonifier_simd_int_t prev = prev_input;
			chunk_processor<registersPerBlock>::process(*this, chunks, prev);

			prev_input		= chunks.template get<registersPerBlock - 1>();
			prev_incomplete = check_incomplete(chunks.template get<registersPerBlock - 1>());
		}

		BNCH_SWT_HOST void checkStep(const uint8_t* src_new) {
			const uint8_t* src_new_local{ src_new };
			for (uint64_t x = 0; x < jsonifierBlocksPerStep; ++x) {
				checkStepImpl(src_new_local + x * bytesPerBlock);
			}
		}

		BNCH_SWT_HOST void checkStepImpl(const uint8_t* src_new) {
			const uint8_t* src{ std::bit_cast<const uint8_t*>(src_new) };
			simd_array<registersPerBlock> chunks = chunk_loader<registersPerBlock>::load(src);

			jsonifier_simd_int_t ascii_or = ascii_checker<registersPerBlock>::or_all(chunks);

			if (isAscii(ascii_or)) {
				error			= opOr(error, prev_incomplete);
				prev_input		= chunks.template get<registersPerBlock - 1>();
				prev_incomplete = jsonifier_simd_int_t{};
				return;
			}

			jsonifier_simd_int_t prev = prev_input;
			chunk_processor<registersPerBlock>::process(*this, chunks, prev);

			prev_input		= chunks.template get<registersPerBlock - 1>();
			prev_incomplete = check_incomplete(chunks.template get<registersPerBlock - 1>());
		}

		BNCH_SWT_HOST bool errors() {
			const jsonifier_simd_int_t combined = opOr(error, prev_incomplete);
			return !opTest(combined);
		}
	};

	template<uint64_t bps> BNCH_SWT_HOST bool validate_utf8(const uint8_t* src_new, uint64_t len) {
		if (len == 0)
			return true;
		const uint8_t* src{ std::bit_cast<const uint8_t*>(src_new) };

		utf8_checker_new checker{};
		checker.reset();

		uint64_t i = 0;

		for (; i + jsonifierBytesPerStep <= len; i += jsonifierBytesPerStep)
			checker.checkStep(src + i);

		if (i < len) {
			alignas(32) uint8_t tmp[jsonifierBytesPerStep];
			std::memset(tmp, 0x41, jsonifierBytesPerStep);
			std::memcpy(tmp, src + i, len - i);
			checker.checkStep(tmp);
		}

		return !checker.errors();
	}

	inline bool validate_utf8(const uint8_t* src, uint64_t len) {
		return validate_utf8<jsonifierBlocksPerStep>(src, len);
	}

	template<typename T, typename... Args> BNCH_SWT_HOST constexpr T& getFirst(T& first, Args&...) noexcept {
		return first;
	}

	struct simd_string_reader {

		template<typename... utf8_validator_type> BNCH_SWT_HOST void resetImpl(const uint8_t* rootIter, uint64_t stringLength, utf8_validator_type&... validator) noexcept {
			static constexpr uint64_t stepBytes = jsonifierBytesPerStep;

			valid										   = true;
			tapecount									   = 0;
			length					   = stringLength;
			index					   = 0;

			while (index + stepBytes <= length) {
				processBlocks<false>(std::bit_cast<const uint8_t*>(rootIter) + index, index, validator...);
				index += stepBytes;
			}

			if (index < length) {
				const uint64_t remaining = length - index;
				uint8_t remainder[stepBytes];
				std::fill_n(remainder, stepBytes, static_cast<uint8_t>(0x20));
				std::copy_n(std::bit_cast<const uint8_t*>(rootIter) + index, remaining, remainder);
				processBlocks<true>(remainder, index, validator...);
				if constexpr (sizeof...(utf8_validator_type) > 0) {
					auto& validator_ref = getFirst(validator...);
					alignas(32) uint8_t tmp[jsonifierBytesPerStep];
					std::memset(tmp, 0x41, jsonifierBytesPerStep);
					std::memcpy(tmp, std::bit_cast<const uint8_t*>(rootIter) + index, remaining);
					validator_ref.checkStep(tmp);
				}

				index += stepBytes;
			}
			if constexpr (sizeof...(utf8_validator_type) > 0) {
				auto& validator_ref = getFirst(validator...);
				valid				= !validator_ref.errors();
			}
		}

		BNCH_SWT_HOST void reset(const uint8_t* rootIter, uint64_t stringLength) noexcept {
			utf8_checker_new utf8Checker{};
			utf8Checker.reset();
			resetImpl(rootIter, stringLength, utf8Checker);
		}

		BNCH_SWT_HOST uint64_t getTapecount() noexcept {
			return tapecount;
		}

		BNCH_SWT_HOST bool isValid() noexcept {
			return valid;
		}

		BNCH_SWT_HOST ~simd_string_reader() noexcept {
		}

	  protected:
		const char* tape{};
		uint64_t index{};
		uint64_t length{};
		uint64_t tapecount{};
		uint64_t capacity{};
		bool valid{ true };

		template<bool leftOver, uint64_t I, typename... utf8_validator_type> BNCH_SWT_HOST void processBlocksImpl(std::array<uint64_t, jsonifierBlocksPerStep>& bitsArr,
			std::array<uint64_t, jsonifierBlocksPerStep>& cntsArr, const uint8_t* blockPtr, [[maybe_unused]] utf8_validator_type&... validator) noexcept {
			simd_array<registersPerBlock> in_vals;
			in_vals.template set<0>(gatherValuesU<jsonifier_simd_int_t>(blockPtr + I * 64));
			if constexpr (registersPerBlock > 1) {
				in_vals.template set<1>(gatherValuesU<jsonifier_simd_int_t>(blockPtr + I * 64 + bytesPerRegister * 1));
				if constexpr (registersPerBlock > 2) {
					in_vals.template set<2>(gatherValuesU<jsonifier_simd_int_t>(blockPtr + I * 64 + bytesPerRegister * 2));
					in_vals.template set<3>(gatherValuesU<jsonifier_simd_int_t>(blockPtr + I * 64 + bytesPerRegister * 3));
				}
			}
			if constexpr (sizeof...(utf8_validator_type) > 0) {
				if constexpr (!leftOver) {
					auto& validator_ref = getFirst(validator...);
					validator_ref.checkBlock(in_vals);
				}
			}
		}

		template<bool leftOver, typename... utf8_validator_type> BNCH_SWT_HOST void processBlocks(const uint8_t* blockPtr, uint64_t stepBaseIndex, utf8_validator_type&... validator) noexcept {
			std::array<uint64_t, jsonifierBlocksPerStep> bitsArr{};
			std::array<uint64_t, jsonifierBlocksPerStep> cntsArr{};
			processBlocksImpl<leftOver, 0>(bitsArr, cntsArr, blockPtr, validator...);
			if constexpr (jsonifierBlocksPerStep > 1) {
				processBlocksImpl<leftOver, 1>(bitsArr, cntsArr, blockPtr, validator...);
				if constexpr (jsonifierBlocksPerStep > 2) {
					processBlocksImpl<leftOver, 2>(bitsArr, cntsArr, blockPtr,
						validator...);
					processBlocksImpl<leftOver, 3>(bitsArr, cntsArr, blockPtr,
						validator...);
					if constexpr (jsonifierBlocksPerStep > 4) {
						processBlocksImpl<leftOver, 4>(bitsArr, cntsArr, blockPtr,
							validator...);
						processBlocksImpl<leftOver, 5>(bitsArr, cntsArr, blockPtr,
							validator...);
						processBlocksImpl<leftOver, 6>(bitsArr, cntsArr, blockPtr,
							validator...);
						processBlocksImpl<leftOver, 7>(bitsArr, cntsArr, blockPtr,
							validator...);
					}
				}
			}

			tapecount += cntsArr[0];
			if constexpr (jsonifierBlocksPerStep > 1) {
				tapecount += cntsArr[1];
				if constexpr (jsonifierBlocksPerStep > 2) {
					tapecount += cntsArr[2];
					tapecount += cntsArr[3];
					if constexpr (jsonifierBlocksPerStep > 4) {
						tapecount += cntsArr[4];
						tapecount += cntsArr[5];
						tapecount += cntsArr[6];
						tapecount += cntsArr[7];
					}
				}
			}
		}
	};

}