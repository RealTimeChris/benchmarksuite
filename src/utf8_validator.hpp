#pragma once
#include <bnch_swt>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <utility>

namespace jsonifier_internal {

	static constexpr uint64_t bytes_per_chunk			 = 32;
	static constexpr uint64_t chunks_per_block_ascii	 = BNCH_SWT_CPB_ASCII;
	static constexpr uint64_t blocks_per_step_ascii		 = BNCH_SWT_BPS_ASCII;
	static constexpr uint64_t bytes_per_block_ascii		 = bytes_per_chunk * chunks_per_block_ascii;
	static constexpr uint64_t chunks_per_block_mixed	 = BNCH_SWT_CPB_MIXED;
	static constexpr uint64_t blocks_per_step_mixed		 = BNCH_SWT_BPS_MIXED;
	static constexpr uint64_t bytes_per_block_mixed		 = bytes_per_chunk * chunks_per_block_mixed;
	static constexpr uint64_t chunks_per_block_multibyte = BNCH_SWT_CPB_MULTIBYTE;
	static constexpr uint64_t blocks_per_step_multibyte	 = BNCH_SWT_BPS_MULTIBYTE;
	static constexpr uint64_t bytes_per_block_multibyte	 = bytes_per_chunk * chunks_per_block_multibyte;

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

	alignas(bnch_swt::cpu_properties::cpu_alignment) static constexpr std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> byte_1_high_table{ [] {
		std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> return_value{};
			for (uint64_t x = 0; x < bnch_swt::cpu_properties::cpu_alignment; ++x) {
			return_value[x] = byte_1_high_table_raw[x % std::size(byte_1_high_table_raw)];
		}
		return return_value;
	}() };

	static constexpr uint8_t byte_1_low_table_raw[]{ CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4, CARRY | OVERLONG_2, CARRY, CARRY, CARRY | TOO_LARGE,
		CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000,
		CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000,
		CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE, CARRY | TOO_LARGE | TOO_LARGE_1000, CARRY | TOO_LARGE | TOO_LARGE_1000 };

	alignas(bnch_swt::cpu_properties::cpu_alignment) static constexpr std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> byte_1_low_table{ [] {
		std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> return_value{};
		for (uint64_t x = 0; x < bnch_swt::cpu_properties::cpu_alignment; ++x) {
			return_value[x] = byte_1_low_table_raw[x % std::size(byte_1_low_table_raw)];
		}
		return return_value;
	}() };

	static constexpr uint8_t byte_2_high_table_raw[]{ TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
		TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4, TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
		TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE, TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT };

	alignas(bnch_swt::cpu_properties::cpu_alignment) static constexpr std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> byte_2_high_table{ [] {
		std::array<uint8_t, bnch_swt::cpu_properties::cpu_alignment> return_value{};
		for (uint64_t x = 0; x < bnch_swt::cpu_properties::cpu_alignment; ++x) {
			return_value[x] = byte_2_high_table_raw[x % std::size(byte_2_high_table_raw)];
		}
		return return_value;
	}() };

	alignas(32) static constexpr uint8_t is_incomplete_max[64] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, static_cast<uint8_t>(0xF0u - 1), static_cast<uint8_t>(0xE0u - 1), static_cast<uint8_t>(0xC0u - 1) };

	template<uint64_t n_blocks, uint64_t total_chunks> struct simd_array {
		__m256i values[total_chunks];

		template<uint64_t index> inline __m256i get() const noexcept {
			return values[index % total_chunks];
		}

		template<uint64_t index> inline void set(__m256i v) noexcept {
			values[index % total_chunks] = v;
		}
	};

	template<uint64_t n_blocks, uint64_t total_chunks> struct chunk_loader {
		template<uint64_t... i> static inline simd_array<n_blocks, total_chunks> load_impl(const uint8_t* src, std::index_sequence<i...>) noexcept {
			simd_array<n_blocks, total_chunks> result;
			((result.template set<i>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * bytes_per_chunk)))), ...);
			return result;
		}

		static inline simd_array<n_blocks, total_chunks> load(const uint8_t* src) noexcept {
			return load_impl(src, std::make_index_sequence<total_chunks>{});
		}
	};

	template<uint64_t n_blocks, uint64_t total_chunks> struct ascii_checker {
		template<uint64_t... i> static inline __m256i or_impl(const simd_array<n_blocks, total_chunks>& chunks, std::index_sequence<i...>) noexcept {
			__m256i result = chunks.template get<0>();
			((result = _mm256_or_si256(result, chunks.template get<i + 1>())), ...);
			return result;
		}

		static inline __m256i or_all(const simd_array<n_blocks, total_chunks>& chunks) noexcept {
			return or_impl(chunks, std::make_index_sequence<total_chunks - 1>{});
		}
	};

	struct utf8_checker {
		__m256i error;
		__m256i prev_input;
		__m256i prev_incomplete;
		__m256i lookup_h;
		__m256i lookup_l;
		__m256i lookup_2;

		inline void reset() {
			error			= _mm256_setzero_si256();
			prev_input		= _mm256_setzero_si256();
			prev_incomplete = _mm256_setzero_si256();
			lookup_h		= _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_1_high_table.data()));
			lookup_l		= _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_1_low_table.data()));
			lookup_2		= _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_2_high_table.data()));
		}

		inline __m256i prev1(__m256i cur, __m256i prev) {
			return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 15);
		}

		inline __m256i prev2(__m256i cur, __m256i prev) {
			return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 14);
		}

		inline __m256i prev3(__m256i cur, __m256i prev) {
			return _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(prev, cur, 0x21), 13);
		}

		inline __m256i check_special_cases(__m256i input, __m256i p1) {
			const __m256i lo_nibble_mask = _mm256_set1_epi8(static_cast<char>(0x0Fu));
			__m256i b1_hi				 = _mm256_shuffle_epi8(lookup_h, _mm256_and_si256(_mm256_srli_epi16(p1, 4), lo_nibble_mask));
			__m256i b1_lo				 = _mm256_shuffle_epi8(lookup_l, _mm256_and_si256(p1, lo_nibble_mask));
			__m256i b2_hi				 = _mm256_shuffle_epi8(lookup_2, _mm256_and_si256(_mm256_srli_epi16(input, 4), lo_nibble_mask));
			return _mm256_and_si256(_mm256_and_si256(b1_hi, b1_lo), b2_hi);
		}

		inline __m256i must_be_2_3_continuation(__m256i p2, __m256i p3) {
			__m256i is_3_or_4 = _mm256_subs_epu8(p2, _mm256_set1_epi8(static_cast<char>(0xE0u - 0x80u)));
			__m256i is_4	  = _mm256_subs_epu8(p3, _mm256_set1_epi8(static_cast<char>(0xF0u - 0x80u)));
			return _mm256_or_si256(is_3_or_4, is_4);
		}

		inline __m256i check_multibyte_lengths(__m256i input, __m256i prev, __m256i sc) {
			__m256i p2	   = prev2(input, prev);
			__m256i p3	   = prev3(input, prev);
			__m256i must23 = must_be_2_3_continuation(p2, p3);
			return _mm256_xor_si256(_mm256_and_si256(must23, _mm256_set1_epi8(static_cast<char>(0x80u))), sc);
		}

		inline __m256i check_incomplete(__m256i input) {
			const __m256i max_val = _mm256_load_si256(reinterpret_cast<const __m256i*>(is_incomplete_max + 32));
			return _mm256_subs_epu8(input, max_val);
		}

		inline void check_chunk(__m256i input, __m256i prev) {
			__m256i p1 = prev1(input, prev);
			__m256i sc = check_special_cases(input, p1);
			error	   = _mm256_or_si256(error, check_multibyte_lengths(input, prev, sc));
		}

		template<uint64_t n_blocks, uint64_t total_chunks> struct chunk_processor {
			template<uint64_t... i>
			static inline void process_impl(utf8_checker& c, const simd_array<n_blocks, total_chunks>& chunks, __m256i& prev, std::index_sequence<i...>) noexcept {
				((c.check_chunk(chunks.template get<i>(), prev), prev = chunks.template get<i>()), ...);
			}

			static inline void process(utf8_checker& c, const simd_array<n_blocks, total_chunks>& chunks, __m256i& prev) noexcept {
				process_impl(c, chunks, prev, std::make_index_sequence<total_chunks>{});
			}
		};

		template<uint64_t n_blocks, uint64_t total_chunks> inline void check_step(const uint8_t* src) {
			simd_array<n_blocks, total_chunks> chunks = chunk_loader<n_blocks, total_chunks>::load(src);

			__m256i ascii_or = ascii_checker<n_blocks, total_chunks>::or_all(chunks);

			if (_mm256_movemask_epi8(ascii_or) == 0) {
				error			= _mm256_or_si256(error, prev_incomplete);
				prev_input		= chunks.template get<total_chunks - 1>();
				prev_incomplete = _mm256_setzero_si256();
				return;
			}

			__m256i prev = prev_input;
			chunk_processor<n_blocks, total_chunks>::process(*this, chunks, prev);

			prev_input		= chunks.template get<total_chunks - 1>();
			prev_incomplete = check_incomplete(chunks.template get<total_chunks - 1>());
		}

		inline bool errors() {
			const __m256i combined = _mm256_or_si256(error, prev_incomplete);
			return !_mm256_testz_si256(combined, combined);
		}
	};

	template<uint64_t cpb, uint64_t bps> inline bool validate_utf8_impl(const uint8_t* src, uint64_t len) {
		if (len == 0)
			return true;

		constexpr uint64_t bytes_per_block = bytes_per_chunk * cpb;
		constexpr uint64_t bytes_per_step  = bytes_per_block * bps;

		utf8_checker checker{};
		checker.reset();

		uint64_t i = 0;
		for (; i + bytes_per_step <= len; i += bytes_per_step)
			checker.check_step<bps, bps * cpb>(src + i);

		for (; i + bytes_per_block <= len; i += bytes_per_block)
			checker.check_step<1, cpb>(src + i);

		if (i < len) {
			alignas(32) uint8_t tmp[bytes_per_block];
			std::memset(tmp, 0x41, bytes_per_block);
			std::memcpy(tmp, src + i, len - i);
			checker.check_step<1, cpb>(tmp);
		}

		return !checker.errors();
	}

	inline bool validate_utf8_ascii(const uint8_t* src, uint64_t len) {
		return validate_utf8_impl<chunks_per_block_ascii, blocks_per_step_ascii>(src, len);
	}

	inline bool validate_utf8_mixed(const uint8_t* src, uint64_t len) {
		return validate_utf8_impl<chunks_per_block_mixed, blocks_per_step_mixed>(src, len);
	}

	inline bool validate_utf8_multibyte(const uint8_t* src, uint64_t len) {
		return validate_utf8_impl<chunks_per_block_multibyte, blocks_per_step_multibyte>(src, len);
	}

	inline bool validate_utf8(const uint8_t* src, uint64_t len) {
		return validate_utf8_impl<chunks_per_block_mixed, blocks_per_step_mixed>(src, len);
	}

}