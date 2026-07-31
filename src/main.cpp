#include <benchmarksuite>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace benchmarksuite;

#if BNCH_SWT_AVX2

static constexpr uint64_t key_count{ 4096 };
static constexpr uint64_t buf_stride{ 32 };
static constexpr uint64_t max_len{ 32 };

using bytes_vec = std::vector<uint8_t>;

BNCH_SWT_HOST bytes_vec make_runtime_keys() {
	static random_generator<uint64_t> gen{};
	bytes_vec result(buf_stride * key_count);
	for (uint64_t x = 0; x < key_count; ++x) {
		uint8_t* dst = result.data() + (x * buf_stride);
		for (uint64_t i = 0; i < buf_stride; ++i) {
			dst[i] = static_cast<uint8_t>(gen.impl() & 0xFF);
		}
	}
	return result;
}

template<uint64_t len> BNCH_SWT_HOST static __m256i load_via_set(const uint8_t* str) noexcept {
	if constexpr (len == 1) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[0]));
	} else if constexpr (len == 2) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 3) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[2]), static_cast<char>(str[1]),
			static_cast<char>(str[0]));
	} else if constexpr (len == 4) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[3]), static_cast<char>(str[2]),
			static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 5) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[4]), static_cast<char>(str[3]),
			static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 6) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[5]), static_cast<char>(str[4]),
			static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 7) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[6]), static_cast<char>(str[5]),
			static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 8) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 9) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 10) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]),
			static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]),
			static_cast<char>(str[0]));
	} else if constexpr (len == 11) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]),
			static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]),
			static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 12) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]),
			static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]),
			static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 13) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]),
			static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]),
			static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 14) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]),
			static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]),
			static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 15) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]),
			static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 16) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]),
			static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]),
			static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]),
			static_cast<char>(str[0]));
	} else if constexpr (len == 17) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]),
			static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]),
			static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]),
			static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 18) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]),
			static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]),
			static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]),
			static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 19) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]),
			static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]),
			static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]),
			static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 20) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]),
			static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]),
			static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]),
			static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 21) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]),
			static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]),
			static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]),
			static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 22) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]),
			static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]),
			static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 23) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]),
			static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]),
			static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]),
			static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]),
			static_cast<char>(str[0]));
	} else if constexpr (len == 24) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]),
			static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]),
			static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]),
			static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]),
			static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 25) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, static_cast<char>(str[24]), static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]),
			static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]),
			static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]),
			static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]),
			static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 26) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, 0, static_cast<char>(str[25]), static_cast<char>(str[24]), static_cast<char>(str[23]), static_cast<char>(str[22]),
			static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]),
			static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]),
			static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]),
			static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 27) {
		return _mm256_set_epi8(0, 0, 0, 0, 0, static_cast<char>(str[26]), static_cast<char>(str[25]), static_cast<char>(str[24]), static_cast<char>(str[23]),
			static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]),
			static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]),
			static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]),
			static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 28) {
		return _mm256_set_epi8(0, 0, 0, 0, static_cast<char>(str[27]), static_cast<char>(str[26]), static_cast<char>(str[25]), static_cast<char>(str[24]),
			static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]),
			static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]),
			static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 29) {
		return _mm256_set_epi8(0, 0, 0, static_cast<char>(str[28]), static_cast<char>(str[27]), static_cast<char>(str[26]), static_cast<char>(str[25]), static_cast<char>(str[24]),
			static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]),
			static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]),
			static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]),
			static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else if constexpr (len == 30) {
		return _mm256_set_epi8(0, 0, static_cast<char>(str[29]), static_cast<char>(str[28]), static_cast<char>(str[27]), static_cast<char>(str[26]), static_cast<char>(str[25]),
			static_cast<char>(str[24]), static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]), static_cast<char>(str[19]),
			static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]), static_cast<char>(str[13]),
			static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]), static_cast<char>(str[7]),
			static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]), static_cast<char>(str[1]),
			static_cast<char>(str[0]));
	} else if constexpr (len == 31) {
		return _mm256_set_epi8(0, static_cast<char>(str[30]), static_cast<char>(str[29]), static_cast<char>(str[28]), static_cast<char>(str[27]), static_cast<char>(str[26]),
			static_cast<char>(str[25]), static_cast<char>(str[24]), static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]), static_cast<char>(str[20]),
			static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]), static_cast<char>(str[14]),
			static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]), static_cast<char>(str[8]),
			static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]), static_cast<char>(str[2]),
			static_cast<char>(str[1]), static_cast<char>(str[0]));
	} else {
		return _mm256_set_epi8(static_cast<char>(str[31]), static_cast<char>(str[30]), static_cast<char>(str[29]), static_cast<char>(str[28]), static_cast<char>(str[27]),
			static_cast<char>(str[26]), static_cast<char>(str[25]), static_cast<char>(str[24]), static_cast<char>(str[23]), static_cast<char>(str[22]), static_cast<char>(str[21]),
			static_cast<char>(str[20]), static_cast<char>(str[19]), static_cast<char>(str[18]), static_cast<char>(str[17]), static_cast<char>(str[16]), static_cast<char>(str[15]),
			static_cast<char>(str[14]), static_cast<char>(str[13]), static_cast<char>(str[12]), static_cast<char>(str[11]), static_cast<char>(str[10]), static_cast<char>(str[9]),
			static_cast<char>(str[8]), static_cast<char>(str[7]), static_cast<char>(str[6]), static_cast<char>(str[5]), static_cast<char>(str[4]), static_cast<char>(str[3]),
			static_cast<char>(str[2]), static_cast<char>(str[1]), static_cast<char>(str[0]));
	}
}

template<uint64_t len> BNCH_SWT_HOST static __m256i load_via_scalar_insert(const uint8_t* str) noexcept {
	__m256i result = _mm256_setzero_si256();
	uint8_t buf[32]{};
	for (uint64_t i = 0; i < len; ++i) {
		buf[i] = str[i];
	}
	std::memcpy(&result, buf, 32);
	return result;
}

template<uint64_t len> BNCH_SWT_HOST static __m256i load_via_memcpy_zeroed(const uint8_t* str) noexcept {
	alignas(32) uint8_t buf[32]{};
	std::memcpy(buf, str, len);
	return _mm256_load_si256(reinterpret_cast<const __m256i*>(buf));
}

template<uint64_t len> BNCH_SWT_HOST static __m256i load_via_memcpy_default_ctor(const uint8_t* str) noexcept {
	alignas(32) __m256i result{};
	std::memcpy(&result, str, len);
	return result;
}

template<auto load_fn> struct throughput_impl {
	BNCH_SWT_HOST static uint64_t impl(const bytes_vec& keys) {
		const uint8_t* base = keys.data();
		for (uint64_t x = 0; x < key_count; ++x) {
			const __m256i v = load_fn(base + (x * buf_stride));
			do_not_optimize_away(v);
		}
		return buf_stride * key_count;
	}
};

template<auto load_fn> struct latency_impl {
	BNCH_SWT_HOST static uint64_t impl(const bytes_vec& keys) {
		const uint8_t* base = keys.data();
		uint64_t index{};
		for (uint64_t x = 0; x < key_count; ++x) {
			const __m256i v		= load_fn(base + (index * buf_stride));
			const uint8_t byte0 = static_cast<uint8_t>(_mm256_extract_epi8(v, 0));
			do_not_optimize_away(v);
			index = (index + 1 + (byte0 & 1)) % key_count;
		}
		return buf_stride * key_count;
	}
};

static constexpr stage_config_data cmp_stage_config{
	.clear_cpu_caches_before_iterations = false,
	.measured_iteration_count			= 200,
	.max_iteration_count				= 20000,
	.convergence_threshold				= 2.0,
	.benchmark_type						= benchmark_types::cpu,
	.max_time_in_s						= 4,
	.rse_threshold						= 2.5,
	.max_k								= 100000,
	.min_k								= 30,
};

using throughput_stage = benchmark_stage<"partial-load-throughput", cmp_stage_config>;
using latency_stage	   = benchmark_stage<"partial-load-latency", cmp_stage_config>;

template<uint64_t len> inline void run_len(const bytes_vec& keys) {
	static constexpr auto test_name_tp	= internal::to_string_literal<static_cast<int64_t>(len)>();
	static constexpr auto test_name_lat = internal::to_string_literal<static_cast<int64_t>(len)>();

	throughput_stage::run_benchmark<test_name_tp, "set_epi8", throughput_impl<load_via_set<len>>>(keys);
	throughput_stage::run_benchmark<test_name_tp, "scalar_insert", throughput_impl<load_via_scalar_insert<len>>>(keys);
	throughput_stage::run_benchmark<test_name_tp, "memcpy_zeroed_buf", throughput_impl<load_via_memcpy_zeroed<len>>>(keys);
	throughput_stage::run_benchmark<test_name_tp, "memcpy_default_ctor", throughput_impl<load_via_memcpy_default_ctor<len>>>(keys);

	latency_stage::run_benchmark<test_name_lat, "set_epi8", latency_impl<load_via_set<len>>>(keys);
	latency_stage::run_benchmark<test_name_lat, "scalar_insert", latency_impl<load_via_scalar_insert<len>>>(keys);
	latency_stage::run_benchmark<test_name_lat, "memcpy_zeroed_buf", latency_impl<load_via_memcpy_zeroed<len>>>(keys);
	latency_stage::run_benchmark<test_name_lat, "memcpy_default_ctor", latency_impl<load_via_memcpy_default_ctor<len>>>(keys);

	std::cout << throughput_stage::get_test_results(test_name_tp.operator std::string()).to_csv() << std::endl;
	std::cout << latency_stage::get_test_results(test_name_lat.operator std::string()).to_csv() << std::endl;
}

template<uint64_t... lens> BNCH_SWT_HOST void run_all_lens(const bytes_vec& keys, std::index_sequence<lens...>) {
	(run_len<lens + 1>(keys), ...);
}

#endif

int main() {
#if BNCH_SWT_AVX2
	benchmarksuite::pin_for_benchmark();

	const bytes_vec keys = make_runtime_keys();

	run_all_lens(keys, std::make_index_sequence<max_len>{});

	std::cout << throughput_stage::get_all_results().to_csv() << std::endl;
	std::cout << latency_stage::get_all_results().to_csv() << std::endl;

	return 0;
#else
	std::cout << "AVX2 not enabled in this build" << std::endl;
	return -1;
#endif
}
