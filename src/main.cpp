#include <bnch_swt>
#include <simdjson.h>
#include "utf8_validator.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bnch_swt;

static constexpr size_t buf_size = 1024 * 1024;

struct utf8_entry {
	uint8_t bytes[4];
	uint8_t len;
};

static std::vector<utf8_entry> build_utf8_table(bool ascii_only, bool multibyte_only) {
	std::vector<utf8_entry> table;
	for (uint32_t cp = 0x0020u; cp <= 0x10FFFFu; ++cp) {
		if (cp >= 0xD800u && cp <= 0xDFFFu)
			continue;
		utf8_entry e{};
		if (cp <= 0x7Fu) {
			if (multibyte_only)
				continue;
			e.bytes[0] = static_cast<uint8_t>(cp);
			e.len	   = 1;
		} else if (cp <= 0x7FFu) {
			if (ascii_only)
				continue;
			e.bytes[0] = static_cast<uint8_t>(0xC0u | (cp >> 6));
			e.bytes[1] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
			e.len	   = 2;
		} else if (cp <= 0xFFFFu) {
			if (ascii_only)
				continue;
			e.bytes[0] = static_cast<uint8_t>(0xE0u | (cp >> 12));
			e.bytes[1] = static_cast<uint8_t>(0x80u | ((cp >> 6) & 0x3Fu));
			e.bytes[2] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
			e.len	   = 3;
		} else {
			if (ascii_only)
				continue;
			e.bytes[0] = static_cast<uint8_t>(0xF0u | (cp >> 18));
			e.bytes[1] = static_cast<uint8_t>(0x80u | ((cp >> 12) & 0x3Fu));
			e.bytes[2] = static_cast<uint8_t>(0x80u | ((cp >> 6) & 0x3Fu));
			e.bytes[3] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
			e.len	   = 4;
		}
		table.push_back(e);
	}
	return table;
}

static std::vector<uint8_t> make_corpus(const std::vector<utf8_entry>& table, size_t n) {
	random_generator<uint32_t, xoshiro_256_seeds::deterministic> rng{};
	std::vector<uint8_t> buf;
	buf.reserve(n + 4);
	const auto max_idx = static_cast<uint32_t>(table.size() - 1);
	while (buf.size() < n) {
		const auto& e = table[rng.impl(0u, max_idx)];
		for (uint8_t i = 0; i < e.len; ++i)
			buf.push_back(e.bytes[i]);
	}
	return buf;
}

static int run_correctness_tests() {
	int failures = 0;

	auto test = [&](const char* label, std::vector<uint8_t> buf, bool expected) {
		bool ours	= jsonifier_internal::validate_utf8(buf.data(), buf.size());
		bool ok	  = (ours == expected);
		if (!ok) {
			++failures;
			std::cout << "FAIL [" << label << "]: ours=" << ours << " expected=" << expected << std::endl;
		} else {
			std::cout << "PASS [" << label << "]: ours=" << ours << " expected=" << expected << std::endl;
		}
	};
	test("empty", {}, true);
	test("single ascii", { 0x41 }, true);
	test("ascii x16", std::vector<uint8_t>(16, 0x41), true);
	test("ascii x64", std::vector<uint8_t>(64, 0x41), true);
	test("ascii x256", std::vector<uint8_t>(256, 0x41), true);
	test("2-byte U+0080", { 0xC2, 0x80 }, true);
	test("2-byte U+07FF", { 0xDF, 0xBF }, true);
	test("3-byte U+0800", { 0xE0, 0xA0, 0x80 }, true);
	test("3-byte U+FFFF", { 0xEF, 0xBF, 0xBF }, true);
	test("4-byte U+10000", { 0xF0, 0x90, 0x80, 0x80 }, true);
	test("4-byte U+10FFFF", { 0xF4, 0x8F, 0xBF, 0xBF }, true);
	test(
		
		"2-byte x128",
		[] {
			std::vector<uint8_t> v;
			for (int i = 0; i < 128; ++i) {
				v.push_back(0xC2);
				v.push_back(0x80);
			}
			return v;
		}(),
		true);
	test(
		
		"3-byte x64",
		[] {
			std::vector<uint8_t> v;
			for (int i = 0; i < 64; ++i) {
				v.push_back(0xE0);
				v.push_back(0xA0);
				v.push_back(0x80);
			}
			return v;
		}(),
		true);
	test(
		
		"4-byte x64",
		[] {
			std::vector<uint8_t> v;
			for (int i = 0; i < 64; ++i) {
				v.push_back(0xF0);
				v.push_back(0x90);
				v.push_back(0x80);
				v.push_back(0x80);
			}
			return v;
		}(),
		true);
	test("2-byte cross chunk",
		{ 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0xC2, 0x80, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
			0x41, 0x41, 0x41, 0x41 },
		true);
	test("3-byte cross chunk",
		{ 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0xE0, 0xA0, 0x80, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
			0x41, 0x41, 0x41, 0x41 },
		true);
	test("4-byte cross chunk",
		{ 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0xF0, 0x90, 0x80, 0x80, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
			0x41, 0x41, 0x41, 0x41 },
		true);
	test(
		
		"4-byte cross block boundary",
		[] {
			std::vector<uint8_t> v(61, 0x41);
			v.push_back(0xF0);
			v.push_back(0x90);
			v.push_back(0x80);
			v.push_back(0x80);
			return v;
		}(),
		true);
	test("mixed ascii+multibyte", { 0x41, 0xC2, 0x80, 0x41, 0xE0, 0xA0, 0x80, 0x41, 0xF0, 0x90, 0x80, 0x80, 0x41, 0x41, 0x41, 0x41 }, true);
	test(
		
		"ascii x64 then 2-byte",
		[] {
			std::vector<uint8_t> v(64, 0x41);
			v.push_back(0xC2);
			v.push_back(0x80);
			return v;
		}(),
		true);
	test(
		
		"ascii x256 then 2-byte",
		[] {
			std::vector<uint8_t> v(256, 0x41);
			v.push_back(0xC2);
			v.push_back(0x80);
			return v;
		}(),
		true);
	test("dangling 2-byte lead", { 0xC2 }, false);
	test("dangling 3-byte lead", { 0xE0 }, false);
	test("dangling 3-byte lead+1", { 0xE0, 0xA0 }, false);
	test("dangling 4-byte lead", { 0xF0 }, false);
	test("dangling 4-byte lead+1", { 0xF0, 0x90 }, false);
	test("dangling 4-byte lead+2", { 0xF0, 0x90, 0x80 }, false);
	test("bad cont: ascii after 2-lead", { 0xC2, 0x41 }, false);
	test("bad cont: ascii after 3-lead", { 0xE0, 0xA0, 0x41 }, false);
	test("bad cont: lead after lead", { 0xC2, 0xC2, 0x80 }, false);
	test("lone continuation", { 0x80 }, false);
	test("lone continuation x4", { 0x80, 0x80, 0x80, 0x80 }, false);
	test("surrogate U+D800", { 0xED, 0xA0, 0x80 }, false);
	test("surrogate U+DFFF", { 0xED, 0xBF, 0xBF }, false);
	test("overlong 2-byte U+0000", { 0xC0, 0x80 }, false);
	test("overlong 2-byte U+007F", { 0xC1, 0xBF }, false);
	test("overlong 3-byte", { 0xE0, 0x80, 0x80 }, false);
	test("overlong 4-byte", { 0xF0, 0x80, 0x80, 0x80 }, false);
	test("too large U+110000", { 0xF4, 0x90, 0x80, 0x80 }, false);
	test("too large 0xF5", { 0xF5, 0x80, 0x80, 0x80 }, false);
	test("0xFF byte", { 0xFF }, false);
	test("0xFE byte", { 0xFE }, false);
	test("bad cont cross chunk",
		{ 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0xC2, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
			0x41, 0x41, 0x41, 0x41 },
		false);
	test("surrogate cross chunk",
		{ 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0xED, 0xA0, 0x80, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
			0x41, 0x41, 0x41, 0x41 },
		false);
	test(
		
		"dangling lead at block end",
		[] {
			std::vector<uint8_t> v(63, 0x41);
			v.push_back(0xC2);
			return v;
		}(),
		false);
	test(
		
		"dangling lead at step end",
		[] {
			std::vector<uint8_t> v(255, 0x41);
			v.push_back(0xC2);
			return v;
		}(),
		false);
	test("valid then lone cont", { 0xC2, 0x80, 0x80 }, false);

	return failures;
}

struct simdjson_validate {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& buf) {
		if (!simdjson::validate_utf8(reinterpret_cast<const char*>(buf.data()), buf.size()))
			throw std::runtime_error{ "simdjson Failed validation!" };
		return buf.size();
	}
};

struct jsonifier_validate_multibyte {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& buf) {
		if (!jsonifier_internal::validate_utf8(buf.data(), buf.size()))
			throw std::runtime_error{ "Jsonifier Failed validation!" };
		return buf.size();
	}
};

struct jsonifier_validate_ascii {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& buf) {
		if (!jsonifier_internal::validate_utf8(buf.data(), buf.size()))
			throw std::runtime_error{ "Jsonifier Failed validation!" };
		return buf.size();
	}
};

struct jsonifier_validate_mixed {
	BNCH_SWT_HOST static uint64_t impl(const std::vector<uint8_t>& buf) {
		if (!jsonifier_internal::validate_utf8(buf.data(), buf.size()))
			throw std::runtime_error{ "Jsonifier Failed validation!" };
		return buf.size();
	}
};

int main() {
	try {
		std::cout << "Blocks Per Step: " << BNCH_SWT_BLOCKS_PER_STEP << std::endl;
		std::cout << "Chunks Per Block: " << BNCH_SWT_CHUNKS_PER_BLOCK << std::endl;
		std::cout << "--- Correctness tests ---" << std::endl;
		int failures = run_correctness_tests();
		if (failures > 0) {
			std::cout << failures << " correctness test(s) FAILED - aborting benchmark" << std::endl;
			return 1;
		}
		std::cout << "All correctness tests passed - proceeding to benchmark" << std::endl << std::endl;

		static constexpr stage_config_data config{ [] {
			stage_config_data return_data{};
			return_data.max_iteration_count = 10000;
			return return_data;
		}() };

		bnch_swt::pin_for_benchmark();

		std::cout << "Building UTF-8 tables..." << std::endl;
		auto ascii_table	 = build_utf8_table(true, false);
		auto mixed_table	 = build_utf8_table(false, false);
		auto multibyte_table = build_utf8_table(false, true);
		std::cout << "ascii entries:     " << ascii_table.size() << std::endl;
		std::cout << "mixed entries:     " << mixed_table.size() << std::endl;
		std::cout << "multibyte entries: " << multibyte_table.size() << std::endl;

		auto ascii_buf	   = make_corpus(ascii_table, buf_size);
		auto mixed_buf	   = make_corpus(mixed_table, buf_size);
		auto multibyte_buf = make_corpus(multibyte_table, buf_size);
		std::cout << "ascii buf size:     " << ascii_buf.size() << std::endl;
		std::cout << "mixed buf size:     " << mixed_buf.size() << std::endl;
		std::cout << "multibyte buf size: " << multibyte_buf.size() << std::endl << std::endl;

		using stage_ascii	  = benchmark_stage<"utf8-validation-ascii", config>;
		using stage_mixed	  = benchmark_stage<"utf8-validation-mixed", config>;
		using stage_multibyte = benchmark_stage<"utf8-validation-multibyte", config>;
		stage_ascii ::run_benchmark<"utf8-validate", "simdjson", simdjson_validate>(ascii_buf);
		stage_ascii ::run_benchmark<"utf8-validate", "jsonifier", jsonifier_validate_ascii>(ascii_buf);
		stage_mixed ::run_benchmark<"utf8-validate", "simdjson", simdjson_validate>(mixed_buf);
		stage_mixed ::run_benchmark<"utf8-validate", "jsonifier", jsonifier_validate_mixed>(mixed_buf);
		stage_multibyte::run_benchmark<"utf8-validate", "simdjson", simdjson_validate>(multibyte_buf);
		stage_multibyte::run_benchmark<"utf8-validate", "jsonifier", jsonifier_validate_multibyte>(multibyte_buf);

		std::cout << stage_ascii ::get_test_results("utf8-validate").to_csv() << std::endl;
		std::cout << stage_mixed ::get_test_results("utf8-validate").to_csv() << std::endl;
		std::cout << stage_multibyte::get_test_results("utf8-validate").to_csv() << std::endl;
	} catch (const std::runtime_error& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}