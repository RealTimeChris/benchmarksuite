#include <cstddef>
#include <cstdint>
#include <array>
#include <bnch_swt/index.hpp>

// Two implementations representing the streaming vs regular variant choice.
// Marked noinline so we can see the dispatch site clearly rather than full
// inlining hiding what's happening.

BNCH_SWT_HOST bool compare_regular(const char* lhs, const char* rhs, std::size_t length) noexcept {
	for (std::size_t i = 0; i < length; ++i) {
		if (lhs[i] != rhs[i])
			return false;
	}
	return true;
}

BNCH_SWT_HOST bool compare_streaming(const char* lhs, const char* rhs, std::size_t length) noexcept {
	// Pretend this uses streaming variants — body intentionally different so
	// the compiler can't recognize them as identical and dedupe.
	bool result = true;
	for (std::size_t i = 0; i < length; ++i) {
		result &= (lhs[i] == rhs[i]);
	}
	return result;
}

using compare_fn = bool (*)(const char*, const char*, std::size_t) noexcept;

// The dispatch table: constexpr, two entries, indexed by a bool.
static constexpr std::array<compare_fn, 2> dispatch_table{ &compare_regular, &compare_streaming };

// Threshold above which we want streaming variant.
static constexpr std::size_t streaming_threshold = 2048;

// The dispatch function — this is what we want to see compiled.
BNCH_SWT_NOINLINE bool dispatch_compare(const char* lhs, const char* rhs, std::size_t length) noexcept {
	return dispatch_table[length >= streaming_threshold](lhs, rhs, length);
}

// For comparison: the equivalent if-branch version.
BNCH_SWT_NOINLINE bool dispatch_compare_branch(const char* lhs, const char* rhs, std::size_t length) noexcept {
	if (length >= streaming_threshold) {
		return compare_streaming(lhs, rhs, length);
	} else {
		return compare_regular(lhs, rhs, length);
	}
}

// Force the compiler to actually emit these by referencing them.
extern "C" int main() {
	bnch_swt::random_generator<uint64_t> rg_01{};
	bnch_swt::random_generator<std::string> rg{};
	auto string_01 = rg.impl(rg_01.impl(4096, 8192));
	dispatch_compare(string_01.data(), string_01.data(), 4096) + dispatch_compare_branch(string_01.data(), string_01.data(), 4096);
	return 0;
}