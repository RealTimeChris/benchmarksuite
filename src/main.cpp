// main.cpp — codegen inspection harness for void-numerics
//
// Purpose: emit one isolated, non-inlinable symbol per integer width for both
// to_chars and from_chars, so the generated assembly can be read per-case
// instead of as part of main()'s inlined soup.
//
// Build for asm (Clang/GCC):
//   clang++ -std=c++23 -O3 -march=native -S -masm=intel main.cpp -o main.s
//   g++     -std=c++23 -O3 -march=native -S -masm=intel main.cpp -o main.s
// Build for asm (MSVC):
//   cl /std:c++latest /O2 /FA /c main.cpp
//
// Then grep the symbols:  probe_to_   probe_from_
//
// To compare against jeaiii in the same TU (so codegen is apples-to-apples),
// define VN_HAVE_JEAIII and provide the include path.

#include <cstdint>
#include <cstddef>
#include <charconv>

// Repoint this to your single-header / aggregated include.
#include <void-numerics>

#if defined(VN_HAVE_JEAIII)
	#include <jeaiii_to_text.h>
#endif

// ---------------------------------------------------------------------------
// Optimization barriers. We need the compiler to treat inputs as opaque at the
// call site (otherwise it folds a known integer into a string literal and the
// whole conversion vanishes), and to not discard the result.
// ---------------------------------------------------------------------------

// Sinks used by all compilers (the from_chars probes store the parsed value
// here regardless of how launder_in is implemented).
static volatile std::uint64_t g_sink_u64;
static volatile char g_sink_char;

#if defined(_MSC_VER)
	#include <intrin.h>
// MSVC has no inline-asm clobber on x64; use a volatile sink instead.
static volatile const void* g_sink_ptr;
// Integer inputs: round-trip through a volatile to defeat const-folding.
template<class T> static inline T launder_in(T v) {
	g_sink_u64 = static_cast<std::uint64_t>(v);
	return static_cast<T>(g_sink_u64);
}
// Pointer inputs (from_chars): can't portably cast a pointer to uint64_t,
// so launder through a volatile void* sink instead.
template<class T> static inline T* launder_in(T* p) {
	g_sink_ptr = p;
	return const_cast<T*>(static_cast<const T*>(const_cast<const void*>(g_sink_ptr)));
}
static inline void consume_ptr(char* p) {
	g_sink_char = *p;
}
	#define VN_NOINLINE __declspec(noinline)
#else
template<class T> static inline T launder_in(T v) {
	asm volatile("" : "+r"(v));// value is now opaque, stays in a reg
	return v;
}
static inline void consume_ptr(char* p) {
	asm volatile("" : : "r"(p) : "memory");
}
	#define VN_NOINLINE __attribute__((noinline))
#endif

// Per-call scratch buffer big enough for any 64-bit decimal + sign.
static char g_buf[32];

// ---------------------------------------------------------------------------
// to_chars probes — one symbol per width. Each takes an opaque value, runs the
// real conversion, and consumes the result so nothing is dead-code eliminated.
// ---------------------------------------------------------------------------
#define VN_DEFINE_TO_PROBE(NAME, TYPE) \
	VN_NOINLINE void probe_to_##NAME(TYPE v) { \
		TYPE x = launder_in(v); \
		auto r = vn::to_chars(g_buf, g_buf + sizeof(g_buf), x); \
		consume_ptr(r.ptr); \
	}

VN_DEFINE_TO_PROBE(u8, std::uint8_t)
VN_DEFINE_TO_PROBE(u16, std::uint16_t)
VN_DEFINE_TO_PROBE(u32, std::uint32_t)
VN_DEFINE_TO_PROBE(u64, std::uint64_t)
VN_DEFINE_TO_PROBE(i8, std::int8_t)
VN_DEFINE_TO_PROBE(i16, std::int16_t)
VN_DEFINE_TO_PROBE(i32, std::int32_t)
VN_DEFINE_TO_PROBE(i64, std::int64_t)

// ---------------------------------------------------------------------------
// from_chars probes. The string pointer/length come in opaque so the parse
// can't be const-folded. We feed back the parsed value through the sink too,
// so the overflow-check / digit-ladder branches all survive.
// ---------------------------------------------------------------------------
#define VN_DEFINE_FROM_PROBE(NAME, TYPE) \
	VN_NOINLINE const char* probe_from_##NAME(const char* p, std::size_t n) { \
		const char* first = launder_in(p); \
		const char* last  = first + n; \
		TYPE out{}; \
		auto r	   = vn::from_chars(first, last, out); \
		g_sink_u64 = static_cast<std::uint64_t>(out); \
		return r.ptr; \
	}

VN_DEFINE_FROM_PROBE(u8, std::uint8_t)
VN_DEFINE_FROM_PROBE(u16, std::uint16_t)
VN_DEFINE_FROM_PROBE(u32, std::uint32_t)
VN_DEFINE_FROM_PROBE(u64, std::uint64_t)
VN_DEFINE_FROM_PROBE(i8, std::int8_t)
VN_DEFINE_FROM_PROBE(i16, std::int16_t)
VN_DEFINE_FROM_PROBE(i32, std::int32_t)
VN_DEFINE_FROM_PROBE(i64, std::int64_t)

// ---------------------------------------------------------------------------
// std:: baseline probes. Identical opaque-in/consume-out treatment so the
// generated asm sits right next to vn's and the same compiler/flags produced
// both. This is the apples-to-apples target for the M1 large-N falloff.
// ---------------------------------------------------------------------------
#define VN_DEFINE_STD_TO_PROBE(NAME, TYPE) \
	VN_NOINLINE void probe_std_to_##NAME(TYPE v) { \
		TYPE x = launder_in(v); \
		auto r = std::to_chars(g_buf, g_buf + sizeof(g_buf), x); \
		consume_ptr(r.ptr); \
	}

VN_DEFINE_STD_TO_PROBE(u8, std::uint8_t)
VN_DEFINE_STD_TO_PROBE(u16, std::uint16_t)
VN_DEFINE_STD_TO_PROBE(u32, std::uint32_t)
VN_DEFINE_STD_TO_PROBE(u64, std::uint64_t)
VN_DEFINE_STD_TO_PROBE(i8, std::int8_t)
VN_DEFINE_STD_TO_PROBE(i16, std::int16_t)
VN_DEFINE_STD_TO_PROBE(i32, std::int32_t)
VN_DEFINE_STD_TO_PROBE(i64, std::int64_t)

#define VN_DEFINE_STD_FROM_PROBE(NAME, TYPE) \
	VN_NOINLINE const char* probe_std_from_##NAME(const char* p, std::size_t n) { \
		const char* first = launder_in(p); \
		const char* last  = first + n; \
		TYPE out{}; \
		auto r	   = std::from_chars(first, last, out); \
		g_sink_u64 = static_cast<std::uint64_t>(out); \
		return r.ptr; \
	}

VN_DEFINE_STD_FROM_PROBE(u8, std::uint8_t)
VN_DEFINE_STD_FROM_PROBE(u16, std::uint16_t)
VN_DEFINE_STD_FROM_PROBE(u32, std::uint32_t)
VN_DEFINE_STD_FROM_PROBE(u64, std::uint64_t)
VN_DEFINE_STD_FROM_PROBE(i8, std::int8_t)
VN_DEFINE_STD_FROM_PROBE(i16, std::int16_t)
VN_DEFINE_STD_FROM_PROBE(i32, std::int32_t)
VN_DEFINE_STD_FROM_PROBE(i64, std::int64_t)

#if defined(VN_HAVE_JEAIII)
	// Same opaque-in/consume-out treatment so the side-by-side asm is fair.
	#define VN_DEFINE_JEAIII_PROBE(NAME, TYPE) \
		VN_NOINLINE void probe_jeaiii_##NAME(TYPE v) { \
			TYPE x	= launder_in(v); \
			char* r = jeaiii::to_text_from_integer(g_buf, x); \
			consume_ptr(r); \
		}
VN_DEFINE_JEAIII_PROBE(u32, std::uint32_t)
VN_DEFINE_JEAIII_PROBE(u64, std::uint64_t)
VN_DEFINE_JEAIII_PROBE(i64, std::int64_t)
#endif

// ---------------------------------------------------------------------------
// A real main so the TU links into a runnable binary too (CI sanity check:
// the conversions must round-trip). Kept tiny; the probes above are the point.
// ---------------------------------------------------------------------------
int main() {
	// Touch every probe so none are stripped before -S in case LTO is on.
	probe_to_u8(0);
	probe_to_u16(0);
	probe_to_u32(0);
	probe_to_u64(0);
	probe_to_i8(0);
	probe_to_i16(0);
	probe_to_i32(0);
	probe_to_i64(0);

	const char s[] = "1234567890123456789";
	probe_from_u8(s, sizeof(s) - 1);
	probe_from_u16(s, sizeof(s) - 1);
	probe_from_u32(s, sizeof(s) - 1);
	probe_from_u64(s, sizeof(s) - 1);
	probe_from_i8(s, sizeof(s) - 1);
	probe_from_i16(s, sizeof(s) - 1);
	probe_from_i32(s, sizeof(s) - 1);
	probe_from_i64(s, sizeof(s) - 1);

	probe_std_to_u8(0);
	probe_std_to_u16(0);
	probe_std_to_u32(0);
	probe_std_to_u64(0);
	probe_std_to_i8(0);
	probe_std_to_i16(0);
	probe_std_to_i32(0);
	probe_std_to_i64(0);
	probe_std_from_u8(s, sizeof(s) - 1);
	probe_std_from_u16(s, sizeof(s) - 1);
	probe_std_from_u32(s, sizeof(s) - 1);
	probe_std_from_u64(s, sizeof(s) - 1);
	probe_std_from_i8(s, sizeof(s) - 1);
	probe_std_from_i16(s, sizeof(s) - 1);
	probe_std_from_i32(s, sizeof(s) - 1);
	probe_std_from_i64(s, sizeof(s) - 1);
	return 0;
}