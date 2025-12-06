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
/// Sep 1, 2024
#pragma once

#include <bnch_swt/config.hpp>

namespace bnch_swt::internal {

	template<typename value_type, typename... arg_types>
	concept invocable = std::is_invocable_v<std::remove_cvref_t<value_type>, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept not_invocable = !invocable<value_type, arg_types...>;

	template<typename value_type, typename... arg_types>
	concept invocable_void = invocable<value_type, arg_types...> && std::is_void_v<std::invoke_result_t<value_type, arg_types...>>;

	template<typename value_type, typename... arg_types>
	concept invocable_not_void = invocable<value_type, arg_types...> && !std::is_void_v<std::invoke_result_t<value_type, arg_types...>>;

	template<typename value_type>
	concept small_trivially_copyable = std::is_trivially_copyable_v<value_type> && (sizeof(value_type) <= sizeof(value_type*));

	template<typename value_type>
	concept large_or_non_trivially_copyable = !std::is_trivially_copyable_v<value_type> || (sizeof(value_type) > sizeof(value_type*));

	inline void const volatile* volatile global_force_escape_pointer;

	BNCH_SWT_HOST static void use_char_pointer(void const volatile* const v) {
		global_force_escape_pointer = v;
	}

#if BNCH_SWT_COMPILER_MSVC

	template<typename value_type> BNCH_SWT_HOST static void do_not_optimize(value_type const& value) {
		use_char_pointer(static_cast<void const volatile* const>(&value));
		_ReadWriteBarrier();
	}

	template<typename value_type> BNCH_SWT_HOST static void do_not_optimize(value_type&& value) {
		use_char_pointer(static_cast<void const volatile* const>(&value));
		_ReadWriteBarrier();
	}

#elif BNCH_SWT_COMPILER_CLANG
	template<typename value_type> BNCH_SWT_HOST static void do_not_optimize(value_type const& value) {
		asm volatile("" : : "r,m"(value) : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST static void do_not_optimize(value_type&& value) {
		asm volatile("" : "+r,m"(value) : : "memory");
	}

#elif BNCH_SWT_COMPILER_GCC
	template<small_trivially_copyable value_type> BNCH_SWT_HOST static void do_not_optimize(value_type const& value) {
		asm volatile("" : : "r,m"(value) : "memory");
	}

	template<large_or_non_trivially_copyable value_type> BNCH_SWT_HOST static void do_not_optimize(value_type const& value) {
		asm volatile("" : : "m"(value) : "memory");
	}

	template<small_trivially_copyable value_type> BNCH_SWT_HOST static void do_not_optimize(value_type&& value) {
		asm volatile("" : "+m,r"(value) : : "memory");
	}

	template<large_or_non_trivially_copyable value_type> BNCH_SWT_HOST static void do_not_optimize(value_type&& value) {
		asm volatile("" : "+m"(value) : : "memory");
	}
#else
	
	template<class value_type> inline BNCH_SWT_HOST static void do_not_optimize(value_type&& value) {
		internal::use_char_pointer(&reinterpret_cast<char const volatile&>(value));
	}

#endif

	BNCH_SWT_HOST static void clobber_memory() {
#if BNCH_SWT_COMPILER_MSVC
		_ReadWriteBarrier();
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GCC
		asm volatile("" ::: "memory");
#endif
	}
}

namespace bnch_swt {

	template<internal::not_invocable value_type> BNCH_SWT_HOST static void do_not_optimize_away(value_type&& value) {
		internal::do_not_optimize(value);
	}

	template<internal::invocable_void function_type, typename... arg_types> BNCH_SWT_HOST static void do_not_optimize_away(function_type&& value, arg_types&&... args) {
		std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::clobber_memory();
	}

	template<internal::invocable_not_void function_type, typename... arg_types> BNCH_SWT_HOST static auto do_not_optimize_away(function_type&& value, arg_types&&... args) {
		auto result_val = std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::do_not_optimize(result_val);
		return result_val;
	}

}
