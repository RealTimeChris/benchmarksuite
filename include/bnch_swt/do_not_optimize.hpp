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

#if BNCH_SWT_COMPILER_MSVC

	inline void const volatile* volatile global_force_escape_pointer;

	BNCH_SWT_HOST void use_char_pointer(char const volatile* const v) {
		global_force_escape_pointer = reinterpret_cast<void const volatile*>(v);
	}

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type const& value) {
		use_char_pointer(&reinterpret_cast<char const volatile&>(value));
		_ReadWriteBarrier();
	}

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type& value) {
		use_char_pointer(&reinterpret_cast<char const volatile&>(value));
		_ReadWriteBarrier();
	}

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type&& value) {
		use_char_pointer(&reinterpret_cast<char const volatile&>(value));
		_ReadWriteBarrier();
	}

#elif BNCH_SWT_COMPILER_CLANG
	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type const& value) {
		asm volatile("" : : "r,m"(value) : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type& value) {
		asm volatile("" : "+r,m"(value) : : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type&& value) {
		asm volatile("" : "+r,m"(value) : : "memory");
	}

#elif BNCH_SWT_COMPILER_GNUCXX
	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type const& value)
		-> std::enable_if_t<std::is_trivially_copyable_v<value_type> && (sizeof(value_type) <= sizeof(value_type*))> {
		asm volatile("" : : "r,m"(value) : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type const& value)
		-> std::enable_if_t<!std::is_trivially_copyable_v<value_type> || (sizeof(value_type) > sizeof(value_type*))> {
		asm volatile("" : : "m"(value) : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type& value)
		-> std::enable_if_t<std::is_trivially_copyable_v<value_type> && (sizeof(value_type) <= sizeof(value_type*))> {
		asm volatile("" : "+m,r"(value) : : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type& value)
		-> std::enable_if_t<!std::is_trivially_copyable_v<value_type> || (sizeof(value_type) > sizeof(value_type*))> {
		asm volatile("" : "+m"(value) : : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type&& value)
		-> std::enable_if_t<std::is_trivially_copyable_v<value_type> && (sizeof(value_type) <= sizeof(value_type*))> {
		asm volatile("" : "+m,r"(value) : : "memory");
	}

	template<typename value_type> BNCH_SWT_HOST auto do_not_optimize(value_type&& value)
		-> std::enable_if_t<!std::is_trivially_copyable_v<value_type> || (sizeof(value_type) > sizeof(value_type*))> {
		asm volatile("" : "+m"(value) : : "memory");
	}
#else

	template<typename value_type> BNCH_SWT_HOST void do_not_optimize(value_type&&) {
	}

#endif

	BNCH_SWT_HOST void clobber_memory() {
#if BNCH_SWT_COMPILER_MSVC
		_ReadWriteBarrier();
#elif BNCH_SWT_COMPILER_CLANG || BNCH_SWT_COMPILER_GNUCXX
		asm volatile("" ::: "memory");
#endif
	}
}

namespace bnch_swt {

	template<internal::not_invocable value_type> BNCH_SWT_HOST void do_not_optimize_away(value_type&& value) {
		internal::do_not_optimize(value);
	}

	template<internal::invocable_void function_type, typename... arg_types> BNCH_SWT_HOST void do_not_optimize_away(function_type&& value, arg_types&&... args) {
		std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::clobber_memory();
	}

	template<internal::invocable_not_void function_type, typename... arg_types> BNCH_SWT_HOST auto do_not_optimize_away(function_type&& value, arg_types&&... args) {
		auto result_val = std::forward<function_type>(value)(std::forward<arg_types>(args)...);
		internal::do_not_optimize(result_val);
		return result_val;
	}

}
