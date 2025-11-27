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

#pragma once

#include <bnch_swt/config.hpp>

namespace bnch_swt {

	static constexpr uint64_t device_alignment{ [] {
		if constexpr (BNCH_SWT_COMPILER_CUDA) {
			return 16ull;
		} else {
			return 64ull;
		}
	}() };

	template<typename value_type_new> struct BNCH_SWT_ALIGN(device_alignment) aligned_const {
		using value_type = value_type_new;
		value_type value{};

		BNCH_SWT_HOST_DEVICE constexpr aligned_const() {
		}
		BNCH_SWT_HOST_DEVICE constexpr aligned_const(const value_type& v) : value(v) {
		}
		BNCH_SWT_HOST_DEVICE constexpr aligned_const(value_type&& v) : value(std::move(v)) {
		}

		BNCH_SWT_HOST_DEVICE constexpr operator const value_type&() const& {
			return value;
		}

		BNCH_SWT_HOST_DEVICE explicit constexpr operator value_type&() & {
			return value;
		}

		BNCH_SWT_HOST_DEVICE explicit constexpr operator value_type&&() && {
			return std::move(value);
		}

		BNCH_SWT_HOST_DEVICE constexpr const value_type* get() const {
			return &value;
		}

		BNCH_SWT_HOST_DEVICE constexpr value_type* get() {
			return &value;
		}

		BNCH_SWT_HOST_DEVICE constexpr const value_type& operator*() const {
			return value;
		}

		BNCH_SWT_HOST_DEVICE constexpr value_type& operator*() {
			return value;
		}

		template<typename value_type_newer> BNCH_SWT_HOST_DEVICE constexpr void emplace(value_type_newer&& value_new) {
			value = std::forward<value_type_newer>(value_new);
		}

		BNCH_SWT_HOST_DEVICE constexpr value_type multiply(const aligned_const& other) const {
			return value * other.value;
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator==(const aligned_const& other) const {
			return value == other.value;
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator!=(const aligned_const& other) const {
			return value != other.value;
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator<(const aligned_const& other) const {
			return value < other.value;
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator>(const aligned_const& other) const {
			return value > other.value;
		}
	};

	template<typename value_type> aligned_const(value_type) -> aligned_const<value_type>;

}
