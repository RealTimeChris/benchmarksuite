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
#include <bnch_swt/benchmarksuite_cpu_properties.hpp>

namespace bnch_swt {

	static constexpr uint64_t device_alignment{ [] {
		if constexpr (BNCH_SWT_COMPILER_CUDA) {
			return 16ULL;
		} else {
			return cpu_properties::cpu_alignment;
		}
	}() };

	template<typename value_type>
	concept derivable_from = std::is_class_v<std::remove_cvref_t<value_type>> && !std::is_final_v<std::remove_cvref_t<value_type>>;

	template<typename value_type_new, uint64_t device_alignment = 16> struct BNCH_SWT_ALIGN(device_alignment) aligned_const {
		using value_type = value_type_new;
		BNCH_SWT_ALIGN(16) value_type value {};

		BNCH_SWT_HOST_DEVICE constexpr operator const value_type&() const& {
			return value;
		}

		BNCH_SWT_HOST_DEVICE explicit constexpr operator value_type&() & {
			return value;
		}

		BNCH_SWT_HOST_DEVICE explicit constexpr operator value_type&&() && {
			return std::move(value);
		}

		BNCH_SWT_HOST_DEVICE constexpr const value_type& operator*() const {
			return value;
		}

		template<typename value_type_newer> BNCH_SWT_HOST_DEVICE constexpr void emplace(value_type_newer&& value_new) {
			value = std::forward<value_type_newer>(value_new);
		}

		BNCH_SWT_HOST_DEVICE value_type& operator*() {
			return value;
		}

		BNCH_SWT_HOST_DEVICE constexpr value_type operator*(const aligned_const& other) const {
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

	template<derivable_from value_type_new, uint64_t device_alignment> struct BNCH_SWT_ALIGN(device_alignment) aligned_const<value_type_new, device_alignment> : public value_type_new {
		using value_type = value_type_new;

		BNCH_SWT_HOST_DEVICE constexpr const value_type& operator*() const {
			return *static_cast<const value_type*>(this);
		}

		template<typename value_type_newer> BNCH_SWT_HOST_DEVICE constexpr void emplace(value_type_newer&& value_new) {
			*static_cast<value_type*>(this) = std::forward<value_type_newer>(value_new);
		}

		BNCH_SWT_HOST_DEVICE value_type& operator*() {
			return *static_cast<value_type*>(this);
		}

		BNCH_SWT_HOST_DEVICE constexpr value_type operator*(const aligned_const& other) const {
			return *static_cast<const value_type*>(this) * *static_cast<const value_type*>(&other);
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator==(const aligned_const& other) const {
			return *static_cast<const value_type*>(this) == *static_cast<const value_type*>(&other);
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator!=(const aligned_const& other) const {
			return *static_cast<const value_type*>(this) != *static_cast<const value_type*>(&other);
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator<(const aligned_const& other) const {
			return *static_cast<const value_type*>(this) < *static_cast<const value_type*>(&other);
		}

		BNCH_SWT_HOST_DEVICE constexpr bool operator>(const aligned_const& other) const {
			return *static_cast<const value_type*>(this) > *static_cast<const value_type*>(&other);
		}
	};

	template<typename value_type> aligned_const(value_type) -> aligned_const<value_type>;

}
