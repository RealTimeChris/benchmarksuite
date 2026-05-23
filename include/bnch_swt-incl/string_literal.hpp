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

#include <bnch_swt-incl/config.hpp>
#include <string_view>
#include <algorithm>
#include <array>

namespace bnch_swt {

	template<uint64_t size_val> struct string_literal {
		using value_type			 = char;
		using const_reference		 = const value_type&;
		using reference				 = value_type&;
		using const_pointer			 = const value_type*;
		using pointer				 = value_type*;
		using size_type				 = uint64_t;

		static constexpr size_type length{ size_val > 0 ? size_val - 1 : 0 };

		BNCH_SWT_HOST constexpr string_literal() noexcept {
		}

		BNCH_SWT_HOST constexpr string_literal(const char *str) noexcept {
			std::copy_n(str, length, data_val);
		}

		template<uint64_t starting_index, uint64_t end_index> constexpr auto substr() const noexcept {
			string_literal<(end_index - starting_index) + 1> return_data{};
			std::copy_n(data_val + starting_index, (end_index - starting_index), return_data.data_val);
			return return_data;
		}

		template<size_type size_new> BNCH_SWT_HOST consteval string_literal<size_new + size_val - 1> operator+(const string_literal<size_new>& str) const noexcept {
			string_literal<size_new + size_val - 1> new_literal{};
			std::copy_n(data_val, size(), new_literal.data_val);
			std::copy_n(str.data_val, str.size(), new_literal.data_val + size());
			return new_literal;
		}

		template<size_type size_new> BNCH_SWT_HOST consteval auto operator+(const value_type (&str)[size_new]) const noexcept {
			string_literal<size_new> new_literal{ str };
			return *this + new_literal;
		}

		template<size_type size_new> BNCH_SWT_HOST consteval friend auto operator+(const value_type (&lhs)[size_new], const string_literal<size_val>& str) noexcept {
			string_literal<size_new> new_literal{ lhs };
			return new_literal + str;
		}

		BNCH_SWT_HOST constexpr reference operator[](size_type index) noexcept {
			return data_val[index];
		}

		BNCH_SWT_HOST constexpr const_reference operator[](size_type index) const noexcept {
			return data_val[index];
		}

		BNCH_SWT_HOST static constexpr size_type size() noexcept {
			return static_cast<size_type>(length);
		}

		template<typename string_types> BNCH_SWT_HOST explicit constexpr operator string_types() const noexcept {
			string_types return_datas{ data_val, size() };
			return return_datas;
		}

		BNCH_SWT_ALIGN(cpu_properties::cpu_alignment) value_type data_val[size_val] {};
	};

	template<uint64_t size> string_literal(const char (&str)[size]) -> string_literal<size>;

	template<uint64_t size> BNCH_SWT_HOST std::ostream& operator<<(std::ostream&os, const string_literal<size>& input) noexcept {
		os.write(input.data_val, input.size());
		return os;
	}

	namespace internal {

		BNCH_SWT_HOST_DEVICE constexpr uint64_t str_len(const char* input) noexcept {
			std::string_view string{ input };
			return string.size();
		}

		template<uint64_t N, typename string_type> constexpr auto string_literal_from_view(string_type str) noexcept {
			string_literal<N + 1> sl{};
			std::copy_n(str.data_val, str.size(), sl.data_val);
			sl[N] = '\0';
			return sl;
		}

		template<typename value_type> constexpr uint64_t count_digits(value_type number) noexcept {
			uint64_t count = 0;
			if (static_cast<int64_t>(number) < 0) {
				number *= -1;
				++count;
			}
			do {
				++count;
				number /= 10;
			} while (number != 0);
			return count;
		}

		template<int64_t number, uint64_t num_digits = count_digits(number)> constexpr string_literal<num_digits + 1> to_string_literal() noexcept {
			char buffer[num_digits + 1]{};
			char* ptr = buffer + num_digits;
			*ptr	  = '\0';
			int64_t temp{};
			if constexpr (number < 0) {
				temp				= number * -1;
				*(ptr - num_digits) = '-';
			} else {
				temp = number;
			}
			do {
				*--ptr = '0' + (temp % 10);
				temp /= 10;
			} while (temp != 0);
			return string_literal<num_digits + 1>{ buffer };
		}

		constexpr char to_lower(char input) noexcept {
			return (input >= 'A' && input <= 'Z') ? (input + 32) : input;
		}

		template<uint64_t size> constexpr auto to_lower(string_literal<size> input) noexcept {
			string_literal<size> output{};
			for (uint64_t x = 0; x < size; ++x) {
				output[x] = to_lower(input[x]);
			}
			return output;
		}

	}

}
