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

#include <bnch_swt/concepts.hpp>
#include <type_traits>
#include <concepts>
#include <cstdint>
#include <variant>

namespace bnch_swt {

	namespace internal {	

		template<const auto& function, uint64_t currentIndex = 0, typename variant_type, typename... arg_types>
		constexpr void visit(variant_type&& variant, arg_types&&... args) noexcept {
			if constexpr (currentIndex < std::variant_size_v<std::remove_cvref_t<variant_type>>) {
				variant_type&& variantNew = std::forward<variant_type>(variant);
				if (variantNew.index() == currentIndex) {
					function(std::get<currentIndex>(std::forward<variant_type>(variantNew)), std::forward<arg_types>(args)...);
					return;
				}
				visit<function, currentIndex + 1>(std::forward<variant_type>(variantNew), std::forward<arg_types>(args)...);
			}
		}

		template<typename value_type> void printValue(std::ostream& os, const value_type& value) {
			os << value;
		}

		template<bool_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << std::boolalpha << value;
		}

		template<string_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "\"" << value << "\"";
		}

		template<string_view_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "\"" << value << "\"";
		}

		template<vector_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "[";
			for (uint64_t x = 0; x < value.size(); ++x) {
				printValue(os, value[x]);
				if (x < value.size() - 1) {
					os << ',';
				}
			}
			os << "]";
		}

		template<map_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			uint64_t index{};
			for (auto iter = value.begin(); iter != value.end(); ++iter) {
				printValue(os, iter->first);
				os << ":";
				printValue(os, iter->second);
				if (index < value.size() - 1) {
					os << ",";
				}
				++index;
			}
			os << "}";
		}

		template<variant_t value_type> void printValue(std::ostream& os, const value_type& value) {
			static constexpr auto lambda = [](auto&& valueNew, auto& osNew) {
				printValue(osNew, valueNew);
			};
			visit<lambda>(value, os);
		}

		template<optional_t value_type> void printValue(std::ostream& os, const value_type& value) {
			if (value.has_value()) {
				printValue(os, value.value());
			}
		}

		template<pair_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			printValue(os, std::get<0>(value));
			os << ",";
			printValue(os, std::get<1>(value));
			os << "}";
		}

		template<tuple_t value_type, uint64_t index> void printValue(std::ostream& os, const value_type& value) {
			if constexpr (index < std::tuple_size_v<std::remove_cvref_t<value_type>>) {
				printValue(os, std::get<index>(value));
				if constexpr (index < std::tuple_size_v<std::remove_cvref_t<value_type>> - 1) {
					os << ",";
				}
				printValue<value_type, index + 1>(os, value);
			}
		}

		template<tuple_t value_type> void printValue(std::ostream& os, const value_type& value) {
			os << "{";
			printValue<value_type, 0>(os, value);
			os << "}";
		}

	}

}

template<bnch_swt::internal::vector_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::map_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::variant_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::optional_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::pair_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}

template<bnch_swt::internal::tuple_t value_type> std::ostream& operator<<(std::ostream& os, const value_type& value) {
	bnch_swt::internal::printValue(os, value);
	return os;
}
