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
#include <bnch_swt/index.hpp>
#include <source_location>
#include <atomic>
#include <thread>

static constexpr size_t total_iterations{ 10000 };
static constexpr size_t measured_iterations{ 10 };
static constexpr size_t wait_notify_cycles{ 1000 };

template<typename value_type> void test_function() {
	std::cout << std::source_location::current().function_name() << std::endl;
}

#include <bnch_swt/index.hpp>
#include <source_location>
#include <atomic>
#include <thread>
#include <algorithm>

template<auto index> using tag = std::integral_constant<size_t, static_cast<size_t>(index)>;

template<auto index, typename derived_type_new> struct variant_elem_base {
	using derived_type = derived_type_new;

	BNCH_SWT_HOST_DEVICE constexpr derived_type& operator[](tag<index>) & noexcept {
		return *reinterpret_cast<derived_type*>(this);
	}

	BNCH_SWT_HOST_DEVICE constexpr const derived_type& operator[](tag<index>) const& noexcept {
		return *reinterpret_cast<const derived_type*>(this);
	}
};

template<typename... bases> struct nihilus_cathedral_base : public bases... {
	using bases::operator[]...;

	template<auto index> BNCH_SWT_HOST_DEVICE decltype(auto) get_core_by_enum() noexcept {
		return (*this)[tag<index>()];
	}
};

template<typename nihilus_cathedral_type, auto index> using get_nihilus_cathedral_base_type_at_index =
	std::remove_cvref_t<decltype(std::declval<nihilus_cathedral_type>().template get_core_by_enum<index>())>;

template<typename... value_types> struct get_nihilus_cathedral_base;

template<size_t... indices, typename... value_types> struct get_nihilus_cathedral_base<std::index_sequence<indices...>, value_types...> {
	using type = nihilus_cathedral_base<variant_elem_base<indices, value_types>...>;
};

template<typename... value_types> using get_nihilus_cathedral_base_t = typename get_nihilus_cathedral_base<std::make_index_sequence<sizeof...(value_types)>, value_types...>::type;

template<typename... types> struct nihilus_variant {
	static_assert(sizeof...(types) <= 255);

	BNCH_SWT_HOST_DEVICE constexpr nihilus_variant() noexcept {
	}

	template<typename type> BNCH_SWT_HOST_DEVICE constexpr nihilus_variant(type&& value) noexcept {
		using decayed_type	   = std::remove_cvref_t<type>;
		constexpr size_t index = get_type_index<decayed_type>();
		construct<index>(std::forward<type>(value));
	}

	template<size_t index, typename... types_new> BNCH_SWT_HOST_DEVICE constexpr void emplace(types_new&&... value) noexcept {
		destroy();
		construct<index>(std::forward<types_new>(value)...);
	}

	template<size_t target_index> BNCH_SWT_HOST_DEVICE constexpr decltype(auto) get() {
		if (active_index != target_index) {
			throw std::runtime_error{ "Incorrect variant access!" };
		}
		return (*reinterpret_cast<get_nihilus_cathedral_base_t<types...>*>(&storage))[tag<target_index>()];
	}

	template<typename function_type, typename... arg_types> BNCH_SWT_HOST_DEVICE void visit(arg_types&&... args) {
		if (active_index == 0xFF)
			return;
		visit_internal<function_type>(std::make_index_sequence<sizeof...(types)>{}, std::forward<arg_types>(args)...);
	}

	BNCH_SWT_HOST_DEVICE ~nihilus_variant() {
		destroy();
	}

  private:
	alignas(std::max(alignof(types)...)) unsigned char storage[std::max({ sizeof(types)... })];
	uint8_t active_index{ 0xFF };

	template<typename type> static consteval size_t get_type_index() {
		size_t index = 0;
		((std::is_same_v<type, types> ? false : (index++, true)) && ...);
		return index;
	}

	template<size_t index, typename... types_new> BNCH_SWT_HOST_DEVICE void construct(types_new&&... values) {
		using target_type = std::tuple_element_t<index, std::tuple<types...>>;
		new (&storage) target_type(std::forward<types_new>(values)...);
		active_index = static_cast<uint8_t>(index);
	}

	template<size_t current_index, typename function_type, typename... arg_types> BNCH_SWT_HOST_DEVICE bool visit_single(arg_types&&... args) {
		if (active_index == current_index) {
			auto& val = (*reinterpret_cast<get_nihilus_cathedral_base_t<types...>*>(&storage))[tag<current_index>()];
			function_type::impl(val, std::forward<arg_types>(args)...);
			return true;
		}
		return false;
	}

	template<typename function_type, size_t... indices, typename... arg_types> BNCH_SWT_HOST_DEVICE void visit_internal(std::index_sequence<indices...>, arg_types&&... args) {
		(visit_single<indices, function_type>(std::forward<arg_types>(args)...) || ...);
	}

	template<size_t current_index> BNCH_SWT_HOST_DEVICE bool destroy_single() {
		if (active_index == current_index) {
			using target_type = std::tuple_element_t<current_index, std::tuple<types...>>;
			reinterpret_cast<target_type*>(&storage)->~target_type();
			return true;
		}
		return false;
	}

	template<size_t... indices> BNCH_SWT_HOST_DEVICE void destroy_internal(std::index_sequence<indices...>) {
		(destroy_single<indices>() || ...);
	}

	BNCH_SWT_HOST_DEVICE void destroy() {
		if (active_index == 0xFF)
			return;
		destroy_internal(std::make_index_sequence<sizeof...(types)>{});
		active_index = 0xFF;
	}
};

struct test_struct_01 {
	size_t value{};
	test_struct_01() : value(1) {
	}
};

struct test_struct_02 {
	size_t value{};
	test_struct_02() : value(1) {
	}
};

struct visitor {
	BNCH_SWT_HOST_DEVICE static void impl(auto&& variant, size_t& counter) {
		counter += variant.value;
	};
};

int main() {
	struct nihilus_visit_benchmark {
		BNCH_SWT_HOST static size_t impl() {
			nihilus_variant<test_struct_01, test_struct_02> v1{ test_struct_01{} };
			size_t counter = 0;
			for (size_t i = 0; i < 1000000; ++i) {
				v1.visit<visitor>(counter);
				bnch_swt::do_not_optimize_away(counter);
			}
			return counter / 10;
		}
	};

	struct std_visit_benchmark {
		BNCH_SWT_HOST static size_t impl() {
			std::variant<test_struct_01, test_struct_02> v1{ test_struct_01{} };
			size_t counter = 0;
			for (size_t i = 0; i < 1000000; ++i) {
				std::visit(
					[&](auto&& arg) {
						visitor::impl(arg, counter);
					},
					v1);
				bnch_swt::do_not_optimize_away(counter);
			}
			return counter / 10;
		}
	};

	bnch_swt::benchmark_stage<"visit_comparison", 1000, 10>::run_benchmark<"nihilus_variant", nihilus_visit_benchmark>();
	bnch_swt::benchmark_stage<"visit_comparison", 1000, 10>::run_benchmark<"std_variant", std_visit_benchmark>();
	bnch_swt::benchmark_stage<"visit_comparison", 1000, 10>::print_results();

	return 0;
}