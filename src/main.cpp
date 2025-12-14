#include <bnch_swt/index.hpp>
#include <iostream>
#include <vector>
#include <string>

template<typename... bases> struct BNCH_SWT_ALIGN(64) static_cathedral : public bases... {
	using bases::operator[]...;

	template<template<typename, typename> typename mixin_type, typename... arg_types> BNCH_SWT_HOST void impl(arg_types&&... args) noexcept {
		(impl_internal_filtered<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
	}

  protected:
	template<template<typename, typename> typename mixin_type, typename base_type, typename... arg_types> BNCH_SWT_HOST void impl_internal_filtered(arg_types&&... args) noexcept {
		if constexpr (mixin_type<int, base_type>::filter()) {
			mixin_type<int, base_type>::impl(*this, std::forward<arg_types>(args)...);
		}
	}
};

template<uint64_t index> struct base_a {
	void operator[](auto) {
	}
};

template<uint64_t index> struct base_b {
	void operator[](auto) {
	}
};

struct fnv1a_64 {
	static constexpr uint64_t offset_basis{ 14695981039346656037ull };
	static constexpr uint64_t prime{ 1099511628211ull };

	BNCH_SWT_HOST_DEVICE static constexpr uint64_t hash(uint64_t value) noexcept {
		uint64_t hash_val{ offset_basis };

		for (uint64_t i = 0; i < 8; ++i) {
			hash_val ^= (value & 0xFFull);
			hash_val *= prime;
			value >>= 8ull;
		}

		return hash_val;
	}
};

template<typename T, typename B> struct work_mixin {
	BNCH_SWT_HOST static constexpr bool filter() {
		return true;
	}
	BNCH_SWT_HOST static void impl(auto& cathedral, uint64_t& counter) {
		counter += 1;
		counter = fnv1a_64::hash(counter);
		bnch_swt::do_not_optimize_away(counter);
	}
};

template<typename cathedral_type> struct benchmark_nihilus_cathedral {
	BNCH_SWT_HOST static uint64_t impl(cathedral_type& cathedral) {
		uint64_t counter_new{};
		for (uint64_t x = 0; x < 1024 * 1024; ++x) {
			cathedral.template impl<work_mixin>(counter_new);
			bnch_swt::do_not_optimize_away(counter_new);
		}
		return counter_new / (2048ull * 2048ull * 16384ull * 2);
	}
};

template<typename tuple_type> struct benchmark_glue_factory {
	BNCH_SWT_HOST static uint64_t impl(tuple_type& glue_factory) {
		uint64_t counter_new{};
		for (uint64_t x = 0; x < 1024 * 1024; ++x) {
			std::apply(
				[&](auto&... base_objs) {
					((work_mixin<int, decltype(base_objs)>::filter() ? work_mixin<int, decltype(base_objs)>::impl(glue_factory, counter_new) : void()), ...);
				},
				glue_factory.data);
			bnch_swt::do_not_optimize_away(counter_new);
		}
		return counter_new / (2048ull * 2048ull * 16384ull * 2);
	}
};

template<bnch_swt::string_literal name> inline void run_cathedral_duel() {
	using namespace bnch_swt;
	static constexpr auto max_iterations{ 200 };
	static constexpr auto measured_iterations{ 10 };

	static_cathedral<base_a<0>, base_b<1>, base_a<2>, base_b<3>> nihilus;

	struct glue_factory_t {
		std::tuple<base_a<0>, base_b<1>, base_a<2>, base_b<3>> data;
	} glue_factory;

	using bench = benchmark_stage<name, max_iterations, measured_iterations, benchmark_types::cpu, false>;

	std::cout << "--- STARTING CATHEDRAL DUEL ---" << std::endl;
	bench::template run_benchmark<"Nihilus_Cathedral", benchmark_nihilus_cathedral<decltype(nihilus)>>(nihilus);
	bench::template run_benchmark<"OpenAI_Glue_Factory", benchmark_glue_factory<decltype(glue_factory)>>(glue_factory);

	bench::print_results(true, true);
}

int main() {
	run_cathedral_duel<"Performance Duel">();
	return 0;
}