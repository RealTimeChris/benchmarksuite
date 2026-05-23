#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <bit>

using float16				   = half;
using float32				   = float;
using float64				   = double;
using uint8					   = unsigned char;
using int8					   = signed char;
using uint16				   = unsigned short;
using int16					   = signed short;
using uint32				   = unsigned int;
using int32					   = signed int;
using uint64				   = std::conditional_t<sizeof(unsigned long) == 8ULL, unsigned long, unsigned long long>;
using int64					   = std::conditional_t<sizeof(signed long) == 8ULL, signed long, signed long long>;
using nothing				   = void;
using forbidden_ptr_type	   = void*;
using const_forbidden_ptr_type = const void*;

template<typename value_type> using base_t = std::remove_cvref_t<value_type>;

template<typename value_type>
concept uint64_types = sizeof(value_type) == 8;

template<uint64_t size> using tag = std::integral_constant<uint64_t, size>;

#define NIHILUS_DEVICE __device__
#define NIHILUS_HOST_DEVICE __host__ __device__
#define NIHILUS_ALIGN(x) alignas(x)

template<typename value_type, uint64_types auto dim_count> struct NIHILUS_ALIGN(sizeof(value_type) * dim_count) member_mixin;

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type)) member_mixin<value_type, 1ULL> {
	using scalar_type = value_type;
	value_type x;
};

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type) * 2ULL) member_mixin<value_type, 2ULL> {
	using scalar_type = value_type;
	value_type x;
	value_type y;
};

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type) * 4ULL) member_mixin<value_type, 4ULL> {
	using scalar_type = value_type;
	value_type x;
	value_type y;
	value_type z;
	value_type w;
};

enum class tag_indexer_errors {
	invalid_index,
};

template<uint64_types auto index> struct element_getter;

template<typename scalar_type_new, uint64_types auto dim_count_new> struct NIHILUS_ALIGN(sizeof(scalar_type_new) * dim_count_new) tag_indexing_mixin
	: public member_mixin<scalar_type_new, dim_count_new> {
	using base_type	  = member_mixin<scalar_type_new, dim_count_new>;
	using scalar_type = typename base_type::scalar_type;
	static constexpr uint64 dim_count{ dim_count_new };

	template<uint64_types auto index> NIHILUS_HOST_DEVICE decltype(auto) operator[](tag<index>) const&& noexcept {
		return device_cast<const scalar_type&&>(element_getter<index>::impl(*this));
	}

	template<uint64_types auto index> NIHILUS_HOST_DEVICE decltype(auto) operator[](tag<index>) const& noexcept {
		return device_cast<const scalar_type&>(element_getter<index>::impl(*this));
	}

	template<uint64_types auto index> NIHILUS_HOST_DEVICE decltype(auto) operator[](tag<index>) && noexcept {
		return device_cast<scalar_type&&>(element_getter<index>::impl(*this));
	}

	template<uint64_types auto index> NIHILUS_HOST_DEVICE decltype(auto) operator[](tag<index>) & noexcept {
		return device_cast<scalar_type&>(element_getter<index>::impl(*this));
	}
};

template<> struct element_getter<0ULL> {
	template<typename derived_type> NIHILUS_HOST_DEVICE static decltype(auto) impl(derived_type&& derived) {
		using scalar_type = typename base_t<derived_type>::scalar_type;
		return forward_as_t<derived_type&&, scalar_type>(derived.x);
	}
};

template<> struct element_getter<1ULL> {
	template<typename derived_type> NIHILUS_HOST_DEVICE static decltype(auto) impl(derived_type&& derived) {
		using scalar_type = typename base_t<derived_type>::scalar_type;
		return forward_as_t<derived_type&&, scalar_type>(derived.y);
	}
};

template<> struct element_getter<2ULL> {
	template<typename derived_type> NIHILUS_HOST_DEVICE static decltype(auto) impl(derived_type&& derived) {
		using scalar_type = typename base_t<derived_type>::scalar_type;
		return forward_as_t<derived_type&&, scalar_type>(derived.z);
	}
};

template<> struct element_getter<3ULL> {
	template<typename derived_type> NIHILUS_HOST_DEVICE static decltype(auto) impl(derived_type&& derived) {
		using scalar_type = typename base_t<derived_type>::scalar_type;
		return forward_as_t<derived_type&&, scalar_type>(derived.w);
	}
};

template<typename value_type> struct x_type_impl {
	using type = value_type;
};

template<typename value_type> using x_type = typename x_type_impl<value_type>::type;

template<auto... rest_types> struct first_val;

template<auto first_new, auto... rest_vals> struct first_val<first_new, rest_vals...> {
	static constexpr auto value{ first_new };
	using type = base_t<decltype(value)>;
};

template<auto... rest_vals> using first_val_t = typename first_val<rest_vals...>::type;

template<auto... rest_vals> static constexpr auto first_val_v = first_val<rest_vals...>::value;

template<typename value_type, uint64_types auto... values> struct NIHILUS_ALIGN(sizeof(value_type) * first_val_v<values...>) nh_vec_base : public value_type {
	using scalar_type = x_type<value_type>;
};

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type)) nh_vec_base<value_type, 1ULL> : public tag_indexing_mixin<value_type, 1ULL> {
	static constexpr uint64 dim_count{ 1ULL };
	using scalar_type = value_type;
};

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type) * 2ULL) nh_vec_base<value_type, 2ULL> : public tag_indexing_mixin<value_type, 2ULL> {
	static constexpr uint64 dim_count{ 2ULL };
	using scalar_type = value_type;
};

template<typename value_type> struct NIHILUS_ALIGN(sizeof(value_type) * 4ULL) nh_vec_base<value_type, 4ULL> : public tag_indexing_mixin<value_type, 4ULL> {
	static constexpr uint64 dim_count{ 4ULL };
	using scalar_type = value_type;
};

template<typename value_type>
concept vector_1_types = sizeof(typename base_t<value_type>::scalar_type) == 1 && sizeof(value_type) == 16;

template<typename value_type>
concept vector_2_types = sizeof(typename base_t<value_type>::scalar_type) == 2 && sizeof(value_type) == 16;

template<typename value_type>
concept vector_4_types = sizeof(typename base_t<value_type>::scalar_type) == 4 && sizeof(value_type) == 16;

template<typename value_type>
concept vector_8_types = sizeof(typename base_t<value_type>::scalar_type) == 8 && sizeof(value_type) == 16;

template<typename value_type> NIHILUS_HOST_DEVICE bool operator==(const nh_vec_base<value_type, 1ULL>& lhs, const nh_vec_base<value_type, 1ULL>& rhs) noexcept {
	return lhs.x == rhs.x;
}

template<typename value_type> NIHILUS_HOST_DEVICE bool operator==(const nh_vec_base<value_type, 2ULL>& lhs, const nh_vec_base<value_type, 2ULL>& rhs) noexcept {
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<typename value_type> NIHILUS_HOST_DEVICE bool operator==(const nh_vec_base<value_type, 4ULL>& lhs, const nh_vec_base<value_type, 4ULL>& rhs) noexcept {
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

template<typename value_type> using nh_quantized = value_type;
using nh_float16_8								 = nh_vec_base<nh_vec_base<float16, 4ULL>, 2ULL>;
using nh_float32_4								 = nh_vec_base<float32, 4ULL>;
using nh_float64_2								 = nh_vec_base<float64, 2ULL>;
using nh_uint8_16								 = nh_vec_base<nh_vec_base<uint8, 4ULL>, 4ULL>;
using nh_int8_16								 = nh_vec_base<nh_vec_base<int8, 4ULL>, 4ULL>;
using nh_uint16_8								 = nh_vec_base<nh_vec_base<uint16, 4ULL>, 2ULL>;
using nh_int16_8								 = nh_vec_base<nh_vec_base<int16, 4ULL>, 2ULL>;
using nh_uint32_4								 = nh_vec_base<uint32, 4ULL>;
using nh_int32_4								 = nh_vec_base<int32, 4ULL>;
using nh_uint64_2								 = nh_vec_base<uint64, 2ULL>;
using nh_int64_2								 = nh_vec_base<int64, 2ULL>;
using nh_bfloat16_8								 = nh_vec_base<nh_vec_base<__nv_bfloat16, 4ULL>, 2ULL>;

template<typename value_type_01, typename value_type_02>
concept static_castable_types = requires { static_cast<value_type_02>(std::declval<value_type_01>()); } &&
	!std::is_same_v<std::remove_cvref_t<value_type_01>, forbidden_ptr_type> && !std::is_same_v<std::remove_cvref_t<value_type_02>, forbidden_ptr_type>;

template<typename value_type_01, typename value_type_02>
concept bit_castable_types = !static_castable_types<value_type_01, value_type_02> && requires { std::bit_cast<value_type_02>(std::declval<value_type_01>()); };

template<typename value_type_01, static_castable_types<value_type_01> value_type_02>
NIHILUS_HOST_DEVICE static constexpr value_type_01 device_cast(value_type_02&& value) noexcept {
	return static_cast<value_type_01>(value);
}

template<typename value_type_01, typename value_type_02> NIHILUS_HOST_DEVICE static constexpr value_type_01 device_cast(value_type_02&& value) noexcept {
	return static_cast<value_type_01>(static_cast<void*>(value));
}

template<typename value_type> struct copy_hbm_to_persisted_l2;
template<typename value_type> struct copy_persisted_l2_to_hbm;
template<typename value_type> struct copy_persisted_l2_to_shared;
template<typename value_type> struct copy_shared_to_persisted_l2;
template<typename value_type> struct load_shared_to_reg;
template<typename value_type> struct store_reg_to_shared;

template<vector_1_types value_type> struct copy_hbm_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.nc.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_1_types value_type> struct copy_persisted_l2_to_hbm<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.wt.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_1_types value_type> struct copy_persisted_l2_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, const value_type* l2_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 "@%%p0 cp.async.ca.shared.global [%0],[%1],16;}\n"
			:
			: "r"(smem_ptr), "l"(l2_src), "l"(index)
			: "memory");
	}
};

template<vector_1_types value_type> struct copy_shared_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* l2_dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.shared.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(l2_dst), "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_1_types value_type> struct load_shared_to_reg<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type& dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		uint32* raw_reg = device_cast<uint32*>(&dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 ld.shared.v4.b32 {%0, %1, %2, %3},[%4];}\n"
			: "+r"(raw_reg[0]), "+r"(raw_reg[1]), "+r"(raw_reg[2]), "+r"(raw_reg[3])
			: "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_1_types value_type> struct store_reg_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, value_type src, uint64 index) {
		uint32 smem_ptr		  = __cvta_generic_to_shared(shared_dst);
		const uint32* raw_reg = device_cast<const uint32*>(&src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 st.shared.v4.b32 [%0],{%1, %2, %3, %4};}\n"
			:
			: "r"(smem_ptr), "r"(raw_reg[0]), "r"(raw_reg[1]), "r"(raw_reg[2]), "r"(raw_reg[3]), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct copy_hbm_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.nc.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct copy_persisted_l2_to_hbm<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.wt.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct copy_persisted_l2_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, const value_type* l2_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 "@%%p0 cp.async.ca.shared.global [%0],[%1],16;}\n"
			:
			: "r"(smem_ptr), "l"(l2_src), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct copy_shared_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* l2_dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.shared.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(l2_dst), "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct load_shared_to_reg<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type& dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		uint32* raw_reg = device_cast<uint32*>(&dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 ld.shared.v4.b32 {%0, %1, %2, %3},[%4];}\n"
			: "+r"(raw_reg[0]), "+r"(raw_reg[1]), "+r"(raw_reg[2]), "+r"(raw_reg[3])
			: "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_2_types value_type> struct store_reg_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, value_type src, uint64 index) {
		uint32 smem_ptr		  = __cvta_generic_to_shared(shared_dst);
		const uint32* raw_reg = device_cast<const uint32*>(&src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 st.shared.v4.b32 [%0],{%1, %2, %3, %4};}\n"
			:
			: "r"(smem_ptr), "r"(raw_reg[0]), "r"(raw_reg[1]), "r"(raw_reg[2]), "r"(raw_reg[3]), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct copy_hbm_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.nc.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct copy_persisted_l2_to_hbm<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.global.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.wt.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct copy_persisted_l2_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, const value_type* l2_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 "@%%p0 cp.async.ca.shared.global [%0],[%1],16;}\n"
			:
			: "r"(smem_ptr), "l"(l2_src), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct copy_shared_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* l2_dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b32 r0, r1, r2, r3;\n"
					 "@%%p0 ld.shared.v4.b32 {r0, r1, r2, r3},[%1];\n"
					 "@%%p0 st.global.v4.b32 [%0],{r0, r1, r2, r3};}\n"
			:
			: "l"(l2_dst), "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct load_shared_to_reg<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type& dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		uint32* raw_reg = device_cast<uint32*>(&dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 ld.shared.v4.b32 {%0, %1, %2, %3},[%4];}\n"
			: "+r"(raw_reg[0]), "+r"(raw_reg[1]), "+r"(raw_reg[2]), "+r"(raw_reg[3])
			: "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_4_types value_type> struct store_reg_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, value_type src, uint64 index) {
		uint32 smem_ptr		  = __cvta_generic_to_shared(shared_dst);
		const uint32* raw_reg = device_cast<const uint32*>(&src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%5,0;\n"
					 "@%%p0 st.shared.v4.b32 [%0],{%1, %2, %3, %4};}\n"
			:
			: "r"(smem_ptr), "r"(raw_reg[0]), "r"(raw_reg[1]), "r"(raw_reg[2]), "r"(raw_reg[3]), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct copy_hbm_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b64 r0, r1;\n"
					 "@%%p0 ld.global.nc.v2.b64 {r0, r1},[%1];\n"
					 "@%%p0 st.global.v2.b64 [%0],{r0, r1};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct copy_persisted_l2_to_hbm<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* dst, const value_type* src, uint64 index) {
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b64 r0, r1;\n"
					 "@%%p0 ld.global.v2.b64 {r0, r1},[%1];\n"
					 "@%%p0 st.global.wt.v2.b64 [%0],{r0, r1};}\n"
			:
			: "l"(dst), "l"(src), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct copy_persisted_l2_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, const value_type* l2_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 "@%%p0 cp.async.ca.shared.global [%0],[%1],16;}\n"
			:
			: "r"(smem_ptr), "l"(l2_src), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct copy_shared_to_persisted_l2<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type* l2_dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%2,0;\n"
					 ".reg.b64 r0, r1;\n"
					 "@%%p0 ld.shared.v2.b64 {r0, r1},[%1];\n"
					 "@%%p0 st.global.v2.b64 [%0],{r0, r1};}\n"
			:
			: "l"(l2_dst), "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct load_shared_to_reg<value_type> {
	NIHILUS_DEVICE static nothing impl(value_type& dst, const_forbidden_ptr_type shared_src, uint64 index) {
		uint32 smem_ptr = __cvta_generic_to_shared(shared_src);
		uint64* raw_reg = device_cast<uint64*>(&dst);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%3,0;\n"
					 "@%%p0 ld.shared.v2.b64 {%0, %1},[%2];}\n"
			: "+l"(raw_reg[0]), "+l"(raw_reg[1])
			: "r"(smem_ptr), "l"(index)
			: "memory");
	}
};

template<vector_8_types value_type> struct store_reg_to_shared<value_type> {
	NIHILUS_DEVICE static nothing impl(forbidden_ptr_type shared_dst, value_type src, uint64 index) {
		uint32 smem_ptr		  = __cvta_generic_to_shared(shared_dst);
		const uint64* raw_reg = device_cast<const uint64*>(&src);
		asm volatile("{.reg.pred %%p0;\n"
					 "setp.eq.u64 %%p0,%3,0;\n"
					 "@%%p0 st.shared.v2.b64 [%0],{%1, %2};}\n"
			:
			: "r"(smem_ptr), "l"(raw_reg[0]), "l"(raw_reg[1]), "l"(index)
			: "memory");
	}
};

template<typename value_type> struct test_struct;

template<vector_1_types value_type> struct test_struct<value_type> {
	static void impl(value_type* value) {
	}
};

template<vector_2_types value_type> struct test_struct<value_type> {
	static void impl(value_type* value) {
	}
};

template<vector_4_types value_type> struct test_struct<value_type> {
	static void impl(value_type* value) {
	}
};

template<vector_8_types value_type> struct test_struct<value_type> {
	static void impl(value_type* value) {
	}
};

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

#define CUDA_CHECK(expr) \
	do { \
		cudaError_t err_ = (expr); \
		if (err_ != cudaSuccess) { \
			std::cerr << "CUDA error " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
			std::exit(1); \
		} \
	} while (0)

template<vector_1_types value_type> __global__ void roundtrip_kernel_v1(value_type* dst, const value_type* src) {
	extern __shared__ unsigned char smem_raw[];
	value_type* smem = device_cast<value_type*>(smem_raw);
	value_type reg{};
	copy_hbm_to_persisted_l2<value_type>::impl(dst, src, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_shared<value_type>::impl(smem, dst, 0ULL);
	asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	load_shared_to_reg<value_type>::impl(reg, smem, 0ULL);
	__syncthreads();
	store_reg_to_shared<value_type>::impl(smem, reg, 0ULL);
	__syncthreads();
	copy_shared_to_persisted_l2<value_type>::impl(dst, smem, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_hbm<value_type>::impl(dst, dst, 0ULL);
}

template<vector_2_types value_type> __global__ void roundtrip_kernel_v2(value_type* dst, const value_type* src) {
	extern __shared__ unsigned char smem_raw[];
	value_type* smem = device_cast<value_type*>(smem_raw);
	value_type reg{};
	copy_hbm_to_persisted_l2<value_type>::impl(dst, src, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_shared<value_type>::impl(smem, dst, 0ULL);
	asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	load_shared_to_reg<value_type>::impl(reg, smem, 0ULL);
	__syncthreads();
	store_reg_to_shared<value_type>::impl(smem, reg, 0ULL);
	__syncthreads();
	copy_shared_to_persisted_l2<value_type>::impl(dst, smem, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_hbm<value_type>::impl(dst, dst, 0ULL);
}

template<vector_4_types value_type> __global__ void roundtrip_kernel_v4(value_type* dst, const value_type* src) {
	extern __shared__ unsigned char smem_raw[];
	value_type* smem = device_cast<value_type*>(smem_raw);
	value_type reg{};
	copy_hbm_to_persisted_l2<value_type>::impl(dst, src, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_shared<value_type>::impl(smem, dst, 0ULL);
	asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	load_shared_to_reg<value_type>::impl(reg, smem, 0ULL);
	__syncthreads();
	store_reg_to_shared<value_type>::impl(smem, reg, 0ULL);
	__syncthreads();
	copy_shared_to_persisted_l2<value_type>::impl(dst, smem, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_hbm<value_type>::impl(dst, dst, 0ULL);
}

template<vector_8_types value_type> __global__ void roundtrip_kernel_v8(value_type* dst, const value_type* src) {
	extern __shared__ unsigned char smem_raw[];
	value_type* smem = device_cast<value_type*>(smem_raw);
	value_type reg{};
	copy_hbm_to_persisted_l2<value_type>::impl(dst, src, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_shared<value_type>::impl(smem, dst, 0ULL);
	asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	load_shared_to_reg<value_type>::impl(reg, smem, 0ULL);
	__syncthreads();
	store_reg_to_shared<value_type>::impl(smem, reg, 0ULL);
	__syncthreads();
	copy_shared_to_persisted_l2<value_type>::impl(dst, smem, 0ULL);
	__syncthreads();
	copy_persisted_l2_to_hbm<value_type>::impl(dst, dst, 0ULL);
}

enum class tier { v1, v2, v4, v8 };

template<typename value_type> constexpr tier tier_of() {
	constexpr uint64 s = sizeof(typename value_type::scalar_type);
	if constexpr (s == 1) {
		return tier::v1;
	} else if constexpr (s == 2) {
		return tier::v2;
	} else if constexpr (s == 4) {
		return tier::v4;
	} else {
		return tier::v8;
	}
}

template<typename value_type> bool run_case(const char* name) {
	constexpr uint64 bytes = sizeof(value_type);
	std::vector<unsigned char> host_in(bytes), host_out(bytes);
	for (uint64 i = 0; i < bytes; ++i) {
		host_in[i] = static_cast<unsigned char>((i * 37 + 11) & 0xFF);
	}

	value_type* dev_src{};
	value_type* dev_dst{};
	CUDA_CHECK(cudaMalloc(&dev_src, bytes));
	CUDA_CHECK(cudaMalloc(&dev_dst, bytes));
	CUDA_CHECK(cudaMemcpy(dev_src, host_in.data(), bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(dev_dst, 0, bytes));

	constexpr tier which = tier_of<value_type>();
	if constexpr (which == tier::v1) {
		roundtrip_kernel_v1<value_type><<<1, 1, bytes>>>(dev_dst, dev_src);
	} else if constexpr (which == tier::v2) {
		roundtrip_kernel_v2<value_type><<<1, 1, bytes>>>(dev_dst, dev_src);
	} else if constexpr (which == tier::v4) {
		roundtrip_kernel_v4<value_type><<<1, 1, bytes>>>(dev_dst, dev_src);
	} else {
		roundtrip_kernel_v8<value_type><<<1, 1, bytes>>>(dev_dst, dev_src);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(host_out.data(), dev_dst, bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(dev_src));
	CUDA_CHECK(cudaFree(dev_dst));

	bool ok = std::memcmp(host_in.data(), host_out.data(), bytes) == 0;
	std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << " (" << bytes << " bytes)" << std::endl;
	if (!ok) {
		for (uint64 i = 0; i < bytes; ++i) {
			if (host_in[i] != host_out[i]) {
				std::cout << "  byte " << i << ": expected 0x" << std::hex << static_cast<uint32>(host_in[i]) << " got 0x" << static_cast<uint32>(host_out[i]) << std::dec
						  << std::endl;
			}
		}
	}
	return ok;
}

int main() {
	int dev_count{};
	CUDA_CHECK(cudaGetDeviceCount(&dev_count));
	if (dev_count == 0) {
		std::cerr << "no CUDA device" << std::endl;
		return 1;
	}
	CUDA_CHECK(cudaSetDevice(0));

	bool all = true;
	all &= run_case<nh_float16_8>("nh_float16_8");
	all &= run_case<nh_float32_4>("nh_float32_4");
	all &= run_case<nh_float64_2>("nh_float64_2");
	all &= run_case<nh_uint8_16>("nh_uint8_16");
	all &= run_case<nh_int8_16>("nh_int8_16");
	all &= run_case<nh_uint16_8>("nh_uint16_8");
	all &= run_case<nh_int16_8>("nh_int16_8");
	all &= run_case<nh_uint32_4>("nh_uint32_4");
	all &= run_case<nh_int32_4>("nh_int32_4");
	all &= run_case<nh_uint64_2>("nh_uint64_2");
	all &= run_case<nh_int64_2>("nh_int64_2");
	all &= run_case<nh_bfloat16_8>("nh_int64_2");

	std::cout << (all ? "ALL PASSED" : "SOME FAILED") << std::endl;
	return all ? 0 : 1;
}