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
/// https://github.com/RealTimeChris/BenchmarkSuite
#include <bnch_swt/index.hpp>
#include <source_location>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstddef>

enum class core_types {
	// Weights.
	attn_q,
	attn_k,
	attn_v,
	attn_output,
	attn_norm,
	ffn_gate,
	ffn_up,
	ffn_down,
	moe_gate,
	moe_experts_gate,
	moe_experts_up,
	moe_experts_down,
	ffn_norm,
	token_embd,
	rope_freqs,
	output_norm,
	output,
	end_of_weights,
	// Global Inputs.
	inp_tokens,
	inp_pos,
	inp_out_ids,
	cache_k,
	cache_v,
	kq_mask,
	benchmark_data,
	end_of_input_only,
	// Token-Embeddings Mega-Kernel.
	inp_embd_get_rows,
	end_of_global_inputs,
	// attn_prep_and_score Mega-Kernel.
	norm_rms_norm,
	attn_norm_mul,
	qcur_mul_mat,
	qcur_reshape,
	qcur_rope,
	kcur_mul_mat,
	kcur_reshape,
	kcur_rope,
	vcur_mul_mat,
	k_cache_view,
	k_cache_view_copy,
	vcur_transpose,
	v_cache_view,
	v_cache_view_copy,
	v_view,
	k_view,
	q_permute,
	kq_mul_mat,
	// attn_and_ffn_out Mega-Kernel (Dense FFN - Llama).
	kq_soft_max,
	kqv_mul_mat,
	kqv_merged_permute,
	kqv_merged_cont,
	kqv_out_mul_mat,
	ffn_inp_add,
	norm_pre_ffn_rms_norm,
	ffn_norm_mul,
	ffn_gate_mul_mat,
	ffn_silu,
	ffn_up_mul_mat,
	ffn_gate_par_mul,
	ffn_out_mul_mat,
	// attn_and_moe_out Mega-Kernel (MoE - Grok).
	moe_inp_add,
	norm_pre_moe_rms_norm,
	moe_norm_mul,
	moe_router_mul_mat,
	moe_router_softmax,
	moe_expert_select,
	moe_expert_gate_mul_mat,
	moe_expert_silu,
	moe_expert_up_mul_mat,
	moe_expert_gate_par_mul,
	moe_expert_down_mul_mat,
	moe_expert_weighted_sum,
	layer_out_add,
	end_of_per_block,
	// global_output_and_sampling Mega-Kernel (Dense FFN - Llama).
	node_1016_get_rows,
	node_1017_get_rows,
	final_ffn_inp_add,
	final_norm_pre_rms_norm,
	final_ffn_norm_mul,
	final_ffn_gate_mul_mat,
	final_ffn_silu,
	final_ffn_up_mul_mat,
	final_ffn_gate_par_mul,
	final_ffn_out_mul_mat,
	// global_output_and_sampling Mega-Kernel (MoE - Grok).
	final_moe_inp_add,
	final_norm_pre_moe_rms_norm,
	final_moe_norm_mul,
	final_moe_router_mul_mat,
	final_moe_router_softmax,
	final_moe_expert_select,
	final_moe_expert_gate_mul_mat,
	final_moe_expert_silu,
	final_moe_expert_up_mul_mat,
	final_moe_expert_gate_par_mul,
	final_moe_expert_down_mul_mat,
	final_moe_expert_weighted_sum,
	final_layer_out_add,
	final_norm_rms_norm,
	result_norm_mul,
	result_output_mul_mat,
	sample_tokens,
	count
};

enum class kernel_types : uint8_t {
	weights,
	global_inputs,
	get_rows,
	rms_norm,
	mul,
	mul_mat,
	mul_mat_moe,
	reshape,
	transpose,
	permute,
	view,
	rope,
	softmax,
	silu,
	copy,
	cont,
	add,
	sub,
	div,
	top_k,
	weighted_sum,
	sample_tokens,
	count,
};

enum class device_types : uint8_t {
	cpu,
	gpu,
	numa,
};

enum class model_arches : uint8_t {
	llama,
	deci,
	falcon,
	baichuan,
	grok,
	gpt2,
	gptj,
	gptneox,
	mpt,
	starcoder,
	refact,
	bert,
	nomic_bert,
	jina_bert_v2,
	bloom,
	stablelm,
	qwen,
	qwen2,
	qwen2moe,
	qwen2vl,
	phi2,
	phi3,
	phimoe,
	plamo,
	codeshell,
	orion,
	internlm2,
	minicpm,
	minicpm3,
	gemma,
	gemma2,
	starcoder2,
	mamba,
	xverse,
	command_r,
	cohere2,
	dbrx,
	olmo,
	olmo2,
	olmoe,
	openelm,
	arctic,
	deepseek,
	deepseek2,
	chatglm,
	bitnet,
	t5,
	t5encoder,
	jais,
	nemotron,
	exaone,
	rwkv6,
	rwkv6qwen2,
	granite,
	granite_moe,
	chameleon,
	wavtokenizer_dec,
	unknown,
	count,
};

enum class kernel_type_profiles : uint8_t {
	fp16_mha,
	fp16_moe,
	bf16_mha,
	bf16_gqa,
	q4_mha,
	q4_gqa,
	q4_moe,
	q8_mha,
	q8_gqa,
	q8_moe,
	mixed_fp16_fp32,
	mixed_bf16_fp32,
	count,
};

enum class model_generations : uint8_t {
	v1,
	v1_v2,
	v1_5,
	v2,
	v3,
	v3_1,
	v3_2,
	count,
};

enum class model_sizes : uint8_t {
	llm_unknown,
	llm_14M,
	llm_17M,
	llm_22M,
	llm_33M,
	llm_60M,
	llm_70M,
	llm_80M,
	llm_109M,
	llm_137M,
	llm_160M,
	llm_220M,
	llm_250M,
	llm_270M,
	llm_335M,
	llm_410M,
	llm_450M,
	llm_770M,
	llm_780M,
	llm_0_5B,
	llm_1B,
	llm_1_3B,
	llm_1_4B,
	llm_1_5B,
	llm_1_6B,
	llm_2B,
	llm_2_8B,
	llm_3B,
	llm_4B,
	llm_6B,
	llm_6_9B,
	llm_7B,
	llm_8B,
	llm_9B,
	llm_11B,
	llm_12B,
	llm_13B,
	llm_14B,
	llm_15B,
	llm_16B,
	llm_20B,
	llm_30B,
	llm_32B,
	llm_34B,
	llm_35B,
	llm_40B,
	llm_46B,
	llm_65B,
	llm_70B,
	llm_314B,
	llm_405B,
	llm_SMALL,
	llm_MEDIUM,
	llm_LARGE,
	llm_XL,
	llm_A1_7B,
	llm_A2_7B,
	llm_8x7B,
	llm_8x22B,
	llm_16x12B,
	llm_16x3_8B,
	llm_10B_128x3_66B,
	llm_57B_A14B,
	llm_27B,
	count,
};

struct model_traits {
	static constexpr const char name[]{ "llama-3.1-8B" };
	static constexpr model_arches model_arch{ model_arches::llama };
	static constexpr model_generations model_generation{ model_generations::v3_1 };
	static constexpr model_sizes model_size{ model_sizes::llm_8B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 4096 * 2;
	static constexpr uint32_t block_count			  = 32;
	static constexpr uint32_t feed_forward_length	  = 14336;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<auto multiple, typename value_type_01 = decltype(multiple)> BNCH_SWT_HOST constexpr value_type_01 round_up_to_multiple(value_type_01 value) noexcept {
	if constexpr ((multiple > 0) && ((multiple & (multiple - 1)) == 0)) {
		constexpr value_type_01 mulSub1{ multiple - 1 };
		return (value + mulSub1) & ~mulSub1;
	} else {
		return ((value + multiple - 1) / multiple) * multiple;
	}
}

template<typename value_type> BNCH_SWT_HOST constexpr decltype(auto) move(value_type&& arg) noexcept {
	return static_cast<std::remove_reference_t<value_type>&&>(arg);
}

template<class value_type_01> BNCH_SWT_HOST constexpr void swap(value_type_01& left, value_type_01& right) noexcept(
	std::is_nothrow_move_constructible_v<value_type_01> && std::is_nothrow_move_assignable_v<value_type_01>) {
	value_type_01 tmp = ::move(left);
	left			  = ::move(right);
	right			  = ::move(tmp);
}

struct cuda_buffer {
	using size_type	 = uint64_t;
	using value_type = std::byte;
	using pointer	 = value_type*;
	BNCH_SWT_HOST cuda_buffer() noexcept {
	}
	BNCH_SWT_HOST cuda_buffer& operator=(const cuda_buffer&) noexcept = delete;
	BNCH_SWT_HOST cuda_buffer(const cuda_buffer&) noexcept			  = delete;

	BNCH_SWT_HOST cuda_buffer& operator=(cuda_buffer&& other) noexcept {
		if (this != &other) {
			::swap(data_val, other.data_val);
			::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_HOST cuda_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_HOST void init(uint64_t size) {
		if (data_val) {
			clear();
		}

		cudaError_t result = cudaMalloc(&data_val, size);
		if (result != cudaSuccess) {
			data_val = nullptr;
			throw std::exception{ "Failed to allocate!" };
		}

		size_val = size;
	}

	BNCH_SWT_HOST void deinit() noexcept {
		clear();
	}

	BNCH_SWT_HOST size_type size() noexcept {
		return size_val;
	}

	template<typename value_type> BNCH_SWT_HOST value_type* data() noexcept {
		return std::bit_cast<value_type*>(data_val);
	}

	template<typename value_type> BNCH_SWT_HOST value_type* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<512ULL>(offset_to_claim);
		pointer return_value	= data_val + aligned_amount;
		return std::bit_cast<value_type*>(return_value);
	}

	BNCH_SWT_HOST ~cuda_buffer() noexcept {
		clear();
	}

  protected:
	size_type size_val{};
	pointer data_val{};

	BNCH_SWT_HOST void clear() noexcept {
		if (data_val) {
			cudaFree(data_val);
			data_val = nullptr;
			size_val = 0;
		}
	}
};

template<typename value_type>
concept integral_types = std::is_integral_v<std::remove_cvref_t<value_type>>;

enum class get_value_type_errors {
	invalid_type,
};

template<typename value_type> using base_t = std::remove_cvref_t<value_type>;

template<typename value_type>
concept r_value_reference_types = std::is_rvalue_reference_v<value_type>;

template<typename value_type>
concept uint_types = std::is_unsigned_v<base_t<value_type>> && integral_types<value_type>;

template<typename value_type>
concept int_types = std::is_signed_v<base_t<value_type>> && integral_types<value_type>;

template<typename value_type>
concept integral8_types = integral_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept integral16_types = integral_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept integral32_types = integral_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept integral64_types = integral_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type>
concept int8_types = int_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept int16_types = int_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept int32_types = int_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept int64_types = int_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type>
concept uint8_types = uint_types<value_type> && sizeof(base_t<value_type>) == 1;

template<typename value_type>
concept uint16_types = uint_types<value_type> && sizeof(base_t<value_type>) == 2;

template<typename value_type>
concept uint32_types = uint_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept uint64_types = uint_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type>
concept float_types = std::floating_point<base_t<value_type>>;

template<typename value_type>
concept float16_types = std::is_same_v<base_t<value_type>, half> || std::is_same_v<base_t<value_type>, bf16_t>;

template<typename value_type>
concept float32_types = float_types<value_type> && sizeof(base_t<value_type>) == 4;

template<typename value_type>
concept float64_types = float_types<value_type> && sizeof(base_t<value_type>) == 8;

template<typename value_type> using x_type = decltype(base_t<value_type>::x);

template<typename value_type>
concept half_cuda_types = std::is_same_v<__half, base_t<value_type>>;

template<typename value_type>
concept cuda_integral8_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept cuda_integral16_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept cuda_integral32_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept cuda_integral64_types = integral_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept int8_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept int16_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept int32_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept int64_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept uint8_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

template<typename value_type>
concept uint16_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

template<typename value_type>
concept uint32_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept uint64_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept float32_cuda_types = float32_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

template<typename value_type>
concept float64_cuda_types = float64_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

template<typename value_type>
concept dim04_types = requires() { base_t<value_type>::w; };

template<typename value_type>
concept dim03_types = requires() { base_t<value_type>::z; } && !dim04_types<value_type>;

template<typename value_type>
concept dim02_types = requires() { base_t<value_type>::y; } && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim01_types = requires() { base_t<value_type>::x; } && !dim02_types<value_type> && !dim03_types<value_type> && !dim04_types<value_type>;

template<typename value_type>
concept dim_types = requires() { base_t<value_type>::x; };

template<typename value_type> struct get_value {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		static_assert(false);
	}
};

enum class data_types : uint64_t {
	f32		= 0,
	f16		= 1,
	q4_0	= 2,
	q4_1	= 3,
	q5_0	= 6,
	q5_1	= 7,
	q8_0	= 8,
	q8_1	= 9,
	q2_k	= 10,
	q3_k	= 11,
	q4_k	= 12,
	q5_k	= 13,
	q6_k	= 14,
	q8_k	= 15,
	iq2_xxs = 16,
	iq2_xs	= 17,
	iq3_xxs = 18,
	iq1_s	= 19,
	iq4_nl	= 20,
	iq3_s	= 21,
	iq2_s	= 22,
	iq4_xs	= 23,
	i8		= 24,
	i16		= 25,
	i32		= 26,
	i64		= 27,
	f64		= 28,
	iq1_m	= 29,
	bf16	= 30,
	tq1_0	= 34,
	tq2_0	= 35,
	mxfp4	= 39,
	count,
};

struct block_q8_0 {};

template<typename derived_type> struct type_traits_base {
	BNCH_SWT_HOST static constexpr uint64_t count_elements(const auto& dims) {
		return cast<uint64_t>(dims[0]) * cast<uint64_t>(dims[1]) * cast<uint64_t>(dims[2]) * cast<uint64_t>(dims[3]);
	}

	BNCH_SWT_HOST static constexpr uint64_t total_byte_size(const auto& dims_new) {
		uint64_t element_count{ count_elements(dims_new) };
		if constexpr (derived_type::block_size == 1) {
			return element_count * derived_type::type_size;
		} else {
			return (element_count + derived_type::block_size - 1) / derived_type::block_size * derived_type::type_size;
		}
	}
};

struct type_traits_dynamic {
	data_types data_type{};
	uint64_t block_size{};
	uint64_t type_size{};
	bool is_quantized{};
};

template<typename data_types> struct type_traits;

template<typename derived_type> struct get_dynamic_type_traits {
	BNCH_SWT_HOST_DEVICE consteval static type_traits_dynamic get_dynamic_type_traits_impl() {
		type_traits_dynamic return_values{};
		return_values.is_quantized = derived_type::is_quantized;
		return_values.block_size   = derived_type::block_size;
		return_values.data_type	   = derived_type::data_type;
		return_values.type_size	   = derived_type::type_size;
		return return_values;
	}
};

template<typename value_type_new> struct type_traits;

template<integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			  public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i8 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i16 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i32 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<base_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i64 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<> struct type_traits<bf16_t> : public type_traits_base<type_traits<bf16_t>>, public get_dynamic_type_traits<type_traits<bf16_t>> {
	using value_type = bf16_t;
	using quant_type = bf16_t;
	inline static constexpr data_types data_type{ data_types::bf16 };
	inline static constexpr uint64_t type_size{ sizeof(bf16_t) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

#if BNCH_SWT_COMPILER_CUDA
template<> struct type_traits<half> : public type_traits_base<type_traits<half>>, public get_dynamic_type_traits<type_traits<half>> {
	using value_type = half;
	using quant_type = half;
	inline static constexpr data_types data_type{ data_types::f16 };
	inline static constexpr uint64_t type_size{ sizeof(half) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i8 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i16 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i32 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<cuda_integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																					public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i64 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<float32_cuda_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				 public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::f32 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<float64_cuda_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				 public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::f64 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

#endif

template<> struct type_traits<float> : public type_traits_base<type_traits<float>>, public get_dynamic_type_traits<type_traits<float>> {
	using value_type = float;
	using quant_type = float;
	inline static constexpr data_types data_type{ data_types::f32 };
	inline static constexpr uint64_t type_size{ sizeof(float) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<> struct type_traits<double> : public type_traits_base<type_traits<double>>, public get_dynamic_type_traits<type_traits<double>> {
	using value_type = double;
	using quant_type = double;
	inline static constexpr data_types data_type{ data_types::f64 };
	inline static constexpr uint64_t type_size{ sizeof(double) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};


template<typename value_type> struct get_value_type;

#if BNCH_SWT_COMPILER_CUDA

template<> struct get_value_type<float> {
	using type = float4;
};

template<> struct get_value_type<double> {
	using type = double4;
};

template<> struct get_value_type<half> {
	using type = uint4;
};

template<> struct get_value_type<bf16_t> {
	using type = uint4;
};

template<> struct get_value_type<uint8_t> {
	using type = uchar4;
};

template<> struct get_value_type<uint16_t> {
	using type = ushort4;
};

template<> struct get_value_type<uint32_t> {
	using type = uint4;
};

template<> struct get_value_type<uint64_t> {
	using type = ulong4;
};

template<> struct get_value_type<int8_t> {
	using type = char4;
};

template<> struct get_value_type<int16_t> {
	using type = short4;
};

template<> struct get_value_type<int32_t> {
	using type = int4;
};

template<> struct get_value_type<int64_t> {
	using type = long4;
};
#else

template<quantized_types value_type_new> struct get_value_type<value_type_new> {
	using type = value_type_new;
};

template<> struct get_value_type<float> {
	using type = float;
};

template<> struct get_value_type<double> {
	using type = double;
};

template<> struct get_value_type<uint8_t> {
	using type = uint8_t;
};

template<> struct get_value_type<uint16_t> {
	using type = uint16_t;
};

template<> struct get_value_type<uint32_t> {
	using type = uint32_t;
};

template<> struct get_value_type<uint64_t> {
	using type = uint64_t;
};

template<> struct get_value_type<int8_t> {
	using type = int8_t;
};

template<> struct get_value_type<int16_t> {
	using type = int16_t;
};

template<> struct get_value_type<int32_t> {
	using type = int32_t;
};

template<> struct get_value_type<int64_t> {
	using type = int64_t;
};

#endif

template<typename value_type> using get_value_type_t = get_value_type<value_type>::type;
template<int8_cuda_types value_type> struct get_value<value_type> {
	template<typename... value_types>
		requires(sizeof...(value_types) == 1)
	BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char1(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char1(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char2(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char3(std::forward<value_types>(args)...);
	}
};

template<int8_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_char4(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short1(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short2(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short3(std::forward<value_types>(args)...);
	}
};

template<int16_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_short4(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int1(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int2(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int3(std::forward<value_types>(args)...);
	}
};

template<int32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_int4(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long1(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long2(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long3(std::forward<value_types>(args)...);
	}
};

template<int64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_long4(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar1(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar2(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar3(std::forward<value_types>(args)...);
	}
};

template<uint8_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uchar4(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort1(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort2(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort3(std::forward<value_types>(args)...);
	}
};

template<uint16_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ushort4(std::forward<value_types>(args)...);
	}
};

template<uint32_types value_type> struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		if constexpr (sizeof...(value_types) == 1) {
			return make_uint1(std::forward<value_types>(args)...);
		} else if constexpr (sizeof...(value_types) == 1) {
			return make_uint2(std::forward<value_types>(args)...);
		} else if constexpr (sizeof...(value_types) == 3) {
			return make_uint3(std::forward<value_types>(args)...);
		} else if constexpr (sizeof...(value_types) == 4) {
			return make_uint4(std::forward<value_types>(args)...);
		}
	}
};

template<uint32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint1(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint2(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint3(std::forward<value_types>(args)...);
	}
};

template<uint32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_uint4(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong1(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong2(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong3(std::forward<value_types>(args)...);
	}
};

template<uint64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_ulong4(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float1(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float2(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float3(std::forward<value_types>(args)...);
	}
};

template<float32_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_float4(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim01_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double1(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim02_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double2(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim03_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double3(std::forward<value_types>(args)...);
	}
};

template<float64_cuda_types value_type>
	requires(dim04_types<value_type>)
struct get_value<value_type> {
	template<typename... value_types> BNCH_SWT_DEVICE static constexpr decltype(auto) impl(value_types&&... args) {
		return make_double4(std::forward<value_types>(args)...);
	}
};

enum class binary_op_types {
	add,
	mul,
	sub,
	div,
};

template<binary_op_types> struct binary_op_core;

template<> struct binary_op_core<binary_op_types::add> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) + static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 += static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::mul> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) * static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 *= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::sub> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) - static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 -= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<> struct binary_op_core<binary_op_types::div> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return std::forward<value_type01>(val01) / static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		val01 /= static_cast<base_t<value_type01>>(std::forward<value_type02>(val02));
	}
};

template<typename value_type, binary_op_types binary_op_type> struct binary_op_base;

template<dim01_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value<value_type01>::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
	}
};

template<dim02_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value<value_type01>::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
	}
};

template<dim03_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value<value_type01>::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y),
			op_core_type::impl(std::forward<value_type01>(val01).z, std::forward<value_type02>(val02).z));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, std::forward<value_type02>(val02).z);
	}
};

template<dim04_types value_type, binary_op_types binary_op_type> struct binary_op_base<value_type, binary_op_type> {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		return get_value<value_type01>::impl(op_core_type::impl(std::forward<value_type01>(val01).x, std::forward<value_type02>(val02).x),
			op_core_type::impl(std::forward<value_type01>(val01).y, std::forward<value_type02>(val02).y),
			op_core_type::impl(std::forward<value_type01>(val01).z, std::forward<value_type02>(val02).z),
			op_core_type::impl(std::forward<value_type01>(val01).w, std::forward<value_type02>(val02).w));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static void impl_in_place(value_type01& val01, value_type02&& val02) {
		using op_core_type = binary_op_core<binary_op_type>;
		op_core_type::impl_in_place(val01.x, std::forward<value_type02>(val02).x);
		op_core_type::impl_in_place(val01.y, std::forward<value_type02>(val02).y);
		op_core_type::impl_in_place(val01.z, std::forward<value_type02>(val02).z);
		op_core_type::impl_in_place(val01.w, std::forward<value_type02>(val02).w);
	}
};

template<binary_op_types binary_op_type> struct binary_op {
	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl(value_type01&& val01, value_type02&& val02) {
		return binary_op_base<value_type01, binary_op_type>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
	}

	template<typename value_type01, typename value_type02> BNCH_SWT_DEVICE static decltype(auto) impl_in_place(value_type01& val01, value_type02&& val02) {
		return binary_op_base<value_type01, binary_op_type>::impl_in_place(val01, std::forward<value_type02>(val02));
	}
};

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator+=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator+(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::add>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator*=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator*(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::mul>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator-=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator-(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::sub>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator/=(value_type01& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl_in_place(val01, std::forward<value_type02>(val02));
}

template<dim_types value_type01, dim_types value_type02> BNCH_SWT_DEVICE decltype(auto) operator/(value_type01&& val01, value_type02&& val02) {
	return binary_op<binary_op_types::div>::impl(std::forward<value_type01>(val01), std::forward<value_type02>(val02));
}

struct cpu_buffer {
	using size_type	 = uint64_t;
	using value_type = std::byte;
	using pointer	 = value_type*;
	BNCH_SWT_HOST cpu_buffer() noexcept {
	}
	BNCH_SWT_HOST cpu_buffer& operator=(const cpu_buffer&) noexcept = delete;
	BNCH_SWT_HOST cpu_buffer(const cpu_buffer&) noexcept			= delete;

	BNCH_SWT_HOST cpu_buffer& operator=(cpu_buffer&& other) noexcept {
		if (this != &other) {
			::swap(data_val, other.data_val);
			::swap(size_val, other.size_val);
		}
		return *this;
	}

	BNCH_SWT_HOST cpu_buffer(cuda_buffer&& other) noexcept {
		*this = std::move(other);
	}

	BNCH_SWT_HOST void init(uint64_t size) noexcept {
		if (data_val.size()) {
			clear();
		}
		data_val.resize(size);

		size_val = size;
	}

	BNCH_SWT_HOST void deinit() noexcept {
		clear();
	}

	BNCH_SWT_HOST size_type size() noexcept {
		return size_val;
	}

	BNCH_SWT_HOST pointer data() noexcept {
		return data_val.data();
	}

	BNCH_SWT_HOST void* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<512ULL>(offset_to_claim);
		pointer return_value	= data_val.data() + aligned_amount;
		return return_value;
	}

	BNCH_SWT_HOST ~cpu_buffer() noexcept {
		clear();
	}

  protected:
	std::vector<value_type> data_val{};
	size_type size_val{};

	BNCH_SWT_HOST void clear() noexcept {
		data_val.clear();
	}
};

template<uint64_t index, typename value_type> struct type_holder {
	using type = value_type;
};

template<typename, typename...> struct type_holder_list;

template<size_t... indices, typename... value_types> struct type_holder_list<std::index_sequence<indices...>, value_types...> : public type_holder<indices, value_types>... {};

template<typename... value_types> struct type_list : public type_holder_list<std::make_index_sequence<sizeof...(value_types)>, value_types...> {};

template<typename value_type, kernel_types> struct cuda_kernel_traits_impl {};

template<typename value_type, kernel_types kernel_type>
	requires(kernel_type == kernel_types::add || kernel_type == kernel_types::mul)
struct cuda_kernel_traits_impl<value_type, kernel_type> {
	static constexpr uint64_t flops_per_byte_moved{ bnch_swt::gpu_properties::flops / bnch_swt::gpu_properties::memory_bw };
	static constexpr uint64_t values_per_execution{ 14 };
};

template<typename value_type, kernel_types kernel_type> struct cuda_kernel_traits : public cuda_kernel_traits_impl<value_type, kernel_type> {
	static constexpr uint64_t register_byte_count{ 255 * 4 };
	static constexpr uint64_t register_value_count{ register_byte_count / sizeof(value_type) };
	static constexpr uint64_t bytes_per_execution{ cuda_kernel_traits_impl<value_type, kernel_type>::values_per_execution * sizeof(value_type) };
	static constexpr uint64_t total_executions{ register_byte_count / bytes_per_execution };
};

template<typename value_type, typename...> struct cuda_sub_kernel;

template<typename value_type, size_t... indices> struct cuda_sub_kernel<value_type, std::index_sequence<indices...>> {
	BNCH_SWT_DEVICE static void impl(value_type* __restrict input_01, value_type* __restrict input_02, value_type* __restrict output, uint64_t base_offset) {
		((*(output + base_offset + indices) = *(input_01 + base_offset + indices) + *(input_02 + base_offset + indices)), ...);
	};
};

template<uint32_types value_type> BNCH_SWT_DEVICE value_type fast_min(value_type value_01, value_type value_02) {
	uint32_t res;
	asm("min.u32 %0, %1, %2;" : "=r"(res) : "r"(value_01), "r"(value_02));
	return res;
}

template<typename value_type, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03>
BNCH_SWT_GLOBAL void test_function(value_type* __restrict input_01, value_type* __restrict input_02, value_type* __restrict output) {
	constexpr uint32_t vectors_per_thread			= cuda_kernel_traits<value_type, kernel_types::add>::total_executions;
	const uint32_t global_thread_id					= blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t total_threads					= gridDim.x * blockDim.x;
	static constexpr uint32_t total_scalar_elements = dim_00 * dim_01 * dim_02 * dim_03;
	static constexpr uint32_t total_vector_elements = (total_scalar_elements + 3) / 4;
	static constexpr uint32_t total_chunks			= (total_vector_elements + vectors_per_thread - 1) / vectors_per_thread;

	for (uint32_t chunk_id = global_thread_id; chunk_id < total_chunks; chunk_id += total_threads) {
		uint32_t base_offset = fast_min(chunk_id * vectors_per_thread, total_vector_elements);
		cuda_sub_kernel<value_type, std::make_index_sequence<vectors_per_thread>>::impl(input_01, input_02, output, base_offset);
	}
};

template<typename value_type, typename...> struct cuda_sub_kernel_branched;

template<typename value_type, size_t... indices> struct cuda_sub_kernel_branched<value_type, std::index_sequence<indices...>> {
	BNCH_SWT_DEVICE static void impl_internal(value_type* __restrict input_01, value_type* __restrict input_02, value_type* __restrict output, uint64_t current_idx,
		uint64_t max_elements) {
		if (current_idx < max_elements) {
			output[0] = input_01[0] + input_02[0];
		}
	}

	BNCH_SWT_DEVICE static void impl(value_type* __restrict input_01, value_type* __restrict input_02, value_type* __restrict output, uint64_t base_offset,
		uint64_t total_vector_elements) {
		(impl_internal(input_01 + indices, input_02 + indices, output + indices, base_offset + indices, total_vector_elements), ...);
	}
};

template<typename value_type, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03>
BNCH_SWT_GLOBAL void test_function_branched(value_type* __restrict input_01, value_type* __restrict input_02, value_type* __restrict output) {
	constexpr uint64_t vectors_per_thread = cuda_kernel_traits<value_type, kernel_types::add>::total_executions;
	const uint64_t global_thread_id		  = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t total_threads		  = gridDim.x * blockDim.x;

	static constexpr uint64_t total_scalar_elements = dim_00 * dim_01 * dim_02 * dim_03;
	static constexpr uint64_t total_vector_elements = (total_scalar_elements + 3) / 4;
	static constexpr uint64_t total_chunks			= (total_vector_elements + vectors_per_thread - 1) / vectors_per_thread;

	for (uint64_t chunk_id = global_thread_id; chunk_id < total_chunks; chunk_id += total_threads) {
		uint64_t base_offset = chunk_id * vectors_per_thread;
		cuda_sub_kernel_branched<value_type, std::make_index_sequence<vectors_per_thread>>::impl(input_01 + base_offset, input_02 + base_offset, output + base_offset, base_offset,
			total_vector_elements);
	}
}

template<typename value_type> struct cpu_baseline {
	BNCH_SWT_HOST static uint64_t impl(value_type* cpu_input_01, value_type* cpu_input_02, value_type* cpu_output, uint64_t output_element_count) {
		for (uint64_t i = 0; i < output_element_count; ++i) {
			cpu_output[i] = cpu_input_01[i] + cpu_input_02[i];
		}
		return output_element_count * sizeof(value_type) * 3;
	};
};

static constexpr uint64_t total_iteration_count{ 4 };
static constexpr uint64_t measured_iterations{ 2 };

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread, uint64_t previous_offset, uint64_t dim_01_new, uint64_t dim_02_new, uint64_t dim_03_new, uint64_t dim_04_new>
struct tensor {
	static constexpr uint64_t dim_01{ dim_01_new };
	static constexpr uint64_t dim_02{ dim_02_new };
	static constexpr uint64_t dim_03{ dim_03_new };
	static constexpr uint64_t dim_04{ dim_04_new };
	static constexpr uint64_t element_count{ dim_01 * dim_02 * dim_03 * dim_04 };
	static constexpr uint64_t padding_elements{ cuda_kernel_traits<value_type, kernel_type>::values_per_execution };
	static constexpr uint64_t byte_count{ round_up_to_multiple<512ULL>(sizeof(value_type) * ((element_count + padding_elements) + vectors_per_thread + 1)) };
	static constexpr uint64_t offset{ previous_offset };
	value_type* data{};
};

struct memory_footprint {
	uint64_t byte_count{};
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread> struct cuda_tensors {
	tensor<value_type, kernel_type, vectors_per_thread, 0, model_traits::embedding_length, model_traits::embedding_length, 1, 1> input_01{};

	static constexpr uint64_t offset_01{ tensor<value_type, kernel_type, vectors_per_thread, 0, model_traits::embedding_length, model_traits::embedding_length, 1, 1>::byte_count };
	tensor<value_type, kernel_type, vectors_per_thread, offset_01, model_traits::embedding_length, model_traits::embedding_length, 1, 1> input_02{};

	static constexpr uint64_t offset_02{ offset_01 +
		tensor<value_type, kernel_type, vectors_per_thread, offset_01, model_traits::embedding_length, model_traits::embedding_length, 1, 1>::byte_count };

	tensor<value_type, kernel_type, vectors_per_thread, offset_02, model_traits::embedding_length, model_traits::embedding_length, 1, 1> output{};
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread> constexpr std::array footprints{
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread>::input_01)::byte_count },
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread>::input_02)::byte_count },
	memory_footprint{ decltype(cuda_tensors<value_type, kernel_type, vectors_per_thread>::output)::byte_count },
};

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread> uint64_t byte_count{ [] {
	uint64_t return_value{};
	for (uint64_t x = 0; x < footprints<value_type, kernel_type, vectors_per_thread>.size(); ++x) {
		return_value += footprints<value_type, kernel_type, vectors_per_thread>[x].byte_count;
	}
	return return_value;
}() };

template<typename value_type, uint64_t value_count, value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()>
BNCH_SWT_HOST void generate_values(void* cuda_memory, void* cpu_memory) {
	static std::vector<value_type> return_values{};
	auto size = return_values.size();
	for (uint64_t x = 0; x < size; ++x) {
		return_values[x] = bnch_swt::random_generator<value_type>::impl(min, max);
	}
	for (uint64_t x = size; x < value_count; ++x) {
		return_values.emplace_back(bnch_swt::random_generator<value_type>::impl(min, max));
	}
	std::memcpy(cpu_memory, return_values.data(), value_count * sizeof(value_type));
	if (auto result = cudaMemcpy(cuda_memory, return_values.data(), value_count * sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice); result) {
		std::cout << "cudaMemcpy Error: " << cudaGetErrorString(result) << std::endl;
	}
	return;
}

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread>
BNCH_SWT_HOST cuda_tensors<value_type, kernel_type, vectors_per_thread> generate_cuda_data(cuda_buffer& buffer, cpu_buffer& cpu_buffer) {
	cuda_tensors<value_type, kernel_type, vectors_per_thread> return_values{};
	return_values.input_01.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.input_01.offset);
	auto* cpu_buffer_ptr		= cpu_buffer.data() + return_values.input_01.offset;
	generate_values<value_type, return_values.input_01.element_count>(std::bit_cast<std::byte*>(return_values.input_01.data), cpu_buffer_ptr);

	return_values.input_02.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.input_02.offset);
	cpu_buffer_ptr				= cpu_buffer.data() + return_values.input_02.offset;
	generate_values<value_type, return_values.input_02.element_count>(std::bit_cast<std::byte*>(return_values.input_02.data), cpu_buffer_ptr);

	return_values.output.data = std::bit_cast<value_type*>(buffer.template data<std::byte>() + return_values.output.offset);
	cpu_buffer_ptr			  = cpu_buffer.data() + return_values.output.offset;
	generate_values<value_type, return_values.output.element_count>(std::bit_cast<std::byte*>(return_values.output.data), cpu_buffer_ptr);
	return return_values;
}

template<typename value_type, kernel_types kernel_type, uint64_t vectors_per_thread> cuda_buffer buffer{ [] {
	cuda_buffer return_values{};
	return_values.init(byte_count<value_type, kernel_type, vectors_per_thread>);
	return return_values;
}() };

struct test_struct {
	constexpr test_struct(uint64_t value = 0) {};
	mutable uint64_t value{};
};

template<const test_struct& value> auto test_function() {
	value.value = 233;
	return value;
}

int main() {
	static constexpr test_struct test_val{};
	std::cout << test_function<test_val>().value << std::endl;
	static constexpr uint64_t total_vector_elements{ model_traits::embedding_length * model_traits::embedding_length };
	static constexpr auto test_function_ptr			= &test_function<get_value_type_t<uint32_t>, model_traits::embedding_length, model_traits::embedding_length, 1, 1>;
	static constexpr auto test_function_ptr_branched = &test_function_branched<get_value_type_t<uint32_t>, model_traits::embedding_length, model_traits::embedding_length, 1, 1>;
	using benchmark							= bnch_swt::benchmark_stage<"kernel-gegen-kernel-addition", total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cuda>;
	using test_benchmark					= bnch_swt::benchmark_stage<"kernel-gegen-kernel-test-addition", total_iteration_count, measured_iterations, bnch_swt::benchmark_types::cpu>;
	cpu_buffer cpu_buffer_val{};
	cpu_buffer finished_cpu_buffer_val{};
	cpu_buffer cpu_reference_buffer{};

	finished_cpu_buffer_val.init(byte_count<uint32_t, kernel_types::add, cuda_kernel_traits<get_value_type<uint32_t>,kernel_types::add>::total_executions>);
	cpu_buffer_val.init(byte_count<uint32_t, kernel_types::add, cuda_kernel_traits<get_value_type<uint32_t>, kernel_types::add>::total_executions>);
	cpu_reference_buffer.init(byte_count<uint32_t, kernel_types::add, cuda_kernel_traits<get_value_type<uint32_t>, kernel_types::add>::total_executions>);

	auto tensors = generate_cuda_data<uint32_t, kernel_types::add, cuda_kernel_traits<get_value_type<uint32_t>, kernel_types::add>::total_executions>(
		buffer<uint32_t, kernel_types::add, cuda_kernel_traits<get_value_type<uint32_t>, kernel_types::add>::total_executions>, cpu_buffer_val);

	auto* cpu_input_01 = std::bit_cast<uint32_t*>(cpu_buffer_val.data() + tensors.input_01.offset);
	auto* cpu_input_02 = std::bit_cast<uint32_t*>(cpu_buffer_val.data() + tensors.input_02.offset);
	auto* cpu_output   = std::bit_cast<uint32_t*>(cpu_reference_buffer.data() + tensors.output.offset);

	test_benchmark::run_benchmark<"cpu_baseline", cpu_baseline<uint32_t>>(cpu_input_01, cpu_input_02, cpu_output, tensors.output.element_count);
	test_benchmark::print_results();

	static constexpr uint64_t total_uint4s = (total_vector_elements + 3) / 4;
	static constexpr uint64_t total_chunks = (total_uint4s + 6 - 1) / 6;// 699,051 chunks
	dim3 block{ 1024, 1, 1 };
	dim3 grid{ (total_chunks + 1023) / 1024, 1, 1 };

	test_function_ptr<<<grid, block>>>(std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_01.data), std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_02.data),
		std::bit_cast<get_value_type_t<uint32_t>*>(tensors.output.data));
	cudaDeviceSynchronize();

	if (auto error = cudaGetLastError(); error) {
		std::cout << "Error: " << cudaGetErrorString(error) << std::endl;
	}

	if (auto result = cudaMemcpy(finished_cpu_buffer_val.data() + tensors.output.offset, tensors.output.data, tensors.output.element_count * sizeof(uint32_t),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		result) {
		std::cout << "cudaMemcpy Error: " << cudaGetErrorString(result) << std::endl;
	}

	auto* gpu_output	= std::bit_cast<uint32_t*>(finished_cpu_buffer_val.data() + tensors.output.offset);
	uint64_t mismatches = 0;
	for (uint64_t i = 0; i < tensors.output.element_count; ++i) {
		if (cpu_output[i] != gpu_output[i]) {
			if (mismatches < 10) {
				std::cout << "Mismatch at index " << i << ": CPU=" << cpu_output[i] << " GPU=" << gpu_output[i] << std::endl;
			}
			++mismatches;
		}
	}

	if (mismatches == 0) {
		std::cout << "✓ VALIDATION PASSED! All " << tensors.output.element_count << " elements match!" << std::endl;
	} else {
		std::cout << "✗ VALIDATION FAILED! " << mismatches << " mismatches out of " << tensors.output.element_count << " elements" << std::endl;
	}
	uint64_t bytes_transferred{ model_traits::embedding_length * model_traits::embedding_length * sizeof(uint32_t) * 3 };

	//benchmark::run_benchmark<"index_unrolled_spiled", test_function_ptr_spilled>(grid, block, 0, bytes_transferred,
	//std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_01.data), std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_02.data),
	//std::bit_cast<get_value_type_t<uint32_t>*>(tensors.output.data), tensors.output.dim_01, tensors.output.dim_02, tensors.output.dim_03, tensors.output.dim_04);

	benchmark::run_benchmark<"index_unrolled_branched", test_function_ptr_branched>(grid, block, 0, bytes_transferred,
		std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_01.data),
		std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_02.data), std::bit_cast<get_value_type_t<uint32_t>*>(tensors.output.data));

	benchmark::run_benchmark<"index_unrolled", test_function_ptr>(grid, block, 0, bytes_transferred, std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_01.data),
		std::bit_cast<get_value_type_t<uint32_t>*>(tensors.input_02.data), std::bit_cast<get_value_type_t<uint32_t>*>(tensors.output.data));

	benchmark::print_results();
	return 0;
}