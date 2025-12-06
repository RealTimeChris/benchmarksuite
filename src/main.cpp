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

enum class composite_core_types {
	weights = 0,
	first	= weights,
	global_inputs,
	fused_norm_ingress,
	fused_kqv_causal_score_stage,
	fused_attn_output_residual,
	fused_ffn_activation_stage,
	fused_ffn_egress_output,
	fused_final_egress_norm,
	fused_logit_sampling_stage,
	count,
};

enum class weight_types : uint8_t {
	attn_q = static_cast<uint8_t>(composite_core_types::count),
	first  = attn_q,
	attn_k,
	attn_v,
	attn_output,
	attn_norm,
	ffn_gate,
	ffn_up,
	ffn_down,
	ffn_norm,
	token_embd,
	rope_freqs,
	output_norm,
	output,
	count,
};

enum class global_input_types : uint8_t {
	inp_tokens = static_cast<uint8_t>(weight_types::count),
	first	   = inp_tokens,
	inp_pos,
	inp_out_ids,
	cache_k,
	cache_v,
	kq_mask,
	benchmark_data,
	count,
};

enum class fused_norm_ingress_types : uint8_t {
	inp_embd_get_rows = static_cast<uint8_t>(global_input_types::count),
	first			  = inp_embd_get_rows,
	norm_rms_norm,
	attn_norm_mul,
	count,
};

enum class fused_kqv_causal_score_stage_types : uint8_t {
	qcur_mul_mat = static_cast<uint8_t>(fused_norm_ingress_types::count),
	first		 = qcur_mul_mat,
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
	kq_soft_max,
	count,
};

enum class fused_attn_output_residual_types : uint8_t {
	kqv_mul_mat = static_cast<uint8_t>(fused_kqv_causal_score_stage_types::count),
	first		= kqv_mul_mat,
	kqv_merged_permute,
	kqv_merged_cont,
	kqv_out_mul_mat,
	ffn_inp_add,
	count,
};

enum class fused_ffn_activation_stage_types : uint8_t {
	norm_pre_ffn_rms_norm = static_cast<uint8_t>(fused_attn_output_residual_types::count),
	first				  = norm_pre_ffn_rms_norm,
	ffn_norm_mul,
	ffn_gate_mul_mat,
	ffn_silu,
	ffn_up_mul_mat,
	ffn_gate_par_mul,
	count,
};

enum class fused_ffn_egress_output_types : uint8_t {
	ffn_out_mul_mat = static_cast<uint8_t>(fused_ffn_activation_stage_types::count),
	first			= ffn_out_mul_mat,
	layer_out_add,
	node_1016_get_rows,
	node_1017_get_rows,
	final_ffn_inp_add,
	final_norm_pre_rms_norm,
	final_ffn_norm_mul,
	final_ffn_gate_mul_mat,
	final_ffn_silu,
	count,
};

enum class fused_final_egress_norm_types : uint8_t {
	final_ffn_up_mul_mat = static_cast<uint8_t>(fused_ffn_egress_output_types::count),
	first				 = final_ffn_up_mul_mat,
	final_ffn_gate_par_mul,
	final_ffn_out_mul_mat,
	final_layer_out_add,
	final_norm_rms_norm,
	result_norm_mul,
	count,
};

enum class fused_logit_sampling_stage_types : uint8_t {
	result_output_mul_mat = static_cast<uint8_t>(fused_final_egress_norm_types::count),
	first				  = result_output_mul_mat,
	sample_tokens,
	count,
};

enum class data_strategy_types : uint8_t {
	none,
	global,
	per_block,
};

enum class transformer_phases {
	prefill,
	decode,
};

enum class sync_types {
	none,
	local_s,
	local_e,
	local_s_e,
	global_s,
	global_e,
	global_s_e,
};

enum class kernel_classes {
	global_input,
	per_block,
	global_output,
};

enum class alloc_classes {
	none,
	allocate_heap,
	allocate_cache,
	mmap,
};

enum class kernel_types : uint8_t {
	weights,
	global_inputs,
	get_rows,
	rms_norm,
	mul,
	mul_mat,
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
	qwen_moe,
	phi,
	phi_moe,
	plamo,
	codeshell,
	orion,
	internlm,
	minicpm,
	gemma,
	mamba,
	xverse,
	command_r,
	cohere,
	dbrx,
	olmo,
	olmo_moe,
	openelm,
	arctic,
	deepseek,
	deepseek_moe,
	chatglm,
	bitnet,
	t5,
	t5encoder,
	jais,
	nemotron,
	exaone,
	rwkv,
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

enum class tokenizer_types : uint8_t {
	none,
	spm,
	bpe,
	wpm,
	ugm,
	rwkv,
	count,
};

enum class tokenizer_pre_types : uint8_t {
	default_pre,
	llama3,
	deepseek_llm,
	deepseek_coder,
	falcon,
	mpt,
	starcoder,
	gpt2,
	refact,
	command_r,
	stablelm2,
	qwen2,
	olmo,
	dbrx,
	smaug,
	poro,
	chatglm3,
	chatglm4,
	viking,
	jais,
	tekken,
	smollm,
	codeshell,
	bloom,
	gpt3_finnish,
	exaone,
	chameleon,
	minerva,
	deepseek3_llm,
	count,
};

enum class token_types : uint8_t {
	undefined_token,
	normal,
	unknown,
	control,
	user_defined,
	unused,
	byte,
	count,
};

enum class tokens : uint16_t {
	undefined	 = 0,
	unknown		 = 1 << 0,
	unused		 = 1 << 1,
	normal		 = 1 << 2,
	control		 = 1 << 3,
	user_defined = 1 << 4,
	byte		 = 1 << 5,
	normalized	 = 1 << 6,
	lstrip		 = 1 << 7,
	rstrip		 = 1 << 8,
	single_word	 = 1 << 9,
	count,
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

template<typename value_type> using base_t = std::remove_cvref_t<value_type>;

template<typename value_type>
concept kernel_types_types = requires() { base_t<value_type>::kernel_type; };

template<typename value_type>
concept unary_kernel_types = kernel_types_types<value_type> &&
	(base_t<value_type>::kernel_type == kernel_types::rms_norm || base_t<value_type>::kernel_type == kernel_types::reshape ||
		base_t<value_type>::kernel_type == kernel_types::transpose || base_t<value_type>::kernel_type == kernel_types::permute ||
		base_t<value_type>::kernel_type == kernel_types::view || base_t<value_type>::kernel_type == kernel_types::silu || base_t<value_type>::kernel_type == kernel_types::cont ||
		base_t<value_type>::kernel_type == kernel_types::top_k);

template<typename value_type>
concept binary_kernel_types = kernel_types_types<value_type> &&
	(base_t<value_type>::kernel_type == kernel_types::mul || base_t<value_type>::kernel_type == kernel_types::add || base_t<value_type>::kernel_type == kernel_types::sub ||
		base_t<value_type>::kernel_type == kernel_types::div || base_t<value_type>::kernel_type == kernel_types::mul_mat ||
		base_t<value_type>::kernel_type == kernel_types::get_rows || base_t<value_type>::kernel_type == kernel_types::softmax ||
		base_t<value_type>::kernel_type == kernel_types::copy);

template<typename value_type>
concept ternary_kernel_types = kernel_types_types<value_type> && (base_t<value_type>::kernel_type == kernel_types::rope);

template<typename value_type>
concept global_alloc_types = base_t<value_type>::data_strategy_type == data_strategy_types::global;

template<typename value_type>
concept per_block_alloc_types = base_t<value_type>::data_strategy_type == data_strategy_types::per_block;

template<typename value_type>
concept integral_types = std::is_integral_v<base_t<value_type>>;

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
concept weights_types = std::is_same_v<weight_types, base_t<decltype(value_type::enum_value)>>;

template<typename value_type>
concept input_only_types = std::is_same_v<global_input_types, base_t<decltype(value_type::enum_value)>>;

template<typename value_type>
concept per_block_types = std::is_same_v<fused_norm_ingress_types, base_t<decltype(value_type::enum_value)>> ||
	std::is_same_v<fused_kqv_causal_score_stage_types, base_t<decltype(value_type::enum_value)>> ||
	std::is_same_v<fused_attn_output_residual_types, base_t<decltype(value_type::enum_value)>> ||
	std::is_same_v<fused_ffn_activation_stage_types, base_t<decltype(value_type::enum_value)>>;

template<typename value_type>
concept global_output_types = std::is_same_v<fused_ffn_egress_output_types, base_t<decltype(value_type::enum_value)>> ||
	std::is_same_v<fused_final_egress_norm_types, base_t<decltype(value_type::enum_value)>> ||
	std::is_same_v<fused_logit_sampling_stage_types, base_t<decltype(value_type::enum_value)>>;

template<typename value_type>
concept active_kernel_types = global_output_types<value_type> || per_block_types<value_type>;

template<typename value_type>
concept active_composite_types =
	std::remove_cvref<value_type>::enum_value != composite_core_types::weights && std::remove_cvref<value_type>::enum_value != composite_core_types::global_inputs;

template<typename value_type>
concept ephemeral_kernel_types =
	base_t<value_type>::kernel_type == kernel_types::view || base_t<value_type>::kernel_type == kernel_types::copy || base_t<value_type>::kernel_type == kernel_types::reshape ||
	base_t<value_type>::kernel_type == kernel_types::permute || base_t<value_type>::kernel_type == kernel_types::transpose || base_t<value_type>::kernel_type == kernel_types::cont;

template<typename value_type>
concept runtime_dimensions_types = requires { base_t<value_type>::dimension_identifiers; };
template<typename value_type>
concept has_count_types = requires() {
	base_t<value_type>::count;
	base_t<value_type>::first;
};
template<typename value_type>
concept enum_types = std::is_enum_v<base_t<value_type>>;
template<typename value_type>
concept printable_enum_types = enum_types<value_type> && has_count_types<value_type>;
template<printable_enum_types auto current_index> consteval std::string_view get_enum_name() {
#if BNCH_SWT_COMPILER_MSVC || BNCH_SWT_COMPILER_CUDA
	constexpr const char* pretty_function_tail{ ">(void)" };
#else
	constexpr const char* pretty_function_tail{ "]" };
#endif
	std::string_view str = std::source_location::current().function_name();
#if BNCH_SWT_COMPILER_GNU
	str			   = str.substr(str.find("=") + 2);
	uint64_t end   = str.find(';');
	str			   = str.substr(0, end);
	uint64_t start = str.find_last_of(':') + 1;
	return str.substr(start);
#else
	str			   = str.substr(str.find("=") + 2);
	uint64_t start = str.find_last_of(':') + 1;
	uint64_t end   = str.find(pretty_function_tail);
	return str.substr(start, end - start);
#endif
}
template<printable_enum_types current_type, uint64_t offset, size_t... I> consteval auto get_enum_names_impl(std::index_sequence<I...>) {
	return std::array<std::string_view, sizeof...(I)>{ get_enum_name<static_cast<current_type>(offset + I)>()... };
}
template<printable_enum_types current_type> consteval auto get_enum_names() {
	constexpr uint64_t begin = static_cast<uint64_t>(current_type::first);
	constexpr uint64_t end	 = static_cast<uint64_t>(current_type::count);
	static_assert(end >= begin, "Enum 'count' must be greater than or equal to 'first'");
	constexpr uint64_t num_elements = end - begin;
	return get_enum_names_impl<current_type, begin>(std::make_index_sequence<num_elements>{});
}
template<printable_enum_types enum_type> std::string_view get_name(enum_type type) {
	static constexpr auto data{ get_enum_names<enum_type>() };
	const uint64_t val	 = static_cast<uint64_t>(type);
	const uint64_t start = static_cast<uint64_t>(enum_type::first);
	if (val >= start && (val - start) < data.size()) {
		return data[val - start];
	}
	return "Unknown Type.";
}
template<printable_enum_types enum_type> std::ostream& operator<<(std::ostream& os, enum_type type) {
	os << get_name(type);
	return os;
}

template<typename derived_type> struct type_traits_base {
	static constexpr uint64_t count_elements(const auto& dims) {
		return static_cast<uint64_t>(dims[0]) * static_cast<uint64_t>(dims[1]) * static_cast<uint64_t>(dims[2]) * static_cast<uint64_t>(dims[3]);
	}

	static constexpr uint64_t total_byte_size(const auto& dims_new) {
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
	consteval static type_traits_dynamic get_dynamic_type_traits_impl() {
		type_traits_dynamic return_values{};
		return_values.is_quantized = derived_type::is_quantized;
		return_values.block_size   = derived_type::block_size;
		return_values.data_type	   = derived_type::data_type;
		return_values.type_size	   = derived_type::type_size;
		return return_values;
	}
};

template<typename value_type_new> struct type_traits;

template<integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<std::remove_cvref_t<value_type_new>>>,
																			  public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i8 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<std::remove_cvref_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i16 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<std::remove_cvref_t<value_type_new>>>,
																			   public get_dynamic_type_traits<type_traits<value_type_new>> {
	using value_type = value_type_new;
	using quant_type = value_type;
	inline static constexpr data_types data_type{ data_types::i32 };
	inline static constexpr uint64_t type_size{ sizeof(value_type) };
	inline static constexpr bool is_quantized{ false };
	inline static constexpr uint64_t block_size{ 1 };
};

template<integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<std::remove_cvref_t<value_type_new>>>,
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

#if NIHILUS_COMPILER_CUDA
template<> struct type_traits<half> : public type_traits_base<type_traits<half>>, public get_dynamic_type_traits<type_traits<half>> {
	using value_type = half;
	using quant_type = half;
	inline static constexpr data_types data_type{ data_types::f16 };
	inline static constexpr uint64_t type_size{ sizeof(half) };
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

template<device_types device_type, typename enum_type> constexpr type_traits_dynamic get_type_traits(enum_type data_type) {
	switch (static_cast<uint64_t>(data_type)) {
		case static_cast<uint64_t>(enum_type::f32): {
			return type_traits<float>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::f16): {
			return type_traits<half>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::i8): {
			return type_traits<int8_t>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::i16): {
			return type_traits<int16_t>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::i32): {
			return type_traits<int32_t>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::i64): {
			return type_traits<int64_t>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::f64): {
			return type_traits<double>::get_dynamic_type_traits_impl();
		}
		case static_cast<uint64_t>(enum_type::bf16): {
			return type_traits<bf16_t>::get_dynamic_type_traits_impl();
		}
		default: {
			return {};
		}
	}
}

constexpr uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
	return (a + b - 1) / b;
}

enum class model_formats {
	nh_void,
	gguf,
	count,
};

enum class max_generation_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_prompt_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_context_length_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class max_batch_size_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class gpu_count_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class exceptions_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

enum class benchmark_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

enum class gpu_rank_type : uint64_t {
	disabled = std::numeric_limits<uint64_t>::min(),
	enabled	 = std::numeric_limits<uint64_t>::max(),
};

enum class dev_type : bool {
	disabled = std::numeric_limits<bool>::min(),
	enabled	 = std::numeric_limits<bool>::max(),
};

struct model_config {
	max_generation_length_type max_generation_length{ static_cast<max_generation_length_type>(std::numeric_limits<uint64_t>::max()) };
	max_prompt_length_type max_prompt_length{ static_cast<max_prompt_length_type>(std::numeric_limits<uint64_t>::max()) };
	max_context_length_type max_context_length{ static_cast<max_context_length_type>(1024) };
	tokenizer_pre_types tokenizer_pre_type{ tokenizer_pre_types::llama3 };
	max_batch_size_type max_batch_size{ static_cast<max_batch_size_type>(1) };
	model_generations model_generation{ model_generations::v3_1 };
	tokenizer_types tokenizer_type{ tokenizer_types::bpe };
	gpu_count_type gpu_count{ static_cast<gpu_count_type>(1ull) };
	model_formats model_format{ model_formats::gguf };
	model_sizes model_size{ model_sizes::llm_405B };
	kernel_type_profiles kernel_type_profile{};
	exceptions_type exceptions{};
	benchmark_type benchmark{};
	device_types device_type{};
	model_arches model_arch{};
	gpu_rank_type gpu_rank{};
	dev_type dev{};

	template<std::same_as<max_generation_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_generation_length = value;
		return return_value;
	}

	template<std::same_as<max_prompt_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_prompt_length = value;
		return return_value;
	}

	template<std::same_as<max_context_length_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_context_length = value;
		return return_value;
	}

	template<std::same_as<tokenizer_pre_types> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.tokenizer_pre_type = value;
		return return_value;
	}

	template<std::same_as<max_batch_size_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.max_batch_size = value;
		return return_value;
	}

	template<std::same_as<model_generations> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_generation = value;
		return return_value;
	}

	template<std::same_as<tokenizer_types> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.tokenizer_type = value;
		return return_value;
	}

	template<std::same_as<gpu_count_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.gpu_count = value;
		return return_value;
	}

	template<std::same_as<model_formats> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_format = value;
		return return_value;
	}

	template<std::same_as<model_sizes> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_size = value;
		return return_value;
	}

	template<std::same_as<kernel_type_profiles> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.kernel_type_profile = value;
		return return_value;
	}

	template<std::same_as<exceptions_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.exceptions = value;
		return return_value;
	}

	template<std::same_as<benchmark_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.benchmark = value;
		return return_value;
	}

	template<std::same_as<device_types> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.device_type = value;
		return return_value;
	}

	template<std::same_as<model_arches> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.model_arch = value;
		return return_value;
	}

	template<std::same_as<gpu_rank_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.gpu_rank = value;
		return return_value;
	}

	template<std::same_as<dev_type> value_type> consteval auto update(const value_type value) const {
		model_config return_value{ *this };
		return_value.dev = value;
		return return_value;
	}
};

template<uint64_t value_01, uint64_t value_02> consteval uint64_t get_updated_value() {
	if constexpr (value_01 == std::numeric_limits<uint64_t>::max()) {
		return ceil_div(value_02, 2);
	} else {
		return value_01;
	}
}

template<model_arches, model_sizes, model_generations> struct model_traits;

template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

enum class model_config_errors {
	context_length_too_large,
	context_length_too_short,
	prompt_length_or_generation_length_too_large,
};

template<const model_config& config> struct model_config_type {
	static constexpr uint64_t max_generation_length = get_updated_value<static_cast<uint64_t>(config.max_generation_length), static_cast<uint64_t>(config.max_context_length)>();
	static constexpr uint64_t max_prompt_length		= get_updated_value<static_cast<uint64_t>(config.max_prompt_length), static_cast<uint64_t>(config.max_context_length)>();
	static constexpr uint64_t max_context_length	= static_cast<uint64_t>(config.max_context_length);
	static constexpr tokenizer_pre_types tokenizer_pre_type	  = config.tokenizer_pre_type;
	static constexpr uint64_t max_batch_size				  = static_cast<uint64_t>(config.max_batch_size);
	static constexpr model_generations model_generation		  = config.model_generation;
	static constexpr tokenizer_types tokenizer_type			  = config.tokenizer_type;
	static constexpr uint64_t gpu_count						  = static_cast<uint64_t>(config.gpu_count);
	static constexpr model_formats model_format				  = config.model_format;
	static constexpr model_sizes model_size					  = config.model_size;
	static constexpr kernel_type_profiles kernel_type_profile = config.kernel_type_profile;
	static constexpr bool exceptions						  = static_cast<bool>(config.exceptions);
	static constexpr bool benchmark							  = static_cast<bool>(config.benchmark);
	static constexpr device_types device_type				  = config.device_type;
	static constexpr model_arches model_arch				  = config.model_arch;
	static constexpr uint64_t gpu_rank						  = static_cast<uint64_t>(config.gpu_rank);
	static constexpr bool dev								  = static_cast<bool>(config.dev);

	static constexpr const model_config& get_config() {
		return config;
	}
};

template<typename search_type, typename... check_types> constexpr uint64_t type_occurrence_count =
	(static_cast<uint64_t>(std::is_same_v<std::remove_cvref_t<search_type>, std::remove_cvref_t<check_types>>) + ...);

template<typename... arg_types>
concept unique_configuration_types = ((type_occurrence_count<arg_types, arg_types...> == 1) && ...);

template<unique_configuration_types... arg_types> inline static consteval auto generate_model_config(arg_types... args) {
	model_config config_new{};
	((config_new = config_new.update(args)), ...);
	return config_new;
};

template<unique_configuration_types... arg_types> inline static consteval auto generate_model_config(model_config config_new, arg_types... args) {
	((config_new = config_new.update(args)), ...);
	return config_new;
};

template<typename value_type> struct uint_pair {
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> multiplicand{};
	bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> shift{};
};

template<typename value_type, value_type...> struct uint_type;

template<uint64_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	value_type lo{};
	value_type hi{};

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type l) : lo{ l } {
	}

	constexpr uint_type(value_type h, value_type l) : lo{ l }, hi{ h } {
	}

	constexpr explicit operator value_type() const {
		return lo;
	}

	constexpr bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	constexpr bool operator!=(const uint_type& other) const {
		return !(*this == other);
	}

	constexpr bool operator<(const uint_type& other) const {
		if (hi != other.hi) {
			return hi < other.hi;
		}
		return lo < other.lo;
	}

	constexpr bool operator>(const uint_type& other) const {
		return other < *this;
	}

	constexpr bool operator<=(const uint_type& other) const {
		return !(*this > other);
	}

	constexpr bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	constexpr uint_type operator~() const {
		return uint_type{ ~hi, ~lo };
	}

	constexpr uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	friend constexpr uint_type operator+(value_type lhs, const uint_type& other) {
		return other + lhs;
	}

	constexpr uint_type operator-(const uint_type& other) const {
		const value_type new_lo = lo - other.lo;
		const value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	constexpr uint_type operator<<(int32_t shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ lo << (shift - 64), 0 };
		}
		return uint_type{ (hi << shift) | (lo >> (64 - shift)), lo << shift };
	}

	constexpr uint_type operator>>(int32_t shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ 0, hi >> (shift - 64) };
		}
		return uint_type{ hi >> shift, (lo >> shift) | (hi << (64 - shift)) };
	}

	constexpr uint_type operator*(const uint_type& other) const {
		value_type u1 = lo >> 32;
		value_type u0 = lo & 0xFFFFFFFF;
		value_type v1 = other.lo >> 32;
		value_type v0 = other.lo & 0xFFFFFFFF;

		value_type t  = u0 * v0;
		value_type w0 = t & 0xFFFFFFFF;
		value_type k  = t >> 32;

		t			  = (u1 * v0) + k;
		value_type w1 = t & 0xFFFFFFFF;
		value_type w2 = t >> 32;

		t = (u0 * v1) + w1;
		k = t >> 32;

		value_type split_hi = (u1 * v1) + w2 + k;
		value_type split_lo = (t << 32) + w0;

		value_type cross_1 = lo * other.hi;
		value_type cross_2 = hi * other.lo;

		return uint_type{ split_hi + cross_1 + cross_2, split_lo };
	}

	constexpr uint_type operator/(const uint_type& other) const {
		if (other.hi == 0 && other.lo == 0) {
			return uint_type{ 0, 0 };
		}

		if (other > *this) {
			return uint_type{ 0, 0 };
		}

		if (other == *this) {
			return uint_type{ 0, 1 };
		}

		uint_type quotient{ 0, 0 };
		uint_type remainder{ 0, 0 };
		const uint_type divisor = other;

		for (int32_t i = 127; i >= 0; --i) {
			remainder = remainder << 1;

			if ((i >= 64 && (hi & (1ULL << (i - 64)))) || (i < 64 && (lo & (1ULL << i)))) {
				remainder.lo |= 1;
			}

			if (remainder >= divisor) {
				remainder = remainder - divisor;
				if (i >= 64) {
					quotient.hi |= (1ULL << (i - 64));
				} else {
					quotient.lo |= (1ULL << i);
				}
			}
		}
		return quotient;
	}

	constexpr uint_type& operator+=(const uint_type& other) {
		*this = *this + other;
		return *this;
	}

	constexpr uint_type& operator-=(const uint_type& other) {
		*this = *this - other;
		return *this;
	}

	constexpr uint_type& operator*=(const uint_type& other) {
		*this = *this * other;
		return *this;
	}

	constexpr uint_type& operator/=(const uint_type& other) {
		*this = *this / other;
		return *this;
	}

	constexpr uint_type& operator<<=(int32_t shift) {
		*this = *this << shift;
		return *this;
	}

	constexpr uint_type& operator>>=(int32_t shift) {
		*this = *this >> shift;
		return *this;
	}

	constexpr value_type lzcnt() const {
		if (hi != 0) {
			value_type x = hi;
			value_type n = 0;
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		} else {
			value_type x = lo;
			value_type n = 64;
			if (x == 0) {
				return 128;
			}
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		}
	}

	consteval static uint_pair<value_type> collect_values() {
		constexpr uint_type div_temp	= divisor_new;
		constexpr uint_type div_minus_1 = divisor_new - 1ULL;
		constexpr value_type l			= 127ULL - div_minus_1.lzcnt();
		constexpr uint_type numerator	= uint_type{ 1ULL } << (64ULL + static_cast<value_type>(l));
		constexpr uint_type m_128		= (numerator + div_temp - 1ULL) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64ULL + l };
	}
};

template<uint32_types value_type, value_type divisor_new> struct uint_type<value_type, divisor_new> {
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	static constexpr uint64_t shift(uint64_t value, uint64_t shift) {
		if (shift == 0ULL) {
			return value;
		}
		if (shift >= 64ULL) {
			return 0ULL;
		}
		return value << shift;
	}

	static constexpr uint64_t div(const uint64_t& lhs, const uint64_t& rhs) {
		if (rhs == 0ULL) {
			return 0ULL;
		}
		if (rhs > lhs) {
			return 0ULL;
		}
		if (rhs == lhs) {
			return 1ULL;
		}
		return static_cast<value_type>(static_cast<uint64_t>(lhs) / static_cast<uint64_t>(rhs));
	}

	static constexpr value_type lzcnt(uint64_t value) {
		if (value == 0ULL) {
			return 64U;
		}
		uint64_t x	 = value;
		value_type n = 0U;
		if (x <= 0x00000000FFFFFFFFULL) {
			n += 32U;
			x <<= 32ULL;
		}
		if (x <= 0x0000FFFFFFFFFFFFULL) {
			n += 16U;
			x <<= 16ULL;
		}
		if (x <= 0x00FFFFFFFFFFFFFFULL) {
			n += 8U;
			x <<= 8ULL;
		}
		if (x <= 0x0FFFFFFFFFFFFFFFULL) {
			n += 4U;
			x <<= 4ULL;
		}
		if (x <= 0x3FFFFFFFFFFFFFFFULL) {
			n += 2U;
			x <<= 2ULL;
		}
		if (x <= 0x7FFFFFFFFFFFFFFFULL) {
			n += 1U;
		}
		return n;
	}

	consteval static uint_pair<value_type> collect_values() {
		if constexpr (divisor_new == 1U) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		constexpr uint64_t div_minus_1{ divisor_new - 1ULL };
		constexpr uint64_t lz{ lzcnt(div_minus_1) };

		if constexpr (lz > 63ULL) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		constexpr uint64_t l{ 63ULL - lz };
		constexpr uint64_t numerator{ shift(1ULL, static_cast<value_type>(32ULL + l)) };
		constexpr uint64_t m_128{ div(numerator + divisor_new - 1ULL, divisor_new) };
		return uint_pair<value_type>{ { static_cast<value_type>(m_128) }, { static_cast<value_type>(32ULL + l) } };
	}
};

template<uint64_types value_type> struct uint_type<value_type> {
	value_type lo{};
	value_type hi{};

	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE constexpr uint_type(value_type h, value_type l = 0) : lo{ l }, hi{ h } {
	}

	BNCH_SWT_HOST_DEVICE explicit operator value_type() const {
		return lo;
	}

	BNCH_SWT_HOST_DEVICE bool operator==(const uint_type& other) const {
		return lo == other.lo && hi == other.hi;
	}

	BNCH_SWT_HOST_DEVICE bool operator>(const uint_type& other) const {
		return other < *this;
	}

	BNCH_SWT_HOST_DEVICE bool operator<(const uint_type& other) const {
		if (hi != other.hi) {
			return hi < other.hi;
		}
		return lo < other.lo;
	}

	BNCH_SWT_HOST_DEVICE bool operator>=(const uint_type& other) const {
		return !(*this < other);
	}

	BNCH_SWT_HOST_DEVICE uint_type operator+(const uint_type& other) const {
		const value_type new_lo = lo + other.lo;
		const value_type new_hi = hi + other.hi + (new_lo < lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator-(const uint_type& other) const {
		const value_type new_lo = lo - other.lo;
		const value_type new_hi = hi - other.hi - (lo < other.lo ? 1 : 0);
		return uint_type{ new_hi, new_lo };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator<<(value_type shift) const {
		if (shift == 0) {
			return *this;
		}
		if (shift >= 128) {
			return uint_type{ 0, 0 };
		}
		if (shift >= 64) {
			return uint_type{ lo << (shift - 64), 0 };
		}
		return uint_type{ (hi << shift) | (lo >> (64 - shift)), lo << shift };
	}

	BNCH_SWT_HOST_DEVICE uint_type operator/(const uint_type& other) const {
		if (other.hi == 0 && other.lo == 0) {
			return uint_type{ 0, 0 };
		}

		if (other > *this) {
			return uint_type{ 0, 0 };
		}

		if (other == *this) {
			return uint_type{ 0, 1 };
		}

		uint_type quotient{ 0, 0 };
		uint_type remainder{ 0, 0 };
		const uint_type divisor_new = other;

		for (std::make_signed_t<value_type> i = 127; i >= 0; --i) {
			remainder = remainder << 1;

			if ((i >= 64 && (hi & (1ULL << (i - 64)))) || (i < 64 && (lo & (1ULL << i)))) {
				remainder.lo |= 1;
			}

			if (remainder >= divisor_new) {
				remainder = remainder - divisor_new;
				if (i >= 64) {
					quotient.hi |= (1ULL << (i - 64));
				} else {
					quotient.lo |= (1ULL << i);
				}
			}
		}
		return quotient;
	}

	BNCH_SWT_HOST_DEVICE value_type lzcnt() const {
		if (hi != 0) {
			value_type x = hi;
			value_type n = 0;
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		} else {
			value_type x = lo;
			value_type n = 64;
			if (x == 0) {
				return 128;
			}
			if (x <= 0x00000000FFFFFFFF) {
				n += 32;
				x <<= 32;
			}
			if (x <= 0x0000FFFFFFFFFFFF) {
				n += 16;
				x <<= 16;
			}
			if (x <= 0x00FFFFFFFFFFFFFF) {
				n += 8;
				x <<= 8;
			}
			if (x <= 0x0FFFFFFFFFFFFFFF) {
				n += 4;
				x <<= 4;
			}
			if (x <= 0x3FFFFFFFFFFFFFFF) {
				n += 2;
				x <<= 2;
			}
			if (x <= 0x7FFFFFFFFFFFFFFF) {
				n += 1;
			}
			return n;
		}
	}

	BNCH_SWT_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		if (divisor_new == 1ULL) {
			return uint_pair<value_type>{ 1ULL, 0ULL };
		}

		uint_type div_temp{ 0ULL, divisor_new };
		uint_type div_minus_1 = div_temp - uint_type{ 0ULL, 1ULL };
		value_type lz		  = div_minus_1.lzcnt();
		if (lz > 127ULL) {
			return uint_pair<value_type>{ 1ULL, 0ULL };
		}
		value_type l		= 127ULL - lz;
		uint_type numerator = uint_type{ 0ULL, 1ULL } << static_cast<value_type>(64ULL + l);
		uint_type m_128		= (numerator + div_temp - uint_type{ 0ULL, 1ULL }) / div_temp;
		return uint_pair<value_type>{ static_cast<value_type>(m_128), 64ULL + l };
	}
};

template<uint32_types value_type> struct uint_type<value_type> {
	BNCH_SWT_HOST_DEVICE constexpr uint_type() {
	}

	BNCH_SWT_HOST_DEVICE static value_type lzcnt(uint64_t value) {
		if (value == 0) {
			return 64;
		}
		uint64_t x	 = value;
		value_type n = 0;
		if (x <= 0x00000000FFFFFFFF) {
			n += 32;
			x <<= 32;
		}
		if (x <= 0x0000FFFFFFFFFFFF) {
			n += 16;
			x <<= 16;
		}
		if (x <= 0x00FFFFFFFFFFFFFF) {
			n += 8;
			x <<= 8;
		}
		if (x <= 0x0FFFFFFFFFFFFFFF) {
			n += 4;
			x <<= 4;
		}
		if (x <= 0x3FFFFFFFFFFFFFFF) {
			n += 2;
			x <<= 2;
		}
		if (x <= 0x7FFFFFFFFFFFFFFF) {
			n += 1;
		}
		return n;
	}

	BNCH_SWT_HOST static uint_pair<value_type> collect_values(value_type divisor_new) {
		if (divisor_new == 1U) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		uint64_t div_minus_1 = divisor_new - 1ULL;
		uint64_t lz			 = lzcnt(div_minus_1);

		if (lz > 63ULL) {
			return uint_pair<value_type>{ { 1U }, { 0U } };
		}

		uint64_t l			 = 63ULL - lz;
		uint64_t numerator	 = 1ULL << static_cast<value_type>(32ULL + l);
		const uint64_t m_128 = (numerator + divisor_new - 1) / divisor_new;
		return uint_pair<value_type>{ { static_cast<value_type>(m_128) }, { static_cast<value_type>(32ULL + l) } };
	}
};

template<typename value_type, value_type const_value_new> struct const_aligned_uint {
	static constexpr bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> const_value{ const_value_new };
};

template<typename value_type, value_type const_value_new> struct aligned_uint : public const_aligned_uint<value_type, const_value_new> {
	mutable bnch_swt::aligned_const<value_type, bnch_swt::device_alignment> value{};
};

template<typename, typename value_type, bool, value_type> struct div_mod_logic;

template<typename value_type> BNCH_SWT_HOST_DEVICE static value_type mul128Generic(value_type u, value_type v, value_type& hi) noexcept {
	value_type u1 = u >> 32;
	value_type u0 = u & 0xFFFFFFFF;
	value_type v1 = v >> 32;
	value_type v0 = v & 0xFFFFFFFF;

	value_type t  = (u0 * v0);
	value_type w0 = t & 0xFFFFFFFF;
	value_type k  = t >> 32;

	t			  = (u1 * v0) + k;
	value_type w1 = t & 0xFFFFFFFF;
	value_type w2 = t >> 32;

	t = (u0 * v1) + w1;
	k = t >> 32;

	hi = (u1 * v1) + w2 + k;
	return (t << 32) + w0;
}

BNCH_SWT_HOST_DEVICE static uint64_t host_umulhi64(uint64_t a, uint64_t b) {
	uint64_t high;
	mul128Generic(a, b, high);
	return high;
}

template<typename derived_type, typename value_type, value_type divisor> struct div_mod_logic<derived_type, value_type, true, divisor> : public aligned_uint<value_type, divisor>,
																																		 public uint_type<value_type> {
	uint_pair<value_type> multiplicand_and_shift{};
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	BNCH_SWT_HOST_DEVICE constexpr div_mod_logic() {
	}

	BNCH_SWT_DEVICE constexpr value_type& get_value() const {
		return static_cast<const aligned_uint<value_type, divisor>*>(this)->value.value;
	}

	BNCH_SWT_HOST void collect_values(value_type d) {
		aligned_uint<value_type, divisor>::value.emplace(d);
		multiplicand_and_shift = uint_type<value_type>::collect_values(d);
	}

	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const {
		if (get_value() == 1) {
			return val;
		}
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
		if constexpr (std::same_as<value_type, uint64_t>) {
			return __umul64hi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 64ULL);
		} else {
			return __umulhi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 32ULL);
		}
#else
		if constexpr (std::same_as<value_type, uint64_t>) {
			uint64_t high_part = host_umulhi64(multiplicand_and_shift.multiplicand, val);
			return high_part >> (multiplicand_and_shift.shift - 64ULL);
		} else {
			return static_cast<value_type>((static_cast<uint64_t>(val) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
		}
#endif
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		return val - (div(val) * get_value());
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator<(value_type lhs, const div_mod_logic& rhs) {
		return lhs < rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>(value_type lhs, const div_mod_logic& rhs) {
		return lhs > rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(value_type lhs, const div_mod_logic& rhs) {
		return lhs >= rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator>=(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(value_type lhs, const div_mod_logic& rhs) {
		return lhs * rhs.value.value;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator*(const div_mod_logic& lhs, value_type rhs) {
		return lhs.value.value * rhs;
	}

	BNCH_SWT_HOST_DEVICE friend value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<typename value_type> BNCH_SWT_HOST_DEVICE consteval bool is_power_of_2(value_type N) {
	return N > 0 && (N & (N - 1)) == 0;
}

template<typename value_type> BNCH_SWT_HOST_DEVICE consteval value_type log2_ct(value_type N) {
	value_type result = 0;
	value_type value  = N;
	while (value >>= 1) {
		++result;
	}
	return result;
}

template<typename derived_type, typename value_type, value_type divisor> struct div_mod_logic<derived_type, value_type, false, divisor>
	: public uint_type<value_type, divisor>, public const_aligned_uint<value_type, divisor> {
	static constexpr value_type bit_count_sub_1{ (sizeof(value_type) * 8ULL) - 1ULL };
	static constexpr value_type bit_count{ sizeof(value_type) * 8ULL };

	BNCH_SWT_DEVICE static constexpr value_type get_value() {
		return derived_type::const_value;
	}

	static constexpr uint_pair<value_type> multiplicand_and_shift{ uint_type<value_type, divisor>::collect_values() };

	BNCH_SWT_HOST_DEVICE value_type div(value_type val) const {
		if constexpr (divisor == 1ULL) {
			return val;
		}
		if constexpr (is_power_of_2(divisor)) {
			static constexpr value_type shift_amount{ log2_ct(divisor) };
			return val >> shift_amount;
		} else {
#if BNCH_SWT_COMPILER_CUDA && defined(__CUDA_ARCH__)
			if constexpr (std::same_as<value_type, uint64_t>) {
				return __umul64hi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 64ULL);
			} else {
				return __umulhi(val, multiplicand_and_shift.multiplicand) >> (multiplicand_and_shift.shift - 32ULL);
			}
#else
			if constexpr (std::same_as<value_type, uint64_t>) {
				uint64_t high_part = host_umulhi64(multiplicand_and_shift.multiplicand, val);
				uint64_t result;
				if constexpr (multiplicand_and_shift.shift >= 64ULL) {
					result = high_part >> (multiplicand_and_shift.shift - 64ULL);
				} else {
					uint64_t low_part = multiplicand_and_shift.multiplicand * val;
					result			  = (high_part << (64ULL - multiplicand_and_shift.shift)) | (low_part >> multiplicand_and_shift.shift);
				}
				return result;
			} else {
				return static_cast<value_type>((static_cast<uint64_t>(val) * multiplicand_and_shift.multiplicand) >> multiplicand_and_shift.shift);
			}
#endif
		}
	}

	BNCH_SWT_HOST_DEVICE value_type mod(value_type val) const {
		if constexpr (is_power_of_2(divisor)) {
			return val & (divisor - 1);
		} else {
			return val - (div(val) * divisor);
		}
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator<(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs < value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs > value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs >= value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator>=(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value >= rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator/(value_type lhs, const div_mod_logic& rhs) {
		return rhs.div(lhs);
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(value_type lhs, const div_mod_logic&) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return lhs * value;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator*(const div_mod_logic&, value_type rhs) {
		constexpr value_type value{ div_mod_logic::get_value() };
		return value * rhs;
	}

	BNCH_SWT_HOST_DEVICE friend constexpr value_type operator%(value_type lhs, const div_mod_logic& rhs) {
		return rhs.mod(lhs);
	}
};

template<typename value_type, value_type static_divisor> struct division {
	BNCH_SWT_DEVICE static value_type div(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			static constexpr value_type shift_amount{ log2_ct(static_divisor) };
			return value >> shift_amount;
		} else {
			static constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.div(value);
		}
	}
};

template<typename value_type, value_type static_divisor> struct modulo {
	BNCH_SWT_DEVICE static value_type mod(value_type value) {
		if constexpr (is_power_of_2(static_divisor)) {
			return value & (static_divisor - 1ULL);
		} else {
			static constexpr div_mod_logic<const_aligned_uint<value_type, static_divisor>, value_type, false, static_divisor> mul_shift{};
			return mul_shift.mod(value);
		}
	}
};

template<typename value_type>
concept dimensions_types = requires() { base_t<value_type>::dimension_identifiers; };

enum class runtime_dimension_value_types : uint8_t {
	none			= 0x0,
	batch_size		= 0x1,
	sequence_length = 0x2,
	other			= 0x4,
	count			= other + 1,
};

struct dimension_identifier {
	constexpr dimension_identifier(runtime_dimension_value_types runtime_dimension_value_type_new, uint64_t index_new = 0) {
		runtime_dimension_value_type = runtime_dimension_value_type_new;
		index						 = static_cast<uint32_t>(index_new);
	}
	explicit constexpr dimension_identifier(uint64_t index_new = 0) {
		index = static_cast<uint32_t>(index_new);
	}
	runtime_dimension_value_types runtime_dimension_value_type{};
	uint32_t index{};
	constexpr bool operator==(const dimension_identifier& other) const {
		return (runtime_dimension_value_type == other.runtime_dimension_value_type) && (index == other.index);
	}

	constexpr explicit operator uint32_t() const {
		return index;
	}

	constexpr dimension_identifier operator-(const dimension_identifier& other) const {
		return dimension_identifier{ runtime_dimension_value_type, index - other.index };
	}
};

template<typename value_type, dimension_identifier dimension_identifier_val_new> struct dimension;

template<typename value_type, dimension_identifier dimension_identifier_val_new>
	requires(dimension_identifier_val_new.runtime_dimension_value_type == runtime_dimension_value_types::none)
struct dimension<value_type, dimension_identifier_val_new>
	: public div_mod_logic<dimension<value_type, dimension_identifier_val_new>, value_type, false, dimension_identifier_val_new.index> {
	static constexpr dimension_identifier dimension_identifier_val{ dimension_identifier_val_new };
};

template<typename value_type, dimension_identifier dimension_identifier_val_new>
	requires(dimension_identifier_val_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct dimension<value_type, dimension_identifier_val_new>
	: public div_mod_logic<dimension<value_type, dimension_identifier_val_new>, value_type, true, dimension_identifier_val_new.index> {
	static constexpr dimension_identifier dimension_identifier_val{ dimension_identifier_val_new };
};

constexpr uint64_t compute_elements(const auto& elems) {
	uint64_t return_value{ 1 };
	for (uint32_t x = 0; x < elems.size(); ++x) {
		return_value *= static_cast<uint64_t>(elems[x]);
	}
	return return_value;
}

constexpr uint64_t compute_elements_identifiers(const auto& elems) {
	uint64_t return_value{ 1 };
	for (uint32_t x = 0; x < elems.size(); ++x) {
		return_value *= static_cast<uint64_t>(elems[x].index);
	}
	return return_value;
}

enum class runtime_dimensions_errors {
	incorrect_dimensions,
	unequal_amount_of_runtime_dimensions_passed,
	incorrect_runtime_dim,
};

consteval bool any_duplicates(const auto& values) {
	bool values_checked[4]{};
	for (uint32_t x = 0; x < 4; ++x) {
		if (values_checked[values[x].index]) {
			return true;
		}
		values_checked[values[x].index] = true;
	}
	return false;
}

consteval bool check_runtime_dimension_types(const auto& runtime_dimension_new) {
	int8_t values[4]{};
	for (uint64_t x = 0; x < 4; ++x) {
		++values[static_cast<uint64_t>(runtime_dimension_new[x].runtime_dimension_value_type)];
	}
	for (uint64_t x = 0; x < 4; ++x) {
		if (values[x] > 1 && x != 0) {
			return false;
		}
	}
	return true;
}

template<uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> using dimension_t =
	std::conditional_t<compute_elements(std::array<uint64_t, 4>{ dim_00, dim_01, dim_02, dim_03 }) >= std::numeric_limits<uint32_t>::max() ||
			dim_00 >= std::numeric_limits<uint32_t>::max() || dim_01 >= std::numeric_limits<uint32_t>::max() || dim_02 >= std::numeric_limits<uint32_t>::max() ||
			dim_03 >= std::numeric_limits<uint32_t>::max(),
		uint64_t, uint32_t>;

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new> struct dimensions_base_00 {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_00_type	 = dimension<dimension_type, dim_00_new>;

	static consteval auto dim_00() {
		return dimension<dimension_type, dim_00_new>{};
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new>
	requires(dim_00_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct dimensions_base_00<dim_00_new, dim_01_new, dim_02_new, dim_03_new> {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_00_type	 = dimension<dimension_type, dim_00_new>;
	dim_00_type d0{};

	auto& dim_00() {
		return d0;
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new> struct dimensions_base_01 {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_01_type	 = dimension<dimension_type, dim_01_new>;

	static consteval auto dim_01() {
		return dimension<dimension_type, dim_01_new>{};
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new>
	requires(dim_01_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct dimensions_base_01<dim_00_new, dim_01_new, dim_02_new, dim_03_new> {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_01_type	 = dimension<dimension_type, dim_01_new>;
	dim_01_type d1{};

	auto& dim_01() {
		return d1;
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new> struct dimensions_base_02 {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_02_type	 = dimension<dimension_type, dim_02_new>;

	static consteval auto dim_02() {
		return dimension<dimension_type, dim_02_new>{};
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new>
	requires(dim_02_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct dimensions_base_02<dim_00_new, dim_01_new, dim_02_new, dim_03_new> {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_02_type	 = dimension<dimension_type, dim_02_new>;
	dim_02_type d2{};

	auto& dim_02() {
		return d2;
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new> struct dimensions_base_03 {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_03_type	 = dimension<dimension_type, dim_03_new>;

	static consteval auto dim_03() {
		return dimension<dimension_type, dim_03_new>{};
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new>
	requires(dim_03_new.runtime_dimension_value_type != runtime_dimension_value_types::none)
struct dimensions_base_03<dim_00_new, dim_01_new, dim_02_new, dim_03_new> {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;
	using dim_03_type	 = dimension<dimension_type, dim_03_new>;
	dim_03_type d3{};

	auto& dim_03() {
		return d3;
	}
};

template<dimension_identifier dim_00_new, dimension_identifier dim_01_new, dimension_identifier dim_02_new, dimension_identifier dim_03_new> struct dimensions
	: public dimensions_base_00<dim_00_new, dim_01_new, dim_02_new, dim_03_new>,
	  public dimensions_base_01<dim_00_new, dim_01_new, dim_02_new, dim_03_new>,
	  public dimensions_base_02<dim_00_new, dim_01_new, dim_02_new, dim_03_new>,
	  public dimensions_base_03<dim_00_new, dim_01_new, dim_02_new, dim_03_new> {
	using dimension_type = dimension_t<dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index>;

	static constexpr std::array<dimension_identifier, 4ull> dimension_identifiers{ dim_00_new, dim_01_new, dim_02_new, dim_03_new };
	template<runtime_dimension_value_types runtime_dimension_value_type_new> static consteval dimension_type get_runtime_dimension_impl() {
		for (uint32_t x = 0; x < 4; ++x) {
			if (dimension_identifiers[x].runtime_dimension_value_type == runtime_dimension_value_type_new) {
				return x;
			}
		}
		return std::numeric_limits<dimension_type>::max();
	}

	template<runtime_dimension_value_types runtime_dimension_value_type_new> static consteval bool runtime_dimension_presence() {
		constexpr dimension_type rt_dim = get_runtime_dimension_impl<runtime_dimension_value_type_new>();
		if constexpr (rt_dim != std::numeric_limits<dimension_type>::max()) {
			return true;
		} else {
			return false;
		}
	}

	template<runtime_dimension_value_types dimension_type_val> void set_rt_dim(uint32_t runtime_value) {
		if constexpr (runtime_dimension_presence<dimension_type_val>()) {
			constexpr dimension_type dim_index = get_runtime_dimension_impl<dimension_type_val>();
			this->template get_rt_dim<dim_index>().collect_values(runtime_value);
		}
	}

	static consteval auto get_dims() {
		constexpr std::array<dimension_type, 4ull> dims_new{ dim_00_new.index, dim_01_new.index, dim_02_new.index, dim_03_new.index };
		return dims_new;
	}

	template<uint32_t index> constexpr decltype(auto) get_rt_dim() {
		if constexpr (index == 0) {
			return dimensions_base_00<dim_00_new, dim_01_new, dim_02_new, dim_03_new>::dim_00();
		} else if constexpr (index == 1) {
			return dimensions_base_01<dim_00_new, dim_01_new, dim_02_new, dim_03_new>::dim_01();
		} else if constexpr (index == 2) {
			return dimensions_base_02<dim_00_new, dim_01_new, dim_02_new, dim_03_new>::dim_02();
		} else if constexpr (index == 3) {
			return dimensions_base_03<dim_00_new, dim_01_new, dim_02_new, dim_03_new>::dim_03();
		} else {
			static_assert(index < 4, "Sorry, but you tried to claim an OOB index.");
		}
	}
};

template<dimensions_types dimension_type, typename...> struct rt_dimensions;

template<dimensions_types runtime_dimensions_type_new> struct rt_dimensions<runtime_dimensions_type_new>
	: public dimensions<runtime_dimensions_type_new::dimension_identifiers[0], runtime_dimensions_type_new::dimension_identifiers[1],
		  runtime_dimensions_type_new::dimension_identifiers[2], runtime_dimensions_type_new::dimension_identifiers[3]> {
	using base_type = dimensions<runtime_dimensions_type_new::dimension_identifiers[0], runtime_dimensions_type_new::dimension_identifiers[1],
		runtime_dimensions_type_new::dimension_identifiers[2], runtime_dimensions_type_new::dimension_identifiers[3]>;
};

template<dimensions_types runtime_dimensions_type, dimensions_types mod_mask_type> struct rt_dimensions<runtime_dimensions_type, mod_mask_type>
	: public rt_dimensions<runtime_dimensions_type> {
	using base_type = rt_dimensions<runtime_dimensions_type>;
	static constexpr std::array<bnch_swt::aligned_const<uint32_t>, 4ULL> mod_mask{ mod_mask_type::get_dims()[0], mod_mask_type::get_dims()[1], mod_mask_type::get_dims()[2],
		mod_mask_type::get_dims()[3] };
};

template<uint32_t dim_00 = 0, uint32_t dim_01 = 1, uint32_t dim_02 = 2, uint32_t dim_03 = 3> consteval auto generate_dimensions() {
	return dimensions<dimension_identifier{ runtime_dimension_value_types::none, dim_00 }, dimension_identifier{ runtime_dimension_value_types::none, dim_01 },
		dimension_identifier{ runtime_dimension_value_types::none, dim_02 }, dimension_identifier{ runtime_dimension_value_types::none, dim_03 }>{};
}

template<dimension_identifier dim_00, dimension_identifier dim_01, dimension_identifier dim_02, dimension_identifier dim_03> consteval auto generate_dimensions() {
	constexpr dimension_identifier corrected_dim_00 = dimension_identifier{ dim_00.runtime_dimension_value_type, dim_00.index };

	constexpr dimension_identifier corrected_dim_01 = dimension_identifier{ dim_01.runtime_dimension_value_type, dim_01.index };

	constexpr dimension_identifier corrected_dim_02 = dimension_identifier{ dim_02.runtime_dimension_value_type, dim_02.index };

	constexpr dimension_identifier corrected_dim_03 = dimension_identifier{ dim_03.runtime_dimension_value_type, dim_03.index };

	return dimensions<corrected_dim_00, corrected_dim_01, corrected_dim_02, corrected_dim_03>{};
}

template<auto... dims> struct get_dimensions_type {
	using type = decltype(generate_dimensions<dims...>());
};

template<auto... dims> using get_dimensions_type_t = get_dimensions_type<dims...>::type;

template<auto... values> struct static_assert_printer_val_inserter {};

template<auto enum_error, typename... value_types> struct static_assert_printer_impl;

template<bool value, auto enum_error, typename... value_types> struct static_assert_printer {
	static constexpr bool impl{ [] {
		if constexpr (!value) {
			static_assert_printer_impl<enum_error, value_types...>::nonexistent_value;
			return false;
		} else {
			return true;
		}
	}() };
};

enum class kernel_trait_static_assert_errors : uint8_t {
	binary_element_count_mismatch,
	ternary_element_count_mismatch,
	quaternary_element_count_mismatch,
	reshape_total_element_count_mismatch,
	copy_total_element_count_mismatch,
	view_total_element_count_mismatch,
	transpose_total_element_count_mismatch,
	permute_total_element_count_mismatch,
	cont_total_element_count_mismatch,
	softmax_mask_not_broadcastable,
};

template<kernel_types kernel_type_new> struct kernel_types_type {
	static constexpr kernel_types kernel_type{ kernel_type_new };
};

template<typename kernel_type, typename... dims_types> struct kernel_traits;

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::weights>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<unary_kernel_types kernel_type, dimensions_types input_dims_01> struct kernel_traits<kernel_type, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<binary_kernel_types kernel_type, dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_type, input_dims_01, input_dims_02> {
	using dimension_type			 = typename input_dims_01::dimension_type;
	static constexpr auto dims_array = input_dims_01::dimension_identifiers;
	static constexpr bool dim0_ok	 = (input_dims_01::dimension_identifiers[0].index == input_dims_02::dimension_identifiers[0].index) ||
		(input_dims_01::dimension_identifiers[0].index == 1) || (input_dims_02::dimension_identifiers[0].index == 1);
	static constexpr bool dim1_ok = (input_dims_01::dimension_identifiers[1].index == input_dims_02::dimension_identifiers[1].index) ||
		(input_dims_01::dimension_identifiers[1].index == 1) || (input_dims_02::dimension_identifiers[1].index == 1);
	static constexpr bool dim2_ok = (input_dims_01::dimension_identifiers[2].index == input_dims_02::dimension_identifiers[2].index) ||
		(input_dims_01::dimension_identifiers[2].index == 1) || (input_dims_02::dimension_identifiers[2].index == 1);
	static constexpr bool dim3_ok = (input_dims_01::dimension_identifiers[3].index == input_dims_02::dimension_identifiers[3].index) ||
		(input_dims_01::dimension_identifiers[3].index == 1) || (input_dims_02::dimension_identifiers[3].index == 1);
	static_assert(static_assert_printer<(dim0_ok && dim1_ok && dim2_ok && dim3_ok), kernel_trait_static_assert_errors::binary_element_count_mismatch,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[1]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[1]>,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[2]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[2]>,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[3]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[3]>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims_array[0], dims_array[1], dims_array[2], dims_array[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::softmax>, input_dims_01, input_dims_02> {
	static constexpr bool dim0_ok = (input_dims_02::dimension_identifiers[0].index == input_dims_01::dimension_identifiers[0].index);
	static constexpr bool dim2_ok = (input_dims_02::dimension_identifiers[2].index == input_dims_01::dimension_identifiers[3].index);
	static_assert(static_assert_printer<(dim0_ok && dim2_ok), kernel_trait_static_assert_errors::softmax_mask_not_broadcastable,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[0]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[0]>,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[1]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[1]>,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[2]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[2]>,
		static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[3]>, static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[3]>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::copy>, input_dims_01, input_dims_02> {
	static constexpr auto dims01			  = input_dims_01::dimension_identifiers;
	static constexpr auto dims02			  = input_dims_02::dimension_identifiers;
	static constexpr uint64_t input_elements  = compute_elements_identifiers(dims01);
	static constexpr uint64_t output_elements = compute_elements_identifiers(dims02);
	static_assert(static_assert_printer<(input_elements == output_elements), kernel_trait_static_assert_errors::copy_total_element_count_mismatch,
		static_assert_printer_val_inserter<input_elements>, static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::softmax>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims, dimensions_types output_dims> struct kernel_traits<kernel_types_type<kernel_types::top_k>, input_dims, output_dims> {
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dimension_identifiers[0], output_dims::dimension_identifiers[1], output_dims::dimension_identifiers[2],
		output_dims::dimension_identifiers[3]>>;
};

template<dimensions_types expert_outputs, dimensions_types router_weights> struct kernel_traits<kernel_types_type<kernel_types::weighted_sum>, expert_outputs, router_weights> {
	static constexpr auto output_dims = expert_outputs::dimension_identifiers;
	using dims_type					  = rt_dimensions<get_dimensions_type_t<output_dims[0], output_dims[1], output_dims[3], 1U>>;
};

template<ternary_kernel_types kernel_type, dimensions_types input_dims_01, dimensions_types input_dims_02, dimensions_types input_dims_03>
struct kernel_traits<kernel_type, input_dims_01, input_dims_02, input_dims_03> {
	static_assert(static_assert_printer<((input_dims_01::dimension_identifiers[0].index == input_dims_02::dimension_identifiers[0].index) ||
											(input_dims_01::dimension_identifiers[0].index == 1) || (input_dims_02::dimension_identifiers[0].index == 1)),
		kernel_trait_static_assert_errors::ternary_element_count_mismatch, static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[0]>,
		static_assert_printer_val_inserter<input_dims_02::dimension_identifiers[0]>>::impl);
	static_assert(static_assert_printer<((input_dims_01::dimension_identifiers[0].index == input_dims_03::dimension_identifiers[0].index) ||
											(input_dims_01::dimension_identifiers[0].index == 1) || (input_dims_03::dimension_identifiers[0].index == 1)),
		kernel_trait_static_assert_errors::ternary_element_count_mismatch, static_assert_printer_val_inserter<input_dims_01::dimension_identifiers[0]>,
		static_assert_printer_val_inserter<input_dims_03::dimension_identifiers[0]>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims, dimensions_types output_dims> struct kernel_traits<kernel_types_type<kernel_types::reshape>, input_dims, output_dims> {
	static constexpr auto dims01			  = input_dims::dimension_identifiers;
	static constexpr auto dims02			  = output_dims::dimension_identifiers;
	static constexpr uint64_t input_elements  = compute_elements_identifiers(dims01);
	static constexpr uint64_t output_elements = compute_elements_identifiers(dims02);
	static_assert(static_assert_printer<(input_elements == output_elements), kernel_trait_static_assert_errors::reshape_total_element_count_mismatch,
		static_assert_printer_val_inserter<input_elements>, static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dimension_identifiers[0], output_dims::dimension_identifiers[1], output_dims::dimension_identifiers[2],
		output_dims::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::view>, input_dims, mod_mask_type> {
	static constexpr auto dims01			  = input_dims::dimension_identifiers;
	static constexpr auto dims02			  = mod_mask_type::dimension_identifiers;
	static constexpr uint64_t input_elements  = compute_elements_identifiers(dims01);
	static constexpr uint64_t output_elements = compute_elements_identifiers(dims02);
	static_assert(static_assert_printer<true, kernel_trait_static_assert_errors::view_total_element_count_mismatch, static_assert_printer_val_inserter<input_elements>,
		static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<mod_mask_type::dimension_identifiers[0], mod_mask_type::dimension_identifiers[1], mod_mask_type::dimension_identifiers[2],
										mod_mask_type::dimension_identifiers[3]>,
		dimensions<dimension_identifier{ dims01[0] - dims02[0] }, dimension_identifier{ dims01[1] - dims02[1] }, dimension_identifier{ dims01[2] - dims02[2] },
			dimension_identifier{ dims01[3] - dims02[3] }>>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::transpose>, input_dims, mod_mask_type> {
	using dimension_type					  = typename input_dims::dimension_type;
	static constexpr auto dims01			  = input_dims::dimension_identifiers;
	static constexpr auto dims02			  = mod_mask_type::dimension_identifiers;
	static constexpr uint64_t input_elements  = compute_elements_identifiers(dims01);
	static constexpr uint64_t output_elements = compute_elements_identifiers(dims02);
	static_assert(static_assert_printer<((static_cast<dimension_type>(dims02[0].index) <= 3 && static_cast<dimension_type>(dims02[1].index) <= 3 &&
											 static_cast<dimension_type>(dims02[2].index) <= 3 && static_cast<dimension_type>(dims02[3].index) <= 3) &&
											!any_duplicates(dims02)),
		kernel_trait_static_assert_errors::transpose_total_element_count_mismatch, static_assert_printer_val_inserter<input_elements>,
		static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims01[static_cast<dimension_type>(dims02[0])], dims01[static_cast<dimension_type>(dims02[1])],
										dims01[static_cast<dimension_type>(dims02[2])], dims01[static_cast<dimension_type>(dims02[3])]>,
		mod_mask_type>;
};

template<dimensions_types input_dims, dimensions_types mod_mask_type> struct kernel_traits<kernel_types_type<kernel_types::permute>, input_dims, mod_mask_type> {
	using dimension_type				   = typename input_dims::dimension_type;
	static constexpr auto input_dims_array = input_dims::dimension_identifiers;
	static constexpr auto permutation	   = mod_mask_type::dimension_identifiers;
	static constexpr auto dims01		   = input_dims_array;
	static constexpr auto dims02 = std::array{ input_dims_array[static_cast<dimension_type>(permutation[0])], input_dims_array[static_cast<dimension_type>(permutation[1])],
		input_dims_array[static_cast<dimension_type>(permutation[2])], input_dims_array[static_cast<dimension_type>(permutation[3])] };
	static constexpr uint64_t input_elements  = compute_elements_identifiers(dims01);
	static constexpr uint64_t output_elements = compute_elements_identifiers(dims02);
	static_assert(static_assert_printer<(input_elements == output_elements), kernel_trait_static_assert_errors::permute_total_element_count_mismatch,
		static_assert_printer_val_inserter<input_elements>, static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<dims02[0], dims02[1], dims02[2], dims02[3]>, mod_mask_type>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::mul_mat>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dimension_identifiers;
	static constexpr auto dims02 = input_dims_02::dimension_identifiers;
	using dims_type				 = rt_dimensions<get_dimensions_type_t<dims02[0], dims01[2], dims02[2], dims02[3]>>;
};

template<dimensions_types input_dims_01, dimensions_types input_dims_02> struct kernel_traits<kernel_types_type<kernel_types::get_rows>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dimension_identifiers;
	static constexpr auto dims02 = input_dims_02::dimension_identifiers;
	using dims_type				 = rt_dimensions<get_dimensions_type_t<dims02[0], dims01[1], dims02[1], dims01[3]>>;
};

template<dimensions_types output_dims, dimensions_types input_dims> struct kernel_traits<kernel_types_type<kernel_types::cont>, output_dims, input_dims> {
	using dimension_type							= typename input_dims::dimension_type;
	static constexpr auto dims01					= input_dims::dimension_identifiers;
	static constexpr auto dims02					= output_dims::dimension_identifiers;
	static constexpr dimension_type input_elements	= static_cast<dimension_type>(compute_elements_identifiers(dims01));
	static constexpr dimension_type output_elements = static_cast<dimension_type>(compute_elements_identifiers(dims02));
	static_assert(static_assert_printer<(input_elements == output_elements), kernel_trait_static_assert_errors::cont_total_element_count_mismatch,
		static_assert_printer_val_inserter<input_elements>, static_assert_printer_val_inserter<output_elements>>::impl);
	using dims_type = rt_dimensions<get_dimensions_type_t<output_dims::dimension_identifiers[0], output_dims::dimension_identifiers[1], output_dims::dimension_identifiers[2],
		output_dims::dimension_identifiers[3]>>;
};

template<dimensions_types input_dims_01> struct kernel_traits<kernel_types_type<kernel_types::sample_tokens>, input_dims_01> {
	using dims_type = rt_dimensions<get_dimensions_type_t<input_dims_01::dimension_identifiers[0], input_dims_01::dimension_identifiers[1], input_dims_01::dimension_identifiers[2],
		input_dims_01::dimension_identifiers[3]>>;
};

template<typename config_type, uint64_t arch_index> struct mega_fused_kernel_launcher;

template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename embedding_type_new, typename logit_type_new, typename token_type_new,
	typename attention_type_new, typename norm_type_new, typename scale_type_new, typename zero_point_type_new, typename kv_cache_type_new, typename mask_type_new,
	typename index_type_new>
struct kernel_type_profile_traits_base {
	using weight_type	  = weight_type_new;
	using activation_type = activation_type_new;
	using compute_type	  = compute_type_new;
	using embedding_type  = embedding_type_new;
	using logit_type	  = logit_type_new;
	using token_type	  = token_type_new;
	using attention_type  = attention_type_new;
	using norm_type		  = norm_type_new;
	using scale_type	  = scale_type_new;
	using zero_point_type = zero_point_type_new;
	using kv_cache_type	  = kv_cache_type_new;
	using mask_type		  = mask_type_new;
	using index_type	  = index_type_new;
};

template<kernel_type_profiles kernel_type_profile> struct kernel_type_profile_traits;

template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_mha> : public kernel_type_profile_traits_base<half,// weight_type
																				   half,// activation_type
																				   float,// compute_type
																				   half,// embedding_type
																				   half,// logit_type
																				   int32_t,// token_token_type
																				   half,// attention_type
																				   half,// norm_type
																				   half,// scale_type
																				   int8_t,// zero_point_type
																				   half,// kv_cache_type
																				   half,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_mha };
	static constexpr const char name[]{ "FP16-MHA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_moe> : public kernel_type_profile_traits_base<half,// weight_type
																				   half,// activation_type
																				   half,// compute_type (Assuming hardware support)
																				   half,// embedding_type
																				   half,// logit_type
																				   int32_t,// token_token_type
																				   half,// attention_type
																				   half,// norm_type
																				   half,// scale_type
																				   int8_t,// zero_point_type
																				   half,// kv_cache_type
																				   half,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_moe };
	static constexpr const char name[]{ "FP16-MoE" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_mha> : public kernel_type_profile_traits_base<bf16_t,// weight_type
																				   bf16_t,// activation_type
																				   bf16_t,// compute_type (Best for high dynamic range)
																				   bf16_t,// embedding_type
																				   bf16_t,// logit_type
																				   int32_t,// token_type
																				   bf16_t,// attention_type
																				   bf16_t,// norm_type
																				   bf16_t,// scale_type
																				   int8_t,// zero_point_type
																				   bf16_t,// kv_cache_type
																				   bf16_t,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_mha };
	static constexpr const char name[]{ "BF16-MHA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_gqa> : public kernel_type_profile_traits_base<bf16_t,// weight_type
																				   bf16_t,// activation_type
																				   bf16_t,// compute_type
																				   bf16_t,// embedding_type
																				   bf16_t,// logit_type
																				   int32_t,// token_type
																				   bf16_t,// attention_type
																				   bf16_t,// norm_type
																				   bf16_t,// scale_type
																				   int8_t,// zero_point_type
																				   bf16_t,// kv_cache_type
																				   bf16_t,// mask_type
																				   int32_t// index_type
																				   > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_gqa };
	static constexpr const char name[]{ "BF16-GQA" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_fp16_fp32> : public kernel_type_profile_traits_base<half,// weight_type (Storage is FP16)
																						  half,// activation_type
																						  float,// compute_type (Compute is FP32)
																						  half,// embedding_type
																						  float,// logit_type
																						  int32_t,// token_type
																						  float,// attention_type
																						  float,// norm_type
																						  half,// scale_type
																						  int8_t,// zero_point_type
																						  half,// kv_cache_type
																						  float,// mask_type
																						  int32_t// index_type
																						  > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_fp16_fp32 };
	static constexpr const char name[]{ "Mixed-FP16/FP32" };
};

template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_bf16_fp32> : public kernel_type_profile_traits_base<bf16_t,// weight_type (Storage is BF16)
																						  bf16_t,// activation_type
																						  float,// compute_type (Compute is FP32)
																						  bf16_t,// embedding_type
																						  float,// logit_type
																						  int32_t,// token_type
																						  float,// attention_type
																						  float,// norm_type
																						  bf16_t,// scale_type
																						  int8_t,// zero_point_type
																						  bf16_t,// kv_cache_type
																						  float,// mask_type
																						  int32_t// index_type
																						  > {
	static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_bf16_fp32 };
	static constexpr const char name[]{ "Mixed-BF16/FP32" };
};

template<model_arches, model_sizes, model_generations> struct model_traits;

template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

template<> struct model_traits<model_arches::llama, model_sizes::llm_3B, model_generations::v3_2> {
	static constexpr const char name[]{ "llama-3.2-3B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_2 };
	static constexpr auto model_size{ model_sizes::llm_3B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 3072;
	static constexpr uint32_t block_count			  = 28;
	static constexpr uint32_t feed_forward_length	  = 8192;
	static constexpr uint32_t attention_head_count	  = 24;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::gemma, model_sizes::llm_27B, model_generations::v3> {
	static constexpr const char name[]{ "gemma3-3-27B" };
	static constexpr auto model_arch{ model_arches::gemma };
	static constexpr auto model_generation{ model_generations::v3 };
	static constexpr auto model_size{ model_sizes::llm_27B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-06f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 262144;
	static constexpr uint32_t embedding_length		  = 5376;
	static constexpr uint32_t block_count			  = 62;
	static constexpr uint32_t feed_forward_length	  = 21504;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 16;
	static constexpr uint32_t rope_dimension_count	  = 168;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = 2688;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-8B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_8B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 4096;
	static constexpr uint32_t block_count			  = 32;
	static constexpr uint32_t feed_forward_length	  = 14336;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-70B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_70B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 8192;
	static constexpr uint32_t block_count			  = 80;
	static constexpr uint32_t feed_forward_length	  = 28672;
	static constexpr uint32_t attention_head_count	  = 64;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1> {
	static constexpr const char name[]{ "llama-3.1-405B" };
	static constexpr auto model_arch{ model_arches::llama };
	static constexpr auto model_generation{ model_generations::v3_1 };
	static constexpr auto model_size{ model_sizes::llm_405B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 500000.0f;
	static constexpr uint32_t vocab_size			  = 128256;
	static constexpr uint32_t embedding_length		  = 16384;
	static constexpr uint32_t block_count			  = 126;
	static constexpr uint32_t feed_forward_length	  = 53248;
	static constexpr uint32_t attention_head_count	  = 128;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v1> {
	static constexpr const char name[]{ "grok-1-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 8192;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v1_5> {
	static constexpr const char name[]{ "grok-1.5-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1_5 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_314B, model_generations::v2> {
	static constexpr const char name[]{ "grok-2-314B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v2 };
	static constexpr auto model_size{ model_sizes::llm_314B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 6144;
	static constexpr uint32_t block_count			  = 64;
	static constexpr uint32_t feed_forward_length	  = 32768;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 48;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 131072;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<> struct model_traits<model_arches::grok, model_sizes::llm_46B, model_generations::v1> {
	static constexpr const char name[]{ "grok-mini-46B" };
	static constexpr auto model_arch{ model_arches::grok };
	static constexpr auto model_generation{ model_generations::v1 };
	static constexpr auto model_size{ model_sizes::llm_46B };
	static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
	static constexpr float rope_freq_base			  = 10000.0f;
	static constexpr uint32_t vocab_size			  = 131072;
	static constexpr uint32_t embedding_length		  = 4096;
	static constexpr uint32_t block_count			  = 32;
	static constexpr uint32_t feed_forward_length	  = 16384;
	static constexpr uint32_t num_experts			  = 8;
	static constexpr uint32_t num_experts_per_tok	  = 2;
	static constexpr uint32_t attention_head_count	  = 32;
	static constexpr uint32_t attention_head_count_kv = 8;
	static constexpr uint32_t rope_dimension_count	  = embedding_length / attention_head_count;
	static constexpr uint32_t context_length		  = 8192;
	static constexpr uint32_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
};

template<typename model_config_type> struct model_dimensions {
	using model_traits_type_new = model_traits_type<model_config_type>;
	enum : uint64_t {
		vocab_size				= model_traits_type_new::vocab_size,
		feed_forward_length		= model_traits_type_new::feed_forward_length,
		attention_head_count	= model_traits_type_new::attention_head_count,
		attention_head_count_kv = model_traits_type_new::attention_head_count_kv,
		rope_dimension_count	= model_traits_type_new::rope_dimension_count,
		n_embd_kv_gqa			= model_traits_type_new::n_embd_kv_gqa,
		block_count				= model_traits_type_new::block_count,
		embedding_length		= model_traits_type_new::embedding_length,
		max_context_length		= model_config_type::max_context_length,
		max_prompt_length		= model_config_type::max_prompt_length,
		max_generation_length	= model_config_type::max_generation_length,
		max_batch_size			= model_config_type::max_batch_size,
	};
};

template<device_types device_type> constexpr alloc_classes alloc_class_weights{ [] {
	if constexpr (device_type == device_types::gpu) {
		return alloc_classes::allocate_heap;
	}
	return alloc_classes::mmap;
}() };

template<typename config_type, auto core_type, typename core_types = std::remove_cvref_t<decltype(core_type)>> struct core_traits;

template<typename config_type> struct core_traits<config_type, weight_types::attn_q, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ull, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::embedding_length, 1ull>> {
	static constexpr auto enum_value{ weight_types::attn_q };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::attn_k, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::n_embd_kv_gqa, 1ULL>> {
	static constexpr auto enum_value{ weight_types::attn_k };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::attn_v, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::n_embd_kv_gqa, 1ULL>> {
	static constexpr auto enum_value{ weight_types::attn_v };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::attn_output, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::embedding_length, 1ULL>> {
	static constexpr auto enum_value{ weight_types::attn_output };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::attn_norm, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, 1ULL, 1ULL>> {
	static constexpr auto enum_value{ weight_types::attn_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::ffn_gate, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::feed_forward_length, 1ULL>> {
	static constexpr auto enum_value{ weight_types::ffn_gate };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::ffn_up, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::feed_forward_length, 1ULL>> {
	static constexpr auto enum_value{ weight_types::ffn_up };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::ffn_down, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::feed_forward_length, model_dimensions<config_type>::embedding_length, 1ULL>> {
	static constexpr auto enum_value{ weight_types::ffn_down };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::ffn_norm, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, 1ULL, 1ULL>> {
	static constexpr auto enum_value{ weight_types::ffn_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::token_embd, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::vocab_size, 1ULL>> {
	static constexpr auto enum_value{ weight_types::token_embd };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::rope_freqs, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::rope_dimension_count / 2, 1ULL, 1ULL>> {
	static constexpr auto enum_value{ weight_types::rope_freqs };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::output_norm, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, 1ULL, 1ULL>> {
	static constexpr auto enum_value{ weight_types::output_norm };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, weight_types::output, weight_types>
	: public rt_dimensions<get_dimensions_type_t<1ULL, model_dimensions<config_type>::embedding_length, model_dimensions<config_type>::vocab_size, 1ULL>> {
	static constexpr auto enum_value{ weight_types::output };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::weights };
	static constexpr alloc_classes alloc_class{ alloc_class_weights<config_type::device_type> };
};

template<typename config_type> struct core_traits<config_type, global_input_types::inp_tokens, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }, dimension_identifier{ runtime_dimension_value_types::none, 1 }>> {
	static constexpr auto enum_value{ global_input_types::inp_tokens };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::inp_pos, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }, dimension_identifier{ runtime_dimension_value_types::none, 1 }>> {
	static constexpr auto enum_value{ global_input_types::inp_pos };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::inp_out_ids, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }, dimension_identifier{ runtime_dimension_value_types::none, 1 },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }>> {
	static constexpr auto enum_value{ global_input_types::inp_out_ids };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::index_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::cache_k, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count_kv },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::rope_dimension_count }>> {
	static constexpr auto enum_value{ global_input_types::cache_k };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::cache_v, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count_kv },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::rope_dimension_count }>> {
	static constexpr auto enum_value{ global_input_types::cache_v };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::kq_mask, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }>> {
	static constexpr auto enum_value{ global_input_types::kq_mask };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::mask_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, global_input_types::benchmark_data, global_input_types>
	: public rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ runtime_dimension_value_types::none, model_dimensions<config_type>::attention_head_count },
		  dimension_identifier{ runtime_dimension_value_types::none, 1 }>> {
	static constexpr auto enum_value{ global_input_types::benchmark_data };
	using output_type = typename kernel_type_profile_traits<config_type::kernel_type_profile>::mask_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::global_inputs };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_norm_ingress_types::inp_embd_get_rows, fused_norm_ingress_types>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, weight_types::token_embd>,
		  core_traits<config_type, global_input_types::inp_tokens>>::dims_type {
	static constexpr auto enum_value{ fused_norm_ingress_types::inp_embd_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::token_embd>;
	using input_type_02 = core_traits<config_type, global_input_types::inp_tokens>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_input };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_norm_ingress_types::norm_rms_norm, fused_norm_ingress_types>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, fused_norm_ingress_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto enum_value{ fused_norm_ingress_types::norm_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_norm_ingress_types::inp_embd_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_norm_ingress_types::attn_norm_mul, fused_norm_ingress_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_norm_ingress_types::norm_rms_norm>,
		  core_traits<config_type, weight_types::attn_norm>>::dims_type {
	static constexpr auto enum_value{ fused_norm_ingress_types::attn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_norm_ingress_types::norm_rms_norm>;
	using input_type_02 = core_traits<config_type, weight_types::attn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_mul_mat, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::attn_q>,
		  core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::qcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::attn_q>;
	using input_type_02 = core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_reshape, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::reshape>, core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::qcur_reshape };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::reshape };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_rope, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::rope>, core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_reshape>,
		  core_traits<config_type, global_input_types::inp_pos>, core_traits<config_type, weight_types::rope_freqs>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::qcur_rope };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_reshape>;
	using input_type_02 = core_traits<config_type, global_input_types::inp_pos>;
	using input_type_03 = core_traits<config_type, weight_types::rope_freqs>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rope };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_mul_mat, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::attn_k>,
		  core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::kcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::attn_k>;
	using input_type_02 = core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_reshape, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::reshape>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length }>>,
		  core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_mul_mat>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::kcur_reshape };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::reshape };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_rope, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::rope>, core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_reshape>,
		  core_traits<config_type, global_input_types::inp_pos>, core_traits<config_type, weight_types::rope_freqs>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::kcur_rope };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_reshape>;
	using input_type_02 = core_traits<config_type, global_input_types::inp_pos>;
	using input_type_03 = core_traits<config_type, weight_types::rope_freqs>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rope };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_mul_mat, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::attn_v>,
		  core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::vcur_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::attn_v>;
	using input_type_02 = core_traits<config_type, fused_norm_ingress_types::attn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, global_input_types::cache_k>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ runtime_dimension_value_types::none, (model_dimensions<config_type>::n_embd_kv_gqa * model_dimensions<config_type>::max_context_length) },
			  dimension_identifier{ runtime_dimension_value_types::none, 1 }, dimension_identifier{ runtime_dimension_value_types::none, 1 }>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::k_cache_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, global_input_types::cache_k>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view_copy, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::copy>, core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_rope>,
		  core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::k_cache_view_copy };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::kcur_rope>;
	using input_type_02 = core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::copy };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_transpose, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::transpose>, core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<0ull, 2ull, 1ull, 3ull>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::vcur_transpose };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::transpose };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, global_input_types::cache_v>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ runtime_dimension_value_types::none, (model_dimensions<config_type>::n_embd_kv_gqa * model_dimensions<config_type>::max_context_length) },
			  dimension_identifier{ runtime_dimension_value_types::none, 1 }, dimension_identifier{ runtime_dimension_value_types::none, 1 }>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::v_cache_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, global_input_types::cache_v>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view_copy, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::copy>, core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_transpose>,
		  core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::v_cache_view_copy };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::vcur_transpose>;
	using input_type_02 = core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::copy };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::v_view, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view_copy>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count }, dimension_identifier{ model_dimensions<config_type>::rope_dimension_count },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::v_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::v_cache_view_copy>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::k_view, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::view>, core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view_copy>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::rope_dimension_count }, dimension_identifier{ model_dimensions<config_type>::attention_head_count },
			  dimension_identifier{ model_dimensions<config_type>::attention_head_count_kv }>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::k_view };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::kv_cache_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::k_cache_view_copy>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::view };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::q_permute, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::permute>, core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_rope>,
		  rt_dimensions<get_dimensions_type_t<0ull, 1ull, 3ull, 2ull>>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::q_permute };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::qcur_rope>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::permute };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::none };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::kq_mul_mat, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, fused_kqv_causal_score_stage_types::k_view>,
		  core_traits<config_type, fused_kqv_causal_score_stage_types::q_permute>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::kq_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::k_view>;
	using input_type_02 = core_traits<config_type, fused_kqv_causal_score_stage_types::q_permute>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_kqv_causal_score_stage_types::kq_soft_max, fused_kqv_causal_score_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::softmax>, core_traits<config_type, fused_kqv_causal_score_stage_types::kq_mul_mat>,
		  core_traits<config_type, global_input_types::kq_mask>>::dims_type {
	static constexpr auto enum_value{ fused_kqv_causal_score_stage_types::kq_soft_max };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::kq_mul_mat>;
	using input_type_02 = core_traits<config_type, global_input_types::kq_mask>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::softmax };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_attn_output_residual_types::kqv_mul_mat, fused_attn_output_residual_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, fused_kqv_causal_score_stage_types::v_view>,
		  core_traits<config_type, fused_kqv_causal_score_stage_types::kq_soft_max>>::dims_type {
	static constexpr auto enum_value{ fused_attn_output_residual_types::kqv_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_kqv_causal_score_stage_types::v_view>;
	using input_type_02 = core_traits<config_type, fused_kqv_causal_score_stage_types::kq_soft_max>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_attn_output_residual_types::kqv_merged_permute, fused_attn_output_residual_types>
	: public kernel_traits<kernel_types_type<kernel_types::permute>, core_traits<config_type, fused_attn_output_residual_types::kqv_mul_mat>,
		  rt_dimensions<get_dimensions_type_t<0ull, 1ull, 3ull, 2ull>>>::dims_type {
	static constexpr auto enum_value{ fused_attn_output_residual_types::kqv_merged_permute };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_attn_output_residual_types::kqv_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::permute };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_attn_output_residual_types::kqv_merged_cont, fused_attn_output_residual_types>
	: public kernel_traits<kernel_types_type<kernel_types::cont>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::embedding_length },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, model_dimensions<config_type>::max_context_length },
			  dimension_identifier{ runtime_dimension_value_types::none, 1 }>>,
		  core_traits<config_type, fused_attn_output_residual_types::kqv_merged_permute>>::dims_type {
	static constexpr auto enum_value{ fused_attn_output_residual_types::kqv_merged_cont };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_attn_output_residual_types::kqv_merged_permute>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::cont };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_attn_output_residual_types::kqv_out_mul_mat, fused_attn_output_residual_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::attn_output>,
		  core_traits<config_type, fused_attn_output_residual_types::kqv_merged_cont>>::dims_type {
	static constexpr auto enum_value{ fused_attn_output_residual_types::kqv_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::attn_output>;
	using input_type_02 = core_traits<config_type, fused_attn_output_residual_types::kqv_merged_cont>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_attn_output_residual_types::ffn_inp_add, fused_attn_output_residual_types>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, fused_attn_output_residual_types::kqv_out_mul_mat>,
		  core_traits<config_type, fused_norm_ingress_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto enum_value{ fused_attn_output_residual_types::ffn_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_attn_output_residual_types::kqv_out_mul_mat>;
	using input_type_02 = core_traits<config_type, fused_norm_ingress_types::inp_embd_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::norm_pre_ffn_rms_norm, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, fused_attn_output_residual_types::ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::norm_pre_ffn_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_attn_output_residual_types::ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::ffn_norm_mul, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_ffn_activation_stage_types::norm_pre_ffn_rms_norm>,
		  core_traits<config_type, weight_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::ffn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_activation_stage_types::norm_pre_ffn_rms_norm>;
	using input_type_02 = core_traits<config_type, weight_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_mul_mat, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_gate>,
		  core_traits<config_type, fused_ffn_activation_stage_types::ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::ffn_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_gate>;
	using input_type_02 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::ffn_silu, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::ffn_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::ffn_up_mul_mat, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_up>,
		  core_traits<config_type, fused_ffn_activation_stage_types::ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::ffn_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_up>;
	using input_type_02 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_par_mul, fused_ffn_activation_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_ffn_activation_stage_types::ffn_silu>,
		  core_traits<config_type, fused_ffn_activation_stage_types::ffn_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_activation_stage_types::ffn_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_silu>;
	using input_type_02 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::ffn_out_mul_mat, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_down>,
		  core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_par_mul>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::ffn_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_down>;
	using input_type_02 = core_traits<config_type, fused_ffn_activation_stage_types::ffn_gate_par_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::layer_out_add, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, fused_ffn_egress_output_types::ffn_out_mul_mat>,
		  core_traits<config_type, fused_attn_output_residual_types::ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::layer_out_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::ffn_out_mul_mat>;
	using input_type_02 = core_traits<config_type, fused_attn_output_residual_types::ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::per_block };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::node_1016_get_rows, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, fused_attn_output_residual_types::kqv_out_mul_mat>,
		  core_traits<config_type, global_input_types::inp_out_ids>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::node_1016_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_attn_output_residual_types::kqv_out_mul_mat>;
	using input_type_02 = core_traits<config_type, global_input_types::inp_out_ids>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::node_1017_get_rows, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::get_rows>, core_traits<config_type, fused_ffn_egress_output_types::layer_out_add>,
		  core_traits<config_type, global_input_types::inp_out_ids>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::node_1017_get_rows };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::layer_out_add>;
	using input_type_02 = core_traits<config_type, global_input_types::inp_out_ids>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::final_ffn_inp_add, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, fused_ffn_egress_output_types::node_1016_get_rows>,
		  core_traits<config_type, fused_ffn_egress_output_types::node_1017_get_rows>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::final_ffn_inp_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::node_1016_get_rows>;
	using input_type_02 = core_traits<config_type, fused_ffn_egress_output_types::node_1017_get_rows>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::final_norm_pre_rms_norm, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, fused_ffn_egress_output_types::final_ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::final_norm_pre_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::final_ffn_norm_mul, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_ffn_egress_output_types::final_norm_pre_rms_norm>,
		  core_traits<config_type, weight_types::ffn_norm>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::final_ffn_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::final_norm_pre_rms_norm>;
	using input_type_02 = core_traits<config_type, weight_types::ffn_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::final_ffn_gate_mul_mat, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_gate>,
		  core_traits<config_type, fused_ffn_egress_output_types::final_ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::final_ffn_gate_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_gate>;
	using input_type_02 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_ffn_egress_output_types::final_ffn_silu, fused_ffn_egress_output_types>
	: public kernel_traits<kernel_types_type<kernel_types::silu>, core_traits<config_type, fused_ffn_egress_output_types::final_ffn_gate_mul_mat>>::dims_type {
	static constexpr auto enum_value{ fused_ffn_egress_output_types::final_ffn_silu };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_gate_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::silu };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::final_ffn_up_mul_mat, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_up>,
		  core_traits<config_type, fused_ffn_egress_output_types::final_ffn_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::final_ffn_up_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_up>;
	using input_type_02 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::final_ffn_gate_par_mul, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_ffn_egress_output_types::final_ffn_silu>,
		  core_traits<config_type, fused_final_egress_norm_types::final_ffn_up_mul_mat>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::final_ffn_gate_par_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_silu>;
	using input_type_02 = core_traits<config_type, fused_final_egress_norm_types::final_ffn_up_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::final_ffn_out_mul_mat, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::ffn_down>,
		  core_traits<config_type, fused_final_egress_norm_types::final_ffn_gate_par_mul>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::final_ffn_out_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::ffn_down>;
	using input_type_02 = core_traits<config_type, fused_final_egress_norm_types::final_ffn_gate_par_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::final_layer_out_add, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::add>, core_traits<config_type, fused_final_egress_norm_types::final_ffn_out_mul_mat>,
		  core_traits<config_type, fused_ffn_egress_output_types::final_ffn_inp_add>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::final_layer_out_add };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_final_egress_norm_types::final_ffn_out_mul_mat>;
	using input_type_02 = core_traits<config_type, fused_ffn_egress_output_types::final_ffn_inp_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::add };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::final_norm_rms_norm, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::rms_norm>, core_traits<config_type, fused_final_egress_norm_types::final_layer_out_add>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::final_norm_rms_norm };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_final_egress_norm_types::final_layer_out_add>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_final_egress_norm_types::result_norm_mul, fused_final_egress_norm_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul>, core_traits<config_type, fused_final_egress_norm_types::final_norm_rms_norm>,
		  core_traits<config_type, weight_types::output_norm>>::dims_type {
	static constexpr auto enum_value{ fused_final_egress_norm_types::result_norm_mul };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_final_egress_norm_types::final_norm_rms_norm>;
	using input_type_02 = core_traits<config_type, weight_types::output_norm>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type> struct core_traits<config_type, fused_logit_sampling_stage_types::result_output_mul_mat, fused_logit_sampling_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::mul_mat>, core_traits<config_type, weight_types::output>,
		  core_traits<config_type, fused_final_egress_norm_types::result_norm_mul>>::dims_type {
	static constexpr auto enum_value{ fused_logit_sampling_stage_types::result_output_mul_mat };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, weight_types::output>;
	using input_type_02 = core_traits<config_type, fused_final_egress_norm_types::result_norm_mul>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_cache };
};

template<typename config_type> struct core_traits<config_type, fused_logit_sampling_stage_types::sample_tokens, fused_logit_sampling_stage_types>
	: public kernel_traits<kernel_types_type<kernel_types::sample_tokens>,
		  rt_dimensions<get_dimensions_type_t<dimension_identifier{ runtime_dimension_value_types::batch_size, model_dimensions<config_type>::max_batch_size },
			  dimension_identifier{ model_dimensions<config_type>::max_generation_length }, dimension_identifier{ runtime_dimension_value_types::none, 1 },
			  dimension_identifier{ runtime_dimension_value_types::sequence_length, 1 }>>>::dims_type {
	static constexpr auto enum_value{ fused_logit_sampling_stage_types::sample_tokens };
	using output_type	= typename kernel_type_profile_traits<config_type::kernel_type_profile>::compute_type;
	using input_type_01 = core_traits<config_type, fused_logit_sampling_stage_types::result_output_mul_mat>;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr kernel_types kernel_type{ kernel_types::sample_tokens };
	static constexpr kernel_classes kernel_class{ kernel_classes::global_output };
	static constexpr alloc_classes alloc_class{ alloc_classes::allocate_heap };
};

template<typename config_type, typename derived_type> struct data_mixin;

template<uint64_types auto multiple, typename value_type_01 = decltype(multiple)> constexpr value_type_01 round_up_to_multiple(value_type_01 value) noexcept {
	if constexpr ((multiple > 0) && ((multiple & (multiple - 1)) == 0)) {
		constexpr value_type_01 mul_sub1{ multiple - 1 };
		return (value + mul_sub1) & ~mul_sub1;
	} else {
		return ((value + multiple - 1) / multiple) * multiple;
	}
}

template<device_types device_type, bool exceptions> struct memory_buffer {
	using value_type = std::byte;
	using alloc		 = std::allocator<value_type>;
	using pointer	 = value_type*;
	using size_type	 = uint64_t;

	memory_buffer() noexcept {
	}

	memory_buffer& operator=(const memory_buffer&) noexcept = delete;
	memory_buffer(const memory_buffer&) noexcept			= delete;

	memory_buffer& operator=(memory_buffer&& other) noexcept {
		if (this != &other) {
			std::swap(offset_val, other.offset_val);
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
		}
		return *this;
	}

	memory_buffer(memory_buffer&& other) noexcept {
		*this = std::move(other);
	}

	void init(uint64_t size) noexcept {
		deinit();
		if (!data_val) {
			static constexpr auto location = std::source_location::current();
		}
		size_val = size;
	}

	void deinit() noexcept {
		if (data_val) {
			data_val   = nullptr;
			offset_val = 0;
			size_val   = 0;
		}
	}

	size_type size() noexcept {
		return size_val;
	}

	pointer data_end() noexcept {
		return data_val + offset_val;
	}

	template<typename value_type_new> value_type_new* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<bnch_swt::device_alignment>(offset_to_claim);
		if (aligned_amount > size_val) {
			static constexpr auto location = std::source_location::current();
		}
		return std::bit_cast<value_type_new*>(data_val + aligned_amount);
	}

	~memory_buffer() noexcept {
		deinit();
	}

  protected:
	pointer data_val{ nullptr };
	size_type offset_val{};
	size_type size_val{};
};

template<typename config_type, global_alloc_types derived_type> struct data_mixin<config_type, derived_type> : public derived_type {
	using output_type = derived_type::output_type;
	using pointer	  = output_type*;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
	static constexpr uint64_t total_required_bytes{ round_up_to_multiple<bnch_swt::device_alignment>(
		type_traits<typename derived_type::output_type>::total_byte_size(derived_type::get_dims())) };

	pointer get_data([[maybe_unused]] uint64_t index = 0) {
		return data;
	}

	pointer* get_data_ptr() {
		return &data;
	}

	void set_data(pointer data_new) {
		data = data_new;
	}

  protected:
	pointer data{ nullptr };
};

template<typename config_type, per_block_alloc_types derived_type> struct data_mixin<config_type, derived_type> : public derived_type {
	using output_type = derived_type::output_type;
	using pointer	  = output_type*;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
	static constexpr uint64_t total_required_bytes{ round_up_to_multiple<bnch_swt::device_alignment>(
		type_traits<typename derived_type::output_type>::total_byte_size(derived_type::get_dims()) * model_dimensions<config_type>::block_count) };

	pointer get_data(uint64_t index = 0) {
		return data[index];
	}

	pointer* get_data_ptr(uint64_t index) {
		return &data[index];
	}

	void set_data(pointer data_new, uint64_t index) {
		data[index] = data_new;
	}

  protected:
	std::array<pointer, model_traits_type<config_type>::block_count> data{};
};

template<typename config_type, typename enum_type> struct core_aggregator;

template<typename config_type> struct composite_core_aggregator;

template<typename base_type> struct math_mixin {};

template<auto index> using tag = std::integral_constant<decltype(index), index>;

template<typename base_type>
	requires(base_type::kernel_type == kernel_types::reshape || base_type::kernel_type == kernel_types::cont)
struct math_mixin<base_type> {
	using dims_type					  = base_type;
	static constexpr auto output_dims = dims_type::get_dims();
	static constexpr auto input_dims  = dims_type::input_type_01::get_dims();
	using dimension_type			  = dimension_t<input_dims[0], input_dims[1], input_dims[2], input_dims[3]>;

	div_mod_logic<dimension<dimension_type, dimension_identifier{ runtime_dimension_value_types::other, 0 }>, dimension_type, true, 0> rt_math_ops_01{};
	div_mod_logic<dimension<dimension_type, dimension_identifier{ runtime_dimension_value_types::other, 0 }>, dimension_type, true, 0> rt_math_ops_02{};

	dimension_type output_stride_0{};
};

template<typename config_type, typename base_type_new> struct memory_planner_impl;

template<typename config_type_new, typename contained_cathedral_type, composite_core_types enum_value_new> struct composite_core_interface : public contained_cathedral_type {
	static constexpr composite_core_types enum_value{ enum_value_new };
	using config_type = config_type_new;
	using last_type	  = typename contained_cathedral_type::last_type;
};

template<typename config_type, typename base_type_new> struct weight_mapper_impl {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return true;
	}
	template<uint64_t size_01, uint64_t size_02> static void impl(base_type& core_val, std::array<std::array<void*, size_01>, size_02>& data) {
		if constexpr (base_type::data_strategy_type == data_strategy_types::per_block) {
			for (uint64_t x = 0; x < model_traits_type<config_type>::block_count; ++x) {
				data[static_cast<uint64_t>(base_type::enum_value) - static_cast<uint64_t>(composite_core_types::count)][x] = static_cast<void*>(core_val.get_data_ptr(x));
			}
		} else {
			data[static_cast<uint64_t>(base_type::enum_value) - static_cast<uint64_t>(composite_core_types::count)][0] = static_cast<void*>(core_val.get_data_ptr());
		}
	}
};

template<typename config_type, typename base_type_new> struct weight_mapper {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return base_type::enum_value == composite_core_types::weights;
	}

	template<uint64_t size_01, uint64_t size_02> static void impl(base_type& core_val, std::array<std::array<void*, size_01>, size_02>& data) {
		core_val.template impl<weight_mapper_impl>(data);
	}
};

template<typename config_type, typename base_type_new> struct dim_updater_impl {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return true;
	}

	static void impl(base_type& core_val, uint32_t max_batch_size, uint32_t sequence_length) {
		core_val.template set_rt_dim<runtime_dimension_value_types::sequence_length>(sequence_length);
		core_val.template set_rt_dim<runtime_dimension_value_types::batch_size>(max_batch_size);
	}
};

template<typename config_type, typename base_type_new> struct dim_updater {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return base_type::enum_value != composite_core_types::weights;
	}

	static void impl(base_type& core_val, uint32_t max_batch_size, uint32_t sequence_length) {
		core_val.template impl<dim_updater_impl>(max_batch_size, sequence_length);
	}
};

template<typename config_type_new, auto enum_value_new> struct core_interface : public data_mixin<config_type_new, core_traits<config_type_new, enum_value_new>>,
																				public math_mixin<core_traits<config_type_new, enum_value_new>> {
	using config_type = config_type_new;
	static constexpr auto enum_value{ enum_value_new };
	uint64_t kernel_iteration_count{};
};

template<typename config_type> struct composite_core_aggregator {
	static constexpr std::array values{ composite_core_types::weights, composite_core_types::global_inputs, composite_core_types::fused_norm_ingress,
		composite_core_types::fused_kqv_causal_score_stage, composite_core_types::fused_attn_output_residual, composite_core_types::fused_ffn_activation_stage,
		composite_core_types::fused_ffn_egress_output, composite_core_types::fused_final_egress_norm, composite_core_types::fused_logit_sampling_stage };
};

template<typename config_type> struct core_aggregator<config_type, weight_types> {
	static constexpr std::array values{ weight_types::attn_q, weight_types::attn_k, weight_types::attn_v, weight_types::attn_output, weight_types::attn_norm,
		weight_types::ffn_gate, weight_types::ffn_up, weight_types::ffn_down, weight_types::ffn_norm, weight_types::token_embd, weight_types::rope_freqs, weight_types::output_norm,
		weight_types::output };
};

template<typename config_type> struct core_aggregator<config_type, global_input_types> {
	static constexpr std::array values{ global_input_types::inp_tokens, global_input_types::inp_pos, global_input_types::inp_out_ids, global_input_types::cache_k,
		global_input_types::cache_v, global_input_types::kq_mask, global_input_types::benchmark_data };
};

template<typename config_type> struct core_aggregator<config_type, fused_norm_ingress_types> {
	static constexpr std::array values{ fused_norm_ingress_types::inp_embd_get_rows, fused_norm_ingress_types::norm_rms_norm, fused_norm_ingress_types::attn_norm_mul };
};

template<typename config_type> struct core_aggregator<config_type, fused_kqv_causal_score_stage_types> {
	static constexpr std::array values{ fused_kqv_causal_score_stage_types::qcur_mul_mat, fused_kqv_causal_score_stage_types::qcur_reshape,
		fused_kqv_causal_score_stage_types::qcur_rope, fused_kqv_causal_score_stage_types::kcur_mul_mat, fused_kqv_causal_score_stage_types::kcur_reshape,
		fused_kqv_causal_score_stage_types::kcur_rope, fused_kqv_causal_score_stage_types::vcur_mul_mat, fused_kqv_causal_score_stage_types::k_cache_view,
		fused_kqv_causal_score_stage_types::k_cache_view_copy, fused_kqv_causal_score_stage_types::vcur_transpose, fused_kqv_causal_score_stage_types::v_cache_view,
		fused_kqv_causal_score_stage_types::v_cache_view_copy, fused_kqv_causal_score_stage_types::v_view, fused_kqv_causal_score_stage_types::k_view,
		fused_kqv_causal_score_stage_types::q_permute, fused_kqv_causal_score_stage_types::kq_mul_mat, fused_kqv_causal_score_stage_types::kq_soft_max };
};

template<typename config_type> struct core_aggregator<config_type, fused_attn_output_residual_types> {
	static constexpr std::array values{ fused_attn_output_residual_types::kqv_mul_mat, fused_attn_output_residual_types::kqv_merged_permute,
		fused_attn_output_residual_types::kqv_merged_cont, fused_attn_output_residual_types::kqv_out_mul_mat, fused_attn_output_residual_types::ffn_inp_add };
};

template<typename config_type> struct core_aggregator<config_type, fused_ffn_activation_stage_types> {
	static constexpr std::array values{ fused_ffn_activation_stage_types::norm_pre_ffn_rms_norm, fused_ffn_activation_stage_types::ffn_norm_mul,
		fused_ffn_activation_stage_types::ffn_gate_mul_mat, fused_ffn_activation_stage_types::ffn_silu, fused_ffn_activation_stage_types::ffn_up_mul_mat,
		fused_ffn_activation_stage_types::ffn_gate_par_mul };
};

template<typename config_type> struct core_aggregator<config_type, fused_ffn_egress_output_types> {
	static constexpr std::array values{ fused_ffn_egress_output_types::ffn_out_mul_mat, fused_ffn_egress_output_types::layer_out_add,
		fused_ffn_egress_output_types::node_1016_get_rows, fused_ffn_egress_output_types::node_1017_get_rows, fused_ffn_egress_output_types::final_ffn_inp_add,
		fused_ffn_egress_output_types::final_norm_pre_rms_norm, fused_ffn_egress_output_types::final_ffn_norm_mul, fused_ffn_egress_output_types::final_ffn_gate_mul_mat,
		fused_ffn_egress_output_types::final_ffn_silu };
};

template<typename config_type> struct core_aggregator<config_type, fused_final_egress_norm_types> {
	static constexpr std::array values{ fused_final_egress_norm_types::final_ffn_up_mul_mat, fused_final_egress_norm_types::final_ffn_gate_par_mul,
		fused_final_egress_norm_types::final_ffn_out_mul_mat, fused_final_egress_norm_types::final_layer_out_add, fused_final_egress_norm_types::final_norm_rms_norm,
		fused_final_egress_norm_types::result_norm_mul };
};

template<typename config_type> struct core_aggregator<config_type, fused_logit_sampling_stage_types> {
	static constexpr std::array values{ fused_logit_sampling_stage_types::result_output_mul_mat, fused_logit_sampling_stage_types::sample_tokens };
};

template<composite_core_types composite_core_type_new> struct composite_core_type_holder {
	static constexpr composite_core_types enum_value{ composite_core_type_new };
	static constexpr auto get_composite_type() {
		if constexpr (enum_value == composite_core_types::weights) {
			return weight_types{};
		} else if constexpr (enum_value == composite_core_types::global_inputs) {
			return global_input_types{};
		} else if constexpr (enum_value == composite_core_types::fused_norm_ingress) {
			return fused_norm_ingress_types{};
		} else if constexpr (enum_value == composite_core_types::fused_kqv_causal_score_stage) {
			return fused_kqv_causal_score_stage_types{};
		} else if constexpr (enum_value == composite_core_types::fused_attn_output_residual) {
			return fused_attn_output_residual_types{};
		} else if constexpr (enum_value == composite_core_types::fused_ffn_activation_stage) {
			return fused_ffn_activation_stage_types{};
		} else if constexpr (enum_value == composite_core_types::fused_ffn_egress_output) {
			return fused_ffn_egress_output_types{};
		} else if constexpr (enum_value == composite_core_types::fused_final_egress_norm) {
			return fused_final_egress_norm_types{};
		} else if constexpr (enum_value == composite_core_types::fused_logit_sampling_stage) {
			return fused_logit_sampling_stage_types{};
		} else {
			return bool{};
		}
	}
	using type = decltype(get_composite_type());
};

template<composite_core_types composite_core_type_new> using composite_core_type_holder_t = composite_core_type_holder<composite_core_type_new>::type;

template<uint64_t index_new> struct composite_type_from_index {
	static constexpr auto get_composite_type() {
		if constexpr (index_new >= static_cast<uint64_t>(weight_types::first) && index_new < static_cast<uint64_t>(weight_types::count)) {
			return weight_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(global_input_types::first) && index_new < static_cast<uint64_t>(global_input_types::count)) {
			return global_input_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_norm_ingress_types::first) && index_new < static_cast<uint64_t>(fused_norm_ingress_types::count)) {
			return fused_norm_ingress_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_kqv_causal_score_stage_types::first) &&
			index_new < static_cast<uint64_t>(fused_kqv_causal_score_stage_types::count)) {
			return fused_kqv_causal_score_stage_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_attn_output_residual_types::first) &&
			index_new < static_cast<uint64_t>(fused_attn_output_residual_types::count)) {
			return fused_attn_output_residual_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_ffn_activation_stage_types::first) &&
			index_new < static_cast<uint64_t>(fused_ffn_activation_stage_types::count)) {
			return fused_ffn_activation_stage_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_ffn_egress_output_types::first) && index_new < static_cast<uint64_t>(fused_ffn_egress_output_types::count)) {
			return fused_ffn_egress_output_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_final_egress_norm_types::first) && index_new < static_cast<uint64_t>(fused_final_egress_norm_types::count)) {
			return fused_final_egress_norm_types{};
		} else if constexpr (index_new >= static_cast<uint64_t>(fused_logit_sampling_stage_types::first) &&
			index_new < static_cast<uint64_t>(fused_logit_sampling_stage_types::count)) {
			return fused_logit_sampling_stage_types{};
		} else {
			return bool{};
		}
	}
	using type = decltype(get_composite_type());
};

template<uint64_t index_new> using composite_type_from_index_t = composite_type_from_index<index_new>::type;

template<auto composite_core_type_new> struct composite_core_type_by_enum_holder {
	static constexpr auto get_composite_type() {
		if constexpr (std::is_same_v<decltype(composite_core_type_new), weight_types>) {
			return composite_core_types::weights;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), global_input_types>) {
			return composite_core_types::global_inputs;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_norm_ingress_types>) {
			return composite_core_types::fused_norm_ingress;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_kqv_causal_score_stage_types>) {
			return composite_core_types::fused_kqv_causal_score_stage;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_attn_output_residual_types>) {
			return composite_core_types::fused_attn_output_residual;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_ffn_activation_stage_types>) {
			return composite_core_types::fused_ffn_activation_stage;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_ffn_egress_output_types>) {
			return composite_core_types::fused_ffn_egress_output;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_final_egress_norm_types>) {
			return composite_core_types::fused_final_egress_norm;
		} else if constexpr (std::is_same_v<decltype(composite_core_type_new), fused_logit_sampling_stage_types>) {
			return composite_core_types::fused_logit_sampling_stage;
		} else {
			return bool{};
		}
	}
	static constexpr auto enum_value{ get_composite_type() };
};

template<auto composite_core_type_new> static constexpr auto composite_core_type_by_enum_holder_v = composite_core_type_by_enum_holder<composite_core_type_new>::enum_value;

template<typename core_traits_type> constexpr void update_index(uint64_t& current_max_index, uint64_t& current_max_value, uint64_t current_new_index, uint64_t current_new_value) {
	if constexpr (active_kernel_types<core_traits_type>) {
		if (current_new_value > current_max_value) {
			current_max_value = current_new_value;
			current_max_index = current_new_index;
		}
	}
}

template<typename config_type, auto enum_value_new, size_t... indices>
constexpr void update_index_impl(uint64_t& current_max_index, uint64_t& current_max_value, std::index_sequence<indices...>) {
	constexpr auto values = core_aggregator<config_type, composite_core_type_holder_t<enum_value_new>>::values;
	(update_index<core_traits<config_type, values[indices]>>(current_max_index, current_max_value, static_cast<uint64_t>(core_traits<config_type, values[indices]>::enum_value),
		 compute_elements(core_traits<config_type, values[indices]>::get_dims())),
		...);
}

template<typename config_type> inline constexpr uint64_t max_tensor_element_index{ []() {
	constexpr auto values = composite_core_aggregator<config_type>::values;
	return [&]<size_t... indices>(std::index_sequence<indices...>) {
		uint64_t max_val{};
		uint64_t index{};
		(update_index_impl<config_type, values[indices]>(index, max_val,
			 std::make_index_sequence<core_aggregator<config_type, composite_core_type_holder_t<values[indices]>>::values.size()>{}),
			...);
		return index;
	}.template operator()(std::make_index_sequence<values.size()>{});
}() };

template<typename... rest_types> struct first;

template<typename first_type, typename... rest_types> struct first<first_type, rest_types...> {
	using type = first_type;
};

template<typename... rest_types> using first_t = typename first<rest_types...>::type;

template<typename... rest_types> struct last;

template<typename last_type, typename... rest_types> struct last<last_type, rest_types...> {
	using type = last<rest_types...>::type;
};

template<typename last_type> struct last<last_type> {
	using type = last_type;
};

template<typename... rest_types> using last_t = typename last<rest_types...>::type;

enum class nihilus_cathedral_errors {
	get_core_by_index_oob,
	invalid_base_cast,
	empty_cathedral_bases_pack,
};

template<auto enum_value, typename... bases_new> struct type_finder;

template<auto enum_value, typename... bases_new>
	requires std::same_as<base_t<decltype(enum_value)>, base_t<decltype(first_t<bases_new...>::enum_value)>>
struct type_finder<enum_value, bases_new...> {
	consteval static size_t find_index() {
		size_t index = 0;
		(((bases_new::enum_value == enum_value) ? true : (++index, false)) || ...);
		return index;
	}

	static constexpr size_t base_index = find_index();
	using type						   = std::tuple_element_t<base_index, std::tuple<bases_new...>>;
};

template<auto enum_value, typename... bases_new>
	requires(!std::same_as<base_t<decltype(enum_value)>, base_t<decltype(first_t<bases_new...>::enum_value)>>)
struct type_finder<enum_value, bases_new...> {
	static constexpr auto target_composite = composite_core_type_by_enum_holder_v<enum_value>;

	consteval static size_t find_index() {
		size_t index = 0;
		(((bases_new::enum_value == target_composite) ? true : (++index, false)) || ...);
		return index;
	}

	static constexpr size_t base_index = find_index();
	using type						   = std::tuple_element_t<base_index, std::tuple<bases_new...>>;
};

template<typename config_type_new, typename... bases> struct nihilus_cathedral : bases... {
	static_assert(static_assert_printer<(sizeof...(bases) > 0), nihilus_cathedral_errors::empty_cathedral_bases_pack>::impl);
	using first_type  = first_t<bases...>;
	using last_type	  = last_t<bases...>;
	using config_type = config_type_new;
	using enum_type	  = decltype(first_type::enum_value);

	static constexpr uint64_t size{ sizeof...(bases) };

	template<template<typename, typename> typename mixin_type, typename... arg_types> constexpr void impl(arg_types&&... args) noexcept {
		(impl_internal_filtered<mixin_type, bases>(args...), ...);
	}

	template<template<typename, typename> typename mixin_type, typename... arg_types> static constexpr void impl_static(arg_types&&... args) noexcept {
		(impl_internal_filtered_static<mixin_type, bases>(args...), ...);
	}

	template<template<typename, typename, auto...> typename mixin_type, auto... values, typename... arg_types> void impl_thread(arg_types&&... args) noexcept {
		(impl_internal_filtered_thread<mixin_type, bases, values...>(args...), ...);
	}

	template<auto enum_value> decltype(auto) operator[](tag<enum_value>) {
		if constexpr (std::same_as<std::remove_cvref_t<decltype(enum_value)>, std::remove_cvref_t<enum_type>>) {
			using base_type = typename type_finder<enum_value, bases...>::type;
			return *static_cast<base_type*>(this);
		} else {
			using base_type = typename type_finder<enum_value, bases...>::type;
			return (*static_cast<base_type*>(this))[tag<enum_value>{}];
		}
	}

	template<uint64_t index_new> decltype(auto) get_core_by_index() const noexcept {
		static_assert(static_assert_printer<(index_new < size), nihilus_cathedral_errors::get_core_by_index_oob, static_assert_printer_val_inserter<index_new>>::impl);
		static constexpr uint64_t index{ static_cast<uint64_t>(index_transform_values[static_cast<uint64_t>(index_new)]) };
		return (*this)[tag<index>()];
	}

	template<enum_type enum_value> static consteval uint64_t get_index_by_enum() noexcept {
		for (uint64_t x = 0; x < size; ++x) {
			if (static_cast<enum_type>(index_transform_values[x]) == enum_value) {
				return x;
			}
		}
		return std::numeric_limits<uint64_t>::max();
	}

	void* intermediate_buffer{ nullptr };

  protected:
	template<template<typename, typename> typename mixin_type, typename base_type, typename... arg_types>
	constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) noexcept {
		if constexpr (mixin_type<config_type, base_type>::filter()) {
			static_assert(static_assert_printer<std::is_base_of_v<base_type, nihilus_cathedral>, nihilus_cathedral_errors::invalid_base_cast>::impl);
			mixin_type<config_type, base_type>::impl(*this, args...);
		}
	}

	template<template<typename, typename> typename mixin_type, typename base_type, typename... arg_types>
	static constexpr void impl_internal_filtered_static([[maybe_unused]] arg_types&&... args) noexcept {
		if constexpr (mixin_type<config_type, base_type>::filter()) {
			static_assert(static_assert_printer<std::is_base_of_v<base_type, nihilus_cathedral>, nihilus_cathedral_errors::invalid_base_cast>::impl);
			mixin_type<config_type, base_type>::impl(args...);
		}
	}

	template<template<typename, typename, auto...> typename mixin_type, typename base_type, auto... values, typename... arg_types>
	void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) noexcept {
		if constexpr (mixin_type<config_type, base_type, values...>::filter()) {
			static_assert(static_assert_printer<std::is_base_of_v<base_type, nihilus_cathedral>, nihilus_cathedral_errors::invalid_base_cast>::impl);
			mixin_type<config_type, base_type, values...>::impl(*this, args...);
		}
	}

	static constexpr uint64_t index_transform_values[sizeof...(bases)]{ static_cast<uint64_t>(bases::enum_value)... };
};


template<typename nihilus_cathedral_type, auto index> using get_nihilus_cathedral_type_at_enum =
	std::remove_cvref_t<decltype(std::declval<nihilus_cathedral_type>().template get_core_by_enum<index>())>;

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type, template<typename, enum_type> typename base_type,
	typename... value_type>
struct get_nihilus_cathedral_array;

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type, template<typename, enum_type> typename base_type, size_t... indices>
struct get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type, std::index_sequence<indices...>> {
	using type = nihilus_cathedral<config_type, base_type<config_type, static_cast<enum_type>(aggregator_type<config_type, enum_type>::values[indices])>...>;
};

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type, template<typename, enum_type> typename base_type>
using get_nihilus_cathedral_array_t = std::remove_cvref_t<typename get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type,
	std::make_index_sequence<static_cast<uint64_t>(aggregator_type<config_type, enum_type>::values.size())>>::type>;

template<typename value_type>
concept composite_core_types_type = std::same_as<composite_core_types, base_t<value_type>>;

template<typename config_type, typename enum_type> struct core {
	using type = get_nihilus_cathedral_array_t<config_type, enum_type, core_aggregator, core_interface>;
};

template<typename config_type, typename enum_type> using core_t = core<config_type, enum_type>::type;

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type, typename... value_type> struct get_nihilus_composite_cathedral_array;

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type, size_t... indices>
struct get_nihilus_composite_cathedral_array<config_type, enum_type, aggregator_type, std::index_sequence<indices...>> {
	using type = nihilus_cathedral<config_type,
		composite_core_interface<config_type, core_t<config_type, composite_core_type_holder_t<aggregator_type<config_type>::values[indices]>>,
			aggregator_type<config_type>::values[indices]>...>;
};

template<typename config_type, typename enum_type, template<typename, typename...> typename aggregator_type> using get_nihilus_composite_cathedral_array_t =
	std::remove_cvref_t<typename get_nihilus_composite_cathedral_array<config_type, enum_type, aggregator_type,
		std::make_index_sequence<static_cast<uint64_t>(aggregator_type<config_type>::values.size())>>::type>;

template<typename config_type, composite_core_types_type enum_type> struct composite_core {
	using type = get_nihilus_composite_cathedral_array_t<config_type, enum_type, composite_core_aggregator>;
};

template<typename config_type> using composite_core_t = composite_core<config_type, composite_core_types>::type;

template<typename config_type, typename base_type_new> struct memory_mapper_impl {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return base_type::alloc_class == alloc_classes::allocate_heap;
	}

	static void impl(base_type& core_val, const uint64_t& plan, memory_buffer<config_type::device_type, config_type::exceptions>& memory_buffer,
		uint64_t& internal_offset) {
		using data_type = typename base_type::output_type;
		if (internal_offset + base_type::total_required_bytes > plan || internal_offset + base_type::total_required_bytes > memory_buffer.size()) {
			static constexpr auto location = std::source_location::current();
			throw std::exception{};
		}
		if constexpr (base_type::data_strategy_type == data_strategy_types::per_block) {
			uint64_t total_required_bytes_for_this_core_traits{ base_type::total_required_bytes };
			uint64_t total_required_bytes_per_block{ total_required_bytes_for_this_core_traits / model_traits_type<config_type>::block_count };

			for (uint64_t x = 0; x < model_traits_type<config_type>::block_count; ++x) {
				data_type* ptr = memory_buffer.template claim_memory<data_type>(internal_offset);
				core_val.set_data(ptr, x);
				internal_offset += total_required_bytes_per_block;
			}
		} else {
			data_type* ptr = memory_buffer.template claim_memory<data_type>(internal_offset);
			core_val.set_data(ptr);
			internal_offset += base_type::total_required_bytes;
		}
	}
};

template<typename config_type, typename base_type_new> struct memory_mapper {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return true;
	}

	static void impl(base_type& core_val, const uint64_t& plan, memory_buffer<config_type::device_type, config_type::exceptions>& memory_buffer, uint64_t& internal_offset) {
		core_val.template impl<memory_mapper_impl>(plan, memory_buffer, internal_offset);
	}
};

template<typename config_type, typename base_type_new> struct kernel_iteration_counter_impl {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return !ephemeral_kernel_types<base_type> && !weights_types<base_type> && !input_only_types<base_type>;
	}

	static void impl([[maybe_unused]] base_type& core_val) {
		uint64_t dims[4]{ core_val.template get_rt_dim<0>().get_value(), core_val.template get_rt_dim<1>().get_value(), core_val.template get_rt_dim<2>().get_value(),
			core_val.template get_rt_dim<3>().get_value() };
		core_val.kernel_iteration_count = ceil_div(type_traits<typename base_type::output_type>::total_byte_size(base_type::get_dims()),
			static_cast<uint32_t>(bnch_swt::gpu_properties::max_persisting_l2_bytes / 2));
	}
};

template<typename config_type, typename base_type_new> struct kernel_iteration_counter {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return true;
	}

	static void impl([[maybe_unused]] base_type& core_val) {
		core_val.template impl<kernel_iteration_counter_impl>();
	}
};

template<typename config_type, typename base_type_new> struct barrett_reduction_updater_impl {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return base_type::kernel_type == kernel_types::reshape || base_type::kernel_type == kernel_types::cont;
	}

	static void impl(base_type& core_val) {		
		using dims_type										 = base_type;
		static constexpr auto output_dims					 = dims_type::get_dims();
		static constexpr auto input_dims					 = base_type::input_type_01::get_dims();
		using dimension_type								 = dimension_t<input_dims[0], input_dims[1], input_dims[2], input_dims[3]>;
		static constexpr dimension_type output_stride_1_base = output_dims[2] * output_dims[3];
		static constexpr dimension_type input_stride_1_base	 = input_dims[2] * input_dims[3];
		using input_type_01									 = typename base_type::input_type_01;
		using enum_type										 = base_t<decltype(base_type::input_type_01::enum_value)>;
		auto& input_rt_dims									 = static_cast<core_t<config_type, enum_type>&>(core_val)[tag<input_type_01::enum_value>{}];
		const dimension_type input_stride_0					 = input_rt_dims.template get_rt_dim<1>() * input_stride_1_base;
		core_val.output_stride_0							 = core_val.template get_rt_dim<1>() * output_stride_1_base;
		const dimension_type input_stride_1					 = input_stride_1_base;
		core_val.rt_math_ops_01.collect_values(input_stride_0);
		core_val.rt_math_ops_02.collect_values(input_stride_1);
	}
};

template<typename config_type, typename base_type_new> struct barrett_reduction_updater {
	using base_type = base_type_new;
	static constexpr bool filter() {
		return true;
	}

	static void impl(base_type& core_val) {
		core_val.template impl<barrett_reduction_updater_impl>();
	}
};

template<typename config_type, typename base_type_new> struct memory_planner_impl {
	using base_type = base_type_new;

	static constexpr bool filter() {
		return base_type::alloc_class == alloc_classes::allocate_heap;
	}

	static constexpr void impl(uint64_t& total_required_bytes) {
		total_required_bytes += base_type::total_required_bytes;
	}
};

template<typename config_type, typename base_type_new> struct memory_planner {
	using base_type = base_type_new;

	static constexpr bool filter() {
		return true;
	}

	static constexpr void impl(uint64_t& total_required_bytes) {
		using enum_type										 = composite_core_type_holder_t<base_type::enum_value>;
		core_t<config_type, enum_type>::template impl_static<memory_planner_impl>(total_required_bytes);
	}
};

template<typename config_type> static uint64_t nihilus_cathedral_memory_plan{ []() {
	uint64_t total_required_bytes{};
	composite_core_t<config_type>::template impl_static<memory_planner>(total_required_bytes);
#if NIHILUS_COMPILER_CUDA
	total_required_bytes += gpu_properties::max_persisting_l2_bytes;
#endif
	return total_required_bytes;
}() };

static constexpr device_types device_type{ device_types::cpu };

int main() {
	static constexpr model_config config_01{ .max_context_length = max_context_length_type{ 1024 },
		.model_generation										 = model_generations::v3_1,
		.model_size												 = model_sizes::llm_8B,
		.device_type											 = device_type,
		.model_arch												 = model_arches::llama };
	//max_tensor_element_index<model_config_type<config_01>>;
	//core_t<model_config_type<config_01>, fused_norm_ingress_types> core{};
	auto total_bytes = nihilus_cathedral_memory_plan<model_config_type<config_01>>;
	composite_core_t<model_config_type<config_01>> core_newer_01{};
	auto& core_newest = core_newer_01[tag<fused_norm_ingress_types::attn_norm_mul>{}];
	core_newest.dim_00();
	auto& core_new{ core_newest };
	auto& core_newester = core_newer_01[tag<composite_core_types::fused_norm_ingress>{}];
	std::array<std::array<void*, static_cast<uint64_t>(80ULL)>, static_cast<uint64_t>(13ULL)> data{};
	core_newer_01.template impl<weight_mapper>(data);
	core_newer_01.template impl<dim_updater>(3, 24);
	core_newer_01.template impl<barrett_reduction_updater>();
	core_newer_01.template impl<kernel_iteration_counter>();
	std::vector<uint8_t> file{};
	*static_cast<void**>(data[0][0]) = static_cast<uint8_t*>(file.data());
	memory_buffer<device_type, false> buffer{};
	buffer.init(nihilus_cathedral_memory_plan<model_config_type<config_01>>);
	uint64_t offset{};
	core_newer_01.template impl<memory_mapper>(nihilus_cathedral_memory_plan<model_config_type<config_01>>, buffer, offset);
	using core_interface_type = std::remove_cvref_t<decltype(core_newester)>;
	core_newester[tag<fused_norm_ingress_types::attn_norm_mul>{}];
	auto result = max_tensor_element_index<model_config_type<config_01>>;
	std::cout << "CURRENT BYTES FOR INTERMEDIATES + KV-CACHE OF META-LLAMA-3-1-70B-FP16: " << result << std::endl;
	//std::cout << "CURRENT BYTES FOR INTERMEDIATES + KV-CACHE OF META-LLAMA-3-1-70B-FP16: " << max_tensor_element_index<model_config_type<config_02>> << std::endl;
	std::cout << "CURRENT BYTES FOR INTERMEDIATES + KV-CACHE OF META-LLAMA-3-1-70B-FP16: " << total_bytes << std::endl;

	//std::cout << "MAX-PROMPT-LENGTH: " << model_config_type<config_01>::max_prompt_length << std::endl;
	//std::cout << "MAX-GENERATION-LENGTH: " << model_config_type<config_01>::max_generation_length << std::endl;
	//std::cout << "MAX-CONTEXT-LENGTH: " << model_config_type<config_01>::max_context_length << std::endl;
	return 0;
}