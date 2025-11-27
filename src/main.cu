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
	static constexpr uint32_t embedding_length		  = 4096;
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
	BNCH_SWT_HOST cuda_buffer(const cuda_buffer&) noexcept			 = delete;

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

	BNCH_SWT_HOST void init(uint64_t size) noexcept {
		if (data_val) {
			clear();
		}

		cudaError_t result = cudaMalloc(&data_val, size);
		if (result != cudaSuccess) {
			data_val = nullptr;
		}

		size_val = size;
	}

	BNCH_SWT_HOST void deinit() noexcept {
		clear();
	}

	BNCH_SWT_HOST size_type size() noexcept {
		return size_val;
	}

	BNCH_SWT_HOST pointer data() noexcept {
		return data_val;
	}

	BNCH_SWT_HOST void* claim_memory(uint64_t offset_to_claim) noexcept {
		uint64_t aligned_amount = round_up_to_multiple<64ull>(offset_to_claim);
		pointer return_value	= data_val + aligned_amount;
		return return_value;
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

template<uint64_t shift, integral_types value_type> BNCH_SWT_HOST_DEVICE constexpr value_type operator<<(const value_type arg, std::integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg << shift_amount;
}

template<uint64_t shift, integral_types value_type> BNCH_SWT_HOST_DEVICE constexpr value_type& operator<<=(value_type& arg, std::integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg = arg << shift_amount;
}

template<uint64_t shift, integral_types value_type> BNCH_SWT_HOST_DEVICE constexpr value_type operator>>(const value_type arg, std::integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg >> shift_amount;
}

template<uint64_t shift, integral_types value_type> BNCH_SWT_HOST_DEVICE constexpr value_type& operator>>=(value_type& arg, std::integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return arg = arg >> shift_amount;
}

template<uint64_t shift> BNCH_SWT_HOST_DEVICE constexpr std::byte operator<<(const std::byte _Arg, std::integral_constant<uint64_t, shift>) noexcept {
	constexpr uint64_t shift_amount{ shift };
	return static_cast<std::byte>(static_cast<unsigned char>(static_cast<unsigned int>(_Arg) << shift_amount));
}

struct cpu_buffer {
	using size_type	 = uint64_t;
	using value_type = std::byte;
	using pointer	 = value_type*;
	BNCH_SWT_HOST cpu_buffer() noexcept {
	}
	BNCH_SWT_HOST cpu_buffer& operator=(const cpu_buffer&) noexcept = delete;
	BNCH_SWT_HOST cpu_buffer(const cpu_buffer&) noexcept			 = delete;

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
		uint64_t aligned_amount = round_up_to_multiple<64ull>(offset_to_claim);
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

template<typename value_type, uint64_t dim_01_new = 1, uint64_t dim_02_new = 1, uint64_t dim_03_new = 1, uint64_t dim_04_new = 1> struct tensor {
	static constexpr uint64_t dim_01{ dim_01_new };
	static constexpr uint64_t dim_02{ dim_02_new };
	static constexpr uint64_t dim_03{ dim_03_new };
	static constexpr uint64_t dim_04{ dim_04_new };
	static constexpr uint64_t element_count{ dim_01 * dim_02 * dim_03 * dim_04 };
	static constexpr uint64_t byte_count{ sizeof(value_type) * element_count };

	void* data{};
};

struct memory_footprint {
	uint64_t byte_count{};
	void* const* data{};
};

struct cuda_tensors {
	tensor<float, model_traits::embedding_length, model_traits::embedding_length, 1, 1> attn_q_weight{};

	tensor<float, model_traits::embedding_length, model_traits::n_embd_kv_gqa, 1, 1> attn_k_weight{};

	tensor<float, model_traits::embedding_length, model_traits::n_embd_kv_gqa, 1, 1> attn_v_weight{};

	tensor<float, model_traits::embedding_length, model_traits::embedding_length, 1, 1> attn_output_weight{};

	tensor<float, model_traits::embedding_length, 1, 1, 1> attn_norm_weight{};

	tensor<float, model_traits::embedding_length, model_traits::feed_forward_length, 1, 1> ffn_gate_weight{};

	tensor<float, model_traits::embedding_length, model_traits::feed_forward_length, 1, 1> ffn_up_weight{};

	tensor<float, model_traits::feed_forward_length, model_traits::embedding_length, 1, 1> ffn_down_weight{};

	tensor<float, model_traits::embedding_length, 1, 1, 1> ffn_norm_weight{};

	tensor<float, model_traits::embedding_length, model_traits::vocab_size, 1, 1> token_embd_weight{};

	tensor<float, model_traits::rope_dimension_count / 2, 1, 1, 1> rope_freqs_weight{};

	tensor<float, model_traits::embedding_length, 1, 1, 1> output_norm_weight{};

	tensor<float, model_traits::embedding_length, model_traits::vocab_size, 1, 1> output_weight{};
};

constexpr cuda_tensors cuda_tensors_val{};

constexpr std::array footprints{ memory_footprint{ cuda_tensors_val.attn_q_weight.byte_count, &cuda_tensors_val.attn_q_weight.data },
	memory_footprint{ cuda_tensors_val.attn_k_weight.byte_count, &cuda_tensors_val.attn_k_weight.data },
	memory_footprint{ cuda_tensors_val.attn_v_weight.byte_count, &cuda_tensors_val.attn_v_weight.data },
	memory_footprint{ cuda_tensors_val.attn_output_weight.byte_count, &cuda_tensors_val.attn_output_weight.data },
	memory_footprint{ cuda_tensors_val.attn_norm_weight.byte_count, &cuda_tensors_val.attn_norm_weight.data },
	memory_footprint{ cuda_tensors_val.ffn_gate_weight.byte_count, &cuda_tensors_val.ffn_gate_weight.data },
	memory_footprint{ cuda_tensors_val.ffn_up_weight.byte_count, &cuda_tensors_val.ffn_up_weight.data },
	memory_footprint{ cuda_tensors_val.ffn_down_weight.byte_count, &cuda_tensors_val.ffn_down_weight.data },
	memory_footprint{ cuda_tensors_val.ffn_norm_weight.byte_count, &cuda_tensors_val.ffn_norm_weight.data },
	memory_footprint{ cuda_tensors_val.token_embd_weight.byte_count, &cuda_tensors_val.token_embd_weight.data },
	memory_footprint{ cuda_tensors_val.rope_freqs_weight.byte_count, &cuda_tensors_val.rope_freqs_weight.data },
	memory_footprint{ cuda_tensors_val.output_norm_weight.byte_count, &cuda_tensors_val.output_norm_weight.data },
	memory_footprint{ cuda_tensors_val.output_weight.byte_count, &cuda_tensors_val.output_weight.data } };

constexpr uint64_t byte_count{ [] {
	uint64_t return_value{};
	for (uint64_t x = 0; x < footprints.size(); ++x) {
		return_value += footprints[x].byte_count;
	}
	return return_value;
}() };

template<typename value_type, uint64_t value_count, value_type min = std::numeric_limits<value_type>::min(), value_type max = std::numeric_limits<value_type>::max()>
BNCH_SWT_HOST void generate_values(void* cuda_memory) {
	std::vector<value_type> return_values{};
	for (uint64_t x = 0; x < value_count; ++x) {
		return_values.emplace_back(bnch_swt::random_generator<value_type>::impl(min, max));
	}
	if (auto result = cudaMemcpy(cuda_memory, return_values.data(), return_values.size() * sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice); result) {
		std::cout << "cudaMemcpy Error: " << cudaGetErrorString(result) << std::endl;
	}
	return;
}

template<typename value_type> BNCH_SWT_HOST void generate_cuda_data(cuda_buffer& buffer) {
	uint64_t current_offset = 0;

	generate_values<value_type, cuda_tensors_val.attn_q_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.attn_q_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.attn_k_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.attn_k_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.attn_v_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.attn_v_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.attn_output_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.attn_output_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.attn_norm_weight.element_count, 0.9f, 1.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.attn_norm_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.ffn_gate_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.ffn_gate_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.ffn_up_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.ffn_up_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.ffn_down_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.ffn_down_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.ffn_norm_weight.element_count, 0.9f, 1.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.ffn_norm_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.token_embd_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.token_embd_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.rope_freqs_weight.element_count, 0.0f, 1.0f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.rope_freqs_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.output_norm_weight.element_count, 0.9f, 1.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
	current_offset += cuda_tensors_val.output_norm_weight.byte_count;

	generate_values<value_type, cuda_tensors_val.output_weight.element_count, -0.1f, 0.1f>(static_cast<std::byte*>(buffer.data()) + current_offset);
}

static constexpr uint64_t total_iterations{ 100 };
static constexpr uint64_t measured_iterations{ 10 };

cuda_buffer buffer{ [] {
	cuda_buffer return_values{};
	return_values.init(byte_count);
	return return_values;
}() };

struct benchmark_test_cpu {
	BNCH_SWT_HOST static uint64_t impl() {
		generate_cuda_data<float>(buffer);
		return buffer.size();
	}
};

struct benchmark_ggml {
	BNCH_SWT_DEVICE static void impl() {
	}
};

struct benchmark_nihilus {
	BNCH_SWT_DEVICE static void impl(cuda_tensors cuda_tensors_val_new) {

	}
};

int main() {
	uint64_t test_byte{};
	test_byte << std::integral_constant<uint64_t, 32>{};
	using benchmark = bnch_swt::benchmark_stage<"kernel-gegen-kernel", total_iterations, measured_iterations, bnch_swt::benchmark_types::cuda>;
	using test_benchmark = bnch_swt::benchmark_stage<"kernel-gegen-kernel-test", 1, 1, bnch_swt::benchmark_types::cpu>;
	generate_cuda_data<float>(buffer);
	dim3 grid{};
	dim3 block{};

	uint64_t bytes_transferred{};
	test_benchmark::run_benchmark<"cuda-setup", benchmark_test_cpu>();
	benchmark::run_benchmark<"ggml", benchmark_ggml>(grid, block, 0, bytes_transferred);

	benchmark::run_benchmark<"nihilus", benchmark_nihilus>(grid, block, 0, bytes_transferred, cuda_tensors_val);

	benchmark::print_results();
	test_benchmark::print_results();
	return 0;
}