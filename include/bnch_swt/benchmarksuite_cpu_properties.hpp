/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/
#pragma once

#include <bnch_swt/aligned_const.hpp>

namespace bnch_swt {

	struct cpu_properties {
	  protected:
		static constexpr aligned_const thread_count_raw{ 32ull };
		static constexpr aligned_const l1_cache_size_raw{ 49152ull };
		static constexpr aligned_const l2_cache_size_raw{ 2097152ull };
		static constexpr aligned_const l3_cache_size_raw{ 37748736ull };
		static constexpr aligned_const cpu_arch_index_raw{ 1ull };
		static constexpr aligned_const cpu_alignment_raw{ 32ull };

	  public:
		static constexpr const uint64_t& thread_count{ *thread_count_raw };
		static constexpr const uint64_t& l1_cache_size{ *l1_cache_size_raw };
		static constexpr const uint64_t& l2_cache_size{ *l2_cache_size_raw };
		static constexpr const uint64_t& l3_cache_size{ *l3_cache_size_raw };
		static constexpr const uint64_t& cpu_arch_index{ *cpu_arch_index_raw };
		static constexpr const uint64_t& cpu_alignment{ *cpu_alignment_raw };
	};

}
