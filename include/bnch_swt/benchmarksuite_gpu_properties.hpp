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

#include <bnch_swt/aligned_const.hpp>

namespace bnch_swt {

	struct gpu_properties {
	  protected:
		static constexpr aligned_const sm_count_raw{ 70ull };
		static constexpr aligned_const max_threads_per_sm_raw{ 1536ull };
		static constexpr aligned_const max_threads_per_block_raw{ 1024ull };
		static constexpr aligned_const warp_size_raw{ 32ull };
		static constexpr aligned_const l2_cache_size_raw{ 50331648ull };
		static constexpr aligned_const shared_mem_per_block_raw{ 49152ull };
		static constexpr aligned_const max_grid_size_x_raw{ 2147483647ull };
		static constexpr aligned_const max_grid_size_y_raw{ 65535ull };
		static constexpr aligned_const max_grid_size_z_raw{ 65535ull };
		static constexpr aligned_const gpu_arch_index_raw{ 4ull };
		static constexpr aligned_const total_threads_raw{ 107520ull };

	  public:
		static constexpr const uint64_t& sm_count{ *sm_count_raw };
		static constexpr const uint64_t& max_threads_per_sm{ *max_threads_per_sm_raw };
		static constexpr const uint64_t& max_threads_per_block{ *max_threads_per_block_raw };
		static constexpr const uint64_t& warp_size{ *warp_size_raw };
		static constexpr const uint64_t& l2_cache_size{ *l2_cache_size_raw };
		static constexpr const uint64_t& shared_mem_per_block{ *shared_mem_per_block_raw };
		static constexpr const uint64_t& max_grid_size_x{ *max_grid_size_x_raw };
		static constexpr const uint64_t& max_grid_size_y{ *max_grid_size_y_raw };
		static constexpr const uint64_t& max_grid_size_z{ *max_grid_size_z_raw };
		static constexpr const uint64_t& total_threads{ *total_threads_raw };
		static constexpr const uint64_t& gpu_arch_index{ *gpu_arch_index_raw };
	};
}
