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
		static constexpr aligned_const<uint64_t, 64> sm_count_raw{ 70ULL };
		static constexpr aligned_const<uint64_t, 64> max_threads_per_sm_raw{ 1536ULL };
		static constexpr aligned_const<uint64_t, 64> max_threads_per_block_raw{ 1024ULL };
		static constexpr aligned_const<uint64_t, 64> warp_size_raw{ 32ULL };
		static constexpr aligned_const<uint64_t, 64> l2_cache_size_raw{ 50331648ULL };
		static constexpr aligned_const<uint64_t, 64> shared_mem_per_block_raw{ 49152ULL };
		static constexpr aligned_const<uint64_t, 64> max_grid_size_x_raw{ 2147483647ULL };
		static constexpr aligned_const<uint64_t, 64> max_grid_size_y_raw{ 65535ULL };
		static constexpr aligned_const<uint64_t, 64> max_grid_size_z_raw{ 65535ULL };
		static constexpr aligned_const<uint64_t, 64> gpu_arch_index_raw{ 4ULL };
		static constexpr aligned_const<uint64_t, 64> total_threads_raw{ 107520ULL };

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
