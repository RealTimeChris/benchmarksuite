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

#include <bnch_swt/config.hpp>

namespace bnch_swt {

	struct BNCH_SWT_ALIGN(64) uint64_holder {
		BNCH_SWT_ALIGN(64) uint64_t value {};

		BNCH_SWT_HOST constexpr operator const uint64_t&() const {
			return value;
		}
	};

	struct cpu_properties {
	  protected:
		static constexpr uint64_holder thread_count_raw{ 32ULL };
		static constexpr uint64_holder l1_cache_size_raw{ 49152ULL };
		static constexpr uint64_holder l2_cache_size_raw{ 2097152ULL };
		static constexpr uint64_holder l3_cache_size_raw{ 37748736ULL };
		static constexpr uint64_holder cpu_arch_index_raw{ 1ULL };
		static constexpr uint64_holder cpu_alignment_raw{ 32ULL };

	  public:
		static constexpr const uint64_t& thread_count{ thread_count_raw };
		static constexpr const uint64_t& l1_cache_size{ l1_cache_size_raw };
		static constexpr const uint64_t& l2_cache_size{ l2_cache_size_raw };
		static constexpr const uint64_t& l3_cache_size{ l3_cache_size_raw };
		static constexpr const uint64_t& cpu_arch_index{ cpu_arch_index_raw };
		static constexpr const uint64_t& cpu_alignment{ cpu_alignment_raw };
	};

}
