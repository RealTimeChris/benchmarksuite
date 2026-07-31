#pragma once

#include <benchmarksuite-incl/config.hpp>

#if BNCH_SWT_PLATFORM_ANDROID

namespace benchmarksuite::internal {

	template<benchmark_types benchmark_types, typename function_type> struct iteration_metric_collector {
		template<typename metric_type, typename... arg_types> BNCH_SWT_NOINLINE static void impl(metric_type& iteration_data, arg_types&&... args) {
			const auto start_clock				= clock_type::now();
			iteration_data.bytes_processed		= static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...));
			const auto end_clock				= clock_type::now();
			iteration_data.time_in_ns			= (end_clock - start_clock).count();
		}
	};

}
#endif
