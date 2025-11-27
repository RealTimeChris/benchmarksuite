// bnch_swt/Counters/AndroidCounters.hpp
#pragma once

#include <bnch_swt/config.hpp>
#include <vector>
#include <chrono>

#if BNCH_SWT_PLATFORM_ANDROID

namespace bnch_swt::internal {

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cpu, count> : public std::vector<event_count> {
		uint64_t currentIndex{};

		BNCH_SWT_HOST event_collector_type() : std::vector<event_count_t>{ count_t } {};

		template<typename function_type, typename... arg_types> BNCH_SWT_HOST void run(arg_types&&... args) {
			const auto startClock = std::chrono::high_resolution_clock::now();
			std::vector<event_count_t>::operator[](currentIndex).bytesProcessedVal.emplace(static_cast<uint64_t>(function_type::impl(std::forward<arg_types>(args)...)));
			const auto endClock = std::chrono::high_resolution_clock::now();
			std::vector<event_count_t>::operator[](currentIndex).elapsed = endClock - startClock;
			++currentIndex;
			return;
		}
	};

}

#endif
