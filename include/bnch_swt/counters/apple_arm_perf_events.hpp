// Original design from:
// =============================================================================
// XNU kperf/kpc
// Available for 64-bit Intel/Apple Silicon, macOS/iOS, with root privileges
//
// References:
//
// XNU source (since xnu 2422.1.72):
// https://github.com/apple/darwin-xnu/blob/main/osfmk/kern/kpc.h
// https://github.com/apple/darwin-xnu/blob/main/bsd/kern/kern_kpc.c
//
// Lightweight PET (Profile Every Thread, since xnu 3789.1.32):
// https://github.com/apple/darwin-xnu/blob/main/osfmk/kperf/pet.c
// https://github.com/apple/darwin-xnu/blob/main/osfmk/kperf/kperf_kpc.c
//
// System Private frameworks (since macOS 10.11, iOS 8.0):
// /System/Library/PrivateFrameworks/kperf.framework
// /System/Library/PrivateFrameworks/kperfdata.framework
//
// Xcode framework (since Xcode 7.0):
// /Applications/Xcode.app/Contents/SharedFrameworks/DVTInstrumentsFoundation.framework
//
// CPU database (plist files)
// macOS (since macOS 10.11):
//     /usr/share/kpep/<name>.plist
// iOS (copied from Xcode, since iOS 10.0, Xcode 8.0):
//     /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform
//     /DeviceSupport/<version>/DeveloperDiskImage.dmg/usr/share/kpep/<name>.plist
//
//
// Created by YaoYuan <ibireme@gmail.com> on 2021.
// Released into the public domain (https://unlicense.org).
// =============================================================================

#pragma once

#include <bnch_swt/config.hpp>

#if BNCH_SWT_PLATFORM_MAC

	#include <mach/mach_time.h>
	#include <sys/sysctl.h>
	#include <sys/kdebug.h>
	#include <iostream>
	#include <unistd.h>
	#include <dlfcn.h>
	#include <cstring>
	#include <array>
	#include <span>
	#include <vector>
	#include <mutex>
	#include <string>
	#include <string_view>
	#include <bit>

namespace bnch_swt::internal {

	struct performance_counters {
		double branch_misses{};
		double instructions{};
		double branches{};
		double cycles{};

		BNCH_SWT_HOST performance_counters(double c, double b, double m, double i) : branch_misses(m), instructions(i), branches(b), cycles(c) {
		}

		BNCH_SWT_HOST performance_counters() : branch_misses{}, instructions{}, branches{}, cycles{} {
		}
	};

	BNCH_SWT_HOST performance_counters operator-(const performance_counters& a, const performance_counters& b) {
		return performance_counters(a.cycles - b.cycles, a.branches - b.branches, a.branch_misses - b.branch_misses, a.instructions - b.instructions);
	}

	class performance_monitor {
	  protected:
		static constexpr uint64_t kpc_class_configurable_mask{ 1ULL << 1ULL };
		static constexpr uint64_t kpc_class_rawpmu_mask{ 1ULL << 3ULL };
		static constexpr uint64_t kpc_class_fixed_mask{ 1ULL << 0ULL };
		static constexpr uint64_t kpc_class_power_mask{ 1ULL << 2ULL };
		static constexpr uint64_t kpc_max_counters{ 32 };

		struct kpep_event {
			const char* description;
			const char* fallback;
			const char* errata;
			const char* alias;
			const char* name;
			uint8_t reserved;
			uint8_t is_fixed;
			uint8_t number;
			uint32_t mask;
			uint8_t umask;
		};

		struct kpep_db {
			uint32_t config_counter_bits;
			kpep_event** fixed_event_arr;
			size_t config_counter_count;
			uint32_t fixed_counter_bits;
			uint32_t power_counter_bits;
			size_t fixed_counter_count;
			size_t power_counter_count;
			const char* marketing_name;
			kpep_event* event_arr;
			uint32_t archtecture;
			const char* cpu_id;
			size_t event_count;
			size_t alias_count;
			size_t reserved_1;
			size_t reserved_2;
			size_t reserved_3;
			const char* name;
			void* plist_data;
			void* event_map;
			void* alias_map;
		};

		struct kpep_config {
			uint32_t config_counter;
			uint32_t power_counter;
			uint64_t* kpc_periods;
			size_t counter_count;
			kpep_event** ev_arr;
			size_t event_count;
			uint32_t reserved;
			uint32_t classes;
			uint32_t* flags;
			size_t* ev_map;
			size_t* ev_idx;
			kpep_db* db;
		};

		enum class kpep_config_error_code {
			kpep_config_error_none					 = 0,
			kpep_config_error_invalid_argument		 = 1,
			kpep_config_error_out_of_memory			 = 2,
			kpep_config_error_io					 = 3,
			kpep_config_error_buffer_too_small		 = 4,
			kpep_config_error_cur_system_unknown	 = 5,
			kpep_config_error_db_path_invalid		 = 6,
			kpep_config_error_db_not_found			 = 7,
			kpep_config_error_db_arch_unsupported	 = 8,
			kpep_config_error_db_version_unsupported = 9,
			kpep_config_error_db_corrupt			 = 10,
			kpep_config_error_event_not_found		 = 11,
			kpep_config_error_conflicting_events	 = 12,
			kpep_config_error_counters_not_forced	 = 13,
			kpep_config_error_event_unavailable		 = 14,
			kpep_config_error_errno					 = 15,
			kpep_config_error_max
		};

		struct lib_symbol {
			const char* name;
			void** impl;
		};

		struct event_alias {
			const char* alias;
			std::array<const char*, 8> names;
		};

		inline static int32_t (*kpc_cpu_string)(char* buf, size_t buf_size);
		inline static uint32_t (*kpc_pmu_version)();
		inline static uint32_t (*kpc_get_counting)();
		inline static int32_t (*kpc_set_counting)(uint32_t classes);
		inline static uint32_t (*kpc_get_thread_counting)();
		inline static int32_t (*kpc_set_thread_counting)(uint32_t classes);
		inline static uint32_t (*kpc_get_config_count)(uint32_t classes);
		inline static int32_t (*kpc_get_config)(uint32_t classes, uint64_t* config);
		inline static int32_t (*kpc_set_config)(uint32_t classes, uint64_t* config);
		inline static uint32_t (*kpc_get_counter_count)(uint32_t classes);
		inline static int32_t (*kpc_get_cpu_counters)(bool all_cpus, uint32_t classes, int32_t* curcpu, uint64_t* buf);
		inline static int32_t (*kpc_get_thread_counters)(uint32_t tid, uint32_t buf_count, uint64_t* buf);
		inline static int32_t (*kpc_force_all_ctrs_set)(int32_t val);
		inline static int32_t (*kpc_force_all_ctrs_get)(int32_t* val_out);
		inline static int32_t (*kpep_config_create)(kpep_db* db, kpep_config** cfg_ptr);
		inline static void (*kpep_config_free)(kpep_config* cfg);
		inline static int32_t (*kpep_config_add_event)(kpep_config* cfg, kpep_event** ev_ptr, uint32_t flag, uint32_t* err);
		inline static int32_t (*kpep_config_force_counters)(kpep_config* cfg);
		inline static int32_t (*kpep_config_kpc)(kpep_config* cfg, uint64_t* buf, size_t buf_size);
		inline static int32_t (*kpep_config_kpc_count)(kpep_config* cfg, size_t* count_ptr);
		inline static int32_t (*kpep_config_kpc_classes)(kpep_config* cfg, uint32_t* classes_ptr);
		inline static int32_t (*kpep_config_kpc_map)(kpep_config* cfg, size_t* buf, size_t buf_size);
		inline static int32_t (*kpep_db_create)(const char* name, kpep_db** db_ptr);
		inline static void (*kpep_db_free)(kpep_db* db);
		inline static int32_t (*kpep_db_event)(kpep_db* db, const char* name, kpep_event** ev_ptr);

		inline static bool lib_inited{ false };
		inline static bool lib_has_err{ false };
		inline static char lib_err_msg[256];
		inline static void* lib_handle_kperf{ nullptr };
		inline static void* lib_handle_kperfdata{ nullptr };
		inline static bool has_events_val{};
		inline static std::mutex mutex{};

		inline static std::array<uint64_t, kpc_max_counters> regs{ 0 };
		inline static std::array<size_t, kpc_max_counters> counter_map{ 0 };
		inline static std::array<uint64_t, kpc_max_counters> counters_0{ 0 };

		static constexpr std::array<const char*, 16> kpep_config_error_names{ "none", "invalid argument", "out of memory", "I/O", "buffer too small", "current system unknown",
			"database path invalid", "database not found", "database architecture unsupported", "database version unsupported", "database corrupt", "event not found",
			"conflicting events", "all counters must be forced", "event unavailable", "check errno" };

		static constexpr std::array<event_alias, 4> profile_events{ {
			{ "cycles", { "FIXED_CYCLES", "CPU_CLK_UNHALTED.THREAD", "CPU_CLK_UNHALTED.CORE" } },
			{ "instructions", { "FIXED_INSTRUCTIONS", "INST_RETIRED.ANY" } },
			{ "branches", { "INST_BRANCH", "BR_INST_RETIRED.ALL_BRANCHES", "INST_RETIRED.ANY" } },
			{ "branch-misses", { "BRANCH_MISPRED_NONSPEC", "BRANCH_MISPREDICT", "BR_MISP_RETIRED.ALL_BRANCHES", "BR_INST_RETIRED.MISPRED" } },
		} };

		BNCH_SWT_HOST static const char* kpep_config_error_desc(int32_t code) {
			if (code >= 0 && static_cast<uint64_t>(code) < kpep_config_error_names.size()) {
				return kpep_config_error_names[static_cast<uint64_t>(code)];
			}
			return "unknown error";
		}

		BNCH_SWT_HOST static bool load_symbols(void* handle, const std::span<const lib_symbol>& symbols, std::string_view lib_name) {
			for (const auto& symbol: symbols) {
				*symbol.impl = dlsym(handle, symbol.name);
				if (!*symbol.impl) {
					std::string error_msg = "Failed to load ";
					error_msg += lib_name;
					error_msg += " function: ";
					error_msg += symbol.name;
					error_msg += ".";
					std::strncpy(lib_err_msg, error_msg.c_str(), sizeof(lib_err_msg) - 1);
					return false;
				}
			}
			return true;
		}

		BNCH_SWT_HOST static kpep_event* get_event(kpep_db* db, const event_alias* alias) {
			for (const char* name: alias->names) {
				if (!name) {
					break;
				}
				kpep_event* ev = nullptr;
				if (kpep_db_event(db, name, &ev) == 0) {
					return ev;
				}
			}
			return nullptr;
		}

	  public:
		BNCH_SWT_HOST static void lib_deinit() {
			lib_inited	= false;
			lib_has_err = false;
			if (lib_handle_kperf) {
				dlclose(lib_handle_kperf);
			}
			if (lib_handle_kperfdata) {
				dlclose(lib_handle_kperfdata);
			}
			lib_handle_kperf	 = nullptr;
			lib_handle_kperfdata = nullptr;
		}

		BNCH_SWT_HOST static bool lib_init() {
			if (lib_inited) {
				return !lib_has_err;
			}

			auto failure_cleanup = []() {
				lib_deinit();
				lib_inited	= true;
				lib_has_err = true;
				return false;
			};

			const char lib_path_kperf[]		= "/System/Library/PrivateFrameworks/kperf.framework/kperf";
			const char lib_path_kperfdata[] = "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata";

			lib_handle_kperf = dlopen(lib_path_kperf, RTLD_LAZY);
			if (!lib_handle_kperf) {
				return failure_cleanup();
			}

			lib_handle_kperfdata = dlopen(lib_path_kperfdata, RTLD_LAZY);
			if (!lib_handle_kperfdata) {
				return failure_cleanup();
			}

			const std::array<lib_symbol, 14> symbols_kperf{ lib_symbol("kpc_pmu_version", std::bit_cast<void**>(&kpc_pmu_version)),
				lib_symbol("kpc_cpu_string", std::bit_cast<void**>(&kpc_cpu_string)), lib_symbol("kpc_set_counting", std::bit_cast<void**>(&kpc_set_counting)),
				lib_symbol("kpc_get_counting", std::bit_cast<void**>(&kpc_get_counting)), lib_symbol("kpc_set_thread_counting", std::bit_cast<void**>(&kpc_set_thread_counting)),
				lib_symbol("kpc_get_thread_counting", std::bit_cast<void**>(&kpc_get_thread_counting)),
				lib_symbol("kpc_get_config_count", std::bit_cast<void**>(&kpc_get_config_count)),
				lib_symbol("kpc_get_counter_count", std::bit_cast<void**>(&kpc_get_counter_count)), lib_symbol("kpc_set_config", std::bit_cast<void**>(&kpc_set_config)),
				lib_symbol("kpc_get_config", std::bit_cast<void**>(&kpc_get_config)), lib_symbol("kpc_get_cpu_counters", std::bit_cast<void**>(&kpc_get_cpu_counters)),
				lib_symbol("kpc_get_thread_counters", std::bit_cast<void**>(&kpc_get_thread_counters)),
				lib_symbol("kpc_force_all_ctrs_set", std::bit_cast<void**>(&kpc_force_all_ctrs_set)),
				lib_symbol("kpc_force_all_ctrs_get", std::bit_cast<void**>(&kpc_force_all_ctrs_get)) };

			const std::array<lib_symbol, 11> symbols_kperfdata{ lib_symbol("kpep_config_create", std::bit_cast<void**>(&kpep_config_create)),
				lib_symbol("kpep_config_free", std::bit_cast<void**>(&kpep_config_free)), lib_symbol("kpep_config_add_event", std::bit_cast<void**>(&kpep_config_add_event)),
				lib_symbol("kpep_config_force_counters", std::bit_cast<void**>(&kpep_config_force_counters)),
				lib_symbol("kpep_config_kpc", std::bit_cast<void**>(&kpep_config_kpc)), lib_symbol("kpep_config_kpc_count", std::bit_cast<void**>(&kpep_config_kpc_count)),
				lib_symbol("kpep_config_kpc_classes", std::bit_cast<void**>(&kpep_config_kpc_classes)),
				lib_symbol("kpep_config_kpc_map", std::bit_cast<void**>(&kpep_config_kpc_map)), lib_symbol("kpep_db_create", std::bit_cast<void**>(&kpep_db_create)),
				lib_symbol("kpep_db_free", std::bit_cast<void**>(&kpep_db_free)), lib_symbol("kpep_db_event", std::bit_cast<void**>(&kpep_db_event)) };

			if (!load_symbols(lib_handle_kperf, symbols_kperf, "kperf")) {
				return failure_cleanup();
			}
			if (!load_symbols(lib_handle_kperfdata, symbols_kperfdata, "kperfdata")) {
				return failure_cleanup();
			}

			lib_inited	= true;
			lib_has_err = false;
			return true;
		}

		BNCH_SWT_HOST static bool setup_performance_counters() {
			std::lock_guard lock{ mutex };
			static bool init   = false;
			if (init) {
				return has_events_val;
			}
			init = true;

			if (!lib_init()) {
				std::cerr << "Error: " << lib_err_msg << std::endl;
				return (has_events_val = false);
			}

			int32_t force_ctrs = 0;
			if (kpc_force_all_ctrs_get(&force_ctrs)) {
				return (has_events_val = false);
			}

			int32_t ret;
			kpep_db* db = nullptr;
			if ((ret = kpep_db_create(nullptr, &db))) {
				return (has_events_val = false);
			}
			kpep_config* cfg = nullptr;
			if ((ret = kpep_config_create(db, &cfg))) {
				return (has_events_val = false);
			}
			if ((ret = kpep_config_force_counters(cfg))) {
				return (has_events_val = false);
			}
			for (const auto& alias: profile_events) {
				kpep_event* ev = get_event(db, &alias);
				if (!ev || kpep_config_add_event(cfg, &ev, 0, nullptr))
					return (has_events_val = false);
			}

			uint32_t classes = 0;
			size_t reg_count = 0;
			if (kpep_config_kpc_classes(cfg, &classes)) {
				return (has_events_val = false);
			}
			if (kpep_config_kpc_count(cfg, &reg_count)) {
				return (has_events_val = false);
			}
			if (kpep_config_kpc_map(cfg, counter_map.data(), sizeof(counter_map))) {
				return (has_events_val = false);
			}
			if (kpep_config_kpc(cfg, regs.data(), sizeof(regs))) {
				return (has_events_val = false);
			}
			if (kpc_force_all_ctrs_set(1)) {
				return (has_events_val = false);
			}
			if ((classes & kpc_class_configurable_mask) && reg_count) {
				if (kpc_set_config(classes, regs.data()))
					return (has_events_val = false);
			}

			if (kpc_set_counting(classes)) {
				return (has_events_val = false);
			}
			if (kpc_set_thread_counting(classes)) {
				return (has_events_val = false);
			}

			return (has_events_val = true);
		}

		BNCH_SWT_HOST static performance_counters get_counters() {
			static bool warned = false;
			if (kpc_get_thread_counters(0, kpc_max_counters, counters_0.data())) {
				if (!warned) {
					std::cerr << "Failed get thread counters." << std::endl;
					warned = true;
				}
				return {};
			}
			return performance_counters{ static_cast<double>(counters_0[counter_map[0]]), static_cast<double>(counters_0[counter_map[2]]),
				static_cast<double>(counters_0[counter_map[3]]), static_cast<double>(counters_0[counter_map[1]]) };
		}

		BNCH_SWT_HOST static bool has_events() {
			return has_events_val;
		}
	};

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cpu, count> : public std::vector<event_count> {
		performance_counters diff{};
		size_t current_index{};

		BNCH_SWT_HOST event_collector_type() : std::vector<event_count>{ count }, diff{}, current_index{} {
			performance_monitor::setup_performance_counters();
		}		

		template<typename function_type, typename... arg_types> BNCH_SWT_HOST void run(arg_types&&... args) {
			if (performance_monitor::has_events()) {
				diff = performance_monitor::get_counters();
			}
			const auto start_clock = clock_type::now();
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(static_cast<size_t>(function_type::impl(std::forward<arg_types>(args)...)));
			const auto end_clock = clock_type::now();
			if (performance_monitor::has_events()) {
				performance_counters end = performance_monitor::get_counters();
				diff					 = end - diff;
				std::vector<event_count>::operator[](current_index).cycles_val.emplace(diff.cycles);
				std::vector<event_count>::operator[](current_index).instructions_val.emplace(diff.instructions);
				std::vector<event_count>::operator[](current_index).branches_val.emplace(diff.branches);
				std::vector<event_count>::operator[](current_index).branch_misses_val.emplace(diff.branch_misses);
			}
			std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(end_clock - start_clock);
			++current_index;
		}
	};
}

#endif