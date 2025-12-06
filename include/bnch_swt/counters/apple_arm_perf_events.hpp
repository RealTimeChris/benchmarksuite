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

		BNCH_SWT_HOST performance_counters& operator-=(const performance_counters& other) {
			cycles -= other.cycles;
			branches -= other.branches;
			branch_misses -= other.branch_misses;
			instructions -= other.instructions;
			return *this;
		}

		BNCH_SWT_HOST performance_counters& min(const performance_counters& other) {
			cycles		  = other.cycles < cycles ? other.cycles : cycles;
			branches	  = other.branches < branches ? other.branches : branches;
			branch_misses = other.branch_misses < branch_misses ? other.branch_misses : branch_misses;
			instructions  = other.instructions < instructions ? other.instructions : instructions;
			return *this;
		}

		BNCH_SWT_HOST performance_counters& operator+=(const performance_counters& other) {
			cycles += other.cycles;
			branches += other.branches;
			branch_misses += other.branch_misses;
			instructions += other.instructions;
			return *this;
		}

		BNCH_SWT_HOST performance_counters& operator/=(double numerator) {
			cycles /= numerator;
			branches /= numerator;
			branch_misses /= numerator;
			instructions /= numerator;
			return *this;
		}
	};

	BNCH_SWT_HOST performance_counters operator-(const performance_counters& a, const performance_counters& b) {
		return performance_counters(a.cycles - b.cycles, a.branches - b.branches, a.branch_misses - b.branch_misses, a.instructions - b.instructions);
	}

	static constexpr uint64_t kpc_class_fixed{ 0 };
	static constexpr uint64_t kpc_class_configurable{ 1 };
	static constexpr uint64_t kpc_class_power{ 2 };
	static constexpr uint64_t kpc_class_rawpmu{ 3 };

	static constexpr uint64_t kpc_class_fixed_mask{ 1u << kpc_class_fixed };
	static constexpr uint64_t kpc_class_configurable_mask{ 1u << kpc_class_configurable };
	static constexpr uint64_t kpc_class_power_mask{ 1u << kpc_class_power };
	static constexpr uint64_t kpc_class_rawpmu_mask{ 1u << kpc_class_rawpmu };

	static constexpr uint64_t kpc_pmu_error{ 0 };
	static constexpr uint64_t kpc_pmu_intel_v3{ 1 };
	static constexpr uint64_t kpc_pmu_arm_apple{ 2 };
	static constexpr uint64_t kpc_pmu_intel_v2{ 3 };
	static constexpr uint64_t kpc_pmu_arm_v2{ 4 };

	static constexpr uint64_t kpc_max_counters{ 32 };

	static constexpr uint64_t kperf_sampler_th_info{ 1U << 0 };
	static constexpr uint64_t kperf_sampler_th_snapshot{ 1U << 1 };
	static constexpr uint64_t kperf_sampler_kstack{ 1U << 2 };
	static constexpr uint64_t kperf_sampler_ustack{ 1U << 3 };
	static constexpr uint64_t kperf_sampler_pmc_thread{ 1U << 4 };
	static constexpr uint64_t kperf_sampler_pmc_cpu{ 1U << 5 };
	static constexpr uint64_t kperf_sampler_pmc_config{ 1U << 6 };
	static constexpr uint64_t kperf_sampler_meminfo{ 1U << 7 };
	static constexpr uint64_t kperf_sampler_th_scheduling{ 1U << 8 };
	static constexpr uint64_t kperf_sampler_th_dispatch{ 1U << 9 };
	static constexpr uint64_t kperf_sampler_tk_snapshot{ 1U << 10 };
	static constexpr uint64_t kperf_sampler_sys_mem{ 1U << 11 };
	static constexpr uint64_t kperf_sampler_th_inscyc{ 1U << 12 };
	static constexpr uint64_t kperf_sampler_tk_info{ 1U << 13 };

	static constexpr uint64_t kperf_action_max{ 32 };
	static constexpr uint64_t kperf_timer_max{ 8 };

	using kpc_config_t = uint64_t;

	static int32_t (*kpc_cpu_string)(char* buf, size_t buf_size);
	static uint32_t (*kpc_pmu_version)();
	static uint32_t (*kpc_get_counting)();
	static int32_t (*kpc_set_counting)(uint32_t classes);
	static uint32_t (*kpc_get_thread_counting)();
	static int32_t (*kpc_set_thread_counting)(uint32_t classes);
	static uint32_t (*kpc_get_config_count)(uint32_t classes);
	static int32_t (*kpc_get_config)(uint32_t classes, kpc_config_t* config);
	static int32_t (*kpc_set_config)(uint32_t classes, kpc_config_t* config);
	static uint32_t (*kpc_get_counter_count)(uint32_t classes);
	static int32_t (*kpc_get_cpu_counters)(bool all_cpus, uint32_t classes, int32_t* curcpu, uint64_t* buf);
	static int32_t (*kpc_get_thread_counters)(uint32_t tid, uint32_t buf_count, uint64_t* buf);
	static int32_t (*kpc_force_all_ctrs_set)(int32_t val);
	static int32_t (*kpc_force_all_ctrs_get)(int32_t* val_out);
	static int32_t (*kperf_action_count_set)(uint32_t count);
	static int32_t (*kperf_action_count_get)(uint32_t* count);
	static int32_t (*kperf_action_samplers_set)(uint32_t actionid, uint32_t sample);
	static int32_t (*kperf_action_samplers_get)(uint32_t actionid, uint32_t* sample);
	static int32_t (*kperf_action_filter_set_by_task)(uint32_t actionid, int32_t port);
	static int32_t (*kperf_action_filter_set_by_pid)(uint32_t actionid, int32_t pid);
	static int32_t (*kperf_timer_count_set)(uint32_t count);
	static int32_t (*kperf_timer_count_get)(uint32_t* count);
	static int32_t (*kperf_timer_period_set)(uint32_t actionid, uint64_t tick);
	static int32_t (*kperf_timer_period_get)(uint32_t actionid, uint64_t* tick);
	static int32_t (*kperf_timer_action_set)(uint32_t actionid, uint32_t timerid);
	static int32_t (*kperf_timer_action_get)(uint32_t actionid, uint32_t* timerid);
	static int32_t (*kperf_timer_pet_set)(uint32_t timerid);
	static int32_t (*kperf_timer_pet_get)(uint32_t* timerid);
	static int32_t (*kperf_sample_set)(uint32_t enabled);
	static int32_t (*kperf_sample_get)(uint32_t* enabled);
	static int32_t (*kperf_reset)();
	static uint64_t (*kperf_ns_to_ticks)(uint64_t ns);
	static uint64_t (*kperf_ticks_to_ns)(uint64_t ticks);
	static uint64_t (*kperf_tick_frequency)();

	static constexpr uint64_t kpep_arch_i386{ 0 };
	static constexpr uint64_t kpep_arch_x86_64{ 1 };
	static constexpr uint64_t kpep_arch_arm{ 2 };
	static constexpr uint64_t kpep_arch_arm64{ 3 };

	struct kpep_event {
		const char* name;
		const char* description;
		const char* errata;
		const char* alias;
		const char* fallback;
		uint32_t mask;
		uint8_t number;
		uint8_t umask;
		uint8_t reserved;
		uint8_t is_fixed;
	};

	struct kpep_db {
		const char* name;
		const char* cpu_id;
		const char* marketing_name;
		void* plist_data;
		void* event_map;
		kpep_event* event_arr;
		kpep_event** fixed_event_arr;
		void* alias_map;
		size_t reserved_1;
		size_t reserved_2;
		size_t reserved_3;
		size_t event_count;
		size_t alias_count;
		size_t fixed_counter_count;
		size_t config_counter_count;
		size_t power_counter_count;
		uint32_t archtecture;
		uint32_t fixed_counter_bits;
		uint32_t config_counter_bits;
		uint32_t power_counter_bits;
	};

	struct kpep_config {
		kpep_db* db;
		kpep_event** ev_arr;
		size_t* ev_map;
		size_t* ev_idx;
		uint32_t* flags;
		uint64_t* kpc_periods;
		size_t event_count;
		size_t counter_count;
		uint32_t classes;
		uint32_t config_counter;
		uint32_t power_counter;
		uint32_t reserved;
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

	static constexpr std::array<const char*, static_cast<uint64_t>(kpep_config_error_code::kpep_config_error_max)> kpep_config_error_names = { "none", "invalid argument",
		"out of memory", "I/O", "buffer too small", "current system unknown", "database path invalid", "database not found", "database architecture unsupported",
		"database version unsupported", "database corrupt", "event not found", "conflicting events", "all counters must be forced", "event unavailable", "check errno" };

	BNCH_SWT_HOST static const char* kpep_config_error_desc(int32_t code) {
		if (0 <= code && static_cast<uint64_t>(code) < static_cast<uint64_t>(kpep_config_error_code::kpep_config_error_max)) {
			return kpep_config_error_names[static_cast<uint64_t>(code)];
		}
		return "unknown error";
	}

	static int32_t (*kpep_config_create)(kpep_db* db, kpep_config** cfg_ptr);
	static void (*kpep_config_free)(kpep_config* cfg);
	static int32_t (*kpep_config_add_event)(kpep_config* cfg, kpep_event** ev_ptr, uint32_t flag, uint32_t* err);
	static int32_t (*kpep_config_remove_event)(kpep_config* cfg, size_t idx);
	static int32_t (*kpep_config_force_counters)(kpep_config* cfg);
	static int32_t (*kpep_config_events_count)(kpep_config* cfg, size_t* count_ptr);
	static int32_t (*kpep_config_events)(kpep_config* cfg, kpep_event** buf, size_t buf_size);
	static int32_t (*kpep_config_kpc)(kpep_config* cfg, kpc_config_t* buf, size_t buf_size);
	static int32_t (*kpep_config_kpc_count)(kpep_config* cfg, size_t* count_ptr);
	static int32_t (*kpep_config_kpc_classes)(kpep_config* cfg, uint32_t* classes_ptr);
	static int32_t (*kpep_config_kpc_map)(kpep_config* cfg, size_t* buf, size_t buf_size);
	static int32_t (*kpep_db_create)(const char* name, kpep_db** db_ptr);
	static void (*kpep_db_free)(kpep_db* db);
	static int32_t (*kpep_db_name)(kpep_db* db, const char** name);
	static int32_t (*kpep_db_aliases_count)(kpep_db* db, size_t* count);
	static int32_t (*kpep_db_aliases)(kpep_db* db, const char** buf, size_t buf_size);
	static int32_t (*kpep_db_counters_count)(kpep_db* db, uint8_t classes, size_t* count);
	static int32_t (*kpep_db_events_count)(kpep_db* db, size_t* count);
	static int32_t (*kpep_db_events)(kpep_db* db, kpep_event** buf, size_t buf_size);
	static int32_t (*kpep_db_event)(kpep_db* db, const char* name, kpep_event** ev_ptr);
	static int32_t (*kpep_event_name)(kpep_event* ev, const char** name_ptr);
	static int32_t (*kpep_event_alias)(kpep_event* ev, const char** alias_ptr);
	static int32_t (*kpep_event_description)(kpep_event* ev, const char** str_ptr);

	struct lib_symbol {
		const char* name;
		void** impl;
	};

	template<typename T> constexpr size_t lib_nelems(const T& x) {
		return sizeof(x) / sizeof((x)[0]);
	}

	#define lib_symbol_def(name) \
		lib_symbol { \
			#name, reinterpret_cast<void**>(&name) \
		}

	static const std::array<lib_symbol, 34> lib_symbols_kperf = {
		lib_symbol_def(kpc_pmu_version),
		lib_symbol_def(kpc_cpu_string),
		lib_symbol_def(kpc_set_counting),
		lib_symbol_def(kpc_get_counting),
		lib_symbol_def(kpc_set_thread_counting),
		lib_symbol_def(kpc_get_thread_counting),
		lib_symbol_def(kpc_get_config_count),
		lib_symbol_def(kpc_get_counter_count),
		lib_symbol_def(kpc_set_config),
		lib_symbol_def(kpc_get_config),
		lib_symbol_def(kpc_get_cpu_counters),
		lib_symbol_def(kpc_get_thread_counters),
		lib_symbol_def(kpc_force_all_ctrs_set),
		lib_symbol_def(kpc_force_all_ctrs_get),
		lib_symbol_def(kperf_action_count_set),
		lib_symbol_def(kperf_action_count_get),
		lib_symbol_def(kperf_action_samplers_set),
		lib_symbol_def(kperf_action_samplers_get),
		lib_symbol_def(kperf_action_filter_set_by_task),
		lib_symbol_def(kperf_action_filter_set_by_pid),
		lib_symbol_def(kperf_timer_count_set),
		lib_symbol_def(kperf_timer_count_get),
		lib_symbol_def(kperf_timer_period_set),
		lib_symbol_def(kperf_timer_period_get),
		lib_symbol_def(kperf_timer_action_set),
		lib_symbol_def(kperf_timer_action_get),
		lib_symbol_def(kperf_sample_set),
		lib_symbol_def(kperf_sample_get),
		lib_symbol_def(kperf_reset),
		lib_symbol_def(kperf_timer_pet_set),
		lib_symbol_def(kperf_timer_pet_get),
		lib_symbol_def(kperf_ns_to_ticks),
		lib_symbol_def(kperf_ticks_to_ns),
		lib_symbol_def(kperf_tick_frequency),
	};

	static const std::array<lib_symbol, 23> lib_symbols_kperfdata = {
		lib_symbol_def(kpep_config_create),
		lib_symbol_def(kpep_config_free),
		lib_symbol_def(kpep_config_add_event),
		lib_symbol_def(kpep_config_remove_event),
		lib_symbol_def(kpep_config_force_counters),
		lib_symbol_def(kpep_config_events_count),
		lib_symbol_def(kpep_config_events),
		lib_symbol_def(kpep_config_kpc),
		lib_symbol_def(kpep_config_kpc_count),
		lib_symbol_def(kpep_config_kpc_classes),
		lib_symbol_def(kpep_config_kpc_map),
		lib_symbol_def(kpep_db_create),
		lib_symbol_def(kpep_db_free),
		lib_symbol_def(kpep_db_name),
		lib_symbol_def(kpep_db_aliases_count),
		lib_symbol_def(kpep_db_aliases),
		lib_symbol_def(kpep_db_counters_count),
		lib_symbol_def(kpep_db_events_count),
		lib_symbol_def(kpep_db_events),
		lib_symbol_def(kpep_db_event),
		lib_symbol_def(kpep_event_name),
		lib_symbol_def(kpep_event_alias),
		lib_symbol_def(kpep_event_description),
	};

	static constexpr const char lib_path_kperf[]{ "/System/Library/PrivateFrameworks/kperf.framework/kperf" };
	static constexpr const char lib_path_kperfdata[]{ "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata" };

	static bool lib_inited	= false;
	static bool lib_has_err = false;
	static char lib_err_msg[256];

	static void* lib_handle_kperf	  = nullptr;
	static void* lib_handle_kperfdata = nullptr;

	BNCH_SWT_HOST static void lib_deinit() {
		lib_inited	= false;
		lib_has_err = false;
		if (lib_handle_kperf)
			dlclose(lib_handle_kperf);
		if (lib_handle_kperfdata)
			dlclose(lib_handle_kperfdata);
		lib_handle_kperf	 = nullptr;
		lib_handle_kperfdata = nullptr;
		for (size_t i = 0; i < lib_nelems(lib_symbols_kperf); i++) {
			const lib_symbol* symbol = &lib_symbols_kperf[i];
			*symbol->impl			 = nullptr;
		}
		for (size_t i = 0; i < lib_nelems(lib_symbols_kperfdata); i++) {
			const lib_symbol* symbol = &lib_symbols_kperfdata[i];
			*symbol->impl			 = nullptr;
		}
	}

	BNCH_SWT_HOST static bool lib_init() {
		auto return_err = [] {
			lib_deinit();
			lib_inited	= true;
			lib_has_err = true;
			return false;
		};

		if (lib_inited) {
			return !lib_has_err;
		}

		lib_handle_kperf = dlopen(lib_path_kperf, RTLD_LAZY);
		if (!lib_handle_kperf) {
			std::string error_string = "Failed to load kperf.framework, message: ";
			error_string += (dlerror() ? dlerror() : "unknown error");
			error_string += ".";
			std::strncpy(lib_err_msg, error_string.c_str(), sizeof(lib_err_msg) - 1);
			return return_err();
		}
		lib_handle_kperfdata = dlopen(lib_path_kperfdata, RTLD_LAZY);
		if (!lib_handle_kperfdata) {
			std::string error_string = "Failed to load kperfdata.framework, message: ";
			error_string += (dlerror() ? dlerror() : "unknown error");
			error_string += ".";
			std::strncpy(lib_err_msg, error_string.c_str(), sizeof(lib_err_msg) - 1);
			return return_err();
		}

		for (size_t i = 0; i < lib_nelems(lib_symbols_kperf); i++) {
			const lib_symbol* symbol = &lib_symbols_kperf[i];
			*symbol->impl			 = dlsym(lib_handle_kperf, symbol->name);
			if (!*symbol->impl) {
				std::string error_string = "Failed to load kperf function: ";
				error_string += symbol->name;
				error_string += ".";
				std::strncpy(lib_err_msg, error_string.c_str(), sizeof(lib_err_msg) - 1);
				return return_err();
			}
		}
		for (size_t i = 0; i < lib_nelems(lib_symbols_kperfdata); i++) {
			const lib_symbol* symbol = &lib_symbols_kperfdata[i];
			*symbol->impl			 = dlsym(lib_handle_kperfdata, symbol->name);
			if (!*symbol->impl) {
				std::string error_string = "Failed to load kperfdata function: ";
				error_string += symbol->name;
				error_string += ".";
				std::strncpy(lib_err_msg, error_string.c_str(), sizeof(lib_err_msg) - 1);
				return return_err();
			}
		}

		lib_inited	= true;
		lib_has_err = false;
		return true;
	}

	#if defined(__arm64__)
	using kd_buf_argtype = uint64_t;
	#else
	using kd_buf_argtype = uintptr_t;
	#endif

	struct kd_buf {
		uint64_t timestamp;
		kd_buf_argtype arg1;
		kd_buf_argtype arg2;
		kd_buf_argtype arg3;
		kd_buf_argtype arg4;
		kd_buf_argtype arg5;
		uint32_t debugid;

	#if defined(__LP64__) || defined(__arm64__)
		uint32_t cpuid;
		kd_buf_argtype unused;
	#endif
	};

	static constexpr uint64_t kdbg_classtype  = 0x10000;
	static constexpr uint64_t kdbg_subclstype = 0x20000;
	static constexpr uint64_t kdbg_rangetype  = 0x40000;
	static constexpr uint64_t kdbg_typenone	  = 0x80000;
	static constexpr uint64_t kdbg_cktypes	  = 0xF0000;
	static constexpr uint64_t kdbg_valcheck	  = 0x00200000U;

	struct kd_regtype {
		uint32_t type;
		uint32_t value1;
		uint32_t value2;
		uint32_t value3;
		uint32_t value4;
	};

	struct kbufinfo_t {
		int32_t nkdbufs;
		int32_t nolog;
		uint32_t flags;
		int32_t nkdthreads;
		int32_t bufid;
	};

	BNCH_SWT_HOST static int32_t kdebug_reset() {
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDREMOVE };
		return sysctl(mib, 3, nullptr, nullptr, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_reinit() {
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDSETUP };
		return sysctl(mib, 3, nullptr, nullptr, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_setreg(kd_regtype* kdr) {
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDSETREG };
		size_t size	   = sizeof(kd_regtype);
		return sysctl(mib, 3, kdr, &size, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_trace_setbuf(int32_t nbufs) {
		int32_t mib[4] = { CTL_KERN, KERN_KDEBUG, KERN_KDSETBUF, nbufs };
		return sysctl(mib, 4, nullptr, nullptr, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_trace_enable(bool enable) {
		int32_t mib[4] = { CTL_KERN, KERN_KDEBUG, KERN_KDENABLE, enable };
		return sysctl(mib, 4, nullptr, nullptr, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_get_bufinfo(kbufinfo_t* info) {
		if (!info) {
			return -1;
		}
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDGETBUF };
		size_t needed  = sizeof(kbufinfo_t);
		return sysctl(mib, 3, info, &needed, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kdebug_trace_read(void* buf, size_t len, size_t* count) {
		if (count) {
			*count = 0;
		}
		if (!buf || !len) {
			return -1;
		}

		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDREADTR };
		int32_t ret	   = sysctl(mib, 3, buf, &len, nullptr, 0);
		if (ret != 0)
			return ret;
		*count = len;
		return 0;
	}

	BNCH_SWT_HOST static int32_t kdebug_wait(size_t timeout_ms, bool* suc) {
		if (timeout_ms == 0) {
			return -1;
		}
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDBUFWAIT };
		size_t val	   = timeout_ms;
		int32_t ret	   = sysctl(mib, 3, nullptr, &val, nullptr, 0);
		if (suc) {
			*suc = !!val;
		}
		return ret;
	}

	static constexpr size_t event_name_max = 8;

	struct event_alias {
		const char* alias;
		std::array<const char*, event_name_max> names;
	};

	static constexpr std::array<event_alias, 4> profile_events = { {
		{ "cycles", { "FIXED_CYCLES", "CPU_CLK_UNHALTED.THREAD", "CPU_CLK_UNHALTED.CORE" } },
		{ "instructions", { "FIXED_INSTRUCTIONS", "INST_RETIRED.ANY" } },
		{ "branches", { "INST_BRANCH", "BR_INST_RETIRED.ALL_BRANCHES", "INST_RETIRED.ANY" } },
		{ "branch-misses", { "BRANCH_MISPRED_NONSPEC", "BRANCH_MISPREDICT", "BR_MISP_RETIRED.ALL_BRANCHES", "BR_INST_RETIRED.MISPRED" } },
	} };

	BNCH_SWT_HOST static kpep_event* get_event(kpep_db* db, const event_alias* alias) {
		for (size_t j = 0; j < event_name_max; j++) {
			const char* name = alias->names[j];
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

	static std::array<kpc_config_t, kpc_max_counters> regs	 = { 0 };
	static std::array<size_t, kpc_max_counters> counter_map	 = { 0 };
	static std::array<uint64_t, kpc_max_counters> counters_0 = { 0 };
	static const size_t ev_count							 = sizeof(profile_events) / sizeof(profile_events[0]);

	BNCH_SWT_HOST static bool setup_performance_counters() {
		static bool init   = false;
		static bool worked = false;

		if (init) {
			return worked;
		}
		init = true;

		if (!lib_init()) {
			std::cerr << "Error: " << lib_err_msg << std::endl;
			return (worked = false);
		}

		int32_t force_ctrs = 0;
		if (kpc_force_all_ctrs_get(&force_ctrs)) {
			std::cerr << "Error: Failed call to kpc_force_all_ctrs_get()" << std::endl;
			return (worked = false);
		}
		int32_t ret;
		kpep_db* db = nullptr;
		if ((ret = kpep_db_create(nullptr, &db))) {
			std::cerr << "Error: cannot load pmc database: " << ret << "." << std::endl;
			return (worked = false);
		}
		std::cout << "loaded db: " << db->name << " (" << db->marketing_name << ")" << std::endl;

		kpep_config* cfg = nullptr;
		if ((ret = kpep_config_create(db, &cfg))) {
			std::cerr << "Failed to create kpep config: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_force_counters(cfg))) {
			std::cerr << "Failed to force counters: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}

		std::array<kpep_event*, ev_count> ev_arr = { nullptr };
		for (size_t i = 0; i < ev_count; i++) {
			const event_alias* alias = &profile_events[i];
			ev_arr[i]				 = get_event(db, alias);
			if (!ev_arr[i]) {
				std::cerr << "Cannot find event: " << alias->alias << "." << std::endl;
				return (worked = false);
			}
		}

		for (size_t i = 0; i < ev_count; i++) {
			kpep_event* ev = ev_arr[i];
			if ((ret = kpep_config_add_event(cfg, &ev, 0, nullptr))) {
				std::cerr << "Failed to add event: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
				return (worked = false);
			}
		}

		uint32_t classes = 0;
		size_t reg_count = 0;
		if ((ret = kpep_config_kpc_classes(cfg, &classes))) {
			std::cerr << "Failed get kpc classes: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc_count(cfg, &reg_count))) {
			std::cerr << "Failed get kpc count: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc_map(cfg, counter_map.data(), sizeof(counter_map)))) {
			std::cerr << "Failed get kpc map: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc(cfg, regs.data(), sizeof(regs)))) {
			std::cerr << "Failed get kpc registers: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}

		if ((ret = kpc_force_all_ctrs_set(1))) {
			std::cerr << "Failed force all ctrs: " << ret << "." << std::endl;
			return (worked = false);
		}
		if ((classes & kpc_class_configurable_mask) && reg_count) {
			if ((ret = kpc_set_config(classes, regs.data()))) {
				std::cerr << "Failed set kpc config: " << ret << "." << std::endl;
				return (worked = false);
			}
		}

		if ((ret = kpc_set_counting(classes))) {
			std::cerr << "Failed set counting: " << ret << "." << std::endl;
			return (worked = false);
		}
		if ((ret = kpc_set_thread_counting(classes))) {
			std::cerr << "Failed set thread counting: " << ret << "." << std::endl;
			return (worked = false);
		}

		return (worked = true);
	}

	BNCH_SWT_HOST static performance_counters get_counters() {
		static bool warned = false;
		int32_t ret;
		if ((ret = kpc_get_thread_counters(0, kpc_max_counters, counters_0.data()))) {
			if (!warned) {
				std::cerr << "Failed get thread counters before: " << ret << "." << std::endl;
				warned = true;
			}
			return {};
		}
		return performance_counters{ static_cast<double>(counters_0[counter_map[0]]), static_cast<double>(counters_0[counter_map[2]]),
			static_cast<double>(counters_0[counter_map[3]]), static_cast<double>(counters_0[counter_map[1]]) };
	}

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cpu, count> : public std::vector<event_count> {
		performance_counters diff{};
		size_t current_index{};
		bool has_events_val{};
		BNCH_SWT_HOST event_collector_type() : std::vector<event_count>{ count }, diff{}, current_index{}, has_events_val{ setup_performance_counters() } {}

		BNCH_SWT_HOST bool has_events() {
			return has_events_val;
		}

		template<typename function_type, typename... arg_types> BNCH_SWT_HOST void run(arg_types&&... args) {
			if (has_events()) {
				diff = get_counters();
			}
			const auto start_clock = clock_type::now();
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(static_cast<size_t>(function_type::impl(std::forward<arg_types>(args)...)));
			const auto end_clock = clock_type::now();
			if (has_events()) {
				performance_counters end = get_counters();
				diff					 = end - diff;
				std::vector<event_count>::operator[](current_index).cycles_val.emplace(diff.cycles);
				std::vector<event_count>::operator[](current_index).instructions_val.emplace(diff.instructions);
				std::vector<event_count>::operator[](current_index).branches_val.emplace(diff.branches);
				std::vector<event_count>::operator[](current_index).branch_misses_val.emplace(diff.branch_misses);
			}
			std::vector<event_count>::operator[](current_index).elapsed_ns_val.emplace(end_clock - start_clock);
			++current_index;
			return;
		}
	};

}

#endif
