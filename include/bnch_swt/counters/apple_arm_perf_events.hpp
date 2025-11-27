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
	#include <array>

namespace bnch_swt::internal {

	struct performance_counters {
		double branch_misses{};
		double instructions{};
		double branches{};
		double cycles{};
		BNCH_SWT_HOST performance_counters(double c, double b, double m, double i) : branch_misses(m), instructions(i), branches(b), cycles(c) {
		}
		BNCH_SWT_HOST performance_counters(double init = 0.0) : branch_misses(init), instructions(init), branches(init), cycles(init) {
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

	#define KPC_CLASS_FIXED (0)
	#define KPC_CLASS_CONFIGURABLE (1)
	#define KPC_CLASS_POWER (2)
	#define KPC_CLASS_RAWPMU (3)

	#define KPC_CLASS_FIXED_MASK (1u << KPC_CLASS_FIXED)
	#define KPC_CLASS_CONFIGURABLE_MASK (1u << KPC_CLASS_CONFIGURABLE)
	#define KPC_CLASS_POWER_MASK (1u << KPC_CLASS_POWER)
	#define KPC_CLASS_RAWPMU_MASK (1u << KPC_CLASS_RAWPMU)

	#define KPC_PMU_ERROR (0)
	#define KPC_PMU_INTEL_V3 (1)
	#define KPC_PMU_ARM_APPLE (2)
	#define KPC_PMU_INTEL_V2 (3)
	#define KPC_PMU_ARM_V2 (4)

	#define KPC_MAX_COUNTERS 32

	#define KPERF_SAMPLER_TH_INFO (1U << 0)
	#define KPERF_SAMPLER_TH_SNAPSHOT (1U << 1)
	#define KPERF_SAMPLER_KSTACK (1U << 2)
	#define KPERF_SAMPLER_USTACK (1U << 3)
	#define KPERF_SAMPLER_PMC_THREAD (1U << 4)
	#define KPERF_SAMPLER_PMC_CPU (1U << 5)
	#define KPERF_SAMPLER_PMC_CONFIG (1U << 6)
	#define KPERF_SAMPLER_MEMINFO (1U << 7)
	#define KPERF_SAMPLER_TH_SCHEDULING (1U << 8)
	#define KPERF_SAMPLER_TH_DISPATCH (1U << 9)
	#define KPERF_SAMPLER_TK_SNAPSHOT (1U << 10)
	#define KPERF_SAMPLER_SYS_MEM (1U << 11)
	#define KPERF_SAMPLER_TH_INSCYC (1U << 12)
	#define KPERF_SAMPLER_TK_INFO (1U << 13)

	#define KPERF_ACTION_MAX (32)

	#define KPERF_TIMER_MAX (8)

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

	BNCH_SWT_HOST static int32_t kperf_lightweight_pet_get(uint32_t* enabled) {
		if (!enabled) {
			return -1;
		}
		size_t size = 4;
		return sysctlbyname("kperf.lightweight_pet", enabled, &size, nullptr, 0);
	}

	BNCH_SWT_HOST static int32_t kperf_lightweight_pet_set(uint32_t enabled) {
		return sysctlbyname("kperf.lightweight_pet", nullptr, nullptr, &enabled, 4);
	}

	#define KPEP_ARCH_I386 0
	#define KPEP_ARCH_X86_64 1
	#define KPEP_ARCH_ARM 2
	#define KPEP_ARCH_ARM64 3

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
		KPEP_CONFIG_ERROR_NONE					 = 0,
		KPEP_CONFIG_ERROR_INVALID_ARGUMENT		 = 1,
		KPEP_CONFIG_ERROR_OUT_OF_MEMORY			 = 2,
		KPEP_CONFIG_ERROR_IO					 = 3,
		KPEP_CONFIG_ERROR_BUFFER_TOO_SMALL		 = 4,
		KPEP_CONFIG_ERROR_CUR_SYSTEM_UNKNOWN	 = 5,
		KPEP_CONFIG_ERROR_DB_PATH_INVALID		 = 6,
		KPEP_CONFIG_ERROR_DB_NOT_FOUND			 = 7,
		KPEP_CONFIG_ERROR_DB_ARCH_UNSUPPORTED	 = 8,
		KPEP_CONFIG_ERROR_DB_VERSION_UNSUPPORTED = 9,
		KPEP_CONFIG_ERROR_DB_CORRUPT			 = 10,
		KPEP_CONFIG_ERROR_EVENT_NOT_FOUND		 = 11,
		KPEP_CONFIG_ERROR_CONFLICTING_EVENTS	 = 12,
		KPEP_CONFIG_ERROR_COUNTERS_NOT_FORCED	 = 13,
		KPEP_CONFIG_ERROR_EVENT_UNAVAILABLE		 = 14,
		KPEP_CONFIG_ERROR_ERRNO					 = 15,
		KPEP_CONFIG_ERROR_MAX
	};

	static std::array<const char*, static_cast<uint64_t>(kpep_config_error_code::KPEP_CONFIG_ERROR_MAX)> kpep_config_error_names = { "none", "invalid argument", "out of memory",
		"I/O", "buffer too small", "current system unknown", "database path invalid", "database not found", "database architecture unsupported", "database version unsupported",
		"database corrupt", "event not found", "conflicting events", "all counters must be forced", "event unavailable", "check errno" };

	static const char* kpep_config_error_desc(int32_t code) {
		if (0 <= code && static_cast<uint64_t>(code) < static_cast<uint64_t>(kpep_config_error_code::KPEP_CONFIG_ERROR_MAX)) {
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

	// -----------------------------------------------------------------------------
	// load kperf/kperfdata dynamic library
	// -----------------------------------------------------------------------------

	struct lib_symbol {
		const char* name;
		void** impl;
	};

	#define lib_nelems(x) (sizeof(x) / sizeof((x)[0]))
	#define lib_symbol_def(name) \
		lib_symbol { \
			#name, reinterpret_cast<void**>(&name) \
		}

	static const std::array<lib_symbol, 34> lib_symbols_kperf{
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

	static const std::array<lib_symbol, 23> lib_symbols_kperfdata{
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

	#define lib_path_kperf "/System/Library/PrivateFrameworks/kperf.framework/kperf"
	#define lib_path_kperfdata "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata"

	static bool lib_inited	= false;
	static bool lib_has_err = false;
	static char lib_err_msg[256];

	static void* lib_handle_kperf	  = nullptr;
	static void* lib_handle_kperfdata = nullptr;

	static void lib_deinit() {
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

	static bool lib_init() {
	#define return_err() \
		do { \
			lib_deinit(); \
			lib_inited	= true; \
			lib_has_err = true; \
			return false; \
		} while (false)

		if (lib_inited)
			return !lib_has_err;

		// load dynamic library
		lib_handle_kperf = dlopen(lib_path_kperf, RTLD_LAZY);
		if (!lib_handle_kperf) {
			std::string error_string{ "Failed to load kperf.framework, message: " };
			error_string += (dlerror() ? dlerror() : "unknown error");
			error_string += ".";
			std::cout << error_string << std::endl;
			return_err();
		}
		lib_handle_kperfdata = dlopen(lib_path_kperfdata, RTLD_LAZY);
		if (!lib_handle_kperfdata) {
			std::string error_string{ "Failed to load kperfdata.framework, message: " };
			error_string += (dlerror() ? dlerror() : "unknown error");
			error_string += ".";
			std::cout << error_string << std::endl;
			return_err();
		}

		// load symbol address from dynamic library
		for (size_t i = 0; i < lib_nelems(lib_symbols_kperf); i++) {
			const lib_symbol* symbol = &lib_symbols_kperf[i];
			*symbol->impl			 = dlsym(lib_handle_kperf, symbol->name);
			if (!*symbol->impl) {
				std::string error_string{ "Failed to load kper function, message: " };
				error_string += (dlerror() ? dlerror() : "unknown error");
				error_string += ".";
				std::cout << error_string << std::endl;
				return_err();
			}
		}
		for (size_t i = 0; i < lib_nelems(lib_symbols_kperfdata); i++) {
			const lib_symbol* symbol = &lib_symbols_kperfdata[i];
			*symbol->impl			 = dlsym(lib_handle_kperfdata, symbol->name);
			if (!*symbol->impl) {
				std::string error_string{ "Failed to load kperfdata function, message: " };
				error_string += (dlerror() ? dlerror() : "unknown error");
				error_string += ".";
				std::cout << error_string << std::endl;
				return_err();
			}
		}

		lib_inited	= true;
		lib_has_err = false;
		return true;

	#undef return_err
	}

	// -----------------------------------------------------------------------------
	// kdebug protected structs
	// https://github.com/apple/darwin-xnu/blob/main/bsd/sys_private/kdebug_private.h
	// -----------------------------------------------------------------------------

	/*
 * Ensure that both LP32 and LP64 variants of arm64 use the same kd_buf
 * structure.
 */
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
		kd_buf_argtype arg5; /* the thread ID */
		uint32_t debugid; /* see <sys/kdebug.h> */

	/*
 * Ensure that both LP32 and LP64 variants of arm64 use the same kd_buf
 * structure.
 */
	#if defined(__LP64__) || defined(__arm64__)
		uint32_t cpuid; /* cpu index, from 0 */
		kd_buf_argtype unused;
	#endif
	};

	/* bits for the type field of kd_regtype */
	#define KDBG_CLASSTYPE 0x10000
	#define KDBG_SUBCLSTYPE 0x20000
	#define KDBG_RANGETYPE 0x40000
	#define KDBG_TYPENONE 0x80000
	#define KDBG_CKTYPES 0xF0000

	/* only trace at most 4 types of events, at the code granularity */
	#define KDBG_VALCHECK 0x00200000U

	struct kd_regtype {
		uint32_t type;
		uint32_t value1;
		uint32_t value2;
		uint32_t value3;
		uint32_t value4;
	};

	struct kbufinfo_t {
		/* number of events that can fit in the buffers */
		int32_t nkdbufs;
		/* set if trace is disabled */
		int32_t nolog;
		/* kd_ctrl_page.flags */
		uint32_t flags;
		/* number of threads in thread map */
		int32_t nkdthreads;
		/* the owning pid */
		int32_t bufid;
	};

	// -----------------------------------------------------------------------------
	// kdebug utils
	// -----------------------------------------------------------------------------

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

		// Note: the input and output units are not the same.
		// input: bytes
		// output: number of kd_buf
		int32_t mib[3] = { CTL_KERN, KERN_KDEBUG, KERN_KDREADTR };
		int32_t ret	   = sysctl(mib, 3, buf, &len, nullptr, 0);
		if (ret != 0) {
			return ret;
		}
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

	// -----------------------------------------------------------------------------
	// Demo
	// -----------------------------------------------------------------------------

	#define EVENT_NAME_MAX 8
	struct event_alias {
		const char* alias;
		std::array<const char*, EVENT_NAME_MAX> names;
	};

	static constexpr std::array<event_alias, 4> profile_events{ [] {
		std::array<event_alias, 4> return_value{};
		return_value[0].alias = "cycles";
		return_value[0].names = { "FIXED_CYCLES", "CPU_CLK_UNHALTED.THREAD", "CPU_CLK_UNHALTED.CORE" };
		return return_value;
	}() };
	/*
		{ { "cycles",
			{
				"FIXED_CYCLES",// Apple A7-A15//CORE_ACTIVE_CYCLE
				"CPU_CLK_UNHALTED.THREAD",// Intel Core 1th-10th
				"CPU_CLK_UNHALTED.CORE",// Intel Yonah, Merom
			} } },
		{ { "instructions",
			{
				"FIXED_INSTRUCTIONS",// Apple A7-A15
				"INST_RETIRED.ANY"// Intel Yonah, Merom, Core 1th-10th
			} } },
		{ { "branches",
			{
				"INST_BRANCH",// Apple A7-A15
				"BR_INST_RETIRED.ALL_BRANCHES",// Intel Core 1th-10th
				"INST_RETIRED.ANY",// Intel Yonah, Merom
			} } },
		{ { "branch-misses",
			{
				"BRANCH_MISPRED_NONSPEC",// Apple A7-A15, since iOS 15, macOS 12
				"BRANCH_MISPREDICT",// Apple A7-A14
				"BR_MISP_RETIRED.ALL_BRANCHES",// Intel Core 2th-10th
				"BR_INST_RETIRED.MISPRED",// Intel Yonah, Merom
			} } },
	};
	*/
	static kpep_event* get_event(kpep_db* db, const event_alias* alias) {
		for (size_t j = 0; j < EVENT_NAME_MAX; j++) {
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

	static std::array<kpc_config_t, KPC_MAX_COUNTERS> regs{ 0 };
	static std::array<size_t, KPC_MAX_COUNTERS> counter_map{ 0 };
	static std::array<uint64_t, KPC_MAX_COUNTERS> counters_0{ 0 };
	static std::array<uint64_t, KPC_MAX_COUNTERS> counters_1{ 0 };
	static const size_t ev_count = sizeof(profile_events) / sizeof(profile_events[0]);

	static bool setup_performance_counters() {
		static bool init   = false;
		static bool worked = false;

		if (init) {
			return worked;
		}
		init = true;

		// load dylib
		if (!lib_init()) {
			std::cout << "Error: " << lib_err_msg << std::endl;
			return (worked = false);
		}

		// check permission
		int32_t force_ctrs = 0;
		if (kpc_force_all_ctrs_get(&force_ctrs)) {
			std::cout << "Permission denied, xnu/kpc requires root privileges." << std::endl;
			return (worked = false);
		}

		int32_t ret;
		// load pmc db
		kpep_db* db = nullptr;
		if ((ret = kpep_db_create(nullptr, &db))) {
			std::cout << "Error: cannot load pmc database: " << ret << "." << std::endl;
			return (worked = false);
		}
		std::cout << "loaded db: " << db->name << " (" << db->marketing_name << ")" << std::endl;

		// create a config
		kpep_config* cfg = nullptr;
		if ((ret = kpep_config_create(db, &cfg))) {
			std::cout << "Failed to create kpep config: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_force_counters(cfg))) {
			std::cout << "Failed to force counters: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}

		// get events
		std::array<kpep_event*, ev_count> ev_arr{ nullptr };
		for (size_t i = 0; i < ev_count; i++) {
			const event_alias* alias = &profile_events[i];
			ev_arr[i]				 = get_event(db, alias);
			if (!ev_arr[i]) {
				std::cout << "Cannot find event: " << alias->alias << "." << std::endl;
				return (worked = false);
			}
		}

		// add event to config
		for (size_t i = 0; i < ev_count; i++) {
			kpep_event* ev = ev_arr[i];
			if ((ret = kpep_config_add_event(cfg, &ev, 0, nullptr))) {
				std::cout << "Failed to add event: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
				return (worked = false);
			}
		}

		// prepare buffer and config
		uint32_t classes = 0;
		size_t reg_count = 0;
		if ((ret = kpep_config_kpc_classes(cfg, &classes))) {
			std::cout << "Failed to get kpc classes: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc_count(cfg, &reg_count))) {
			std::cout << "Failed to get kpc count: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc_map(cfg, counter_map.data(), sizeof(counter_map)))) {
			std::cout << "Failed to get kpc map: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}
		if ((ret = kpep_config_kpc(cfg, regs.data(), sizeof(regs)))) {
			std::cout << "Failed to get kpc registers: " << ret << " (" << kpep_config_error_desc(ret) << ")." << std::endl;
			return (worked = false);
		}

		// set config to kernel
		if ((ret = kpc_force_all_ctrs_set(1))) {
			std::cout << "Failed to force all ctrs: " << ret << "." << std::endl;
			return (worked = false);
		}
		if ((classes & KPC_CLASS_CONFIGURABLE_MASK) && reg_count) {
			if ((ret = kpc_set_config(classes, regs.data()))) {
				std::cout << "Failed to set kpc config: " << ret << "." << std::endl;
				return (worked = false);
			}
		}

		// run counting
		if ((ret = kpc_set_counting(classes))) {
			std::cout << "Failed to set counting: " << ret << "." << std::endl;
			return (worked = false);
		}
		if ((ret = kpc_set_thread_counting(classes))) {
			std::cout << "Failed to set thread counting: " << ret << "." << std::endl;
			return (worked = false);
		}

		return (worked = true);
	}

	BNCH_SWT_HOST performance_counters get_counters() {
		static bool warned = false;
		int32_t ret;
		// get counters before
		if ((ret = kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters_0.data()))) {
			if (!warned) {
				std::cout << "Failed to get thread counters before: " << ret << "." << std::endl;
				warned = true;
			}
			return 1;
		}
		return performance_counters{ static_cast<double>(counters_0[counter_map[0]]), static_cast<double>(counters_0[counter_map[2]]),
			static_cast<double>(counters_0[counter_map[3]]), static_cast<double>(counters_0[counter_map[1]]) };
	}

	template<typename event_count, uint64_t count> struct event_collector_type<event_count, benchmark_types::cpu, count> : public std::vector<event_count> {
		performance_counters diff{};
		size_t current_index{};
		bool has_events_val{};
		BNCH_SWT_HOST event_collector_type() : std::vector<event_count>{ count }, diff(0), current_index{}, has_events_val{ setup_performance_counters() } {
		}
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
			std::vector<event_count>::operator[](current_index).elapsed = end_clock - start_clock;
			++current_index;
			return;
		}
	};

}

#endif
