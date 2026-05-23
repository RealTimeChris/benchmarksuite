
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

#if BNCH_SWT_COMPILER_CLANG
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-libc-call"
	#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

#include <cstdint>
#include <cstdio>
#include <vector>

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <windows.h>
	#include <intrin.h>
#elif BNCH_SWT_PLATFORM_MAC
	#ifndef xnu_static_assert_struct_size
		#define xnu_static_assert_struct_size(...)
	#endif
	#ifndef xnu_static_assert_struct_size_kernel_user
		#define xnu_static_assert_struct_size_kernel_user(...)
	#endif

	#include <pthread.h>
	#include <sys/qos.h>
	#include <mach/mach.h>
	#include <mach/thread_policy.h>
	#include <mach/thread_act.h>
#elif BNCH_SWT_PLATFORM_LINUX
	#ifndef _GNU_SOURCE
		#define _GNU_SOURCE
	#endif
	#include <pthread.h>
	#include <sched.h>
	#include <sys/resource.h>
	#include <unistd.h>
	#include <cpuid.h>
	#include <errno.h>
#endif

namespace bnch_swt {

#if BNCH_SWT_PLATFORM_WINDOWS || (BNCH_SWT_PLATFORM_LINUX && BNCH_SWT_ARCH_X64)

	struct cpuid_regs {
		uint32_t eax, ebx, ecx, edx;
	};

	inline cpuid_regs cpuid_call(uint32_t leaf, uint32_t subleaf) noexcept {
		cpuid_regs r{};
	#if BNCH_SWT_PLATFORM_WINDOWS
		int regs[4];
		__cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
		r.eax = static_cast<uint32_t>(regs[0]);
		r.ebx = static_cast<uint32_t>(regs[1]);
		r.ecx = static_cast<uint32_t>(regs[2]);
		r.edx = static_cast<uint32_t>(regs[3]);
	#else
		__cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
	#endif
		return r;
	}

	inline bool is_intel_hybrid() noexcept {
		auto vendor		 = cpuid_call(0, 0);
		const bool intel = (vendor.ebx == 0x756e6547u) && (vendor.edx == 0x49656e69u) && (vendor.ecx == 0x6c65746eu);
		if (!intel || vendor.eax < 0x7)
			return false;
		auto feat = cpuid_call(0x7, 0);
		if (!((feat.edx >> 15) & 0x1u))
			return false;
		return cpuid_call(0, 0).eax >= 0x1A;
	}

	inline bool current_cpu_is_pcore() noexcept {
		return ((cpuid_call(0x1A, 0).eax >> 24) & 0xffu) == 0x40u;
	}

#endif

#if BNCH_SWT_PLATFORM_WINDOWS

	inline bool index_to_processor_number(DWORD index, PROCESSOR_NUMBER& out) noexcept {
		DWORD seen		  = 0;
		const WORD groups = GetActiveProcessorGroupCount();
		for (WORD g = 0; g < groups; ++g) {
			const DWORD in_group = GetActiveProcessorCount(g);
			if (index < seen + in_group) {
				out.Group	 = g;
				out.Number	 = static_cast<BYTE>(index - seen);
				out.Reserved = 0;
				return true;
			}
			seen += in_group;
		}
		return false;
	}

	inline int find_first_pcore() noexcept {
		if (!is_intel_hybrid())
			return -1;

		const DWORD total = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
		HANDLE self		  = GetCurrentThread();

		GROUP_AFFINITY original{};
		GetThreadGroupAffinity(self, &original);

		int found = -1;
		for (DWORD i = 0; i < total; ++i) {
			PROCESSOR_NUMBER pn{};
			if (!index_to_processor_number(i, pn))
				continue;

			GROUP_AFFINITY ga{};
			ga.Group = pn.Group;
			ga.Mask	 = static_cast<KAFFINITY>(1ull) << pn.Number;

			if (!SetThreadGroupAffinity(self, &ga, nullptr))
				continue;
			SwitchToThread();

			if (current_cpu_is_pcore()) {
				found = static_cast<int>(i);
				break;
			}
		}

		SetThreadGroupAffinity(self, &original, nullptr);
		return found;
	}

	inline bool pin_for_benchmark() noexcept {
		HANDLE self = GetCurrentThread();

		bool aff_ok		= false;
		const int pcore = find_first_pcore();
		if (pcore >= 0) {
			PROCESSOR_NUMBER pn{};
			if (index_to_processor_number(static_cast<DWORD>(pcore), pn)) {
				GROUP_AFFINITY ga{};
				ga.Group = pn.Group;
				ga.Mask	 = static_cast<KAFFINITY>(1ull) << pn.Number;
				aff_ok	 = SetThreadGroupAffinity(self, &ga, nullptr) != 0;
			}
		} else {
			PROCESSOR_NUMBER pn{};
			GetCurrentProcessorNumberEx(&pn);
			GROUP_AFFINITY ga{};
			ga.Group = pn.Group;
			ga.Mask	 = static_cast<KAFFINITY>(1ull) << pn.Number;
			aff_ok	 = SetThreadGroupAffinity(self, &ga, nullptr) != 0;
		}

		SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
		const bool prio_ok = SetThreadPriority(self, THREAD_PRIORITY_TIME_CRITICAL) != 0;
		SetThreadPriorityBoost(self, TRUE);

		if (!aff_ok)
			std::fprintf(stderr, "[bench] affinity pin failed\n");
		if (!prio_ok)
			std::fprintf(stderr, "[bench] priority raise failed\n");
		return aff_ok && prio_ok;
	}

#elif BNCH_SWT_PLATFORM_LINUX

	inline int find_first_pcore() noexcept {
	#if defined(__x86_64__) || defined(__i386__)
		if (!is_intel_hybrid())
			return -1;

		const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
		if (ncpu <= 0)
			return -1;

		pthread_t self = pthread_self();
		cpu_set_t original;
		CPU_ZERO(&original);
		pthread_getaffinity_np(self, sizeof(original), &original);

		int found = -1;
		for (long i = 0; i < ncpu; ++i) {
			cpu_set_t one;
			CPU_ZERO(&one);
			CPU_SET(static_cast<int>(i), &one);
			if (pthread_setaffinity_np(self, sizeof(one), &one) != 0)
				continue;
			sched_yield();
			if (current_cpu_is_pcore()) {
				found = static_cast<int>(i);
				break;
			}
		}

		pthread_setaffinity_np(self, sizeof(original), &original);
		return found;
	#else
		return -1;
	#endif
	}

	inline bool pin_for_benchmark() noexcept {
		pthread_t self = pthread_self();

		cpu_set_t target;
		CPU_ZERO(&target);
		const int pcore = find_first_pcore();
		if (pcore >= 0) {
			CPU_SET(pcore, &target);
		} else {
			CPU_SET(sched_getcpu(), &target);
		}
		const bool aff_ok = pthread_setaffinity_np(self, sizeof(target), &target) == 0;

		bool prio_ok = false;
		sched_param sp{};
		sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
		if (pthread_setschedparam(self, SCHED_FIFO, &sp) == 0) {
			prio_ok = true;
		} else {
			errno		 = 0;
			const int rc = setpriority(PRIO_PROCESS, 0, -20);
			prio_ok		 = (rc == 0 && errno == 0);
		}

		if (!aff_ok)
			std::fprintf(stderr, "[bench] affinity pin failed\n");
		if (!prio_ok)
			std::fprintf(stderr, "[bench] priority raise failed (need CAP_SYS_NICE or root for SCHED_FIFO)\n");
		return aff_ok && prio_ok;
	}

#elif BNCH_SWT_PLATFORM_MAC

	inline bool pin_for_benchmark() noexcept {
		const int qos_rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

		thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());

		thread_affinity_policy_data_t aff{ 1 };
		thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&aff), THREAD_AFFINITY_POLICY_COUNT);

		thread_precedence_policy_data_t prec{ 63 };
		thread_policy_set(mach_thread, THREAD_PRECEDENCE_POLICY, reinterpret_cast<thread_policy_t>(&prec), THREAD_PRECEDENCE_POLICY_COUNT);

		if (qos_rc != 0) {
			std::fprintf(stderr, "[bench] QoS set failed (rc=%d) - scheduler will not bias toward P-cores\n", qos_rc);
			return false;
		}
		return true;
	}

#else
	inline bool pin_for_benchmark() noexcept {
		std::fprintf(stderr, "[bench] unsupported platform - no pin performed\n");
		return false;
	}
#endif

}
#if BNCH_SWT_COMPILER_CLANG
	#pragma clang diagnostic pop
#endif
