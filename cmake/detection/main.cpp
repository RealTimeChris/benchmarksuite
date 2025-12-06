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
// cmake/detection/main.cpp

#if defined(BNCH_SWT_DETECT_GPU_PROPERTIES)
	#include <cuda_runtime.h>
	#include <iostream>
	#include <cstdint>

int main() {
	cudaDeviceProp deviceProp;
	cudaError_t result = cudaGetDeviceProperties(&deviceProp, 0);

	if (result != cudaSuccess) {
		std::cout << "CUDA_ERROR=1" << std::endl;
		return 1;
	}

	uint32_t gpu_arch_index = 0;
	if (deviceProp.major == 9) {
		gpu_arch_index = 1;
	} else if (deviceProp.major == 10) {
		gpu_arch_index = 2;
	} else if (deviceProp.major == 11) {
		gpu_arch_index = 3;
	} else if (deviceProp.major == 12) {
		gpu_arch_index = 4;
	} else {
		gpu_arch_index = 0;
	}

	std::cout << "SM_COUNT=" << deviceProp.multiProcessorCount << std::endl;
	std::cout << "ALIGNMENT=" << std::max({ deviceProp.textureAlignment, deviceProp.surfaceAlignment, ( size_t )256 }) << std::endl;
	std::cout << "MAX_THREADS_PER_SM=" << deviceProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "MAX_THREADS_PER_BLOCK=" << deviceProp.maxThreadsPerBlock << std::endl;
	std::cout << "WARP_SIZE=" << deviceProp.warpSize << std::endl;
	std::cout << "L2_CACHE_SIZE=" << deviceProp.l2CacheSize << std::endl;
	std::cout << "SHARED_MEM_PER_BLOCK=" << deviceProp.sharedMemPerBlock << std::endl;
	std::cout << "MEMORY_BUS_WIDTH=" << deviceProp.memoryBusWidth << std::endl;
	int clock_rate;
	cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0);
	std::cout << "MEMORY_CLOCK_RATE=" << clock_rate << std::endl;
	std::cout << "MAJOR_COMPUTE_CAPABILITY=" << deviceProp.major << std::endl;
	std::cout << "MINOR_COMPUTE_CAPABILITY=" << deviceProp.minor << std::endl;
	std::cout << "MAX_GRID_SIZE_X=" << deviceProp.maxGridSize[0] << std::endl;
	std::cout << "MAX_GRID_SIZE_Y=" << deviceProp.maxGridSize[1] << std::endl;
	std::cout << "MAX_GRID_SIZE_Z=" << deviceProp.maxGridSize[2] << std::endl;
	std::cout << "MAX_BLOCK_SIZE_X=" << deviceProp.maxThreadsPerBlock << std::endl;
	std::cout << "MAX_PERSISTING_L2_BYTES=" << deviceProp.persistingL2CacheMaxSize << std::endl;
	std::cout << "GPU_ARCH_INDEX=" << gpu_arch_index << std::endl;
	std::cout << "HAS_CUDA_9=" << (deviceProp.major == 9) << std::endl;
	std::cout << "HAS_CUDA_10=" << (deviceProp.major == 10) << std::endl;
	std::cout << "HAS_CUDA_11=" << (deviceProp.major == 11) << std::endl;
	std::cout << "HAS_CUDA_12=" << (deviceProp.major == 12) << std::endl;
	std::cout << "GPU_SUCCESS=1" << std::endl;

	return 0;
}
#elif defined(BNCH_SWT_DETECT_CPU_PROPERTIES)
	#include <cstring>
	#include <cstdint>
	#include <cstdlib>
	#include <iostream>
	#include <thread>
	#include <vector>

	#if defined(_MSC_VER)
		#include <intrin.h>
	#elif defined(__GNUC__) || defined(__clang__)
		#include <cpuid.h>
	#endif

	#if defined(_WIN32) || defined(_WIN64)
		#include <Windows.h>
	#endif

	#if defined(__linux__) || defined(__ANDROID__)
		#include <fstream>
		#include <string>
	#endif

	#if defined(__APPLE__) && defined(__MACH__)
		#include <sys/sysctl.h>
		#include <sys/types.h>
		#include <string>
	#endif

	#if defined(__aarch64__) || defined(_M_ARM64)
		#if defined(__linux__)
			#include <sys/auxv.h>
			#include <asm/hwcap.h>
		#elif defined(__APPLE__)
			#include <sys/sysctl.h>
		#endif
	#endif

enum class instruction_set {
	FALLBACK = 0x0,
	AVX2	 = 0x1,
	AVX512f	 = 0x2,
	NEON	 = 0x4,
	SVE2	 = 0x8,
};

enum class cache_level {
	one	  = 1,
	two	  = 2,
	three = 3,
};

	#if defined(__aarch64__) || defined(_M_ARM64)
inline static uint32_t detect_supported_architectures() {
	uint32_t host_isa = static_cast<uint32_t>(instruction_set::NEON);

		#if defined(__linux__) && defined(HWCAP_SVE)
	unsigned long hwcap = getauxval(AT_HWCAP);
	if (hwcap & HWCAP_SVE) {
		host_isa |= static_cast<uint32_t>(instruction_set::SVE2);
	}
		#endif

	return host_isa;
}

	#elif defined(__x86_64__) || defined(_M_X64)
static constexpr uint32_t cpuid_avx2_bit	 = 1ul << 5;
static constexpr uint32_t cpuid_avx512_bit	 = 1ul << 16;
static constexpr uint64_t cpuid_avx256_saved = 1ULL << 2;
static constexpr uint64_t cpuid_avx512_saved = 7ULL << 5;
static constexpr uint32_t cpuid_osx_save	 = (1ul << 26) | (1ul << 27);

inline static void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
		#if defined(_MSC_VER)
	int32_t cpu_info[4];
	__cpuidex(cpu_info, *eax, *ecx);
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
		#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "c"(c));
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
		#endif
}

inline static uint64_t xgetbv() {
		#if defined(_MSC_VER)
	return _xgetbv(0);
		#else
	uint32_t eax, edx;
	asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (( uint64_t )edx << 32) | eax;
		#endif
}

inline static uint32_t detect_supported_architectures() {
	std::uint32_t eax	   = 0;
	std::uint32_t ebx	   = 0;
	std::uint32_t ecx	   = 0;
	std::uint32_t edx	   = 0;
	std::uint32_t host_isa = static_cast<uint32_t>(instruction_set::FALLBACK);

	eax = 0x1;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if ((ecx & cpuid_osx_save) != cpuid_osx_save) {
		return host_isa;
	}

	uint64_t xcr0 = xgetbv();
	if ((xcr0 & cpuid_avx256_saved) == 0) {
		return host_isa;
	}

	eax = 0x7;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if (ebx & cpuid_avx2_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
	}

	if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
		return host_isa;
	}

	if (ebx & cpuid_avx512_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX512f);
	}

	return host_isa;
}

	#else
inline static uint32_t detect_supported_architectures() {
	return static_cast<uint32_t>(instruction_set::FALLBACK);
}
	#endif

inline uint64_t get_cache_size(cache_level level) {
	#if defined(_WIN32) || defined(_WIN64)
	DWORD bufferSize = 0;
	std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};
	GetLogicalProcessorInformation(nullptr, &bufferSize);
	if (bufferSize == 0)
		return 0;
	buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

	if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
		return 0;
	}

	for (const auto& i: buffer) {
		if (i.Relationship == RelationCache && i.Cache.Level == static_cast<int32_t>(level)) {
			if (level == cache_level::one && i.Cache.Type == CacheData) {
				return i.Cache.Size;
			} else if (level != cache_level::one && i.Cache.Type == CacheUnified) {
				return i.Cache.Size;
			}
		}
	}
	return 0;

	#elif defined(__linux__) || defined(__ANDROID__)
	auto get_cache_size_from_file = [](const std::string& index) -> uint64_t {
		const std::string cacheFilePath = "/sys/devices/system/cpu/cpu0/cache/index" + index + "/size";
		std::ifstream file(cacheFilePath);
		if (!file.is_open())
			return 0ULL;

		std::string sizeStr;
		file >> sizeStr;
		uint64_t size = std::stoul(sizeStr);
		if (sizeStr.find('K') != std::string::npos)
			size *= 1024;
		else if (sizeStr.find('M') != std::string::npos)
			size *= 1024 * 1024;
		return static_cast<uint64_t>(size);
	};

	if (level == cache_level::one)
		return get_cache_size_from_file("0");
	std::string idx = (level == cache_level::two) ? "2" : "3";
	return get_cache_size_from_file(idx);

	#elif defined(__APPLE__)
	auto get_cache_size_for_mac = [](const char* cacheType) {
		uint64_t cacheSize = 0;
		uint64_t size	   = sizeof(cacheSize);
		std::string query  = std::string("hw.") + cacheType + "cachesize";
		if (sysctlbyname(query.c_str(), &cacheSize, &size, nullptr, 0) != 0)
			return 0ULL;
		return cacheSize;
	};

	if (level == cache_level::one)
		return get_cache_size_for_mac("l1d");
	if (level == cache_level::two)
		return get_cache_size_for_mac("l2");
	return get_cache_size_for_mac("l3");
	#endif

	return 0;
}

enum class host_cxx_compilers {
	clang,
	gnu,
	msvc,
};

int main() {
	const uint32_t thread_count	 = std::thread::hardware_concurrency();
	const uint32_t supported_isa = detect_supported_architectures();
	const uint64_t l1_cache_size = get_cache_size(cache_level::one);
	const uint64_t l2_cache_size = get_cache_size(cache_level::two);
	const uint64_t l3_cache_size = get_cache_size(cache_level::three);

	host_cxx_compilers host_cxx_compiler{};
	#if __clang__
	host_cxx_compiler = host_cxx_compilers::clang;
	#elif __GNUC__
	host_cxx_compiler = host_cxx_compilers::gnu;
	#elif _MSC_VER
	host_cxx_compiler = host_cxx_compilers::msvc;
	#endif

	std::cout << "HOST_CXX_COMPILER=" << static_cast<uint32_t>(host_cxx_compiler) << std::endl;
	std::cout << "THREAD_COUNT=" << thread_count << std::endl;
	std::cout << "INSTRUCTION_SET=" << supported_isa << std::endl;
	std::cout << "HAS_AVX2=" << ((supported_isa & static_cast<uint32_t>(instruction_set::AVX2)) ? 1 : 0) << std::endl;
	std::cout << "HAS_AVX512=" << ((supported_isa & static_cast<uint32_t>(instruction_set::AVX512f)) ? 1 : 0) << std::endl;
	std::cout << "HAS_NEON=" << ((supported_isa & static_cast<uint32_t>(instruction_set::NEON)) ? 1 : 0) << std::endl;
	std::cout << "HAS_SVE2=" << ((supported_isa & static_cast<uint32_t>(instruction_set::SVE2)) ? 1 : 0) << std::endl;
	std::cout << "L1_CACHE_SIZE=" << l1_cache_size << std::endl;
	std::cout << "L2_CACHE_SIZE=" << l2_cache_size << std::endl;
	std::cout << "L3_CACHE_SIZE=" << l3_cache_size << std::endl;
	std::cout << "CPU_SUCCESS=1" << std::endl;
	return 0;
}
#else
int main() {
	return -1;
}
#endif
