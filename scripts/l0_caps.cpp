// Level Zero capability + allocation probe for the EXL3 MoE bring-up (P0-05).
//
// Reads device limits from the Level Zero loader (not SYCL/torch) and performs
// a ladder of device allocations to prove whether relaxed allocation limits are
// in effect. Emits a single JSON object on stdout; logs go to stderr.
//
// Build:
//   g++ -O2 -std=c++17 -I tools/sysroot/usr/include scripts/l0_caps.cpp \
//       -L/usr/lib -lze_loader -o scripts/l0_caps
// Run (capture both configurations, since the env var is read at zeInit):
//   scripts/l0_caps > logs/moe-preflight/l0-capabilities.json
//   UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 scripts/l0_caps \
//       > logs/moe-preflight/l0-capabilities-relaxed.json

#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char *result_str(ze_result_t r) {
    switch (r) {
    case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    default: return "ZE_RESULT_OTHER";
    }
}

std::string json_escape(const std::string &in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') { out += "\\n"; }
        else if (static_cast<unsigned char>(c) < 0x20) { out += ' '; }
        else { out += c; }
    }
    return out;
}

void check(ze_result_t r, const char *what) {
    if (r != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "l0_caps: %s failed: %s\n", what, result_str(r));
    }
}

constexpr size_t MiB = 1024ull * 1024ull;
constexpr size_t GiB = 1024ull * MiB;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *relaxed_env = std::getenv("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS");

    ze_result_t r = zeInit(0);
    if (r != ZE_RESULT_SUCCESS) {
        std::printf("{\"error\":\"zeInit\",\"result\":\"%s\"}\n", result_str(r));
        return 1;
    }

    uint32_t driver_count = 0;
    zeDriverGet(&driver_count, nullptr);
    std::vector<ze_driver_handle_t> drivers(driver_count);
    zeDriverGet(&driver_count, drivers.data());

    std::printf("{\n");
    std::printf("  \"probe\": \"l0_caps\",\n");
    std::printf("  \"relaxed_allocation_env\": %s,\n",
                relaxed_env ? ("\"" + json_escape(relaxed_env) + "\"").c_str() : "null");
    std::printf("  \"driver_count\": %u,\n", driver_count);

    bool first_device = true;
    for (uint32_t d = 0; d < driver_count; ++d) {
        ze_driver_properties_t dp{};
        dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
        check(zeDriverGetProperties(drivers[d], &dp), "zeDriverGetProperties");

        uint32_t ext_count = 0;
        zeDriverGetExtensionProperties(drivers[d], &ext_count, nullptr);
        std::vector<ze_driver_extension_properties_t> exts(ext_count);
        zeDriverGetExtensionProperties(drivers[d], &ext_count, exts.data());

        uint32_t device_count = 0;
        zeDeviceGet(drivers[d], &device_count, nullptr);
        std::vector<ze_device_handle_t> devices(device_count);
        zeDeviceGet(drivers[d], &device_count, devices.data());

        std::printf("  \"driver_%u\": {\n", d);
        std::printf("    \"driver_version_raw\": %u,\n", dp.driverVersion);
        std::printf("    \"device_count\": %u,\n", device_count);
        std::printf("    \"relaxed_allocation_extension\": %s,\n",
                    [&]() {
                        for (const auto &e : exts) {
                            if (std::strstr(e.name, "relaxed_allocation_limits") ||
                                std::strstr(e.name, "RELAXED_ALLOCATION_LIMITS"))
                                return "true";
                        }
                        return "false";
                    }());
        std::printf("    \"extensions\": [");
        bool first_ext = true;
        for (const auto &e : exts) {
            // Keep the manifest readable: only surface extensions relevant to
            // memory, topology and profiling decisions for this project.
            const char *interesting[] = {"relaxed_allocation_limits",
                                         "device_memory_properties",
                                         "device_usablemem_size_properties",
                                         "xe_device_properties",
                                         "memory_compression_hints",
                                         "memory_free_policies",
                                         "external_memmap_sysmem",
                                         "eu_count",
                                         "device_cache_line_size"};
            bool keep = false;
            for (const char *needle : interesting) {
                if (std::strstr(e.name, needle) ||
                    std::strstr(e.name, "RELAXED_ALLOCATION_LIMITS")) {
                    keep = true;
                    break;
                }
            }
            if (std::strstr(e.name, "ZE_extension_relaxed_allocation")) keep = true;
            if (!keep) continue;
            std::printf("%s\"%s\"", first_ext ? "" : ", ", json_escape(e.name).c_str());
            first_ext = false;
        }
        std::printf("]\n");
        std::printf("  },\n");

        for (uint32_t di = 0; di < device_count; ++di) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            check(zeDeviceGetProperties(devices[di], &props), "zeDeviceGetProperties");

            ze_device_compute_properties_t compute{};
            compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
            check(zeDeviceGetComputeProperties(devices[di], &compute),
                  "zeDeviceGetComputeProperties");

            ze_device_module_properties_t module{};
            module.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
            check(zeDeviceGetModuleProperties(devices[di], &module),
                  "zeDeviceGetModuleProperties");

            uint32_t mem_count = 0;
            zeDeviceGetMemoryProperties(devices[di], &mem_count, nullptr);
            std::vector<ze_device_memory_properties_t> mems(mem_count);
            for (auto &m : mems) m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
            zeDeviceGetMemoryProperties(devices[di], &mem_count, mems.data());

            uint32_t cache_count = 0;
            zeDeviceGetCacheProperties(devices[di], &cache_count, nullptr);
            std::vector<ze_device_cache_properties_t> caches(cache_count);
            for (auto &c : caches) c.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
            zeDeviceGetCacheProperties(devices[di], &cache_count, caches.data());

            std::printf("  \"device_%u\": {\n", di);
            std::printf("    \"name\": \"%s\",\n", json_escape(props.name).c_str());
            std::printf("    \"type\": %u,\n", static_cast<unsigned>(props.type));
            std::printf("    \"vendor_id\": \"0x%04x\",\n", props.vendorId);
            std::printf("    \"device_id\": \"0x%04x\",\n", props.deviceId);
            unsigned long long mem_total = 0;
            for (const auto &m : mems) mem_total += m.totalSize;
            std::printf("    \"total_memory_bytes\": %llu,\n", mem_total);
            std::printf("    \"max_mem_alloc_bytes\": %llu,\n",
                        static_cast<unsigned long long>(props.maxMemAllocSize));
            std::printf("    \"max_hardware_contexts\": %u,\n", props.maxHardwareContexts);
            std::printf("    \"core_clock_mhz\": %u,\n", props.coreClockRate);
            std::printf("    \"timer_resolution_ns\": %llu,\n",
                        static_cast<unsigned long long>(props.timerResolution));
            std::printf("    \"topology\": {\"slices\": %u, \"subslices\": %u, \"eus\": %u, "
                        "\"threads_per_eu\": %u, \"simd_width\": %u},\n",
                        props.numSlices, props.numSubslicesPerSlice, props.numEUsPerSubslice,
                        props.numThreadsPerEU, props.physicalEUSimdWidth);
            std::printf("    \"compute\": {\"max_total_group\": %u, \"max_group\": [%u,%u,%u], "
                        "\"max_count\": [%u,%u,%u], \"max_slm_bytes\": %u, "
                        "\"subgroup_sizes\": [%u,%u]},\n",
                        compute.maxTotalGroupSize, compute.maxGroupSizeX, compute.maxGroupSizeY,
                        compute.maxGroupSizeZ, compute.maxGroupCountX,
                        compute.maxGroupCountY, compute.maxGroupCountZ,
                        compute.maxSharedLocalMemory,
                        compute.subGroupSizes[0],
                        compute.numSubGroupSizes > 1 ? compute.subGroupSizes[1] : 0);
            std::printf("    \"module\": {\"spirv\": \"%u.%u\", \"fp16\": %u, \"fp32\": %u, "
                        "\"fp64\": %u, \"max_args_bytes\": %u},\n",
                        ZE_MAJOR_VERSION(module.spirvVersionSupported),
                        ZE_MINOR_VERSION(module.spirvVersionSupported), module.fp16flags,
                        module.fp32flags, module.fp64flags, module.maxArgumentsSize);
            std::printf("    \"memory\": [");
            for (uint32_t mi = 0; mi < mem_count; ++mi) {
                std::printf("%s{\"name\": \"%s\", \"total_bytes\": %llu, "
                            "\"max_clock_mhz\": %u, \"bus_width_bits\": %u}",
                            mi ? ", " : "", json_escape(mems[mi].name).c_str(),
                            static_cast<unsigned long long>(mems[mi].totalSize),
                            mems[mi].maxClockRate, mems[mi].maxBusWidth);
            }
            std::printf("],\n");
            std::printf("    \"caches\": [");
            for (uint32_t ci = 0; ci < cache_count; ++ci) {
                std::printf("%s%llu", ci ? ", " : "",
                            static_cast<unsigned long long>(caches[ci].cacheSize));
            }
            std::printf("],\n");

            // Context is required for device allocations.
            ze_context_desc_t cdesc{};
            cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
            ze_context_handle_t context = nullptr;
            check(zeContextCreate(drivers[d], &cdesc, &context), "zeContextCreate");

            std::printf("    \"allocation_ladder\": [");
            if (context) {
                ze_device_mem_alloc_desc_t adesc{};
                adesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
                const size_t ladder[] = {256 * MiB, 512 * MiB, 1 * GiB, 2 * GiB, 4 * GiB};
                bool first_alloc = true;
                for (size_t bytes : ladder) {
                    void *ptr = nullptr;
                    ze_result_t ar = zeMemAllocDevice(context, &adesc, bytes, 4096,
                                                      devices[di], &ptr);
                    std::printf("%s{\"bytes\": %llu, \"gib\": %.2f, \"result\": \"%s\", "
                                "\"ok\": %s}",
                                first_alloc ? "" : ", ",
                                static_cast<unsigned long long>(bytes),
                                static_cast<double>(bytes) / static_cast<double>(GiB),
                                result_str(ar), ar == ZE_RESULT_SUCCESS ? "true" : "false");
                    first_alloc = false;
                    if (ar == ZE_RESULT_SUCCESS) {
                        // Device allocations are not guaranteed host-visible on
                        // all driver paths; allocation success is the signal we
                        // need, so release immediately without touching pages.
                        zeMemFree(context, ptr);
                    }
                }
                zeContextDestroy(context);
            }
            std::printf("]\n");
            std::printf("  }%s\n", (di + 1 < device_count) ? "," : "");
            first_device = false;
        }
        (void)first_device;
    }
    std::printf("}\n");
    return 0;
}
