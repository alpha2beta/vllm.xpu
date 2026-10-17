// PCIe link bandwidth probe for the Intel Arc Pro B60 (tasks.md Phase 0).
//
// Why: `sudo lspci -vv` and the kernel's own sysfs link attributes both report
// Gen1 x1 (2.5 GT/s) on the switch->GPU hop of BOTH cards, while the upstream
// ports report Gen4 x8. Gen1 x1 would cap host<->device traffic at ~250 MB/s.
// Register reads are root-only beyond offset 64, so instead of arguing about
// what the registers mean, measure what the link actually does.
//
// Method: large pinned-host <-> device copies over Level Zero, timed on the
// client clock. No root required.
//
// Build:
//   g++ -O2 -std=c++17 -I<cookbook-or-ainfer>/include l0_bw.cpp \
//       -o l0_bw /usr/lib/x86_64-linux-gnu/libze_loader.so.1
// Usage: l0_bw [device_index]
#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  int pick = 0;
  if (argc > 1) pick = std::atoi(argv[1]);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrivers = 0;
  CHECK(zeDriverGet(&nDrivers, nullptr));
  std::vector<ze_driver_handle_t> drivers(nDrivers);
  CHECK(zeDriverGet(&nDrivers, drivers.data()));
  if (nDrivers == 0) {
    std::fprintf(stderr, "no Level Zero driver\n");
    return 1;
  }
  ze_driver_handle_t drv = drivers[0];

  uint32_t nDev = 0;
  CHECK(zeDeviceGet(drv, &nDev, nullptr));
  std::vector<ze_device_handle_t> devs(nDev);
  CHECK(zeDeviceGet(drv, &nDev, devs.data()));

  std::printf("Level Zero enumerated %u device(s):\n", nDev);
  for (uint32_t i = 0; i < nDev; i++) {
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    CHECK(zeDeviceGetProperties(devs[i], &p));
    double memGiB = double(p.maxMemAllocSize);
    std::printf("  [%u] %-40s  type=%d  maxAlloc=%.2f GiB\n", i, p.name,
                (int)p.type, memGiB / (1024.0 * 1024.0 * 1024.0));
  }
  if (pick < 0 || pick >= int(nDev)) {
    std::fprintf(stderr, "device index %d out of range\n", pick);
    return 1;
  }
  ze_device_handle_t dev = devs[pick];
  ze_device_properties_t dp{};
  dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  CHECK(zeDeviceGetProperties(dev, &dp));
  std::printf("\nprobing: [%d] %s\n\n", pick, dp.name);

  ze_context_desc_t cdesc{};
  cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t ctx = nullptr;
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // 64 MiB working set, host-pinned so the copy really crosses PCIe.
  const size_t BUF = 64ull << 20;
  const int ROUNDS = 64;  // 64 * 64 MiB = 4 GiB per direction

  ze_host_mem_alloc_desc_t hdesc{};
  hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void* host = nullptr;
  CHECK(zeMemAllocHost(ctx, &hdesc, BUF, 64, &host));

  ze_device_mem_alloc_desc_t ddesc{};
  ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void* device = nullptr;
  CHECK(zeMemAllocDevice(ctx, &ddesc, BUF, 64, dev, &device));
  std::printf("allocated %zu MiB host (pinned) + %zu MiB device\n\n",
              BUF >> 20, BUF >> 20);

  ze_command_queue_desc_t qdesc{};
  qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qdesc.ordinal = 0;
  qdesc.index = 0;
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_queue_handle_t queue = nullptr;
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &queue));

  ze_command_list_desc_t ldesc{};
  ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &list));

  auto run_batch = [&](bool h2d) -> double {
    CHECK(zeCommandListReset(list));
    for (int r = 0; r < ROUNDS; r++) {
      if (h2d)
        CHECK(zeCommandListAppendMemoryCopy(list, device, host, BUF, nullptr, 0, nullptr));
      else
        CHECK(zeCommandListAppendMemoryCopy(list, host, device, BUF, nullptr, 0, nullptr));
    }
    CHECK(zeCommandListClose(list));
    ze_command_list_handle_t one = list;
    auto t0 = clk::now();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &one, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    auto t1 = clk::now();
    return secs(t0, t1);
  };

  // Warmup (JIT/driver setup, page pinning).
  run_batch(true);
  run_batch(false);

  const double bytes = double(ROUNDS) * double(BUF);

  double best_h2d = 0, sum_h2d = 0;
  const int TRIALS = 5;
  for (int i = 0; i < TRIALS; i++) {
    double t = run_batch(true);
    double gbs = bytes / t / 1e9;
    sum_h2d += gbs;
    if (gbs > best_h2d) best_h2d = gbs;
    std::printf("  H2D trial %d: %.3f s -> %.2f GB/s\n", i + 1, t, gbs);
  }
  double h2d_mean = sum_h2d / TRIALS;

  double best_d2h = 0, sum_d2h = 0;
  for (int i = 0; i < TRIALS; i++) {
    double t = run_batch(false);
    double gbs = bytes / t / 1e9;
    sum_d2h += gbs;
    if (gbs > best_d2h) best_d2h = gbs;
    std::printf("  D2H trial %d: %.3f s -> %.2f GB/s\n", i + 1, t, gbs);
  }
  double d2h_mean = sum_d2h / TRIALS;

  std::printf("\n===== RESULT =====\n");
  std::printf("H2D: mean %.2f GB/s, best %.2f GB/s\n", h2d_mean, best_h2d);
  std::printf("D2H: mean %.2f GB/s, best %.2f GB/s\n", d2h_mean, best_d2h);

  double observed = (best_h2d > best_d2h) ? best_h2d : best_d2h;
  std::printf("\nReference (practical) PCIe bandwidths:\n");
  std::printf("  Gen1 x1  ~0.25 GB/s     Gen3 x4  ~3.94 GB/s\n");
  std::printf("  Gen2 x1  ~0.98 GB/s     Gen4 x4  ~7.88 GB/s\n");
  std::printf("  Gen3 x1  ~1.97 GB/s     Gen4 x8  ~15.75 GB/s\n");

  std::printf("\nInterpretation: measured best %.2f GB/s\n", observed);
  if (observed < 0.4)
    std::printf("  -> consistent with Gen1 x1: the downgrade is REAL.\n");
  else if (observed < 1.0)
    std::printf("  -> clearly above Gen1 x1: link is wider/faster than reported.\n");
  else
    std::printf("  -> far above Gen1 x1 (which caps at ~0.25 GB/s):\n"
                "     the Gen1 x1 register reading does NOT describe the\n"
                "     effective link. NOT a link downgrade.\n");

  CHECK(zeCommandQueueDestroy(queue));
  CHECK(zeCommandListDestroy(list));
  CHECK(zeMemFree(ctx, device));
  CHECK(zeMemFree(ctx, host));
  CHECK(zeContextDestroy(ctx));
  return 0;
}
