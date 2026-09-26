// Minimal probe: does the SYCL runtime on this device actually allow device USM?
//   set +u; source /opt/intel/oneapi/setvars.sh; set -u
//   export CPATH=$PWD/tools/sysroot/usr/include
//   icpx -fsycl scripts/sycl_usm_probe.cpp -o /tmp/sycl_usm_probe && /tmp/sycl_usm_probe
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
  auto d = sycl::device(sycl::default_selector_v);
  std::cout << "device: " << d.get_info<sycl::info::device::name>() << "\n";
  std::cout << "usm_device_allocations: "
            << d.get_info<sycl::info::device::usm_device_allocations>() << "\n";
  std::cout << "usm_host_allocations:   "
            << d.get_info<sycl::info::device::usm_host_allocations>() << "\n";
  std::cout << "usm_shared_allocations: "
            << d.get_info<sycl::info::device::usm_shared_allocations>() << "\n";

  sycl::queue q{d};
  void* p = nullptr;
  try {
    p = sycl::malloc_device(1024, q);
    std::cout << "malloc_device(1024): " << (p ? "OK" : "NULL") << "\n";
  } catch (const std::exception& e) {
    std::cout << "malloc_device threw: " << e.what() << "\n";
  }
  if (p) sycl::free(p, q);

  try {
    auto* q_p = sycl::malloc_device<int>(64, q);
    std::cout << "malloc_device<int>[64]: " << (q_p ? "OK" : "NULL") << "\n";
    if (q_p) {
      q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { q_p[i] = 42; }).wait();
      std::cout << "kernel write OK, readback=" << q_p[0] << "\n";
      sycl::free(q_p, q);
    }
  } catch (const std::exception& e) {
    std::cout << "queue path threw: " << e.what() << "\n";
  }
  return 0;
}
