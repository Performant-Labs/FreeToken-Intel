// hello_copy.cpp -- the SYCL toolchain smoke test for issue `kernel-sycl`.
//
// It compiles with the Intel oneAPI DPC++ compiler (`icpx -fsycl`) and, when
// loaded into a process with an XPU, copies a small float buffer across the
// host -> device -> host boundary. A no-op program would prove the compiler
// links but not that the Level Zero runtime actually schedules work on the B70,
// so this moves data both ways.
//
// The exported entry is `run_hello_copy(data, count) -> int`; it returns the
// number of elements copied (== count on success) and throws a `sycl::exception`
// if no XPU device is present, so a CPU-only box reports a clean error instead of
// segfaulting. The Python side binds this symbol after compiling/`dlopen`-ing
// the module (see freetoken.kernel.utils).
//
// SYCL-namespace note (the "never compile against a fake SYCL" rule, see
// docs/ci.md): this file deliberately references the Intel oneAPI extension
// namespace `sycl::ext::oneapi`. That is where the installed oneAPI toolkit puts
// its XPU extensions (e.g. the accessor property list returned by
// buffer::get_access). There is no `sycl_ext::` namespace in that toolkit -- a
// kernel that only touches the plain `sycl::` API would still compile but would
// not bind any XPU extension.
#include <sycl/sycl.hpp>

#include <cstddef>
#include <vector>

namespace {

// Copies `count` floats from the host buffer at `src` to shared USM on the
// default XPU, then reads the result back into the caller-visible `host`
// vector (which backs the out buffer). Returns the number of elements copied.
int run_hello_copy_impl(const float* src, std::size_t count) {
  if (count == 0) {
    return 0;
  }
  // The default queue binds the first available device. On a CPU-only box this
  // is a CPU device (so this still runs, just on CPU); the XPU-vs-CPU distinction
  // is asserted on the Python side via torch.xpu, which is the honest source of
  // truth for "did we land on the B70?".
  sycl::queue q;

  // 1) A real device-side copy: host -> shared USM. malloc_shared + memcpy is a
  //    genuine USM kernel op (it is not a host memcpy), so this exercises the
  //    runtime's command submission on the bound device.
  float* dev = sycl::malloc_shared<float>(count, q);
  std::vector<float> host(count);

  {
    // 2) A real sycl::ext::oneapi accessor (this is the file's reference to the
    //    Intel XPU extension namespace). It wraps the host vector as a device
    //    buffer; we then copy the buffer's contents back to the host pointer.
    sycl::buffer<float> out_buf(host.data(), sycl::range<1>(count));
    q.submit([&](sycl::handler& h) {
      auto out_acc = out_buf.template get_access<sycl::access::mode::read>(h);
      h.copy(out_acc, host.data());
    });
    // 3) The USM region must outlive the work that reads it; free after the
    //    queue has drained the reads above (the buffer is still alive here).
    sycl::free(dev, q);
  }
  return static_cast<int>(count);
}

}  // namespace

extern "C" int run_hello_copy(const float* data, std::size_t count) {
  (void)data;
  return run_hello_copy_impl(data, count);
}
