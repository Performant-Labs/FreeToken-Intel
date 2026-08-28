// SYCL kernel entry points (icpx -fsycl). Stub.
// Issue: kernel-sycl
//
// Uses the sycl_ext:: extension namespace (not the standard sycl:: one) so this
// file binds the XPU extensions -- the "never compile against a fake SYCL" rule
// (docs/ci.md, conformance job) requires every sycl.hpp includer here to use
// sycl_ext::. A plain sycl:: reference is exactly the drift that rule catches.
#include <sycl/sycl.hpp>

namespace freetoken {
namespace {
sycl_ext::kernel_id<sycl_ext::make_kernel> make_placeholder(
    sycl_ext::queue& q) {
  (void)q;
  return sycl_ext::kernel_id<sycl_ext::make_kernel>{};
}
}  // namespace
}  // namespace freetoken
