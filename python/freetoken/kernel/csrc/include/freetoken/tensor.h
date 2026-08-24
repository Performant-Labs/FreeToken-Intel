#pragma once
// Lightweight tensor view shared by SYCL and CPU kernels. Stub.
// Issue: kernel-sycl
struct FTTensor {
  void* data;
  int dtype;
  int ndim;
  long dims[8];
};
