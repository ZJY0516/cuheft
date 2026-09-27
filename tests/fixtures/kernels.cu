// Test fixture for cuheft, compiled with nvcc by tests/cli.rs.

// Not inlined, so ptxas emits it as a local `$kernel$callee` symbol inside
// each calling kernel's .text section.
__device__ __noinline__ float scale(float x, float s) { return x * s + 1.0f; }

template <int N>
__global__ void scale_kernel(float* out, const float* in, float s) {
  // Static shared memory is a NOBITS section: 16 KiB and 32 KiB declared,
  // no bytes in the file
  __shared__ float tile[N];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x % N] = in[i];
  __syncthreads();
  out[i] = scale(tile[(threadIdx.x + 1) % N], s);
}

template __global__ void scale_kernel<4096>(float*, const float*, float);
template __global__ void scale_kernel<8192>(float*, const float*, float);

extern "C" __global__ void plain_c_kernel(int* out) { out[threadIdx.x] = threadIdx.x; }

// A dynamically indexed local array lives in local memory, giving the kernel
// a non-zero stack frame
__global__ void local_array_kernel(float* out, int n) {
  float buf[64];
  for (int i = 0; i < 64; ++i) buf[(i * 7 + n) % 64] = out[i];
  out[threadIdx.x] = buf[n % 64] + buf[(n + threadIdx.x) % 64];
}
