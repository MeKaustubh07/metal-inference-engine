#include <metal_stdlib>
using namespace metal;

// Sum of one value per thread over the whole threadgroup: simd_sum inside each 32-wide SIMD group, the SIMD-group
// partials through threadgroup memory, then one more simd_sum. Every thread gets the total.
inline float threadgroup_sum(float v, threadgroup float* partial, uint sg, uint lane, uint ntg) {
    v = simd_sum(v);
    if (lane == 0) partial[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = lane < (ntg + 31) / 32 ? partial[lane] : 0.0f;
        p = simd_sum(p);
        if (lane == 0) partial[0] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);       // partial is reused by the next sum
    return total;
}

// LayerNorm without bias (Cohere2 / Tiny Aya), fp32 weights: one threadgroup per row (token). Unlike RMSNorm the
// row's mean is subtracted first: mean, then the variance of the centred values, then w * (x - mean) / sqrt(var + eps),
// in the order of ops.layer_norm and transformers.
kernel void layer_norm(device float* out        [[buffer(0)]],
                       device const float* x    [[buffer(1)]],
                       device const float* w    [[buffer(2)]],
                       constant float& eps      [[buffer(3)]],
                       constant uint& d         [[buffer(4)]],
                       uint row  [[threadgroup_position_in_grid]],
                       uint tid  [[thread_position_in_threadgroup]],
                       uint ntg  [[threads_per_threadgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    device const float* xr = x + row * d;
    float acc = 0.0f;
    for (uint j = tid; j < d; j += ntg) acc += xr[j];
    float mean = threadgroup_sum(acc, partial, sg, lane, ntg) / float(d);
    acc = 0.0f;
    for (uint j = tid; j < d; j += ntg) { float c = xr[j] - mean; acc += c * c; }
    float inv = precise::rsqrt(threadgroup_sum(acc, partial, sg, lane, ntg) / float(d) + eps);
    for (uint j = tid; j < d; j += ntg) out[row * d + j] = w[j] * ((xr[j] - mean) * inv);
}
