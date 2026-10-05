#include <metal_stdlib>
using namespace metal;

// Both decode-attention kernels are templates over T, the type the KV cache stores: float (fp32 cache) or bfloat
// (bf16 cache, half the bytes). Every K/V element is widened to float as it is read, so all arithmetic stays fp32
// and a bf16 cache gives exactly what the fp32 kernel gives on the widened copy. Instantiated at the end of the file:
// the float versions keep their names, the bfloat ones end in _bf16.

// Decode attention: ONE new query per head against all S cached keys/values (GQA-aware).
// One threadgroup per query head.  q: [Hq, d]  k, v: [S, Hkv, d]  scores (scratch): [Hq, S]  out: [Hq, d]
// Steps: scores = q.k / sqrt(d) -> max -> exp and sum (numerically stable softmax) -> weighted sum of v.
// No mask is needed: the single query is the newest position, so every cached key is in its past.
template <typename T>
kernel void attention_decode(device float* out        [[buffer(0)]],
                             device const float* q    [[buffer(1)]],
                             device const T* k        [[buffer(2)]],
                             device const T* v        [[buffer(3)]],
                             device float* scores     [[buffer(4)]],
                             constant uint& S         [[buffer(5)]],
                             constant uint& n_kv      [[buffer(6)]],
                             constant uint& group     [[buffer(7)]],
                             constant uint& d         [[buffer(8)]],
                             uint h    [[threadgroup_position_in_grid]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint ntg  [[threads_per_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint kvh = h / group;
    device const float* qh = q + h * d;
    device float* sc = scores + (ulong)h * S;
    float scale = precise::rsqrt(float(d));

    // 1. scores and running max
    float m = -INFINITY;
    for (uint s = tid; s < S; s += ntg) {
        device const vec<T, 4>* ks = (device const vec<T, 4>*)(k + ((ulong)s * n_kv + kvh) * d);
        device const float4* q4 = (device const float4*)qh;
        float dotv = 0.0f;
        for (uint j = 0; j < d / 4; ++j) dotv += dot(q4[j], float4(ks[j]));   // d is a multiple of 4 (64 to 256)
        dotv *= scale;
        sc[s] = dotv;
        m = max(m, dotv);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : -INFINITY; t = simd_max(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. exponentials and their sum
    float sum = 0.0f;
    for (uint s = tid; s < S; s += ntg) { float e = precise::exp(sc[s] - m); sc[s] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : 0.0f; t = simd_sum(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float inv = 1.0f / red[0];

    // 3. weighted sum of values. The threadgroup is split into ntg/d groups; group p handles positions
    //    p, p + ngroups, ... for one output dimension each, then the partial sums are added.
    threadgroup float part[1024];
    uint ngroups = max(1u, ntg / d);
    uint j = tid % d, p = tid / d;
    if (p < ngroups) {
        float acc = 0.0f;
        for (uint s = p; s < S; s += ngroups) acc += sc[s] * float(v[((ulong)s * n_kv + kvh) * d + j]);
        part[p * d + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p == 0) {
        float acc = 0.0f;
        for (uint g = 0; g < ngroups; ++g) acc += part[g * d + j];
        out[h * d + j] = acc * inv;
    }
}

// Paged, batched decode attention (vLLM-style): one threadgroup per (query head, sequence). Keys/values are read in
// place from the paged pool through each sequence's block table, so no per-sequence gather into a contiguous copy.
// q/out: [B, Hq, d]; k, v: one layer of the pool [blocks, bs, Hkv, d]; tables: [B, max_nb]; lens: [B];
// scores (scratch): [B, Hq, max_len]. Same three steps as attention_decode.
template <typename T>
kernel void paged_attention_decode(device float* out        [[buffer(0)]],
                                   device const float* q    [[buffer(1)]],
                                   device const T* k        [[buffer(2)]],
                                   device const T* v        [[buffer(3)]],
                                   device float* scores     [[buffer(4)]],
                                   device const int* tables [[buffer(5)]],
                                   device const int* lens   [[buffer(6)]],
                                   constant uint& max_nb    [[buffer(7)]],
                                   constant uint& max_len   [[buffer(8)]],
                                   constant uint& bs        [[buffer(9)]],
                                   constant uint& n_kv      [[buffer(10)]],
                                   constant uint& group     [[buffer(11)]],
                                   constant uint& d         [[buffer(12)]],
                                   uint2 tgp  [[threadgroup_position_in_grid]],
                                   uint2 tpos [[thread_position_in_threadgroup]],
                                   uint2 tgs  [[threads_per_threadgroup]],
                                   uint sg    [[simdgroup_index_in_threadgroup]],
                                   uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint h = tgp.x, bi = tgp.y, tid = tpos.x, ntg = tgs.x, Hq = n_kv * group;
    uint kvh = h / group, S = uint(lens[bi]);
    device const int* table = tables + (ulong)bi * max_nb;
    device const float* qh = q + ((ulong)bi * Hq + h) * d;
    device float* sc = scores + ((ulong)bi * Hq + h) * max_len;
    float scale = precise::rsqrt(float(d));
    #define KV_ROW(s) (((ulong)table[(s) / bs] * bs + (s) % bs) * n_kv + kvh) * d     // position -> pool offset

    float m = -INFINITY;
    for (uint s = tid; s < S; s += ntg) {
        device const vec<T, 4>* ks = (device const vec<T, 4>*)(k + KV_ROW(s));
        device const float4* q4 = (device const float4*)qh;
        float dotv = 0.0f;
        for (uint j = 0; j < d / 4; ++j) dotv += dot(q4[j], float4(ks[j]));
        dotv *= scale;
        sc[s] = dotv;
        m = max(m, dotv);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : -INFINITY; t = simd_max(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint s = tid; s < S; s += ntg) { float e = precise::exp(sc[s] - m); sc[s] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : 0.0f; t = simd_sum(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float inv = 1.0f / red[0];

    threadgroup float part[1024];
    uint ngroups = max(1u, ntg / d);
    uint j = tid % d, p = tid / d;
    if (p < ngroups) {
        float acc = 0.0f;
        for (uint s = p; s < S; s += ngroups) acc += sc[s] * float(v[KV_ROW(s) + j]);
        part[p * d + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p == 0) {
        float acc = 0.0f;
        for (uint g = 0; g < ngroups; ++g) acc += part[g * d + j];
        out[((ulong)bi * Hq + h) * d + j] = acc * inv;
    }
    #undef KV_ROW
}

// The kernels the backend calls, by name (the parameter attributes come from the templates above).
#define ATTENTION_DECODE(T, NAME) \
    template [[host_name(NAME)]] kernel void attention_decode<T>(device float*, device const float*, \
        device const T*, device const T*, device float*, constant uint&, constant uint&, constant uint&, \
        constant uint&, uint, uint, uint, uint, uint);
#define PAGED_ATTENTION_DECODE(T, NAME) \
    template [[host_name(NAME)]] kernel void paged_attention_decode<T>(device float*, device const float*, \
        device const T*, device const T*, device float*, device const int*, device const int*, constant uint&, \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint2, uint2, uint, \
        uint);
ATTENTION_DECODE(float, "attention_decode")
ATTENTION_DECODE(bfloat, "attention_decode_bf16")
PAGED_ATTENTION_DECODE(float, "paged_attention_decode")
PAGED_ATTENTION_DECODE(bfloat, "paged_attention_decode_bf16")
#undef ATTENTION_DECODE
#undef PAGED_ATTENTION_DECODE
