#include <metal_stdlib>
using namespace metal;

// RoPE, HF "rotate_half" pairing: number i pairs with number i + d/2.
// One thread per (token, head, pair). x/out: [T, H, d] contiguous, positions: [T], inv_freq: [d/2].
// The frequencies come from the host, computed as transformers computes them (fp32, on the CPU): a pow() here
// differed from them in the last bit, and the angle position * freq turns that into an error that grows with the
// position (2e-4 relative at 4095, 4e-4 at 8191).
kernel void rope(device float* out           [[buffer(0)]],
                 device const float* x       [[buffer(1)]],
                 device const int* positions [[buffer(2)]],
                 device const float* inv_freq [[buffer(3)]],
                 constant uint& n_heads      [[buffer(4)]],
                 constant uint& d            [[buffer(5)]],
                 constant uint& n_total      [[buffer(6)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n_total) return;
    uint half_d = d / 2;
    uint i = gid % half_d;                     // pair index
    uint th = gid / half_d;                    // flat (token, head)
    uint t = th / n_heads;
    float ang = float(positions[t]) * inv_freq[i];
    float c = precise::cos(ang), s = precise::sin(ang);
    uint base = th * d;
    float a = x[base + i], b = x[base + i + half_d];
    out[base + i] = a * c - b * s;
    out[base + i + half_d] = a * s + b * c;
}
