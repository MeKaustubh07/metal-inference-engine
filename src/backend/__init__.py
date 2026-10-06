"""Compute backends.

Apple GPUs: PyTorch's MPS allocator puts every tensor of 10-512 MiB in a 1 GiB heap until the process is "under memory
pressure", which by default starts at 1.4x Metal's recommended working set: 7.46 GiB on an 8 GB M2, so never. Tiny Aya
INT8 with an fp32 KV cache at 4.8K tokens then held 5.21 of the 5.33 GiB Metal recommends; while another process used
the GPU too, Metal aborted command buffers for lack of memory, and PyTorch 2.14, which never reads a command buffer's
status, returned garbage without an error. From a low watermark of 0.75x (4.0 GiB here) the allocator sizes each heap
to its request and frees empty ones: 4.70 GiB, decode exact. 0.75, because a 1 GiB heap made just below it still ends
under 0.95x the recommended size. It is read once, when torch first allocates on the GPU, so it is set here, before any
backend or model allocates; a value already in the environment wins.
"""
import os

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.75")
