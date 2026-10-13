"""Occupies a given amount of GPU memory (GiB, argv[1]) with one torch allocation and sleeps until terminated.
Used by ollama_offload_under_vram_pressure.py to imitate co-resident GPU consumers (Docling, the embedder, the
desktop) that were present in the March 2026 notebook runs but not in the isolated runner probes."""
import sys, time, torch
gib = float(sys.argv[1])
x = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda"); x.fill_(1); torch.cuda.synchronize()
print("ready", flush=True)
while True: time.sleep(1)
