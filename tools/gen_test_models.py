#!/usr/bin/env python3
"""Generate test models of increasing sizes for GPU vs CPU benchmarking."""
import sys
import os

# Add parent dir to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.export_model import export_dummy

sizes = [
    # (name, d_model, layers, ffn_hidden, heads, d_head)
    ("test_d128_l4",  128,  4,  512,   4, 32),
    ("test_d256_l4",  256,  4,  1024,  8, 32),
    ("test_d512_l4",  512,  4,  2048,  8, 64),
    ("test_d1024_l2", 1024, 2,  4096, 16, 64),
]

models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

for name, d_model, n_layers, ffn_hidden, n_heads, d_head in sizes:
    config = {
        "d_model": d_model,
        "n_layers": n_layers,
        "ffn_hidden": ffn_hidden,
        "n_heads": n_heads,
        "d_head": d_head,
        "vocab_size": 1000,
        "max_seq_len": 128,
        "has_bias": True,
        "weight_tying": True,
    }
    out_path = os.path.join(models_dir, f"{name}.mlm")
    print(f"Generating {name}... (d={d_model}, L={n_layers}, ffn={ffn_hidden})")
    export_dummy(out_path, config)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  -> {out_path} ({size_kb:.0f} KB)")
    print()

print("All models generated!")
