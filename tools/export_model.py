#!/usr/bin/env python3
"""
Export a TinyStories model from HuggingFace to the mmllm binary format (.mlm).

Usage:
    python3 tools/export_model.py --model <hf_model_name> --output <output.mlm>

Examples:
    # Export TinyStories-1M (if available on HF)
    python3 tools/export_model.py --model tiny-stories-1M --output models/tinystories-1m.mlm

    # Export a NanoGPT-style model from a local checkpoint
    python3 tools/export_model.py --checkpoint checkpoint.pt --config config.json --output models/model.mlm

The .mlm format packs all weights into RGBA32F-compatible layout (4 floats per texel).
"""

import struct
import sys
import os
import json
import argparse
import numpy as np

MAGIC = b'MMLM'
VERSION = 1


def write_model_header(f, config):
    """Write the model header."""
    header = struct.pack(
        '<4sI'  # magic (4 bytes) + version (uint32)
        '9i'    # 9 int32 config fields (vocab_size through weight_tying)
        '8i',   # 8 int32 reserved
        MAGIC, VERSION,
        config['vocab_size'],
        config['d_model'],
        config['n_layers'],
        config['n_heads'],
        config['d_head'],
        config['ffn_hidden'],
        config['max_seq_len'],
        int(config.get('has_bias', True)),
        int(config.get('weight_tying', True)),
        int(config.get('flags', 0)),            # reserved[0]: bit0 = attention not scaled (GPT-Neo)
        int(config.get('eos_token', -1)) + 1,   # reserved[1]: EOS token id + 1 (0 = none)
        int(config.get('local_window', 0)),     # reserved[2]: local attention window
        int(config.get('local_mask', 0)),       # reserved[3]: bit i = layer i is local
        *([0] * 4),
    )
    f.write(header)


def write_tensor(f, data):
    """
    Write a 2D float tensor as raw row-major float32.
    Model::load() reads rows*cols floats and packs them into RGBA32F texels itself.
    data: numpy array of shape [rows, cols]
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    assert data.ndim == 2, data.shape
    f.write(data.tobytes())


def export_huggingface(model_name, output_path):
    """
    Export a HuggingFace TinyStories model.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed.")
        print("  pip install transformers torch")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Detect architecture type
    state_dict = model.state_dict()
    keys = set(state_dict.keys())
    is_gptneo = any('q_proj' in k for k in keys)
    arch_type = "GPT-Neo" if is_gptneo else "GPT-2"
    print(f"Detected architecture: {arch_type}")

    # Infer ffn_hidden from MLP weight shape if not in config
    sample_mlp_key = None
    for k in keys:
        if 'mlp.c_fc.weight' in k:
            sample_mlp_key = k
            break
    ffn_hidden = getattr(config, 'intermediate_size', None)
    if ffn_hidden is None and sample_mlp_key:
        ffn_hidden = state_dict[sample_mlp_key].shape[0 if is_gptneo else 1]
        print(f"Inferred ffn_hidden={ffn_hidden} from {sample_mlp_key}")
    elif ffn_hidden is None:
        ffn_hidden = 4 * config.hidden_size  # fallback

    # Extract config
    cfg = {
        'vocab_size': config.vocab_size,
        'd_model': config.hidden_size,
        'n_layers': config.num_layers if hasattr(config, 'num_layers') else config.num_hidden_layers,
        'n_heads': config.num_heads if hasattr(config, 'num_heads') else config.num_attention_heads,
        'd_head': config.hidden_size // (config.num_heads if hasattr(config, 'num_heads') else config.num_attention_heads),
        'ffn_hidden': ffn_hidden,
        'max_seq_len': getattr(config, 'max_position_embeddings', 512),
        'has_bias': True,
        'flags': 1 if is_gptneo else 0,   # GPT-Neo does not scale attention scores
        'local_window': int(getattr(config, 'window_size', 0) or 0) if is_gptneo else 0,
        'local_mask': sum(1 << i for i, t in enumerate(getattr(config, 'attention_layers', []) or [])
                          if t == 'local') if is_gptneo else 0,
        'eos_token': (getattr(config, 'eos_token_id', None)
                      if getattr(config, 'eos_token_id', None) is not None
                      else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1)),
        'weight_tying': True,  # mmllm always uses token_embed for LM head
    }

    print(f"Config: vocab={cfg['vocab_size']} d={cfg['d_model']} "
          f"layers={cfg['n_layers']} heads={cfg['n_heads']}")

    f = open(output_path, 'wb')
    write_model_header(f, cfg)

    D = cfg['d_model']
    H = cfg['ffn_hidden']

    # Token embedding (also used as LM head via weight tying)
    write_tensor(f, state_dict['transformer.wte.weight'].cpu().numpy())

    # Position embedding
    write_tensor(f, state_dict['transformer.wpe.weight'].cpu().numpy())

    # Per-layer weights
    for i in range(cfg['n_layers']):
        prefix = f'transformer.h.{i}.'

        if is_gptneo:
            # GPT-Neo: separate Q, K, V projections [D, D] each -> fuse to [D, 3*D]
            q = state_dict[f'{prefix}attn.attention.q_proj.weight'].cpu().numpy()  # [D, D]
            k = state_dict[f'{prefix}attn.attention.k_proj.weight'].cpu().numpy()  # [D, D]
            v = state_dict[f'{prefix}attn.attention.v_proj.weight'].cpu().numpy()  # [D, D]
            wqkv = np.concatenate([q, k, v], axis=0)  # [3*D, D]
            write_tensor(f, wqkv.T)  # [D, 3*D]

            # Attention output projection: [D, D]
            o = state_dict[f'{prefix}attn.attention.out_proj.weight'].cpu().numpy()
            write_tensor(f, o.T)  # [D, D]
        else:
            # GPT-2: fused QKV [3*D, D] -> transpose to [D, 3*D]
            # Conv1D weight is already [D, 3*D] (x @ W), q|k|v concatenated: no transpose
            q = state_dict[f'{prefix}attn.c_attn.weight'].cpu().numpy()  # [D, 3*D]
            write_tensor(f, q)

            # Attention output: [D, D]
            o = state_dict[f'{prefix}attn.c_proj.weight'].cpu().numpy()  # [D, D], already x @ W
            write_tensor(f, o)

        # FFN weights (same for GPT-2 and GPT-Neo)
        w1 = state_dict[f'{prefix}mlp.c_fc.weight'].cpu().numpy()
        write_tensor(f, w1.T if is_gptneo else w1)  # -> [D, H]

        w2 = state_dict[f'{prefix}mlp.c_proj.weight'].cpu().numpy()
        write_tensor(f, w2.T if is_gptneo else w2)  # -> [H, D]

        # Layer norms
        write_tensor(f, state_dict[f'{prefix}ln_1.weight'].cpu().numpy().reshape(1, -1))
        write_tensor(f, state_dict[f'{prefix}ln_1.bias'].cpu().numpy().reshape(1, -1))
        write_tensor(f, state_dict[f'{prefix}ln_2.weight'].cpu().numpy().reshape(1, -1))
        write_tensor(f, state_dict[f'{prefix}ln_2.bias'].cpu().numpy().reshape(1, -1))

        # Attention biases: GPT-Neo has no q/k/v bias (zeros) but does have out_proj.bias
        if is_gptneo:
            write_tensor(f, np.zeros((1, 3 * D), dtype=np.float32))
            write_tensor(f, state_dict[f'{prefix}attn.attention.out_proj.bias'].cpu().numpy().reshape(1, -1))
        else:
            write_tensor(f, state_dict[f'{prefix}attn.c_attn.bias'].cpu().numpy().reshape(1, -1))
            write_tensor(f, state_dict[f'{prefix}attn.c_proj.bias'].cpu().numpy().reshape(1, -1))

    # Final layer norm
    write_tensor(f, state_dict['transformer.ln_f.weight'].cpu().numpy().reshape(1, -1))
    write_tensor(f, state_dict['transformer.ln_f.bias'].cpu().numpy().reshape(1, -1))

    # FFN biases for every layer: up-projection bias [H], then down-projection bias [D]
    for i in range(cfg['n_layers']):
        prefix = f'transformer.h.{i}.'
        write_tensor(f, state_dict[f'{prefix}mlp.c_fc.bias'].cpu().numpy().reshape(1, -1))
        write_tensor(f, state_dict[f'{prefix}mlp.c_proj.bias'].cpu().numpy().reshape(1, -1))

    f.close()
    file_size = os.path.getsize(output_path)
    print(f"Exported to {output_path} ({file_size / 1024:.1f} KB)")


def export_dummy(output_path, config_override=None):
    """
    Export a dummy model with random weights for testing.
    """
    cfg = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'd_head': 32,
        'ffn_hidden': 512,
        'max_seq_len': 128,
        'has_bias': True,
        'weight_tying': True,
    }
    if config_override:
        cfg.update(config_override)

    np.random.seed(42)

    f = open(output_path, 'wb')
    write_model_header(f, cfg)

    D = cfg['d_model']
    H = cfg['ffn_hidden']
    V = cfg['vocab_size']
    L = cfg['n_layers']
    T = cfg['max_seq_len']

    # Token embedding
    write_tensor(f, np.random.randn(V, D).astype(np.float32) * 0.02)
    # Position embedding
    write_tensor(f, np.random.randn(T, D).astype(np.float32) * 0.02)

    for _ in range(L):
        write_tensor(f, np.random.randn(D, 3*D).astype(np.float32) * 0.02)  # wqkv
        write_tensor(f, np.random.randn(D, D).astype(np.float32) * 0.02)    # wo
        write_tensor(f, np.random.randn(D, H).astype(np.float32) * 0.02)    # wg1
        write_tensor(f, np.random.randn(H, D).astype(np.float32) * 0.02)    # wg2
        write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln1_gain
        write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln1_bias
        write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln2_gain
        write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln2_bias
        write_tensor(f, np.zeros((1, 3*D), dtype=np.float32))  # bqkv
        write_tensor(f, np.zeros((1, D), dtype=np.float32))    # bo

    write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln_final_gain
    write_tensor(f, np.zeros((1, D), dtype=np.float32))  # ln_final_bias

    for _ in range(L):
        write_tensor(f, np.zeros((1, H), dtype=np.float32))  # bg1
        write_tensor(f, np.zeros((1, D), dtype=np.float32))  # bg2

    f.close()
    file_size = os.path.getsize(output_path)
    print(f"Dummy model exported to {output_path} ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Export model to .mlm format')
    parser.add_argument('--model', type=str, help='HuggingFace model name')
    parser.add_argument('--checkpoint', type=str, help='PyTorch checkpoint path')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--dummy', action='store_true', help='Export a dummy model for testing')
    parser.add_argument('--output', type=str, default='models/model.mlm',
                       help='Output .mlm file path')

    args = parser.parse_args()

    if args.dummy:
        config = {}
        if args.config:
            with open(args.config) as f:
                config = json.load(f) | config
        export_dummy(args.output, config)
    elif args.model:
        export_huggingface(args.model, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 tools/export_model.py --dummy --output models/test.mlm")
        print("  python3 tools/export_model.py --model tiny-stories-1M --output models/tinystories-1m.mlm")


if __name__ == '__main__':
    main()
