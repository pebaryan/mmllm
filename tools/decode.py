#!/usr/bin/env python3
"""
Decode token IDs (comma-separated or space-separated) to text.

Usage:
    python3 tools/decode.py --model <hf_model_name> "1 2 3 4"
    python3 tools/decode.py --model tiny-stories-1M "42 128 256"

This reads token IDs from stdin or command line args and prints the decoded text.
Useful for converting the output of mmllm back to readable text.

Example workflow:
    # On Macmini:
    ./mmllm --model models/tinystories-1m.mlm --tokens 20 --prompt 42
    42 128 256 512 1
    
    # On dev machine:
    python3 tools/decode.py --model tiny-stories-1M "42 128 256 512 1"
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Decode token IDs to text')
    parser.add_argument('--model', type=str, default='roneneldan/TinyStories-1M',
                       help='HuggingFace model name for tokenizer')
    parser.add_argument('tokens', type=str, nargs='*',
                       help='Token IDs to decode (space-separated)')

    args = parser.parse_args()

    # Read tokens from args or stdin
    token_strs = []
    if args.tokens:
        token_strs = ' '.join(args.tokens)
    else:
        token_strs = sys.stdin.read().strip()

    # Parse token IDs
    tokens = []
    for t in token_strs.replace(',', ' ').split():
        t = t.strip()
        if t:
            try:
                tokens.append(int(t))
            except ValueError:
                print(f"Skipping invalid token: '{t}'", file=sys.stderr)

    if not tokens:
        print("No tokens to decode. Provide token IDs as arguments or pipe them in.")
        print("Example: python3 tools/decode.py --model tiny-stories '42 128 256'")
        return

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"Loading tokenizer for '{args.model}'...", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load '{args.model}': {e}", file=sys.stderr)
        print("Try a different model name, e.g. 'roneneldan/TinyStories-1M'", file=sys.stderr)
        sys.exit(1)

    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(decoded)


if __name__ == '__main__':
    main()
