"""
HeartMuLa Turbo - Optimized Music Generation with torch.compile and Persistent Caching

This script provides significant speedups through:
1. torch.compile() with persistent caching (compiled artifacts saved to disk)
2. Optimized model components (backbone, decoder, flow matching, scalar model)
3. Optional warmup to trigger compilation before actual inference

First run: Slow (~2-5 minutes) as models are compiled
Subsequent runs: Fast (~5-10 seconds) as compiled artifacts are loaded from cache
"""

import os
import sys
import time
import argparse

# ============================================================================
# PERSISTENT CACHE CONFIGURATION - Must be set BEFORE importing torch
# ============================================================================
CACHE_DIR = os.path.expanduser("~/.cache/heartlib_compiled")
os.makedirs(CACHE_DIR, exist_ok=True)

# Enable persistent compilation caching
os.environ["TORCH_COMPILE_CACHE_DIR"] = CACHE_DIR
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

# Optional: Reduce recompilation overhead
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import torch
from heartlib import HeartMuLaGenPipeline


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got: {value}")


def str2dtype(value):
    value = value.lower()
    if value == "float32" or value == "fp32":
        return torch.float32
    elif value == "float16" or value == "fp16":
        return torch.float16
    elif value == "bfloat16" or value == "bf16":
        return torch.bfloat16
    else:
        raise argparse.ArgumentTypeError(f"Dtype not recognized: {value}")


def str2device(value):
    value = value.lower()
    return torch.device(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HeartMuLa Turbo - Optimized Music Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model checkpoints")
    parser.add_argument("--version", type=str, default="3B",
                        choices=["300M", "400M", "3B", "7B"],
                        help="Model version/size")
    
    # Input/Output
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt",
                        help="Path to lyrics file or lyrics string")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt",
                        help="Path to tags file or tags string")
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3",
                        help="Output audio file path")
    
    # Generation parameters
    parser.add_argument("--max_audio_length_ms", type=int, default=240_000,
                        help="Maximum audio length in milliseconds")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (1.0 = no CFG, 2x faster)")
    
    # Device/dtype configuration
    parser.add_argument("--mula_device", type=str2device, default="cuda",
                        help="Device for HeartMuLa model")
    parser.add_argument("--codec_device", type=str2device, default="cuda",
                        help="Device for HeartCodec model")
    parser.add_argument("--mula_dtype", type=str2dtype, default="bfloat16",
                        help="Data type for HeartMuLa model")
    parser.add_argument("--codec_dtype", type=str2dtype, default="float32",
                        help="Data type for HeartCodec model")
    
    # Optimization options
    parser.add_argument("--compile", type=str2bool, default=True,
                        help="Enable torch.compile() optimization")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--warmup", type=str2bool, default=False,
                        help="Run warmup pass to trigger compilation")
    parser.add_argument("--warmup_length_ms", type=int, default=5000,
                        help="Warmup audio length in milliseconds")
    
    return parser.parse_args()


def compile_pipeline(pipe, compile_mode="reduce-overhead", verbose=True):
    """
    Compile the pipeline components with persistent caching.
    
    Compiled artifacts are saved to ~/.cache/heartlib_compiled and reused
    on subsequent runs for instant loading.
    
    NOTE: HeartCodec flow_matching.estimator is SKIPPED due to Inductor
    compatibility issues with positional encoding (NaN bounds checking bug).
    The HeartMuLa components provide the main speedup anyway.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Compiling pipeline with mode: {compile_mode}")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"{'='*60}\n")
    
    compile_start = time.time()
    
    # Access models to ensure they're loaded (triggers lazy loading if enabled)
    mula = pipe.mula
    codec = pipe.codec
    
    compiled_count = 0
    
    # Compile HeartMuLa backbone (the main LLM - biggest speedup)
    if verbose:
        print("[1/3] Compiling HeartMuLa backbone...")
    try:
        mula.backbone = torch.compile(
            mula.backbone,
            mode=compile_mode,
            fullgraph=False,  # Allow graph breaks for flexibility
        )
        compiled_count += 1
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed to compile backbone: {e}")
    
    # Compile HeartMuLa decoder (smaller transformer for codebook prediction)
    if verbose:
        print("[2/3] Compiling HeartMuLa decoder...")
    try:
        mula.decoder = torch.compile(
            mula.decoder,
            mode=compile_mode,
            fullgraph=False,
        )
        compiled_count += 1
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed to compile decoder: {e}")
    
    # SKIP: HeartCodec flow_matching.estimator
    # This has Inductor compatibility issues with positional encoding
    # causing "TypeError: Invalid NaN comparison" during compilation.
    # The speedup from HeartMuLa compilation is the main benefit anyway.
    if verbose:
        print("[3/3] Skipping HeartCodec flow_matching.estimator (Inductor incompatible)")
        print("      Reason: Positional encoding causes NaN bounds checking errors")
    
    # Compile scalar model (audio decoder) - optional, try with fallback
    if verbose:
        print("[+] Attempting to compile HeartCodec scalar model...")
    try:
        codec.scalar_model = torch.compile(
            codec.scalar_model,
            mode=compile_mode,
            fullgraph=False,
        )
        compiled_count += 1
        if verbose:
            print("      ✓ scalar_model compiled successfully")
    except Exception as e:
        if verbose:
            print(f"      ⚠️  Skipped scalar_model: {e}")
    
    if verbose:
        print(f"\nCompilation setup complete in {time.time() - compile_start:.2f}s")
        print(f"Successfully compiled {compiled_count} components")
        print("(Actual compilation happens on first forward pass)\n")
    
    return pipe


def warmup_compiled_model(pipe, warmup_length_ms=5000, verbose=True):
    """
    Run a warmup pass to trigger actual compilation.
    
    This is optional but recommended for:
    - First-time runs (populates the cache)
    - Benchmarking (ensures compiled model is ready)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Running warmup pass to trigger compilation...")
        print("(This may take a few minutes on first run)")
        print(f"{'='*60}\n")
    
    warmup_start = time.time()
    
    # Use simple inputs for warmup
    dummy_input = {
        "tags": "pop, upbeat, electronic, dance",
        "lyrics": "la la la warmup test audio generation"
    }
    
    # Create temp output path
    import tempfile
    warmup_output = os.path.join(tempfile.gettempdir(), "heartlib_warmup.mp3")
    
    with torch.no_grad():
        pipe(
            dummy_input,
            max_audio_length_ms=warmup_length_ms,
            save_path=warmup_output,
            cfg_scale=1.0,  # No CFG for faster warmup
            topk=50,
            temperature=1.0,
        )
    
    # Clean up warmup file
    if os.path.exists(warmup_output):
        os.remove(warmup_output)
    
    warmup_time = time.time() - warmup_start
    if verbose:
        print(f"\nWarmup complete in {warmup_time:.2f}s")
        print("Compiled model is now cached for fast subsequent runs.\n")
    
    return warmup_time


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("HeartMuLa Turbo - Optimized Music Generation")
    print(f"{'='*60}")
    print(f"Model: {args.version}")
    print(f"Compile: {args.compile} (mode: {args.compile_mode})")
    print(f"Warmup: {args.warmup}")
    print(f"{'='*60}\n")
    
    # Load pipeline
    print("Loading models...")
    load_start = time.time()
    
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device={
            "mula": torch.device(args.mula_device),
            "codec": torch.device(args.codec_device),
        },
        dtype={
            "mula": args.mula_dtype,
            "codec": args.codec_dtype,
        },
        version=args.version,
        lazy_load=False,  # Must load models to compile them
    )
    print(f"Models loaded in {time.time() - load_start:.2f}s")
    
    # Apply torch.compile if enabled
    if args.compile:
        pipe = compile_pipeline(pipe, compile_mode=args.compile_mode)
        
        # Optional warmup to trigger compilation
        if args.warmup:
            warmup_compiled_model(pipe, warmup_length_ms=args.warmup_length_ms)
    
    # Run actual inference
    print(f"\n{'='*60}")
    print("Generating music...")
    print(f"  Lyrics: {args.lyrics}")
    print(f"  Tags: {args.tags}")
    print(f"  Max length: {args.max_audio_length_ms / 1000:.1f}s")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"{'='*60}\n")
    
    inference_start = time.time()
    
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    
    inference_time = time.time() - inference_start
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Output: {args.save_path}")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
