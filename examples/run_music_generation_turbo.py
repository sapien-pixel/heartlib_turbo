"""
HeartMuLa Turbo - Optimized Music Generation with Multiple Optimizations

Optimizations applied:
1. torch.compile() with persistent caching
2. Flash Attention (enabled by default on A100/H100)
3. Reduced flow matching steps (configurable)
4. Optional CFG disable for 2x speedup
5. Disable progress bars for reduced overhead
6. Memory-efficient attention backends

First run: Slow (~2-5 minutes) as models are compiled
Subsequent runs: Fast as compiled artifacts are loaded from cache
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
import torch.nn.functional as F
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
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default recommended since CUDA graphs fail)")
    parser.add_argument("--warmup", type=str2bool, default=False,
                        help="Run warmup pass to trigger compilation")
    parser.add_argument("--warmup_length_ms", type=int, default=5000,
                        help="Warmup audio length in milliseconds")
    
    # NEW: Additional optimizations
    parser.add_argument("--flow_steps", type=int, default=10,
                        help="Number of flow matching steps (fewer = faster, 5-10 recommended)")
    parser.add_argument("--disable_progress", type=str2bool, default=True,
                        help="Disable tqdm progress bars for reduced overhead")
    parser.add_argument("--flash_attention", type=str2bool, default=True,
                        help="Enable Flash Attention if available")
    
    return parser.parse_args()


def enable_flash_attention(verbose=True):
    """Enable Flash Attention and memory-efficient attention backends."""
    if verbose:
        print("\nConfiguring attention backends:")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        if verbose:
            print("  ⚠️  CUDA not available, skipping attention optimization")
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    
    if verbose:
        print(f"  GPU: {gpu_name}")
        print(f"  Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    # Enable Flash Attention (requires compute capability >= 8.0 for best performance)
    try:
        # Enable Flash SDP (Scaled Dot Product) attention
        torch.backends.cuda.enable_flash_sdp(True)
        if verbose:
            print("  ✓ Flash Attention (Flash SDP) enabled")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Flash SDP not available: {e}")
    
    try:
        # Enable memory-efficient attention as fallback
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if verbose:
            print("  ✓ Memory-efficient attention enabled")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Memory-efficient SDP not available: {e}")
    
    # Enable math SDP as final fallback
    try:
        torch.backends.cuda.enable_math_sdp(True)
        if verbose:
            print("  ✓ Math SDP fallback enabled")
    except Exception:
        pass
    
    # Additional CUDA optimizations
    try:
        torch.backends.cudnn.benchmark = True
        if verbose:
            print("  ✓ cuDNN benchmark mode enabled")
    except Exception:
        pass
    
    return


def compile_pipeline(pipe, compile_mode="default", verbose=True):
    """
    Compile the pipeline components with persistent caching.
    
    NOTE: HeartCodec flow_matching.estimator is SKIPPED due to Inductor
    compatibility issues with positional encoding (NaN bounds checking bug).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Compiling pipeline with mode: {compile_mode}")
        print(f"Cache directory: {CACHE_DIR}")
        print(f"{'='*60}\n")
    
    compile_start = time.time()
    
    # Access models to ensure they're loaded
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
            fullgraph=False,
        )
        compiled_count += 1
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Failed to compile backbone: {e}")
    
    # Compile HeartMuLa decoder
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
    
    # SKIP flow_matching.estimator (Inductor incompatible)
    if verbose:
        print("[3/3] Skipping flow_matching.estimator (Inductor incompatible)")
    
    # Compile scalar model
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
            print(f"      ⚠️  Skipped: {e}")
    
    if verbose:
        print(f"\nCompilation setup complete in {time.time() - compile_start:.2f}s")
        print(f"Successfully compiled {compiled_count} components\n")
    
    return pipe


def optimized_postprocess(pipe, model_outputs, save_path, flow_steps=10, 
                          disable_progress=True, verbose=True):
    """
    Optimized postprocessing with configurable flow matching steps.
    
    Default is 10 steps. Reducing to 5-7 can give ~50% speedup on codec
    with minimal quality loss.
    """
    import torchaudio
    
    frames = model_outputs["frames"].to(pipe.codec_device)
    
    if verbose:
        print(f"  Flow matching steps: {flow_steps}")
    
    # Call detokenize with custom num_steps
    wav = pipe.codec.detokenize(
        frames, 
        num_steps=flow_steps,
        disable_progress=disable_progress,
    )
    
    pipe._unload()
    torchaudio.save(save_path, wav.to(torch.float32).cpu(), 48000)


def optimized_forward(pipe, model_inputs, max_audio_length_ms, temperature, 
                      topk, cfg_scale, disable_progress=True):
    """
    Optimized forward pass with optional progress bar disable.
    """
    from tqdm import tqdm
    
    prompt_tokens = model_inputs["tokens"].to(pipe.mula_device)
    prompt_tokens_mask = model_inputs["tokens_mask"].to(pipe.mula_device)
    continuous_segment = model_inputs["muq_embed"].to(pipe.mula_device)
    starts = model_inputs["muq_idx"]
    prompt_pos = model_inputs["pos"].to(pipe.mula_device)
    frames = []

    bs_size = 2 if cfg_scale != 1.0 else 1
    pipe.mula.setup_caches(bs_size)
    
    with torch.autocast(device_type=pipe.mula_device.type, dtype=pipe.mula_dtype):
        curr_token = pipe.mula.generate_frame(
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
            continuous_segments=continuous_segment,
            starts=starts,
        )
    frames.append(curr_token[0:1,])

    def _pad_audio_token(token):
        padded_token = (
            torch.ones(
                (token.shape[0], pipe._parallel_number),
                device=token.device,
                dtype=torch.long,
            )
            * pipe.config.empty_id
        )
        padded_token[:, :-1] = token
        padded_token = padded_token.unsqueeze(1)
        padded_token_mask = torch.ones_like(
            padded_token, device=token.device, dtype=torch.bool
        )
        padded_token_mask[..., -1] = False
        return padded_token, padded_token_mask

    max_audio_frames = max_audio_length_ms // 80
    
    # Use tqdm or simple range based on disable_progress
    iterator = range(max_audio_frames)
    if not disable_progress:
        iterator = tqdm(iterator)

    for i in iterator:
        curr_token, curr_token_mask = _pad_audio_token(curr_token)
        with torch.autocast(
            device_type=pipe.mula_device.type, dtype=pipe.mula_dtype
        ):
            curr_token = pipe.mula.generate_frame(
                tokens=curr_token,
                tokens_mask=curr_token_mask,
                input_pos=prompt_pos[..., -1:] + i + 1,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=None,
                starts=None,
            )
        if torch.any(curr_token[0:1, :] >= pipe.config.audio_eos_id):
            break
        frames.append(curr_token[0:1,])
    
    frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
    pipe._unload()
    return {"frames": frames}


def warmup_compiled_model(pipe, warmup_length_ms=5000, flow_steps=10, verbose=True):
    """Run a warmup pass to trigger compilation."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running warmup pass to trigger compilation...")
        print(f"{'='*60}\n")
    
    warmup_start = time.time()
    
    dummy_input = {
        "tags": "pop, upbeat, electronic, dance",
        "lyrics": "la la la warmup test audio generation"
    }
    
    import tempfile
    warmup_output = os.path.join(tempfile.gettempdir(), "heartlib_warmup.mp3")
    
    with torch.no_grad():
        # Preprocess
        preprocess_kwargs = {"cfg_scale": 1.0}
        model_inputs = pipe.preprocess(dummy_input, **preprocess_kwargs)
        
        # Forward with optimized function
        model_outputs = optimized_forward(
            pipe, model_inputs, warmup_length_ms, 
            temperature=1.0, topk=50, cfg_scale=1.0,
            disable_progress=True
        )
        
        # Postprocess with optimized function
        optimized_postprocess(
            pipe, model_outputs, warmup_output,
            flow_steps=flow_steps, disable_progress=True,
            verbose=False
        )
    
    if os.path.exists(warmup_output):
        os.remove(warmup_output)
    
    if verbose:
        print(f"\nWarmup complete in {time.time() - warmup_start:.2f}s\n")


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("HeartMuLa Turbo - Optimized Music Generation")
    print(f"{'='*60}")
    print(f"Model: {args.version}")
    print(f"Compile: {args.compile} (mode: {args.compile_mode})")
    print(f"CFG Scale: {args.cfg_scale} {'(2x faster!)' if args.cfg_scale == 1.0 else ''}")
    print(f"Flow Steps: {args.flow_steps} (default: 10)")
    print(f"Flash Attention: {args.flash_attention}")
    print(f"Progress Bars: {'disabled' if args.disable_progress else 'enabled'}")
    print(f"{'='*60}\n")
    
    # Enable Flash Attention
    if args.flash_attention:
        enable_flash_attention(verbose=True)
    
    # Load pipeline
    print("\nLoading models...")
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
        lazy_load=False,
    )
    print(f"Models loaded in {time.time() - load_start:.2f}s")
    
    # Apply torch.compile if enabled
    if args.compile:
        pipe = compile_pipeline(pipe, compile_mode=args.compile_mode)
        
        if args.warmup:
            warmup_compiled_model(
                pipe, 
                warmup_length_ms=args.warmup_length_ms,
                flow_steps=args.flow_steps
            )
    
    # Run inference
    print(f"\n{'='*60}")
    print("Generating music...")
    print(f"  Lyrics: {args.lyrics}")
    print(f"  Tags: {args.tags}")
    print(f"  Max length: {args.max_audio_length_ms / 1000:.1f}s")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Flow steps: {args.flow_steps}")
    print(f"{'='*60}\n")
    
    inference_start = time.time()
    
    with torch.no_grad():
        # Preprocess
        preprocess_kwargs = {"cfg_scale": args.cfg_scale}
        model_inputs = pipe.preprocess(
            {"lyrics": args.lyrics, "tags": args.tags}, 
            **preprocess_kwargs
        )
        
        # Forward with optimized function
        forward_start = time.time()
        model_outputs = optimized_forward(
            pipe, model_inputs, args.max_audio_length_ms,
            temperature=args.temperature, topk=args.topk, 
            cfg_scale=args.cfg_scale,
            disable_progress=args.disable_progress
        )
        forward_time = time.time() - forward_start
        
        # Postprocess with optimized function
        postprocess_start = time.time()
        optimized_postprocess(
            pipe, model_outputs, args.save_path,
            flow_steps=args.flow_steps,
            disable_progress=args.disable_progress,
            verbose=True
        )
        postprocess_time = time.time() - postprocess_start
    
    inference_time = time.time() - inference_start
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Output: {args.save_path}")
    print(f"  HeartMuLa (token gen): {forward_time:.2f}s")
    print(f"  HeartCodec (audio):    {postprocess_time:.2f}s")
    print(f"  Total inference time:  {inference_time:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
