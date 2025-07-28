"""
token_reduction_example.py

Example script demonstrating how to apply training-free token reduction 
to OpenVLA models for improved efficiency.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add the project root to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.robot.token_reduction_utils import (
    apply_token_reduction_to_vla,
    monkey_patch_vision_backbone_with_reduction,
    benchmark_token_reduction,
    apply_50_percent_topk_reduction,
    apply_spatial_pooling_reduction,
)


def main():
    """Main example function."""
    
    # Configuration
    MODEL_PATH = "openvla/openvla-7b"  # Or your local path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading OpenVLA model from {MODEL_PATH}...")
    
    # Load model and processor
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Create sample input
    print("Creating sample input...")
    sample_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    prompt = "What action should the robot take to pick up the object?"
    
    # Prepare inputs
    inputs = processor(prompt, sample_image).to(DEVICE, dtype=torch.bfloat16)
    
    print(f"Original number of patches: {vla.model.vision_backbone.get_num_patches()}")
    
    # Example 1: Apply 50% token reduction using top-k magnitude
    print("\n=== Example 1: 50% Top-K Magnitude Reduction ===")
    vla_reduced = apply_50_percent_topk_reduction(vla)
    print(f"Reduced number of patches: {vla_reduced.model.vision_backbone.get_num_patches()}")
    
    # Test inference
    with torch.no_grad():
        output = vla_reduced.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"Generated response: {response}")
    
    # Example 2: Apply spatial pooling reduction
    print("\n=== Example 2: Spatial Pooling (2x2) Reduction ===")
    # First restore original model (create fresh copy)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    
    vla_spatial = apply_spatial_pooling_reduction(vla, pool_size=2)
    print(f"Spatial pooling patches: {vla_spatial.model.vision_backbone.get_num_patches()}")
    
    # Example 3: Custom configuration
    print("\n=== Example 3: Custom Configuration ===")
    # Restore original model
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    
    vla_custom = apply_token_reduction_to_vla(
        vla,
        method="attention_score",
        reduction_ratio=0.25,  # Keep only 25% of tokens
        keep_cls_token=True,
    )
    print(f"Custom reduction patches: {vla_custom.model.vision_backbone.get_num_patches()}")
    
    # Example 4: Quick monkey-patch for experimentation
    print("\n=== Example 4: Quick Monkey-Patch ===")
    # Restore original model
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    
    original_patches = vla.model.vision_backbone.get_num_patches()
    vla_patched = monkey_patch_vision_backbone_with_reduction(
        vla, method="topk_magnitude", reduction_ratio=0.75
    )
    patched_patches = vla_patched.model.vision_backbone.get_num_patches()
    
    print(f"Original: {original_patches} patches")
    print(f"Patched: {patched_patches} patches")
    print(f"Reduction ratio: {patched_patches/original_patches:.2f}")
    
    # Example 5: Benchmark different methods (optional - can be slow)
    print("\n=== Example 5: Benchmarking (Optional) ===")
    benchmark_choice = input("Run benchmark? (y/n): ").lower().strip()
    
    if benchmark_choice == 'y':
        # Restore original model for fair comparison
        vla = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        
        print("Running benchmark (this may take a while)...")
        results = benchmark_token_reduction(
            vla,
            inputs,
            methods=["topk_magnitude", "spatial_pool"],
            ratios=[0.25, 0.5, 0.75]
        )
        
        print("\nBenchmark Results:")
        print(f"Baseline time: {results['baseline']['time']:.4f}s")
        print(f"Baseline patches: {results['baseline']['num_patches']}")
        
        for method in ["topk_magnitude", "spatial_pool"]:
            print(f"\n{method.upper()}:")
            for ratio in [0.25, 0.5, 0.75]:
                result = results[method][f"ratio_{ratio}"]
                print(f"  Ratio {ratio}: {result['time']:.4f}s "
                      f"(speedup: {result['speedup']:.2f}x, "
                      f"patches: {result['num_patches']})")
    
    print("\n=== Examples Complete ===")
    print("Token reduction has been successfully applied to OpenVLA!")
    print("\nNext steps:")
    print("1. Fine-tune with reduced tokens using vla-scripts/finetune.py")
    print("2. Evaluate performance on your specific tasks")
    print("3. Experiment with different reduction methods and ratios")


def demonstrate_fused_backbone():
    """Demonstrate token reduction with fused backbones (DINOv2 + SigLIP)."""
    print("\n=== Fused Backbone Example ===")
    
    # This would work with models that use fused backbones
    # For demonstration, we'll show the concept
    
    print("For fused backbones (DINOv2 + SigLIP):")
    print("- Token reduction is applied AFTER concatenation")
    print("- Both DINOv2 and SigLIP features are processed together")
    print("- The reducer sees patches of shape (B, N, 2*D)")
    print("- Reduction preserves the combined feature representation")
    
    # Example configuration for fused backbone
    fused_config = {
        "method": "topk_magnitude",
        "reduction_ratio": 0.5,
        "keep_cls_token": True,
    }
    
    print(f"Example config for fused backbone: {fused_config}")


if __name__ == "__main__":
    main()
    demonstrate_fused_backbone() 