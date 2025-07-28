"""
token_reduction_utils.py

Utility functions for applying token reduction to OpenVLA models.
Provides easy-to-use interfaces for different use cases.
"""

from typing import Dict, Optional, Union
import torch.nn as nn
from transformers import AutoModelForVision2Seq

from prismatic.models.backbones.vision.token_reducer import ReducedVisionBackbone


def apply_token_reduction_to_vla(
    vla: nn.Module,
    method: str = "topk_magnitude",
    reduction_ratio: float = 0.5,
    spatial_merge_size: int = 2,
    keep_cls_token: bool = True,
) -> nn.Module:
    """
    Apply token reduction to an OpenVLA model's vision backbone.
    
    Args:
        vla: OpenVLA model (either from HF or native)
        method: Reduction method ("topk_magnitude", "spatial_pool", "random", "attention_score")
        reduction_ratio: Fraction of tokens to keep (0.0 < ratio <= 1.0)
        spatial_merge_size: Size of spatial pooling window (for spatial_pool method)
        keep_cls_token: Whether to preserve CLS token if present
        
    Returns:
        Modified VLA model with token reduction applied
    """
    # Configuration for token reducer
    reducer_config = {
        "method": method,
        "reduction_ratio": reduction_ratio,
        "spatial_merge_size": spatial_merge_size,
        "keep_cls_token": keep_cls_token,
    }
    
    # Determine if this is HF model or native model and find vision backbone
    vision_backbone = None
    
    # Try different possible locations for vision backbone
    # For OpenVLAForActionPrediction (HF model), vision_backbone is a direct attribute
    if hasattr(vla, 'vision_backbone'):
        vision_backbone = vla.vision_backbone
        vla.vision_backbone = ReducedVisionBackbone(vision_backbone, reducer_config)
        
    elif hasattr(vla, 'model') and hasattr(vla.model, 'vision_backbone'):
        # Native OpenVLA model with .model attribute (some versions)
        vision_backbone = vla.model.vision_backbone
        vla.model.vision_backbone = ReducedVisionBackbone(vision_backbone, reducer_config)
        
    else:
        # Try to find other possible locations
        possible_attrs = [attr for attr in dir(vla) if 'vision' in attr.lower() or 'backbone' in attr.lower()]
        raise ValueError(
            f"Unable to locate vision_backbone in the provided model. "
            f"Model type: {type(vla)}, "
            f"Available vision/backbone attributes: {possible_attrs}, "
            f"All attributes: {[attr for attr in dir(vla) if not attr.startswith('_')]}"
        )
    
    return vla


def create_reduced_vision_backbone_config(
    base_vision_backbone_id: str,
    method: str = "topk_magnitude",
    reduction_ratio: float = 0.5,
    spatial_merge_size: int = 2,
    keep_cls_token: bool = True,
) -> Dict:
    """
    Create a configuration for a reduced vision backbone that can be used
    in model configs or registry entries.
    
    Args:
        base_vision_backbone_id: ID of the base vision backbone
        method: Reduction method
        reduction_ratio: Fraction of tokens to keep
        spatial_merge_size: Size of spatial pooling window
        keep_cls_token: Whether to preserve CLS token
        
    Returns:
        Configuration dictionary
    """
    return {
        "base_vision_backbone_id": base_vision_backbone_id,
        "reducer_config": {
            "method": method,
            "reduction_ratio": reduction_ratio,
            "spatial_merge_size": spatial_merge_size,
            "keep_cls_token": keep_cls_token,
        }
    }


def monkey_patch_vision_backbone_with_reduction(
    vla: nn.Module,
    method: str = "topk_magnitude", 
    reduction_ratio: float = 0.5,
) -> nn.Module:
    """
    Quick monkey-patch approach for experimentation.
    This modifies the forward method of the vision backbone in-place.
    
    Args:
        vla: OpenVLA model
        method: Reduction method
        reduction_ratio: Fraction of tokens to keep
        
    Returns:
        Modified VLA model
    """
    from prismatic.models.backbones.vision.token_reducer import TokenReducer
    
    # Get vision backbone - try different possible locations
    backbone = None
    # For OpenVLAForActionPrediction (HF model), vision_backbone is a direct attribute
    if hasattr(vla, 'vision_backbone'):
        backbone = vla.vision_backbone
    elif hasattr(vla, 'model') and hasattr(vla.model, 'vision_backbone'):
        backbone = vla.model.vision_backbone
    else:
        possible_attrs = [attr for attr in dir(vla) if 'vision' in attr.lower() or 'backbone' in attr.lower()]
        raise ValueError(
            f"Unable to locate vision_backbone in the provided model. "
            f"Model type: {type(vla)}, "
            f"Available vision/backbone attributes: {possible_attrs}"
        )
    
    # Create reducer
    reducer = TokenReducer(method=method, reduction_ratio=reduction_ratio)
    
    # Store original methods
    backbone._original_forward = backbone.forward
    backbone._original_get_num_patches = backbone.get_num_patches
    
    # Patch forward method
    def new_forward(pixel_values):
        patches = backbone._original_forward(pixel_values)
        return reducer(patches)
    
    # Patch get_num_patches method
    def new_get_num_patches():
        original_patches = backbone._original_get_num_patches()
        return int(original_patches * reducer.get_reduction_ratio_actual(original_patches))
    
    # Apply patches
    backbone.forward = new_forward
    backbone.get_num_patches = new_get_num_patches
    backbone._token_reducer = reducer  # Store for reference
    
    return vla


def benchmark_token_reduction(
    vla: nn.Module,
    sample_input: Dict,
    methods: Optional[list] = None,
    ratios: Optional[list] = None,
) -> Dict:
    """
    Benchmark different token reduction methods and ratios.
    
    Args:
        vla: OpenVLA model
        sample_input: Sample input dictionary with pixel_values and input_ids
        methods: List of methods to test (default: ["topk_magnitude", "spatial_pool"])
        ratios: List of reduction ratios to test (default: [0.25, 0.5, 0.75])
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    import torch
    
    if methods is None:
        methods = ["topk_magnitude", "spatial_pool", "attention_score"]
    if ratios is None:
        ratios = [0.25, 0.5, 0.75]
    
    results = {}
    
    # Get vision backbone for patch count
    def get_vision_backbone(model):
        # For OpenVLAForActionPrediction (HF model), vision_backbone is a direct attribute
        if hasattr(model, 'vision_backbone'):
            return model.vision_backbone
        elif hasattr(model, 'model') and hasattr(model.model, 'vision_backbone'):
            return model.model.vision_backbone
        else:
            possible_attrs = [attr for attr in dir(model) if 'vision' in attr.lower() or 'backbone' in attr.lower()]
            raise ValueError(
                f"Cannot find vision backbone in model. "
                f"Model type: {type(model)}, "
                f"Available vision/backbone attributes: {possible_attrs}"
            )
    
    # Baseline (no reduction)
    with torch.no_grad():
        start_time = time.time()
        baseline_output = vla(**sample_input)
        baseline_time = time.time() - start_time
        
    results["baseline"] = {
        "time": baseline_time,
        "num_patches": get_vision_backbone(vla).get_num_patches(),
        "output_shape": baseline_output.logits.shape if hasattr(baseline_output, 'logits') else None,
    }
    
    # Test different configurations
    for method in methods:
        results[method] = {}
        for ratio in ratios:
            # Create a copy for testing (or restore original)
            test_vla = monkey_patch_vision_backbone_with_reduction(vla, method, ratio)
            
            with torch.no_grad():
                start_time = time.time()
                output = test_vla(**sample_input)
                test_time = time.time() - start_time
                
            results[method][f"ratio_{ratio}"] = {
                "time": test_time,
                "speedup": baseline_time / test_time,
                "num_patches": get_vision_backbone(test_vla).get_num_patches(),
                "output_shape": output.logits.shape if hasattr(output, 'logits') else None,
            }
            
            # Restore original backbone for next test
            backbone = get_vision_backbone(test_vla)
            if hasattr(backbone, '_original_forward'):
                backbone.forward = backbone._original_forward
                backbone.get_num_patches = backbone._original_get_num_patches
                delattr(backbone, '_original_forward')
                delattr(backbone, '_original_get_num_patches')
                delattr(backbone, '_token_reducer')
    
    return results


# Example usage functions
def apply_50_percent_topk_reduction(vla: nn.Module) -> nn.Module:
    """Quick function to apply 50% token reduction using top-k magnitude."""
    return apply_token_reduction_to_vla(
        vla, 
        method="topk_magnitude", 
        reduction_ratio=0.5
    )


def apply_spatial_pooling_reduction(vla: nn.Module, pool_size: int = 2) -> nn.Module:
    """Quick function to apply spatial pooling reduction."""
    return apply_token_reduction_to_vla(
        vla,
        method="spatial_pool",
        reduction_ratio=1.0 / (pool_size ** 2),  # Automatic ratio based on pool size
        spatial_merge_size=pool_size
    ) 