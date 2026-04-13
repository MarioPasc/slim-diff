# TASK 01 — Decoupled-Bottleneck U-Net Variant

## Context

SLIM-Diff is a compact joint diffusion model for synthesizing paired FLAIR MRI slices and lesion masks. The model uses MONAI's `DiffusionModelUNet` with a **shared bottleneck**: both image and mask channels pass through a single bottleneck block at the deepest level (256 channels, 20×20 spatial resolution). This shared bottleneck is the paper's central architectural inductive bias — it forces the model to learn a joint representation that regularises against memorisation in the low-data regime (N=78 patients).

For the ICIP 2026 camera-ready revision, we committed to a **decoupled-bottleneck ablation**: a parameter-matched variant (same total ~26.9M parameters) where the bottleneck has **independent paths for image-related and mask-related features**. This serves simultaneously as (a) the missing baseline comparison, (b) the ablation isolating the shared-bottleneck contribution, and (c) within-framework comparison context.

## Your Deliverable

Implement a `DecoupledDiffusionModelUNet` class and integrate it into the model factory so that it can be selected via YAML configuration. Everything else in the pipeline (training, generation, evaluation) must work with the new variant without modification.

## Files You Own (create or modify)

- `src/diffusion/model/decoupled_unet.py` — **CREATE**: new module with the decoupled variant
- `src/diffusion/model/factory.py` — **MODIFY**: add `bottleneck_mode` support to `build_model()`
- `src/diffusion/model/__init__.py` — **MODIFY**: export new class
- `src/diffusion/tests/test_decoupled_unet.py` — **CREATE**: unit tests

## Files You Must NOT Modify

- `src/diffusion/training/lit_modules.py`
- `src/diffusion/training/runners/*.py`
- `src/diffusion/data/*`
- `src/diffusion/losses/*`

## Architecture Specification

### Current Architecture (shared bottleneck)

The U-Net is built via MONAI's `DiffusionModelUNet` with:

```yaml
model:
  spatial_dims: 2
  in_channels: 2   # [FLAIR, mask]
  out_channels: 2  # [FLAIR, mask]
  channels: [64, 128, 256, 256]
  attention_levels: [false, false, true, true]
  num_res_blocks: 2
  num_head_channels: 32
  norm_num_groups: 32
```

MONAI's `DiffusionModelUNet` internally has:
- `self.input_blocks`: encoder (ModuleList)
- `self.middle_block`: bottleneck at 256 channels, 20×20 spatial
- `self.output_blocks`: decoder (ModuleList)

The `middle_block` typically consists of: ResBlock → AttentionBlock → ResBlock, all at 256 channels.

### Decoupled Architecture

Replace the single `middle_block` with two **independent** processing paths:

1. Project from 256 channels to two halves via 1×1 convolution: `proj_split: Conv2d(256, 256, 1)` that splits into two 128-channel tensors.
2. Path A: processes channels [0:128] through its own ResBlock(128) → GroupNorm → SiLU → Conv(128,128,3) → ... (mirroring the original ResBlock structure)
3. Path B: processes channels [128:256] independently with the same architecture
4. Concatenate outputs back to 256 channels
5. Project back: `proj_merge: Conv2d(256, 256, 1)` for the decoder

The key constraint: **the conditioning embedding e = t_emb + c_emb must be injected identically into both paths** (same as the original middle_block does with its ResBlocks). This means each independent ResBlock must receive the same `emb` tensor.

### Parameter Matching Strategy

This is critical. The total parameter count of the decoupled variant must be within 1% of the shared variant (26.9M ± ~270K). Strategy:

1. Build the shared model, count parameters: `sum(p.numel() for p in model.parameters())`
2. Build the decoupled model with half-channel bottleneck paths
3. Compare counts. The split projection layers (1×1 convs) add a small overhead. Compensate by:
   - Reducing the number of groups in GroupNorm of the bottleneck paths
   - Or adjusting the MLP conditioning dimension slightly
4. If exact matching is infeasible, accept ≤1% deviation and document the exact counts

### Implementation Approach

The cleanest approach is **post-construction surgery** on the MONAI model:

```python
class DecoupledDiffusionModelUNet(nn.Module):
    """DiffusionModelUNet with decoupled (independent) bottleneck paths.
    
    Replaces the shared middle_block with two independent processing
    paths that split and merge features at the bottleneck level.
    """
    
    def __init__(self, base_model: DiffusionModelUNet, bottleneck_channels: int = 256):
        super().__init__()
        # Steal all submodules from the base model
        self.input_blocks = base_model.input_blocks
        self.output_blocks = base_model.output_blocks
        # ... (copy other attributes)
        
        # Replace middle_block with decoupled version
        self.middle_block = DecoupledMiddleBlock(
            original_middle_block=base_model.middle_block,
            channels=bottleneck_channels,
        )
    
    def forward(self, x, timesteps, context=None, class_labels=None):
        # Must replicate MONAI's forward pass exactly,
        # only the middle_block call differs
        ...
```

**Alternative approach** (simpler, recommended): build the standard `DiffusionModelUNet` and then do:

```python
model = DiffusionModelUNet(...)
original_middle = model.middle_block
model.middle_block = DecoupledMiddleBlock(original_middle, channels=256)
```

This preserves MONAI's forward pass logic. Verify that MONAI's `forward()` calls `self.middle_block(h, emb)` with the embedding tensor.

**Important**: Inspect MONAI's source code (`monai.networks.nets.diffusion_model_unet`) to confirm:
1. How `middle_block` is called in the forward pass
2. What arguments it receives (h, emb, context?)
3. Whether there are residual connections around the middle_block

## Config Interface

Add to the YAML config under `model:`:

```yaml
model:
  bottleneck_mode: "shared"  # "shared" (default, current behavior) or "decoupled"
```

The `build_model()` function in `factory.py` should:
1. Build the standard `DiffusionModelUNet` as before
2. If `bottleneck_mode == "decoupled"`, wrap/modify the middle_block
3. Log the parameter counts of both configurations

## Acceptance Criteria (Testable)

All tests must pass on a machine with 8GB VRAM or CPU-only.

### Test 1: Forward Pass Equivalence
```python
def test_forward_pass_shapes():
    """Both variants produce identical output shapes."""
    # Build both models with small config for testing
    cfg = small_test_config()  # channels=[16, 32, 64, 64], 160x160
    model_shared = build_model_variant(cfg, "shared")
    model_decoupled = build_model_variant(cfg, "decoupled")
    
    x = torch.randn(2, 2, 160, 160)
    t = torch.randint(0, 1000, (2,))
    labels = torch.randint(0, 60, (2,))
    
    out_s = model_shared(x, timesteps=t, class_labels=labels)
    out_d = model_decoupled(x, timesteps=t, class_labels=labels)
    
    assert out_s.shape == out_d.shape == (2, 2, 160, 160)
```

### Test 2: Parameter Count Matching
```python
def test_parameter_count_matching():
    """Total parameters differ by <1%."""
    cfg = full_config()  # channels=[64, 128, 256, 256]
    model_shared = build_model_variant(cfg, "shared")
    model_decoupled = build_model_variant(cfg, "decoupled")
    
    n_shared = sum(p.numel() for p in model_shared.parameters())
    n_decoupled = sum(p.numel() for p in model_decoupled.parameters())
    
    ratio = abs(n_shared - n_decoupled) / n_shared
    assert ratio < 0.01, f"Parameter mismatch: {n_shared} vs {n_decoupled} ({ratio:.2%})"
    
    # Log exact counts for the paper
    print(f"Shared: {n_shared:,} | Decoupled: {n_decoupled:,} | Δ: {ratio:.4%}")
```

### Test 3: Gradient Flow
```python
def test_gradient_flow_decoupled():
    """Gradients flow through both independent paths."""
    cfg = small_test_config()
    model = build_model_variant(cfg, "decoupled")
    
    x = torch.randn(1, 2, 160, 160, requires_grad=True)
    t = torch.randint(0, 1000, (1,))
    labels = torch.randint(0, 60, (1,))
    
    out = model(x, timesteps=t, class_labels=labels)
    loss = out.sum()
    loss.backward()
    
    # Check that both paths received gradients
    for name, param in model.named_parameters():
        if "middle_block" in name and param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
```

### Test 4: Independence Verification
```python
def test_paths_are_independent():
    """The two bottleneck paths do not share parameters."""
    cfg = small_test_config()
    model = build_model_variant(cfg, "decoupled")
    
    path_a_params = set()
    path_b_params = set()
    for name, param in model.named_parameters():
        if "path_a" in name or "branch_0" in name:
            path_a_params.add(id(param))
        elif "path_b" in name or "branch_1" in name:
            path_b_params.add(id(param))
    
    assert len(path_a_params) > 0, "No path_a parameters found"
    assert len(path_b_params) > 0, "No path_b parameters found"
    assert path_a_params.isdisjoint(path_b_params), "Paths share parameters!"
```

### Test 5: Config Backward Compatibility
```python
def test_default_config_unchanged():
    """Without bottleneck_mode in config, behavior is identical to current."""
    cfg = load_existing_config()  # loads jsddpm.yaml
    model, encoder = build_model(cfg)
    # Should work exactly as before — no bottleneck_mode key means shared
    assert isinstance(model, DiffusionModelUNet)
```

### Test 6: Checkpoint Compatibility
```python
def test_shared_checkpoint_loads():
    """Existing shared-bottleneck checkpoints load into the shared variant."""
    # This ensures we didn't break the existing model
    cfg = full_config()
    cfg.model.bottleneck_mode = "shared"
    model, _ = build_model(cfg)
    # Should have the same state_dict keys as before
```

## Anti-Patterns — Do NOT:

- Do NOT create a second copy of the entire U-Net. The encoder, decoder, skip connections, and conditioning must be shared.
- Do NOT change the forward signature. The model must accept the same arguments as `DiffusionModelUNet.forward()`.
- Do NOT modify the conditioning mechanism (AdaGN/class embeddings). Both variants receive the same conditioning.
- Do NOT change any training hyperparameters or loss functions.
- Do NOT add new dependencies beyond what's already in `pyproject.toml`.

## Useful References

- MONAI source: `monai.networks.nets.diffusion_model_unet.DiffusionModelUNet`
- The `middle_block` in MONAI is typically a sequential of `ResBlock, AttentionBlock, ResBlock`
- The paper's channel progression: `[64, 128, 256, 256]` with attention at levels 2 and 3
- Total reported parameter count: **26.9M**
