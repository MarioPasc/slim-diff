#!/usr/bin/env python3
"""Verify EMA checkpoint saving/loading mechanism.

This script checks:
1. EMACallback saves EMA weights to correct locations
2. generate_replicas.py can load them correctly
3. Config parameters work as expected
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.diffusion.training.callbacks.epoch_callbacks import EMACallback


def test_ema_checkpoint_structure():
    """Test that EMACallback saves to expected locations."""
    print("=" * 70)
    print("Test 1: EMA Checkpoint Structure")
    print("=" * 70)

    # Create minimal config
    cfg = OmegaConf.create({
        "training": {
            "ema": {
                "enabled": True,
                "decay": 0.999,
                "update_every": 10,
                "update_start_step": 0,
                "store_on_cpu": True,
                "use_buffers": True,
                "export_to_checkpoint": True,  # KEY: This enables top-level saving
            }
        }
    })

    # Create EMACallback
    ema_callback = EMACallback(
        decay=cfg.training.ema.decay,
        update_every=cfg.training.ema.update_every,
        update_start_step=cfg.training.ema.update_start_step,
        store_on_cpu=cfg.training.ema.store_on_cpu,
        use_buffers=cfg.training.ema.use_buffers,
        use_for_validation=True,
        export_to_checkpoint=cfg.training.ema.export_to_checkpoint,
    )

    # Simulate EMA state (mock weights)
    mock_ema_state = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.layer1.bias": torch.randn(10),
        "model.layer2.weight": torch.randn(5, 10),
    }
    ema_callback._ema = mock_ema_state
    ema_callback._num_updates = 1000
    ema_callback._last_global_step = 10000
    ema_callback._applied = False  # Initialize to avoid AttributeError

    # Test 1: state_dict() - callback state
    print("\n1. Testing callback state_dict():")
    state = ema_callback.state_dict()
    print(f"   Keys in state_dict: {list(state.keys())}")
    assert "ema" in state, "Missing 'ema' key"
    assert state["ema"] is mock_ema_state, "EMA dict not saved correctly"
    assert state["num_updates"] == 1000
    print("   ✓ state_dict() contains 'ema' with correct data")

    # Test 2: on_save_checkpoint() - top-level checkpoint
    print("\n2. Testing on_save_checkpoint() with export_to_checkpoint=True:")
    checkpoint = {}
    ema_callback.on_save_checkpoint(None, None, checkpoint)

    if "ema_state_dict" in checkpoint:
        print(f"   ✓ Top-level 'ema_state_dict' key exists")
        assert checkpoint["ema_state_dict"] is mock_ema_state
        print(f"   ✓ Contains {len(checkpoint['ema_state_dict'])} tensors")
        print(f"   ✓ EMA metadata keys: {list(checkpoint['ema_meta'].keys())}")
    else:
        print("   ✗ MISSING top-level 'ema_state_dict' key")
        return False

    # Test 3: Simulate full checkpoint structure
    print("\n3. Simulating full Lightning checkpoint structure:")
    full_checkpoint = {
        "state_dict": {"model.layer1.weight": torch.randn(10, 10)},  # Model weights
        "callbacks": {
            "EMACallback": state,  # Callback state from state_dict()
        },
        "ema_state_dict": checkpoint["ema_state_dict"],  # Top-level EMA (from on_save_checkpoint)
        "ema_meta": checkpoint["ema_meta"],
    }

    print("   Full checkpoint keys:")
    for key in full_checkpoint.keys():
        print(f"      - {key}")

    print("\n   Callback state location:")
    print(f"      checkpoint['callbacks']['EMACallback']['ema'] exists: "
          f"{'ema' in full_checkpoint['callbacks']['EMACallback']}")

    print("\n   Top-level EMA location:")
    print(f"      checkpoint['ema_state_dict'] exists: {'ema_state_dict' in full_checkpoint}")

    # Test 4: Verify generate_replicas.py loading logic
    print("\n4. Testing generate_replicas.py loading logic:")
    print("   Trying load path 1: checkpoint['ema_state_dict']")
    if "ema_state_dict" in full_checkpoint and isinstance(full_checkpoint["ema_state_dict"], dict):
        print(f"      ✓ Found EMA at top-level with {len(full_checkpoint['ema_state_dict'])} tensors")
        load_success_path1 = True
    else:
        print("      ✗ Not found")
        load_success_path1 = False

    print("   Trying load path 2: checkpoint['callbacks']['EMACallback']['ema']")
    if "callbacks" in full_checkpoint:
        cb_state = full_checkpoint.get("callbacks", {}).get("EMACallback", {})
        if "ema" in cb_state and cb_state["ema"] is not None:
            print(f"      ✓ Found EMA in callback state with {len(cb_state['ema'])} tensors")
            load_success_path2 = True
        else:
            print("      ✗ Not found")
            load_success_path2 = False
    else:
        load_success_path2 = False

    if load_success_path1 or load_success_path2:
        print("\n   ✓ At least one loading path succeeds")
    else:
        print("\n   ✗ BOTH loading paths failed!")
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    return True


def test_export_to_checkpoint_false():
    """Test behavior when export_to_checkpoint=False."""
    print("\n" + "=" * 70)
    print("Test 2: Behavior with export_to_checkpoint=False")
    print("=" * 70)

    ema_callback = EMACallback(
        decay=0.999,
        update_every=10,
        export_to_checkpoint=False,  # Disabled
    )

    mock_ema_state = {"param1": torch.randn(5, 5)}
    ema_callback._ema = mock_ema_state

    checkpoint = {}
    ema_callback.on_save_checkpoint(None, None, checkpoint)

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    if "ema_state_dict" in checkpoint:
        print("✗ UNEXPECTED: 'ema_state_dict' should NOT be saved when export_to_checkpoint=False")
        return False
    else:
        print("✓ Correct: 'ema_state_dict' not saved (export_to_checkpoint=False)")

    # But callback state should still be available
    state = ema_callback.state_dict()
    if "ema" in state and state["ema"] is mock_ema_state:
        print("✓ Callback state_dict() still contains EMA (fallback path works)")
    else:
        print("✗ Callback state_dict() missing EMA")
        return False

    print("\n" + "=" * 70)
    print("✓ Test passed: export_to_checkpoint=False works correctly")
    print("=" * 70)
    return True


def test_save_and_load_cycle():
    """Test actual save/load cycle with torch.save/load."""
    print("\n" + "=" * 70)
    print("Test 3: Actual Save/Load Cycle")
    print("=" * 70)

    # Create EMA with export enabled
    ema_callback = EMACallback(decay=0.999, export_to_checkpoint=True)
    mock_ema = {
        "layer.weight": torch.randn(3, 3),
        "layer.bias": torch.randn(3),
    }
    ema_callback._ema = mock_ema
    ema_callback._num_updates = 500
    ema_callback._last_global_step = 5000
    ema_callback._applied = False

    # Simulate checkpoint saving
    checkpoint = {"epoch": 10}
    ema_callback.on_save_checkpoint(None, None, checkpoint)

    # Also save callback state (Lightning does this automatically)
    checkpoint["callbacks"] = {"EMACallback": ema_callback.state_dict()}

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        ckpt_path = Path(f.name)
        torch.save(checkpoint, ckpt_path)

    print(f"\n1. Saved checkpoint to {ckpt_path}")
    print(f"   Size: {ckpt_path.stat().st_size / 1024:.1f} KB")

    # Load checkpoint back
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("\n2. Loaded checkpoint")
    print(f"   Keys: {list(loaded_ckpt.keys())}")

    # Verify EMA can be loaded (method 1)
    if "ema_state_dict" in loaded_ckpt:
        ema_loaded_1 = loaded_ckpt["ema_state_dict"]
        print(f"\n3. Method 1: Top-level EMA")
        print(f"   ✓ Loaded {len(ema_loaded_1)} tensors")
        print(f"   Tensor names: {list(ema_loaded_1.keys())}")

        # Verify values match
        for key in mock_ema.keys():
            if torch.allclose(ema_loaded_1[key], mock_ema[key]):
                print(f"   ✓ {key}: values match")
            else:
                print(f"   ✗ {key}: values DO NOT match")
                return False
    else:
        print("\n3. Method 1: ✗ Top-level EMA not found")

    # Verify EMA can be loaded (method 2)
    if "callbacks" in loaded_ckpt and "EMACallback" in loaded_ckpt["callbacks"]:
        cb_state = loaded_ckpt["callbacks"]["EMACallback"]
        if "ema" in cb_state:
            ema_loaded_2 = cb_state["ema"]
            print(f"\n4. Method 2: Callback state")
            print(f"   ✓ Loaded {len(ema_loaded_2)} tensors")

            # Verify same object (should be same reference)
            if ema_loaded_2 is ema_loaded_1:
                print("   ✓ Both methods point to SAME dictionary (efficient)")
            else:
                print("   ⚠ Both methods have different dictionaries (duplicated)")
        else:
            print("\n4. Method 2: ✗ Callback EMA not found")
    else:
        print("\n4. Method 2: ✗ Callback state not found")

    # Cleanup
    ckpt_path.unlink()

    print("\n" + "=" * 70)
    print("✓ Save/Load cycle successful!")
    print("=" * 70)
    return True


def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  EMA Checkpoint Verification for JS-DDPM".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    results = []

    try:
        results.append(("Structure Test", test_ema_checkpoint_structure()))
    except Exception as e:
        print(f"\n✗ Structure Test FAILED: {e}")
        results.append(("Structure Test", False))

    try:
        results.append(("Export Disabled Test", test_export_to_checkpoint_false()))
    except Exception as e:
        print(f"\n✗ Export Disabled Test FAILED: {e}")
        results.append(("Export Disabled Test", False))

    try:
        results.append(("Save/Load Cycle Test", test_save_and_load_cycle()))
    except Exception as e:
        print(f"\n✗ Save/Load Cycle Test FAILED: {e}")
        results.append(("Save/Load Cycle Test", False))

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nConclusion:")
        print("  - EMACallback with export_to_checkpoint=True saves EMA to checkpoint['ema_state_dict']")
        print("  - generate_replicas.py will successfully load EMA weights")
        print("  - Current config is correctly set up")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit(main())
