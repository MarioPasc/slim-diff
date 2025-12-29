"""Test multi-component logic for z-bin priors.

Verifies that the first N bins correctly keep multiple components
while higher bins keep only the largest component.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.ndimage import label

from src.diffusion.utils.zbin_priors import load_zbin_priors


@pytest.mark.skipif(
    not Path("/media/mpascual/Sandisk2TB/research/epilepsy/data").exists(),
    reason="Data directory not found"
)
class TestMultiComponentPriors:
    """Test that multi-component logic is correctly applied."""

    @pytest.fixture
    def config(self):
        """Load configuration."""
        config_path = Path("slurm/jsddpm_baseline/jsddpm_baseline.yaml")
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")
        return OmegaConf.load(config_path)

    @pytest.fixture
    def cache_dir(self, config):
        """Get cache directory."""
        cache_dir = Path(config.data.cache_dir)
        if not cache_dir.exists():
            pytest.skip(f"Cache not found: {cache_dir}")
        return cache_dir

    @pytest.fixture
    def priors(self, cache_dir, config):
        """Load z-bin priors."""
        pp_cfg = config.postprocessing.zbin_priors
        if not pp_cfg.get("enabled", False):
            pytest.skip("Z-bin priors not enabled in config")

        priors_path = cache_dir / pp_cfg.priors_filename
        if not priors_path.exists():
            pytest.skip(
                f"Priors file not found: {priors_path}. "
                "Run cache builder with postprocessing.zbin_priors.enabled=true"
            )

        z_bins = config.conditioning.z_bins
        return load_zbin_priors(cache_dir, pp_cfg.priors_filename, z_bins)

    def test_multicomponent_count_verification(self, config, cache_dir, priors):
        """Verify that first N bins have correct number of components.

        Checks:
        1. For bins < n_first_bins: should have max_components_for_first_bins
           components (unless fewer components exist in the image)
        2. For bins >= n_first_bins: should have exactly 1 component

        Generates visualization showing component counts per bin.
        """
        pp_cfg = config.postprocessing.zbin_priors
        z_bins = config.conditioning.z_bins

        n_first_bins = pp_cfg.get("n_first_bins", 0)
        max_components_for_first_bins = pp_cfg.get("max_components_for_first_bins", 1)

        # Count components in each prior
        component_counts = []
        expected_counts = []
        bin_indices = []

        for z_bin in range(z_bins):
            if z_bin not in priors:
                continue

            prior = priors[z_bin]

            # Count connected components
            labeled, n_components = label(prior)

            component_counts.append(n_components)
            bin_indices.append(z_bin)

            # Determine expected count
            if z_bin < n_first_bins:
                # First bins: expect max_components_for_first_bins
                # (but could be fewer if image doesn't have that many)
                expected_counts.append(max_components_for_first_bins)
            else:
                # Higher bins: expect exactly 1
                expected_counts.append(1)

        # Verify counts
        failures = []
        for z_bin, actual, expected in zip(bin_indices, component_counts, expected_counts):
            if z_bin < n_first_bins:
                # For first bins: actual should be <= expected
                # (could be fewer if image has fewer components)
                if actual > expected:
                    failures.append(
                        f"Bin {z_bin}: has {actual} components, "
                        f"expected at most {expected}"
                    )
            else:
                # For higher bins: should be exactly 1
                if actual != expected:
                    failures.append(
                        f"Bin {z_bin}: has {actual} components, "
                        f"expected exactly {expected}"
                    )

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot actual counts
        ax.bar(bin_indices, component_counts, color="steelblue",
               edgecolor="black", label="Actual components")

        # Mark expected counts with horizontal lines
        for i, (z_bin, expected) in enumerate(zip(bin_indices, expected_counts)):
            if z_bin < n_first_bins:
                # First bins: show max as dotted line
                ax.plot([z_bin - 0.4, z_bin + 0.4], [expected, expected],
                       'r--', linewidth=2, alpha=0.7)
            else:
                # Higher bins: show expected as solid line
                ax.plot([z_bin - 0.4, z_bin + 0.4], [expected, expected],
                       'g-', linewidth=2, alpha=0.7)

        # Mark the boundary between first_bins and rest
        if n_first_bins > 0 and n_first_bins in bin_indices:
            ax.axvline(n_first_bins - 0.5, color="orange", linestyle="--",
                      linewidth=2, label=f"n_first_bins = {n_first_bins}")

        ax.set_xlabel("Z-bin", fontsize=12)
        ax.set_ylabel("Number of Components", fontsize=12)
        ax.set_title(
            f"Component Count Verification\n"
            f"First {n_first_bins} bins: max {max_components_for_first_bins} components, "
            f"Rest: 1 component",
            fontsize=14,
            fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        # Add text annotation
        textstr = (
            f"n_first_bins: {n_first_bins}\n"
            f"max_components_for_first_bins: {max_components_for_first_bins}\n"
            f"Total bins checked: {len(bin_indices)}"
        )
        ax.text(0.98, 0.98, textstr,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()

        # Save
        output_dir = Path("docs/data_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "multicomponent_verification.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n✓ Saved component count verification to: {output_path}")

        # Print summary
        print(f"\nComponent Count Summary:")
        print(f"  First {n_first_bins} bins (multi-component):")
        for z_bin in range(min(n_first_bins, len(bin_indices))):
            if z_bin < len(component_counts):
                actual = component_counts[z_bin]
                expected = expected_counts[z_bin]
                status = "✓" if actual <= expected else "✗"
                print(f"    Bin {z_bin}: {actual}/{expected} components {status}")

        print(f"\n  Remaining bins (single-component):")
        for i, z_bin in enumerate(bin_indices):
            if z_bin >= n_first_bins:
                actual = component_counts[i]
                expected = expected_counts[i]
                status = "✓" if actual == expected else "✗"
                print(f"    Bin {z_bin}: {actual}/{expected} components {status}")

        # Fail if any violations found
        if failures:
            pytest.fail("\n".join(["Component count violations:"] + failures))

    def test_visualize_component_differences(self, config, cache_dir, priors):
        """Visualize side-by-side comparison of first bins vs later bins.

        Shows how multi-component logic preserves multiple brain structures
        in low bins while single-component logic is used in higher bins.
        """
        pp_cfg = config.postprocessing.zbin_priors
        z_bins = config.conditioning.z_bins

        n_first_bins = pp_cfg.get("n_first_bins", 0)

        if n_first_bins == 0:
            pytest.skip("n_first_bins is 0, no multi-component bins to visualize")

        # Select representative bins
        # First bin (multi-component) and a mid-range bin (single-component)
        low_bin = 0
        mid_bin = min(n_first_bins + 5, z_bins - 1)

        if low_bin not in priors or mid_bin not in priors:
            pytest.skip("Required bins not found in priors")

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for row_idx, (z_bin, title) in enumerate([
            (low_bin, f"Bin {low_bin} (Multi-component)"),
            (mid_bin, f"Bin {mid_bin} (Single-component)")
        ]):
            prior = priors[z_bin]

            # Count components
            labeled, n_components = label(prior)
            component_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background

            # Panel 1: Prior ROI
            axes[row_idx, 0].imshow(prior, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f"{title}\nPrior ROI")
            axes[row_idx, 0].axis("off")

            # Panel 2: Labeled components
            # Create colormap for components
            component_colors = np.zeros((*prior.shape, 3))
            colors = plt.cm.Set3(np.linspace(0, 1, n_components))

            for comp_idx in range(1, n_components + 1):
                mask = labeled == comp_idx
                component_colors[mask] = colors[comp_idx - 1, :3]

            axes[row_idx, 1].imshow(component_colors)
            axes[row_idx, 1].set_title(
                f"Components (n={n_components})\n"
                f"Sizes: {', '.join(str(s) for s in sorted(component_sizes, reverse=True)[:3])}"
            )
            axes[row_idx, 1].axis("off")

            # Panel 3: Coverage statistics
            coverage = prior.sum() / prior.size
            axes[row_idx, 2].text(
                0.5, 0.5,
                f"Z-bin: {z_bin}\n\n"
                f"Components: {n_components}\n\n"
                f"Coverage: {coverage:.1%}\n\n"
                f"Total pixels: {prior.sum()}\n\n"
                f"Largest: {component_sizes[0] if len(component_sizes) > 0 else 0} px",
                ha="center", va="center",
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            )
            axes[row_idx, 2].axis("off")

        fig.suptitle(
            "Multi-Component vs Single-Component Priors Comparison",
            fontsize=16,
            fontweight="bold"
        )
        fig.tight_layout()

        # Save
        output_dir = Path("docs/data_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "multicomponent_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n✓ Saved multi-component comparison to: {output_path}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
