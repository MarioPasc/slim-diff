"""Test wrapper for z-bin prior visualizations.

Runs the visualization script and verifies outputs are generated.
Uses configuration from src/diffusion/config/jsddpm.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.diffusion.scripts.visualize_zbin_priors import (
    compare_priors_with_cache_distribution,
    visualize_prior_statistics,
    visualize_priors_overview,
    visualize_priors_with_patient,
)
from src.diffusion.utils.zbin_priors import load_zbin_priors


@pytest.mark.skipif(
    not Path("/media/mpascual/Sandisk2TB/research/epilepsy/data").exists(),
    reason="Data directory not found"
)
class TestVisualizePriors:
    """Test z-bin prior visualization generation."""

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

    @pytest.fixture
    def output_dir(self):
        """Create output directory."""
        output_dir = Path("docs/data_visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def test_visualize_priors_overview(self, config, priors, output_dir):
        """Test overview grid visualization generation."""
        z_bins = config.conditioning.z_bins
        z_range = tuple(config.data.slice_sampling.z_range)

        visualize_priors_overview(priors, z_bins, z_range, output_dir)

        # Verify output exists
        output_path = output_dir / "zbin_priors_overview.png"
        assert output_path.exists(), f"Output not generated: {output_path}"

        print(f"\n✓ Generated: {output_path}")

    def test_visualize_priors_with_patient(self, config, priors, output_dir):
        """Test patient-specific visualization generation."""
        subject_id = "MRIe_087"

        visualize_priors_with_patient(priors, config, output_dir, subject_id)

        # Verify at least one output exists
        patient_dir = output_dir / subject_id
        assert patient_dir.exists(), f"Patient directory not created: {patient_dir}"

        # Find generated images
        generated_images = list(patient_dir.glob("**/*.png"))
        assert len(generated_images) > 0, "No patient visualizations generated"

        print(f"\n✓ Generated {len(generated_images)} patient visualizations")
        for img in generated_images:
            print(f"  - {img.relative_to(output_dir)}")

    def test_visualize_prior_statistics(self, config, priors, output_dir):
        """Test statistical analysis visualization generation."""
        z_bins = config.conditioning.z_bins

        visualize_prior_statistics(priors, z_bins, output_dir)

        # Verify output exists
        output_path = output_dir / "zbin_priors_statistics.png"
        assert output_path.exists(), f"Output not generated: {output_path}"

        print(f"\n✓ Generated: {output_path}")

    def test_compare_priors_with_cache_distribution(
        self, config, cache_dir, priors, output_dir
    ):
        """Test cache distribution comparison visualization."""
        z_bins = config.conditioning.z_bins

        compare_priors_with_cache_distribution(priors, z_bins, cache_dir, output_dir)

        # Verify output exists
        output_path = output_dir / "zbin_priors_vs_distribution.png"
        assert output_path.exists(), f"Output not generated: {output_path}"

        print(f"\n✓ Generated: {output_path}")

    def test_all_visualizations(
        self, config, cache_dir, priors, output_dir
    ):
        """Generate all visualizations in one test."""
        print("\n" + "=" * 60)
        print("Generating All Z-Bin Prior Visualizations")
        print("=" * 60)

        z_bins = config.conditioning.z_bins
        z_range = tuple(config.data.slice_sampling.z_range)
        subject_id = "MRIe_087"

        # 1. Overview
        print("\n[1/4] Creating priors overview grid...")
        visualize_priors_overview(priors, z_bins, z_range, output_dir)

        # 2. Patient slices
        print(f"\n[2/4] Creating patient visualizations for {subject_id}...")
        visualize_priors_with_patient(priors, config, output_dir, subject_id)

        # 3. Statistics
        print("\n[3/4] Creating statistical analysis...")
        visualize_prior_statistics(priors, z_bins, output_dir)

        # 4. Distribution comparison
        print("\n[4/4] Creating cache distribution comparison...")
        compare_priors_with_cache_distribution(priors, z_bins, cache_dir, output_dir)

        print("\n" + "=" * 60)
        print("✓ All visualizations complete!")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")

        # Verify key outputs exist
        expected_outputs = [
            "zbin_priors_overview.png",
            "zbin_priors_statistics.png",
            "zbin_priors_vs_distribution.png",
        ]

        for expected in expected_outputs:
            output_path = output_dir / expected
            assert output_path.exists(), f"Missing output: {output_path}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
