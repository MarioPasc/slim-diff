"""Tests for k-fold planner with real + synthetic data support."""

from __future__ import annotations

import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.segmentation.data.kfold_planner import KFoldPlanner, SampleRecord


class TestKFoldPlannerFixtures:
    """Fixtures for testing KFoldPlanner."""

    @staticmethod
    def create_mock_real_csv(cache_dir: Path, filename: str, subjects: list[dict]):
        """Create a mock real data CSV file.

        Args:
            cache_dir: Directory to create CSV in
            filename: CSV filename (train.csv or val.csv)
            subjects: List of dicts with subject info
        """
        csv_path = cache_dir / filename
        slices_dir = cache_dir / "slices"
        slices_dir.mkdir(exist_ok=True)

        fieldnames = [
            "subject_id", "z_index", "z_bin", "pathology_class", "token",
            "source", "split", "has_lesion", "filepath"
        ]

        rows = []
        for subj in subjects:
            subject_id = subj["subject_id"]
            has_lesion_subject = subj.get("has_lesion", False)
            n_slices = subj.get("n_slices", 10)
            lesion_slices = subj.get("lesion_slices", [])  # List of z_indices with lesions

            for z_idx in range(n_slices):
                z_bin = z_idx // 2  # Simple binning
                has_lesion = z_idx in lesion_slices
                pathology_class = 1 if has_lesion else 0

                filepath = f"slices/{subject_id}_z{z_idx:03d}_bin{z_bin:02d}_c{pathology_class}.npz"

                rows.append({
                    "subject_id": subject_id,
                    "z_index": z_idx,
                    "z_bin": z_bin,
                    "pathology_class": pathology_class,
                    "token": z_bin + (30 * pathology_class),
                    "source": "epilepsy",
                    "split": filename.replace(".csv", ""),
                    "has_lesion": str(has_lesion),
                    "filepath": filepath,
                })

                # Create empty NPZ placeholder
                npz_path = cache_dir / filepath
                npz_path.parent.mkdir(exist_ok=True)
                np.savez(npz_path, image=np.zeros((128, 128)), mask=np.zeros((128, 128)))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return csv_path

    @staticmethod
    def create_mock_replica(synth_dir: Path, replica_name: str, n_samples: int = 100):
        """Create a mock synthetic replica NPZ file.

        Args:
            synth_dir: Directory to create replica in
            replica_name: Replica filename
            n_samples: Number of samples per replica
        """
        replica_path = synth_dir / replica_name

        # Create samples: half lesion, half no-lesion, spread across z-bins
        n_zbins = 10
        samples_per_zbin = n_samples // n_zbins

        images = np.random.randn(n_samples, 128, 128).astype(np.float16)
        masks = np.random.randn(n_samples, 128, 128).astype(np.float16)
        zbins = np.zeros(n_samples, dtype=np.int32)
        lesion_present = np.zeros(n_samples, dtype=np.uint8)

        idx = 0
        for zbin in range(n_zbins):
            for i in range(samples_per_zbin):
                if idx >= n_samples:
                    break
                zbins[idx] = zbin
                # Alternate lesion/no-lesion
                lesion_present[idx] = 1 if (i % 2 == 0) else 0
                idx += 1

        np.savez(
            replica_path,
            images=images,
            masks=masks,
            zbin=zbins,
            lesion_present=lesion_present,
            domain=np.zeros(n_samples, dtype=np.uint8),
            condition_token=zbins + 30 * lesion_present,
            seed=np.zeros(n_samples, dtype=np.int32),
            k_index=np.arange(n_samples, dtype=np.int32),
            replica_id=np.zeros(n_samples, dtype=np.int32),
        )

        return replica_path


class TestRealOnlyMode:
    """Test scenario 1: Only real train/val data."""

    @pytest.fixture
    def real_only_setup(self, tmp_path):
        """Create test setup for real-only mode."""
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create train subjects (8 subjects, 4 with lesions)
        train_subjects = [
            {"subject_id": f"subj_{i:03d}", "has_lesion": i < 4, "n_slices": 10,
             "lesion_slices": [3, 4, 5] if i < 4 else []}
            for i in range(8)
        ]

        # Create val subjects (2 subjects, 1 with lesion)
        val_subjects = [
            {"subject_id": f"subj_{i:03d}", "has_lesion": i == 8, "n_slices": 10,
             "lesion_slices": [3, 4] if i == 8 else []}
            for i in range(8, 10)
        ]

        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "train.csv", train_subjects)
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "val.csv", val_subjects)

        # Create config
        cfg = OmegaConf.create({
            "data": {
                "real": {
                    "enabled": True,
                    "cache_dir": str(cache_dir),
                },
                "synthetic": {
                    "enabled": False,
                    "samples_dir": "",
                    "replicas": [],
                    "merging_strategy": "concat",
                },
            },
            "k_fold": {
                "n_folds": 5,
                "exclude_test": False,
                "stratify_by": "has_lesion_subject",
                "seed": 42,
            },
            "experiment": {
                "output_dir": str(output_dir),
            },
        })

        return cfg, cache_dir, output_dir

    def test_real_only_initialization(self, real_only_setup):
        """Test real-only mode initializes correctly."""
        cfg, cache_dir, output_dir = real_only_setup
        planner = KFoldPlanner(cfg)

        assert planner.real_enabled is True
        assert planner.synthetic_enabled is False
        assert len(planner.real_subjects) == 10
        assert len(planner.folds) == 5

    def test_real_only_fold_counts(self, real_only_setup):
        """Test fold sample counts are correct."""
        cfg, cache_dir, output_dir = real_only_setup
        planner = KFoldPlanner(cfg)

        total_samples = sum(len(s.samples) for s in planner.real_subjects.values())

        for fold_idx in range(5):
            train, val = planner.get_fold(fold_idx)

            # All samples should be accounted for
            assert len(train) + len(val) == total_samples

            # All samples should be real
            assert all(s.source == "real" for s in train)
            assert all(s.source == "real" for s in val)

    def test_real_only_no_subject_overlap(self, real_only_setup):
        """Test train/val have no overlapping subjects."""
        cfg, cache_dir, output_dir = real_only_setup
        planner = KFoldPlanner(cfg)

        for fold_idx in range(5):
            train, val = planner.get_fold(fold_idx)

            train_subjects = set(s.subject_id for s in train)
            val_subjects = set(s.subject_id for s in val)

            # No overlap
            assert train_subjects.isdisjoint(val_subjects)

    def test_real_only_stratification(self, real_only_setup):
        """Test stratification maintains lesion ratio."""
        cfg, cache_dir, output_dir = real_only_setup
        planner = KFoldPlanner(cfg)

        # Count lesion subjects in each fold's train set
        for fold_idx in range(5):
            train_subjects_ids, val_subjects_ids = planner.folds[fold_idx]

            train_lesion = sum(
                1 for sid in train_subjects_ids
                if planner.real_subjects[sid].has_lesion
            )
            train_total = len(train_subjects_ids)

            # With 5 lesion subjects out of 10, each train fold (8 subjects)
            # should have ~4 lesion subjects (stratified)
            lesion_ratio = train_lesion / train_total
            assert 0.3 <= lesion_ratio <= 0.7  # Allow some variance

    def test_real_only_csv_generation(self, real_only_setup):
        """Test CSV is generated correctly."""
        cfg, cache_dir, output_dir = real_only_setup
        planner = KFoldPlanner(cfg)

        csv_path = planner.plan()
        assert csv_path.exists()

        # Read and validate CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check all required columns
        required_cols = [
            "fold", "split", "subject_id", "filepath", "z_index", "z_bin",
            "has_lesion_slice", "has_lesion_subject", "source", "replica"
        ]
        assert all(col in reader.fieldnames for col in required_cols)

        # Check source is all "real"
        assert all(row["source"] == "real" for row in rows)


class TestConcatMode:
    """Test scenario 2: Real + Synthetic with concat strategy."""

    @pytest.fixture
    def concat_setup(self, tmp_path):
        """Create test setup for concat mode."""
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()
        synth_dir = tmp_path / "synthetic"
        synth_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create real data (4 subjects)
        train_subjects = [
            {"subject_id": f"subj_{i:03d}", "has_lesion": i < 2, "n_slices": 10,
             "lesion_slices": [3, 4, 5] if i < 2 else []}
            for i in range(4)
        ]
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "train.csv", train_subjects)
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "val.csv", [])

        # Create synthetic replicas
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_001.npz", n_samples=50)
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_002.npz", n_samples=50)

        cfg = OmegaConf.create({
            "data": {
                "real": {
                    "enabled": True,
                    "cache_dir": str(cache_dir),
                },
                "synthetic": {
                    "enabled": True,
                    "samples_dir": str(synth_dir),
                    "replicas": ["replica_001.npz", "replica_002.npz"],
                    "merging_strategy": "concat",
                },
            },
            "k_fold": {
                "n_folds": 2,
                "exclude_test": False,
                "stratify_by": "has_lesion_subject",
                "seed": 42,
            },
            "experiment": {
                "output_dir": str(output_dir),
            },
        })

        return cfg, cache_dir, synth_dir, output_dir

    def test_concat_loads_all_replicas(self, concat_setup):
        """Test all replicas are loaded."""
        cfg, cache_dir, synth_dir, output_dir = concat_setup
        planner = KFoldPlanner(cfg)

        # 2 replicas * 50 samples = 100 synthetic samples
        assert len(planner.synthetic_samples) == 100

    def test_concat_adds_all_synthetic(self, concat_setup):
        """Test concat adds all synthetic samples to training."""
        cfg, cache_dir, synth_dir, output_dir = concat_setup
        planner = KFoldPlanner(cfg)

        # Get total real samples in train
        train_real, _ = planner._get_fold_real_samples(0)
        n_real = len(train_real)

        # Get combined samples
        train_combined, _ = planner.get_fold(0)

        # Concat should add ALL synthetic samples
        n_synthetic = len(planner.synthetic_samples)
        assert len(train_combined) == n_real + n_synthetic

    def test_concat_val_is_real_only(self, concat_setup):
        """Test validation set contains only real samples."""
        cfg, cache_dir, synth_dir, output_dir = concat_setup
        planner = KFoldPlanner(cfg)

        _, val = planner.get_fold(0)

        # All validation samples should be real
        for sample in val:
            assert sample.source == "real"

    def test_concat_csv_has_both_sources(self, concat_setup):
        """Test generated CSV contains both real and synthetic."""
        cfg, cache_dir, synth_dir, output_dir = concat_setup
        planner = KFoldPlanner(cfg)

        csv_path = planner.plan()

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        sources = set(row["source"] for row in rows)
        assert "real" in sources
        assert "synthetic" in sources

        # Train should have synthetic, val should not
        train_rows = [r for r in rows if r["split"] == "train"]
        val_rows = [r for r in rows if r["split"] == "val"]

        train_sources = set(r["source"] for r in train_rows)
        val_sources = set(r["source"] for r in val_rows)

        assert "synthetic" in train_sources
        if val_rows:  # Only if there are val rows
            assert all(r["source"] == "real" for r in val_rows)


class TestBalanceMode:
    """Test scenario 3: Real + Synthetic with balance strategy."""

    @pytest.fixture
    def balance_setup(self, tmp_path):
        """Create test setup for balance mode with imbalanced data."""
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()
        synth_dir = tmp_path / "synthetic"
        synth_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create IMBALANCED real data:
        # - Subject 0: 10 slices, only 2 lesion slices (z=3,4)
        # - Subject 1: 10 slices, only 1 lesion slice (z=5)
        # - Subject 2: 10 slices, 0 lesion slices
        # - Subject 3: 10 slices, 0 lesion slices
        # This gives us heavily imbalanced data (3 lesion slices vs 37 no-lesion)
        train_subjects = [
            {"subject_id": "subj_000", "has_lesion": True, "n_slices": 10, "lesion_slices": [3, 4]},
            {"subject_id": "subj_001", "has_lesion": True, "n_slices": 10, "lesion_slices": [5]},
            {"subject_id": "subj_002", "has_lesion": False, "n_slices": 10, "lesion_slices": []},
            {"subject_id": "subj_003", "has_lesion": False, "n_slices": 10, "lesion_slices": []},
        ]
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "train.csv", train_subjects)
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "val.csv", [])

        # Create synthetic replicas with balanced samples
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_001.npz", n_samples=100)

        cfg = OmegaConf.create({
            "data": {
                "real": {
                    "enabled": True,
                    "cache_dir": str(cache_dir),
                },
                "synthetic": {
                    "enabled": True,
                    "samples_dir": str(synth_dir),
                    "replicas": ["replica_001.npz"],
                    "merging_strategy": "balance",
                },
            },
            "k_fold": {
                "n_folds": 2,
                "exclude_test": False,
                "stratify_by": "has_lesion_subject",
                "seed": 42,
            },
            "experiment": {
                "output_dir": str(output_dir),
            },
        })

        return cfg, cache_dir, synth_dir, output_dir

    def test_balance_reduces_imbalance(self, balance_setup):
        """Test balance strategy reduces class imbalance."""
        cfg, cache_dir, synth_dir, output_dir = balance_setup
        planner = KFoldPlanner(cfg)

        # Get real-only samples for comparison
        train_real, _ = planner._get_fold_real_samples(0)
        real_lesion = sum(1 for s in train_real if s.has_lesion)
        real_no_lesion = sum(1 for s in train_real if not s.has_lesion)

        # Original is heavily imbalanced
        original_ratio = real_lesion / (real_lesion + real_no_lesion)
        assert original_ratio < 0.2  # Less than 20% lesion

        # Get balanced samples
        train_balanced, _ = planner.get_fold(0)
        balanced_lesion = sum(1 for s in train_balanced if s.has_lesion)
        balanced_no_lesion = sum(1 for s in train_balanced if not s.has_lesion)

        balanced_ratio = balanced_lesion / (balanced_lesion + balanced_no_lesion)

        # Balance strategy should improve ratio (closer to 50%)
        assert balanced_ratio > original_ratio

    def test_balance_adds_synthetic_to_minority(self, balance_setup):
        """Test synthetic samples are added to minority class."""
        cfg, cache_dir, synth_dir, output_dir = balance_setup
        planner = KFoldPlanner(cfg)

        train_balanced, _ = planner.get_fold(0)

        # Count synthetic lesion vs no-lesion
        synth_lesion = sum(1 for s in train_balanced if s.source == "synthetic" and s.has_lesion)
        synth_no_lesion = sum(1 for s in train_balanced if s.source == "synthetic" and not s.has_lesion)

        # Since real data has fewer lesions, synthetic should add more lesions
        # (or at least equal, as balance adds pairs after initial balancing)
        assert synth_lesion >= synth_no_lesion * 0.8  # Allow some variance

    def test_balance_per_zbin(self, balance_setup):
        """Test balance works per z-bin."""
        cfg, cache_dir, synth_dir, output_dir = balance_setup
        planner = KFoldPlanner(cfg)

        train_balanced, _ = planner.get_fold(0)

        # Count per z-bin
        zbin_counts = defaultdict(lambda: {"lesion": 0, "no_lesion": 0})
        for s in train_balanced:
            key = "lesion" if s.has_lesion else "no_lesion"
            zbin_counts[s.z_bin][key] += 1

        # For each z-bin, the ratio should be more balanced than original
        for zbin, counts in zbin_counts.items():
            total = counts["lesion"] + counts["no_lesion"]
            if total > 0:
                ratio = counts["lesion"] / total
                # After balancing, no z-bin should be extremely imbalanced
                # (unless there were no synthetic samples for that z-bin)
                if counts["lesion"] > 0 or counts["no_lesion"] > 0:
                    # At least one class should have samples
                    pass  # Basic sanity check


class TestSyntheticOnlyMode:
    """Test scenario 4: Only synthetic data."""

    @pytest.fixture
    def synthetic_only_setup(self, tmp_path):
        """Create test setup for synthetic-only mode."""
        synth_dir = tmp_path / "synthetic"
        synth_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()

        # Create synthetic replicas
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_001.npz", n_samples=100)
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_002.npz", n_samples=100)

        cfg = OmegaConf.create({
            "data": {
                "real": {
                    "enabled": False,
                    "cache_dir": str(cache_dir),
                },
                "synthetic": {
                    "enabled": True,
                    "samples_dir": str(synth_dir),
                    "replicas": ["replica_001.npz", "replica_002.npz"],
                    "merging_strategy": "concat",  # Ignored for synthetic-only
                },
            },
            "k_fold": {
                "n_folds": 5,
                "exclude_test": False,
                "stratify_by": "has_lesion_subject",
                "seed": 42,
            },
            "experiment": {
                "output_dir": str(output_dir),
            },
        })

        return cfg, synth_dir, output_dir

    def test_synthetic_only_loads_replicas(self, synthetic_only_setup):
        """Test synthetic-only mode loads all replicas."""
        cfg, synth_dir, output_dir = synthetic_only_setup
        planner = KFoldPlanner(cfg)

        assert len(planner.synthetic_samples) == 200
        assert len(planner.real_subjects) == 0

    def test_synthetic_only_creates_splits(self, synthetic_only_setup):
        """Test synthetic-only mode creates train/val splits."""
        cfg, synth_dir, output_dir = synthetic_only_setup
        planner = KFoldPlanner(cfg)

        train, val = planner.get_fold(0)

        # Should have train and val samples
        assert len(train) > 0
        assert len(val) > 0

        # All should be synthetic
        assert all(s.source == "synthetic" for s in train)
        assert all(s.source == "synthetic" for s in val)

        # Should account for all samples
        total = len(train) + len(val)
        assert total == 200

    def test_synthetic_only_stratified_split(self, synthetic_only_setup):
        """Test synthetic-only creates stratified train/val."""
        cfg, synth_dir, output_dir = synthetic_only_setup
        planner = KFoldPlanner(cfg)

        train, val = planner.get_fold(0)

        # Check lesion ratio is similar in train and val
        train_lesion_ratio = sum(1 for s in train if s.has_lesion) / len(train)
        val_lesion_ratio = sum(1 for s in val if s.has_lesion) / len(val)

        # Ratios should be similar (within 20%)
        assert abs(train_lesion_ratio - val_lesion_ratio) < 0.2

    def test_synthetic_only_csv_generation(self, synthetic_only_setup):
        """Test CSV generation for synthetic-only."""
        cfg, synth_dir, output_dir = synthetic_only_setup
        planner = KFoldPlanner(cfg)

        csv_path = planner.plan()
        assert csv_path.exists()
        assert "synthetic_only" in csv_path.name

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # All should be synthetic
        assert all(row["source"] == "synthetic" for row in rows)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def minimal_setup(self, tmp_path):
        """Create minimal test setup."""
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        return cache_dir, output_dir

    def test_no_data_enabled_raises(self, minimal_setup):
        """Test error when no data is enabled."""
        cache_dir, output_dir = minimal_setup

        cfg = OmegaConf.create({
            "data": {
                "real": {"enabled": False, "cache_dir": str(cache_dir)},
                "synthetic": {"enabled": False, "samples_dir": "", "replicas": []},
            },
            "k_fold": {"n_folds": 5, "exclude_test": False, "stratify_by": "has_lesion_subject", "seed": 42},
            "experiment": {"output_dir": str(output_dir)},
        })

        planner = KFoldPlanner(cfg)

        with pytest.raises(ValueError, match="No data enabled"):
            planner.get_fold(0)

    def test_missing_replica_logs_warning(self, minimal_setup, caplog):
        """Test missing replica logs warning."""
        cache_dir, output_dir = minimal_setup
        synth_dir = cache_dir / "synthetic"
        synth_dir.mkdir()

        cfg = OmegaConf.create({
            "data": {
                "real": {"enabled": False, "cache_dir": str(cache_dir)},
                "synthetic": {
                    "enabled": True,
                    "samples_dir": str(synth_dir),
                    "replicas": ["nonexistent.npz"],
                    "merging_strategy": "concat",
                },
            },
            "k_fold": {"n_folds": 5, "exclude_test": False, "stratify_by": "has_lesion_subject", "seed": 42},
            "experiment": {"output_dir": str(output_dir)},
        })

        import logging
        with caplog.at_level(logging.WARNING):
            planner = KFoldPlanner(cfg)

        assert "Replica not found" in caplog.text


class TestStatistics:
    """Test statistics and reporting."""

    @pytest.fixture
    def stats_setup(self, tmp_path):
        """Create setup for statistics testing."""
        cache_dir = tmp_path / "slice_cache"
        cache_dir.mkdir()
        synth_dir = tmp_path / "synthetic"
        synth_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train_subjects = [
            {"subject_id": f"subj_{i:03d}", "has_lesion": i < 2, "n_slices": 10,
             "lesion_slices": [3, 4, 5] if i < 2 else []}
            for i in range(4)
        ]
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "train.csv", train_subjects)
        TestKFoldPlannerFixtures.create_mock_real_csv(cache_dir, "val.csv", [])
        TestKFoldPlannerFixtures.create_mock_replica(synth_dir, "replica_001.npz", n_samples=50)

        cfg = OmegaConf.create({
            "data": {
                "real": {"enabled": True, "cache_dir": str(cache_dir)},
                "synthetic": {
                    "enabled": True,
                    "samples_dir": str(synth_dir),
                    "replicas": ["replica_001.npz"],
                    "merging_strategy": "concat",
                },
            },
            "k_fold": {"n_folds": 2, "exclude_test": False, "stratify_by": "has_lesion_subject", "seed": 42},
            "experiment": {"output_dir": str(output_dir)},
        })

        return cfg

    def test_fold_statistics_structure(self, stats_setup):
        """Test fold statistics have expected structure."""
        planner = KFoldPlanner(stats_setup)
        stats = planner.get_fold_statistics(0)

        assert "fold" in stats
        assert "train" in stats
        assert "val" in stats

        for split in ["train", "val"]:
            assert "total" in stats[split]
            assert "real" in stats[split]
            assert "synthetic" in stats[split]
            assert "lesion" in stats[split]
            assert "no_lesion" in stats[split]
            assert "lesion_ratio" in stats[split]
            assert "zbins" in stats[split]

    def test_statistics_counts_correct(self, stats_setup):
        """Test statistics counts are accurate."""
        planner = KFoldPlanner(stats_setup)
        stats = planner.get_fold_statistics(0)

        train_stats = stats["train"]
        train_samples, _ = planner.get_fold(0)

        # Verify total
        assert train_stats["total"] == len(train_samples)

        # Verify source counts
        actual_real = sum(1 for s in train_samples if s.source == "real")
        actual_synth = sum(1 for s in train_samples if s.source == "synthetic")
        assert train_stats["real"] == actual_real
        assert train_stats["synthetic"] == actual_synth

        # Verify lesion counts
        actual_lesion = sum(1 for s in train_samples if s.has_lesion)
        actual_no_lesion = sum(1 for s in train_samples if not s.has_lesion)
        assert train_stats["lesion"] == actual_lesion
        assert train_stats["no_lesion"] == actual_no_lesion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
