# SLIM-Diff Publication Checklist for ICIP 2026

This checklist covers all steps required before submitting the paper to ICIP 2026.

---

## Pre-Submission Checklist

### 1. Code Preparation

- [ ] **Run test suite**
  ```bash
  ~/.conda/envs/jsddpm/bin/pytest src/diffusion/tests/ -v
  ```

- [ ] **Verify all CLI commands work**
  ```bash
  slimdiff-cache --help
  slimdiff-train --help
  slimdiff-generate --help
  slimdiff-generate-spec --help
  slimdiff-metrics --help
  ```

- [ ] **Test generation with a checkpoint**
  ```bash
  slimdiff-generate-spec --spec configs/examples/generation_spec_example.json --dry-run
  ```

- [ ] **Clean up any debug/development code**

- [ ] **Verify example configs have correct placeholder paths**
  - `configs/examples/cache_example.yaml`
  - `configs/examples/training_example.yaml`
  - `configs/examples/generation_spec_example.json`
  - `configs/examples/metrics_example.yaml`

---

### 2. Prepare Model Weights for Zenodo

- [ ] **Create directory for weights**
  ```bash
  mkdir -p ~/zenodo_upload/slimdiff_weights_v1.0
  ```

- [ ] **Copy and rename checkpoints**
  ```bash
  # Base path
  BASE="/media/mpascual/Sandisk2TB/research/jsddpm/results/epilepsy/icip2026/runs/self_cond_ablation"
  OUT="~/zenodo_upload/slimdiff_weights_v1.0"

  # x0-prediction models
  cp "$BASE/self_cond_p_0.0/x0_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp1.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/x0_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp2.0_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/x0_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp2.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.5/x0_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp1.5_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/x0_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp2.0_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/x0_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_x0_lp2.5_sc0.5.ckpt"

  # velocity-prediction models
  cp "$BASE/self_cond_p_0.0/velocity_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp1.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/velocity_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp2.0_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/velocity_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp2.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.5/velocity_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp1.5_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/velocity_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp2.0_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/velocity_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_velocity_lp2.5_sc0.5.ckpt"

  # epsilon-prediction models
  cp "$BASE/self_cond_p_0.0/epsilon_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp1.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/epsilon_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp2.0_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.0/epsilon_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp2.5_sc0.0.ckpt"
  cp "$BASE/self_cond_p_0.5/epsilon_lp_1.5/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp1.5_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/epsilon_lp_2.0/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp2.0_sc0.5.ckpt"
  cp "$BASE/self_cond_p_0.5/epsilon_lp_2.5/checkpoints/"*.ckpt "$OUT/slimdiff_epsilon_lp2.5_sc0.5.ckpt"
  ```

- [ ] **Create README for weights**
  ```bash
  # Create a README.txt in the weights folder explaining each model
  ```

- [ ] **Verify all 18 checkpoints are present**
  ```bash
  ls -la ~/zenodo_upload/slimdiff_weights_v1.0/*.ckpt | wc -l  # Should be 18
  ```

---

### 3. Upload to arXiv

- [ ] **Prepare arXiv submission**
  - Compile final PDF
  - Prepare source files (.tex, figures, .bbl)
  - Create `00README.XXX` if needed

- [ ] **Submit to arXiv**
  - Go to https://arxiv.org/submit
  - Select category: `cs.CV` (primary), `eess.IV` (cross-list)
  - Upload files and submit

- [ ] **Wait for arXiv ID** (usually within 24-48 hours)
  - Note the arXiv ID: `arXiv:XXXX.XXXXX`

- [ ] **Update repository with arXiv ID**

  Files to update:
  - [ ] `README.md` - Line 4: arXiv badge URL
  - [ ] `README.md` - Line 313: arXiv preprint citation

  ```bash
  # Replace XXXX.XXXXX with actual arXiv ID
  sed -i 's/XXXX\.XXXXX/2501.12345/g' README.md
  ```

---

### 4. Publish to PyPI

#### 4.1 Test on TestPyPI First

- [ ] **Install build tools**
  ```bash
  ~/.conda/envs/jsddpm/bin/pip install build twine
  ```

- [ ] **Clean previous builds**
  ```bash
  rm -rf dist/ build/ *.egg-info
  ```

- [ ] **Build package**
  ```bash
  cd /home/mpascual/research/code/slim-diff
  ~/.conda/envs/jsddpm/bin/python -m build
  ```

- [ ] **Check package contents**
  ```bash
  tar -tzf dist/slim_diff-1.0.0.tar.gz | head -20
  ```

- [ ] **Upload to TestPyPI**
  ```bash
  ~/.conda/envs/jsddpm/bin/python -m twine upload --repository testpypi dist/*
  ```
  - Username: `__token__`
  - Password: Your TestPyPI API token

- [ ] **Test installation from TestPyPI**
  ```bash
  # Create fresh test environment
  python -m venv /tmp/test_slimdiff
  source /tmp/test_slimdiff/bin/activate

  # Install from TestPyPI (with PyPI for dependencies)
  pip install --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      slim-diff

  # Verify
  slimdiff-train --help
  slimdiff-generate-spec --help

  # Cleanup
  deactivate
  rm -rf /tmp/test_slimdiff
  ```

#### 4.2 Publish to Production PyPI

- [ ] **Upload to PyPI**
  ```bash
  ~/.conda/envs/jsddpm/bin/python -m twine upload dist/*
  ```
  - Username: `__token__`
  - Password: Your PyPI API token

- [ ] **Verify installation from PyPI**
  ```bash
  pip install slim-diff
  slimdiff-train --help
  ```

- [ ] **Check PyPI page**: https://pypi.org/project/slim-diff/

---

### 5. Publish to Zenodo

#### 5.1 Upload Model Weights

- [ ] **Go to Zenodo**: https://zenodo.org/deposit/new

- [ ] **Fill metadata**:
  - **Title**: SLIM-Diff: Pretrained Models for Epilepsy Lesion Synthesis (v1.0)
  - **Upload type**: Dataset
  - **Authors**: Same as paper (with affiliations and ORCIDs)
  - **Description**:
    ```
    Pretrained model weights for SLIM-Diff, a compact joint diffusion model
    for synthesizing paired FLAIR MRI slices and lesion masks.

    This release includes 18 model configurations:
    - 3 prediction types: x₀, velocity (v), epsilon (ε)
    - 3 Lₚ norm values: 1.5, 2.0, 2.5
    - 2 self-conditioning probabilities: 0.0, 0.5

    Recommended model: slimdiff_x0_lp2.25_sc0.5.ckpt

    For usage instructions, see: https://github.com/MarioPasc/slim-diff
    ```
  - **License**: MIT License
  - **Keywords**: diffusion models, medical imaging, epilepsy, FCD, MRI, FLAIR, deep learning
  - **Related identifiers**:
    - GitHub repo: `https://github.com/MarioPasc/slim-diff` (isSupplementTo)
    - arXiv: `https://arxiv.org/abs/XXXX.XXXXX` (isSupplementTo)
    - PyPI: `https://pypi.org/project/slim-diff/` (isSupplementTo)

- [ ] **Upload weight files** (all 18 `.ckpt` files)

- [ ] **Publish and get DOI**
  - Note the DOI: `10.5281/zenodo.XXXXXXX`

- [ ] **Update repository with Zenodo DOI**

  Files to update:
  - [ ] `README.md` - Line 5: Zenodo badge
  - [ ] `README.md` - Lines 82, 88, 99-127, 134: Download URLs

  ```bash
  # Replace XXXXXXX with actual Zenodo record number
  sed -i 's/zenodo\.XXXXXXX/zenodo.1234567/g' README.md
  ```

#### 5.2 (Optional) Archive Code on Zenodo

- [ ] **Link GitHub to Zenodo** (for automatic releases)
  - Go to: https://zenodo.org/account/settings/github/
  - Enable the `slim-diff` repository

- [ ] **Create GitHub release**
  ```bash
  git tag -a v1.0.0 -m "SLIM-Diff v1.0.0 - ICIP 2026"
  git push origin v1.0.0
  ```
  - Go to GitHub → Releases → Create release from tag

- [ ] **Zenodo automatically creates DOI for code**

---

### 6. Final Repository Updates

- [ ] **Commit all placeholder updates**
  ```bash
  git add README.md
  git commit -m "Update arXiv and Zenodo identifiers"
  git push origin main
  ```

- [ ] **Verify all links work**
  - [ ] arXiv badge links to paper
  - [ ] Zenodo badge links to weights
  - [ ] PyPI badge links to package
  - [ ] All download links work
  - [ ] GitHub clone URL is correct

- [ ] **Check README renders correctly on GitHub**

---

### 7. Paper Submission to ICIP 2026

- [ ] **Verify paper includes**:
  - [ ] GitHub URL in paper
  - [ ] arXiv reference (if allowed)
  - [ ] Zenodo DOI for weights

- [ ] **Submit to ICIP 2026**
  - Submission portal: [ICIP 2026 submission site]
  - Deadline: [Check ICIP 2026 deadlines]

---

## Post-Acceptance Updates

After paper is accepted at ICIP 2026:

- [ ] **Update citation in README.md**
  - Uncomment the `@inproceedings` citation (lines 318-330)
  - Comment out or remove the arXiv citation

- [ ] **Update paper PDF on arXiv**
  - Add "Accepted at ICIP 2026" note

- [ ] **Create new Zenodo version** (if needed)
  - Update description to mention ICIP 2026 acceptance

---

## Quick Reference: Placeholder Locations

| Placeholder | Location | Description |
|-------------|----------|-------------|
| `XXXX.XXXXX` | `README.md:4,313` | arXiv ID |
| `XXXXXXX` | `README.md:5,82,88,99-127,134` | Zenodo record number |

---

## Useful Links

- **arXiv**: https://arxiv.org/submit
- **PyPI**: https://pypi.org/
- **TestPyPI**: https://test.pypi.org/
- **Zenodo**: https://zenodo.org/
- **ICIP 2026**: [Add conference URL when available]

---

## Contact

For questions about this checklist, contact: mpascual@uma.es
