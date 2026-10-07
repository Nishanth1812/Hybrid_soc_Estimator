# SOC Data Quality and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent invalid simulation data from contaminating training, reduce SOC endpoint saturation, and make evaluation plots and metrics represent complete test simulations.

**Architecture:** Add validation at the simulation boundary before scaling, constrain generated profiles using a configurable maximum SOC excursion, and keep evaluation metrics/plots derived from the same flattened prediction arrays. Diagnostics will optionally reshape predictions into fixed-length simulation blocks so trajectory plots do not accidentally show only an arbitrary prefix.

**Tech Stack:** Python 3.11, NumPy, PyBaMM, Matplotlib, PyTorch, unittest, ZIP packaging through `scripts/build_upload_zip.py`.

## Global Constraints

- Reject non-finite or physically impossible terminal voltage; do not silently clip invalid voltage.
- Keep the existing LSTM architecture unchanged until the regenerated data is valid.
- Preserve compatibility with existing evaluation tests and callers.
- Do not overwrite existing generated result directories; package source and data through the existing ZIP builder.

---

### Task 1: Add simulation-output validation and profile safeguards

**Files:**
- Modify: `data_pipeline/simulation/base_simulation.py`
- Modify: `data_pipeline/generation/generate_profiles.py`
- Modify: `config/config.py`
- Modify: `data_pipeline/generation/dataset_generator.py`
- Test: `tests/test_simulation_validation.py`

**Interfaces:**
- Produce `validate_simulation_output(output, voltage_min=2.0, voltage_max=4.4)` in `data_pipeline/simulation/base_simulation.py`, raising `RuntimeError` with the first invalid index/value.
- Produce `profile_soc_excursion(current_profile_a, capacity_ah, sampling_period_s)` in `data_pipeline/generation/generate_profiles.py`, returning the maximum positive and negative SOC change implied by the profile.
- `generate_current_profile` will reject profiles whose implied SOC excursion exceeds the configured `max_soc_excursion`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_validate_simulation_output_rejects_invalid_voltage():
    output = make_output(voltage_v=np.array([3.7, 500.0], dtype=np.float64))
    with pytest.raises(RuntimeError, match="physically invalid voltage"):
        validate_simulation_output(output)

def test_profile_soc_excursion_reports_charge_and_discharge_headroom():
    profile = np.array([5.0, -10.0, 0.0], dtype=np.float64)
    positive, negative = profile_soc_excursion(profile, capacity_ah=5.0, sampling_period_s=1)
    assert positive == pytest.approx(1.0 / 3600.0)
    assert negative == pytest.approx(2.0 / 3600.0)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the interfaces do not exist.**

Run: `python -m unittest tests.test_simulation_validation -v`

- [ ] **Step 3: Implement the validator and excursion calculation.**

Validate matching lengths, finite values, and voltage bounds before returning from `BaseSimulation.run`. Add `max_soc_excursion` to `SIM_CONFIG` and regenerate/retry profiles that exceed it.

- [ ] **Step 4: Run the focused tests and the existing simulation/profile tests.**

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 5: Commit the slice.**

Commit message: `fix: reject invalid simulation voltage and oversized profiles`

### Task 2: Make metrics and trajectory plots simulation-aware

**Files:**
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/evaluation_plots.py`
- Modify: `run_system.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_evaluation_outputs.py`

**Interfaces:**
- Produce `calculate_grouped_metrics(y_true, y_pred, group_size)` returning macro-averaged scalar metrics plus per-group metric rows.
- Extend `plot_prediction_diagnostics(..., group_size=None)`; when `group_size` is supplied, plot one complete group and label it with its group index.
- Preserve existing behavior when `group_size` is omitted.

- [ ] **Step 1: Write failing tests for grouped metrics and complete trajectory selection.**

```python
def test_grouped_metrics_are_macro_averaged():
    result = calculate_grouped_metrics(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 0.5, 1.0, 0.5]),
        group_size=2,
    )
    assert result["group_count"] == 2
    assert len(result["groups"]) == 2

def test_tracking_plot_uses_one_complete_group(tmp_path):
    plot_prediction_diagnostics(true, pred, tmp_path, group_size=4, group_index=1)
    assert (tmp_path / "soc_tracking.png").exists()
```

- [ ] **Step 2: Run the focused tests and verify the new assertions fail.**

Run: `python -m unittest tests.test_metrics tests.test_evaluation_outputs -v`

- [ ] **Step 3: Implement grouped metrics and group-aware plots.**

Use `group_size = timesteps - sequence_length + 1` in `run_system.py`, write both overall and grouped metrics to `metrics.json`, and make the tracking title identify the selected simulation.

- [ ] **Step 4: Run the focused tests and inspect generated PNG dimensions/non-zero sizes.**

Run: `python -m unittest tests.test_metrics tests.test_evaluation_outputs tests.test_evaluation_run -v`

- [ ] **Step 5: Commit the slice.**

Commit message: `fix: make SOC diagnostics simulation-aware`

### Task 3: Verify the pipeline and rebuild artifacts

**Files:**
- Modify: `README.md`
- Modify: `scripts/build_upload_zip.py` only if packaging needs the new files.
- Test: all existing tests plus focused regression tests.

- [ ] **Step 1: Run the complete test suite.**

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 2: Run static source checks and confirm no debug instrumentation remains.**

Run: `Select-String -Path data_pipeline,evaluation,tests -Pattern '\[DEBUG-' -SimpleMatch`

- [ ] **Step 3: Build the code, dataset, and Kaggle bundle ZIPs with the repository script.**

Run: `python scripts/build_upload_zip.py`

- [ ] **Step 4: Inspect ZIP contents and report their paths and sizes.**

Run: `Get-ChildItem Hybrid_soc_Estimator_*.zip | Select-Object Name,Length,LastWriteTime`

- [ ] **Step 5: Commit the documentation and packaging slice.**

Commit message: `docs: document validated SOC evaluation artifacts`
