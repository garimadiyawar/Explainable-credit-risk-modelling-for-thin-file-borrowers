"""
run_full_pipeline.py
---------------------
Master orchestration script. Runs all pipeline steps in order.
Use --skip-tuning to skip the slow Optuna step (uses default params).
Use --synthetic to auto-generate synthetic data if data/ folder is empty.

Usage:
    python run_full_pipeline.py                  # full run, real data
    python run_full_pipeline.py --synthetic      # generate + run synthetic
    python run_full_pipeline.py --skip-tuning    # skip Optuna (faster)
    python run_full_pipeline.py --synthetic --skip-tuning
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_FILES, VERBOSE


def log(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run_step(module_name: str, step_label: str):
    log(f"Running: {step_label}")
    t0 = time.time()
    import importlib
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  ✓ {step_label} completed in {elapsed:.1f}s")


def check_data_present():
    missing = [k for k, v in DATA_FILES.items() if not os.path.exists(v)]
    return len(missing) == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Credit Risk Pipeline")
    parser.add_argument("--synthetic",    action="store_true",
                        help="Generate synthetic data before running pipeline")
    parser.add_argument("--skip-tuning",  action="store_true",
                        help="Skip Optuna hyperparameter tuning (use default params)")
    parser.add_argument("--steps",        type=str, default=None,
                        help="Comma-separated step numbers to run, e.g. '1,2,3'")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  CREDIT RISK PIPELINE — DISSERTATION REPLICATION")
    print("  Explainable Credit Risk Modelling for Thin-File Borrowers")
    print("="*60)

    # ── Data check ────────────────────────────────────────────────────────────
    if args.synthetic:
        print("\n[Synthetic mode] Generating synthetic dataset ...")
        import generate_synthetic_data
        generate_synthetic_data.main()
    elif not check_data_present():
        print("\nERROR: Dataset files not found in data/ folder.")
        print("Options:")
        print("  1. Download from https://www.kaggle.com/c/home-credit-default-risk/data")
        print("     and place CSVs in data/")
        print("  2. Run with --synthetic flag to use generated data")
        sys.exit(1)

    # ── Step selection ────────────────────────────────────────────────────────
    all_steps = {
        1: ("01_data_loading",          "Step 01: Data Loading"),
        2: ("02_preprocessing",         "Step 02: Preprocessing"),
        3: ("03_feature_engineering",   "Step 03: Feature Engineering"),
        4: ("04_train_test_split",      "Step 04: Train-Test Split"),
        5: ("05_model_training",        "Step 05: Model Training"),
        6: ("06_hyperparameter_tuning", "Step 06: Hyperparameter Tuning (Optuna)"),
        7: ("07_explainability",        "Step 07: Explainability (SHAP + LIME)"),
        8: ("08_fairness_evaluation",   "Step 08: Fairness Evaluation"),
        9: ("09_results_reporting",     "Step 09: Results Reporting"),
    }

    if args.steps:
        step_nums = [int(s.strip()) for s in args.steps.split(",")]
    else:
        step_nums = list(all_steps.keys())

    # Remove tuning if skip requested
    if args.skip_tuning and 6 in step_nums:
        step_nums.remove(6)
        print("\n[--skip-tuning] Skipping Step 06 (Optuna). Using default hyperparameters.")

    # ── Execute ────────────────────────────────────────────────────────────────
    total_start = time.time()
    completed   = []
    failed      = []

    for step_num in step_nums:
        if step_num not in all_steps:
            print(f"\nWarning: Step {step_num} not recognised — skipping.")
            continue

        module_name, step_label = all_steps[step_num]
        try:
            run_step(module_name, step_label)
            completed.append(step_label)
        except Exception as e:
            print(f"\n  ✗ {step_label} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append((step_label, str(e)))
            # Ask whether to continue
            resp = input("\nContinue to next step? [y/N]: ").strip().lower()
            if resp != "y":
                break

        # After Step 05, if tuning was run (Step 06), re-run Step 05 with tuned params
        if step_num == 6 and 5 in step_nums:
            print("\n[Post-tuning] Re-running Step 05 with Optuna-tuned parameters ...")
            try:
                run_step("05_model_training", "Step 05 (Retrain with tuned params)")
                completed.append("Step 05 (retrain)")
            except Exception as e:
                print(f"  Retrain failed: {e}")
                failed.append(("Step 05 retrain", str(e)))

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Completed : {len(completed)} steps")
    print(f"  Failed    : {len(failed)} steps")
    if failed:
        print("\n  Failed steps:")
        for name, err in failed:
            print(f"    ✗ {name}: {err[:80]}")
    print("\n  Outputs:")
    print("    models/         — trained model files (.joblib)")
    print("    outputs/        — intermediate arrays and JSON results")
    print("    plots/          — SHAP and evaluation plots (.png)")
    print("    reports/        — Excel summary + text tables")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
