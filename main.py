import importlib.util
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_STEPS = [
    ("01_ingest.py", "run_ingest"),
    ("02_features.py", "build_features"),
    ("03_train_avm.py", "train_and_evaluate"),
    ("04_ltv_lgd_stress.py", "run_collateral_stress_test"),
]


def run_step(filename: str, func_name: str) -> None:
    module_path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getattr(module, func_name)()


def main() -> None:
    total_start = time.perf_counter()
    print("=================================================================")
    print("  AVM & COLLATERAL RISK PIPELINE (WARSZAWA MORTGAGE PORTFOLIO)")
    print("=================================================================\n")

    for idx, (filename, func_name) in enumerate(PIPELINE_STEPS, start=1):
        step_start = time.perf_counter()
        print(f"\n>>> [{idx}/{len(PIPELINE_STEPS)}] Uruchamiam: src/{filename} ...")
        run_step(filename, func_name)
        elapsed = time.perf_counter() - step_start
        print(f"<<< Zakończono src/{filename} w {elapsed:.2f} s.\n")
        print("-" * 65)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nCały pipeline zakończony pomyślnie w {total_elapsed:.2f} s.")


if __name__ == "__main__":
    main()