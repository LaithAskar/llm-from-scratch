"""Sanity-check the training environment before starting Part 1.

Run: python verify_setup.py
Expected: all checks pass and CUDA device prints as RTX 4060 (or whatever GPU).
"""
import sys


def main() -> int:
    failures: list[str] = []

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        failures.append("Python >= 3.10 required")

    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        failures.append("torch missing")
        return _report(failures)

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        failures.append(
            "CUDA not available. Reinstall with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (torch was built with): {torch.version.cuda}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"Matmul test: ok ({tuple(y.shape)})")

    for pkg in ("tiktoken", "numpy", "matplotlib"):
        try:
            __import__(pkg)
            print(f"{pkg}: ok")
        except ImportError:
            failures.append(f"{pkg} missing")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed. Ready to start Part 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
