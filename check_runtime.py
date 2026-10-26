"""Verify the interpreter and actual imports without loading model weights."""
import importlib
import sys

CORE_MODULES = ("requests", "PIL", "numpy", "scipy", "mediapipe", "torch",
                "torchvision", "open_clip", "cv2", "onnxruntime", "sklearn", "gradio")


def main():
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 13):
        print("Python 3.13 is required by the Windows launcher.")
        return 1
    modules = ("insightface",) if "--optional" in sys.argv else CORE_MODULES
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(name)
            print(f"  {name}: {type(exc).__name__}: {exc}")
    print("Failed imports: " + (", ".join(failed) if failed else "none"))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
