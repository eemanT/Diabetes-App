"""
One-shot migration: strip `quantization_config` from ann.keras so it loads on
older Keras builds. Makes a backup at ann.keras.bak.
"""
import os, json, shutil, zipfile, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "ann.keras")
BACKUP_PATH = MODEL_PATH + ".bak"


def strip_quantization(obj):
    """Recursively remove every `quantization_config` key from a dict/list tree."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            strip_quantization(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_quantization(v)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Backup written: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")

    # Read the archive into memory, patch config.json, write a new archive.
    with zipfile.ZipFile(MODEL_PATH, "r") as zin:
        members = {name: zin.read(name) for name in zin.namelist()}

    if "config.json" not in members:
        raise RuntimeError("config.json missing from .keras archive")

    cfg = json.loads(members["config.json"].decode("utf-8"))
    strip_quantization(cfg)
    members["config.json"] = json.dumps(cfg).encode("utf-8")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".keras", dir=os.path.dirname(MODEL_PATH))
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zout:
            for name, data in members.items():
                zout.writestr(name, data)
        os.replace(tmp_path, MODEL_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Patched {MODEL_PATH} (removed all `quantization_config` entries).")


if __name__ == "__main__":
    main()
