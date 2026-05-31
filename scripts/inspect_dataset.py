from __future__ import annotations

from datasets import load_dataset
from rich import print


def main() -> None:
    ds = load_dataset("edi45/gearxai-dds-seu", "windows_100", split="train")
    print(ds)
    print(ds.features)
    print(ds[0].keys())
    print({key: ds[0][key] for key in ds[0] if key != "signal"})
    print(f"signal rows={len(ds[0]['signal'])}, channels={len(ds[0]['signal'][0])}")


if __name__ == "__main__":
    main()
