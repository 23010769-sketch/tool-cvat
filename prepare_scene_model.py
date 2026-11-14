"""Chuẩn bị SegFormer-B0 tại revision cố định; chỉ tải model, không kết nối CVAT."""

from pathlib import Path

import requests

from prepare_bdd_model import download_checked


REPOSITORY = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"
DIRECTORY = Path(__file__).parent / "models" / "segformer-b0-cityscapes"
FILES = {
    "config.json": (1680, "649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9"),
    "preprocessor_config.json": (272, "d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7"),
    "pytorch_model.bin": (14957601, "027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c"),
}


def main():
    try:
        with requests.Session() as session:
            for name, (size, digest) in FILES.items():
                url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
                download_checked(url, DIRECTORY / name, digest, size, session=session)
        print("[OK] SegFormer-B0: đã xác minh checksum weights và configs; chưa kết nối CVAT.")
        return 0
    except Exception as error:
        print(f"[DỪNG] {type(error).__name__}: chưa chuẩn bị được model segmentation; không dùng weights chưa xác minh.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
