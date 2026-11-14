"""Offline semantic draft for the 20 cached Job 908 frames, never a CVAT writer.

Predictions are drafts. Review is recorded separately; no existing ground truth is used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_bdd_model import file_hash
from prepare_scene_model import DIRECTORY, FILES
from scene_detector import CITYSCAPES_NAMES


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "job908_review"
OUTPUT = ROOT / "outputs" / "job908_semantic"
NAMES = tuple(name.replace(" ", "_") for name in CITYSCAPES_NAMES)
PALETTE = np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
                    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
                    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
                    (0,80,100),(0,0,230),(119,11,32)],dtype=np.uint8)


def render(number, category, confidence):
    with Image.open(INPUT / f"frame_{number:06d}.png") as source:
        pixels = np.asarray(source.convert("RGB"))
    colors = np.zeros_like(pixels)
    valid = category != 255
    colors[valid] = PALETTE[category[valid]]
    overlay = pixels.copy()
    overlay[valid] = (pixels[valid].astype(float)*.57 + colors[valid]*.43).astype(np.uint8)
    # Show source and its overlay at full resolution, allowing boundary inspection.
    sheet = Image.new("RGB", (pixels.shape[1], pixels.shape[0]*2+80), (24,24,24))
    sheet.paste(Image.fromarray(pixels), (0,25))
    sheet.paste(Image.fromarray(overlay), (0,pixels.shape[0]+50))
    draw = ImageDraw.Draw(sheet)
    draw.text((12,6), f"Frame {number} | original", fill="white")
    draw.text((12,pixels.shape[0]+31), "Semantic draft | void = uncolored | requires visual review", fill="white")
    sheet.save(OUTPUT / f"review_{number:06d}.jpg", quality=93)
    Image.fromarray(category).save(OUTPUT / f"classes_{number:06d}.png")
    return {NAMES[k]:int((category==k).sum()) for k in range(19) if (category==k).any()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args(argv)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    scope=json.loads((INPUT/"complete.json").read_text(encoding="utf-8"))
    if (scope["job_id"],scope["task_id"])!=(908,211):
        raise ValueError("Wrong cached scope")
    model=None
    if not args.render_only:
        import torch
        from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
        torch.set_num_threads(4)
        for name,(_,expected) in FILES.items():
            if file_hash(DIRECTORY/name)!=expected:
                raise ValueError("Unverified semantic model")
        settings=json.loads((DIRECTORY/"preprocessor_config.json").read_text())
        settings.pop("feature_extractor_type",None)
        if "reduce_labels" in settings:
            settings["do_reduce_labels"]=settings.pop("reduce_labels")
        processor=SegformerImageProcessor(**settings)
        model=SegformerForSemanticSegmentation.from_pretrained(str(DIRECTORY),local_files_only=True,
                        use_safetensors=False,weights_only=True).eval()
        if tuple(model.config.id2label[k] for k in range(19))!=CITYSCAPES_NAMES:
            raise ValueError("Wrong model class order")
    report=[]
    for number in scope["frames"]:
        cached=OUTPUT/f"prediction_{number:06d}.npz"
        if model is not None and not cached.exists():
            with Image.open(INPUT/f"frame_{number:06d}.png") as image:
                w,h=image.size
                data=processor(images=image.convert("RGB"),return_tensors="pt",
                               size={"height":576,"width":1024})
                with torch.inference_mode():
                    logits=model(**data).logits
                    logits=torch.nn.functional.interpolate(logits,size=(h,w),mode="bilinear",align_corners=False)[0]
                    confidence,category=logits.softmax(dim=0).max(dim=0)
                np.savez_compressed(cached,category=category.numpy().astype(np.uint8),
                                    confidence=confidence.numpy().astype(np.float16))
        with np.load(cached) as data:
            category=data["category"].copy(); confidence=data["confidence"]
        corrections=OUTPUT/f"corrections_{number:06d}.json"
        if corrections.exists():
            import cv2
            for edit in json.loads(corrections.read_text(encoding="utf-8")):
                bitmap=np.zeros_like(category)
                if edit.get("kind")=="ellipse":
                    cv2.ellipse(bitmap,tuple(edit["center"]),tuple(edit["axes"]),edit.get("angle",0),
                                0,360,1,edit.get("thickness",3))
                elif edit.get("kind")=="line":
                    cv2.polylines(bitmap,[np.asarray(edit["points"],dtype=np.int32)],False,1,edit.get("thickness",3))
                else:
                    cv2.fillPoly(bitmap,[np.asarray(edit["points"],dtype=np.int32)],1)
                selected=bitmap.astype(bool)
                if "replace" in edit:
                    selected &= np.isin(category,[255 if n=="void" else NAMES.index(n) for n in edit["replace"]])
                if edit["class"]=="nearest":
                    invalid=np.isin(category,[NAMES.index(n) for n in edit["replace"]]) | (category==255)
                    if invalid.all():
                        raise ValueError("No neighboring class available")
                    _,nearest=cv2.distanceTransformWithLabels(invalid.astype(np.uint8),cv2.DIST_L2,5,
                                                              labelType=cv2.DIST_LABEL_PIXEL)
                    lookup=np.zeros(int(nearest.max())+1,dtype=np.uint8)
                    lookup[nearest[~invalid]]=category[~invalid]
                    category[selected]=lookup[nearest[selected]]
                else:
                    class_id=255 if edit["class"]=="void" else NAMES.index(edit["class"])
                    category[selected]=class_id
        counts=render(number,category,confidence)
        report.append({"frame":number,"pixels":counts,"void":int((category==255).sum()),
                       "low_confidence_pixels":int(((confidence<.55)&(category!=255)).sum()),
                       "reviewed":False,"source_sha256":hashlib.sha256((INPUT/f"frame_{number:06d}.png").read_bytes()).hexdigest()})
        print(f"Frame {number}: {len(counts)} predicted classes; DRAFT",flush=True)
    (OUTPUT/"draft_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Offline drafts only. No CVAT requests or writes.")


if __name__=="__main__":
    main()
