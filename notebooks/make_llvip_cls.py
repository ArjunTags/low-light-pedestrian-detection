"""Turn LLVIP (detection) into a 2-class classification dataset: person vs background.

Output layout (what the BAM notebook expects for DATA_DIR):
    DST/train/{background,person}/*.jpg
    DST/test/{background,person}/*.jpg

Usage:
    python make_llvip_cls.py --src /home/arjun/lli-pedestrian/data/llvip/LLVIP \
                             --dst /home/arjun/lli-pedestrian/data/llvip_cls
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def square_crop(box, W, H, margin):
    """Square window around the box (keeps the person's aspect ratio: no squashing when resized later)."""
    x1, y1, x2, y2 = box
    s = min(max(x2 - x1, y2 - y1) * (1 + margin), W, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - s / 2, 0), W - s)
    top = min(max(cy - s / 2, 0), H - s)
    return int(left), int(top), int(left + s), int(top + s)


def overlap_frac(win, box):
    """Fraction of the window covered by the box."""
    l, t, r, b = win
    iw = max(0, min(r, box[2]) - max(l, box[0]))
    ih = max(0, min(b, box[3]) - max(t, box[1]))
    return iw * ih / float((r - l) * (b - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LLVIP folder (contains Annotations/ and visible/)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--modality", default="visible", choices=["visible", "infrared"])
    ap.add_argument("--size", type=int, default=64, help="saved crop size (notebook resizes again to IMG_SIZE)")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--min-height", type=int, default=20, help="skip person boxes shorter than this (pixels)")
    ap.add_argument("--max-bg-overlap", type=float, default=0.02)
    ap.add_argument("--max-per-class", type=int, default=None, help="cap per class and split (optional)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    src, dst = Path(a.src), Path(a.dst)
    xmls = {p.stem: p for p in (src / "Annotations").rglob("*.xml")}
    img_root = src / a.modality
    imgs = sorted(p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXT)
    assert xmls and imgs, f"no annotations/images found under {src}"

    # Split: use LLVIP's own train/test folders. A random per-crop split would put near-identical
    # neighbouring video frames in both sets (leakage) and inflate every score.
    split_of = {}
    for p in imgs:
        parts = {x.lower() for x in p.parts}
        if "train" in parts:
            split_of[p] = "train"
        elif "test" in parts:
            split_of[p] = "test"
    if len(split_of) != len(imgs):
        print("No train/test folders found: using first 80% of frames (sorted by id) as train, rest as test")
        cut = int(0.8 * len(imgs))
        split_of = {p: ("train" if i < cut else "test") for i, p in enumerate(imgs)}

    for sp in ("train", "test"):
        for c in ("person", "background"):
            (dst / sp / c).mkdir(parents=True, exist_ok=True)

    counts = {(s, c): 0 for s in ("train", "test") for c in ("person", "background")}
    order = imgs[:]
    rng.shuffle(order)                       # so a --max-per-class cap samples across the whole set

    for n, p in enumerate(order, 1):
        sp = split_of[p]
        if p.stem not in xmls:
            continue
        boxes = [b for b in read_boxes(xmls[p.stem]) if (b[3] - b[1]) >= a.min_height]
        if not boxes:
            continue
        cap = a.max_per_class
        if cap and counts[(sp, "person")] >= cap and counts[(sp, "background")] >= cap:
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size

        sides = []
        for i, b in enumerate(boxes):
            win = square_crop(b, W, H, a.margin)
            sides.append(win[2] - win[0])
            if cap and counts[(sp, "person")] >= cap:
                continue
            img.crop(win).resize((a.size, a.size), Image.BICUBIC).save(
                dst / sp / "person" / f"{p.stem}_{i}.jpg", quality=95)
            counts[(sp, "person")] += 1

        # One background patch per person, same size distribution as the person crops
        # (otherwise the classifier could cheat by learning patch scale instead of content).
        for i in range(len(boxes)):
            if cap and counts[(sp, "background")] >= cap:
                break
            s = min(rng.choice(sides), W, H)
            for _ in range(50):
                l, t = rng.randint(0, W - s), rng.randint(0, H - s)
                win = (l, t, l + s, t + s)
                if all(overlap_frac(win, b) <= a.max_bg_overlap for b in boxes):
                    img.crop(win).resize((a.size, a.size), Image.BICUBIC).save(
                        dst / sp / "background" / f"{p.stem}_{i}.jpg", quality=95)
                    counts[(sp, "background")] += 1
                    break
        if n % 1000 == 0:
            print(f"{n}/{len(order)} frames | {counts}")

    print("\nDone. Crops written to", dst)
    for (sp, c), v in sorted(counts.items()):
        print(f"  {sp:5s} {c:11s} {v}")
    print('\nIn the notebook set:  DATA_DIR = r"%s"' % dst)


if __name__ == "__main__":
    main()
