"""Convert COCO-Stuff fine-grained GT (png values 0-181, 255=unlabeled) to the
27-class coarse protocol used by the evaluation ('stuff', NUM_CLASSES=27).

Never touches the source dataset: reads raw pngs from --dataroot, writes
<name>_27labelTrainIds.png into a SEPARATE --outroot tree (refuses to run if
outroot == dataroot, refuses to overwrite unless --resume skips existing).
Output value domain is guaranteed {0..26, 255} by LUT construction.

Mapping provenance
------------------
FINE_TO_COARSE below is the canonical cocostuff-27 fine->coarse grouping of the
IIC / PiCIE / STEGO lineage, copied verbatim from STEGO src/data.py (
https://github.com/mhamilton723/STEGO, master, fetched 2026-07-15).
STEGO is MIT-licensed, Copyright (c) Mark Hamilton; see THIRD_PARTY_NOTICES.md.
Keys = fine png pixel values (0-181 = official cocostuff classId-1);
values = coarse indices 0-26 whose order matches coco_stuff_classes_27 in the
upstream ReCLIP++ utils/preprocess.py (0=electronic ... 9=person ... 11=vehicle
... 26=water), verified by spot checks: 0(person)->9, 1-8(bicycle..boat)->11
(vehicle), 105(clouds)->23(sky), 170-176(wall-*)->18(wall), 179-180(window-*)->19.

Usage (only val2017 is needed for evaluation):
  python tools/convert_coco_stuff27_labels.py \
      --dataroot /path/to/cocostuff \
      --outroot  /path/to/COCOStuff27 \
      --splits val2017 [--limit 8] [--resume]
where /path/to/cocostuff/annotations/<split>/*.png are the COCO-Stuff 164k
fine label maps.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

FINE_TO_COARSE = {
    0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8,
    11: 8, 12: 8, 13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7,
    21: 7, 22: 7, 23: 7, 24: 7, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6,
    31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10, 37: 10, 38: 10, 39: 10,
    40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5,
    50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2,
    60: 2, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3,
    70: 3, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1,
    80: 1, 81: 1, 82: 1, 83: 4, 84: 4, 85: 4, 86: 4, 87: 4, 88: 4, 89: 4,
    90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22, 97: 15, 98: 25,
    99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
    107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13,
    115: 13, 116: 13, 117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15,
    123: 22, 124: 21, 125: 21, 126: 24, 127: 20, 128: 22, 129: 15, 130: 17,
    131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17, 137: 25, 138: 16,
    139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
    147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26,
    155: 15, 156: 23, 157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22,
    163: 25, 164: 15, 165: 20, 166: 17, 167: 17, 168: 22, 169: 14, 170: 18,
    171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18, 177: 26, 178: 26,
    179: 19, 180: 19, 181: 24,
}
IGNORE = 255


def build_lut():
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for fine, coarse in FINE_TO_COARSE.items():
        lut[fine] = coarse
    lut[IGNORE] = IGNORE
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True, help='COCO-Stuff root with annotations/<split>/*.png')
    ap.add_argument('--outroot', required=True, help='output root (must differ from --dataroot)')
    ap.add_argument('--splits', default='val2017')
    ap.add_argument('--limit', type=int, default=0, help='smoke: convert only N files per split')
    ap.add_argument('--resume', action='store_true', help='skip files whose output already exists')
    args = ap.parse_args()

    src_root = os.path.abspath(args.dataroot)
    out_root = os.path.abspath(args.outroot)
    if out_root == src_root:
        sys.exit('FATAL: outroot must differ from dataroot (never write into the source dataset)')

    lut = build_lut()
    grand_hist = np.zeros(256, dtype=np.int64)
    unexpected_src = {}
    all_ignore = 0
    total = 0
    t0 = time.time()

    for split in args.splits.split(','):
        in_dir = os.path.join(src_root, 'annotations', split)
        out_dir = os.path.join(out_root, 'annotations', split)
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(f for f in os.listdir(in_dir)
                       if f.endswith('.png') and not f.endswith('_27labelTrainIds.png'))
        if args.limit:
            names = names[:args.limit]
        print('[%s] %d raw labels' % (split, len(names)), flush=True)
        for i, f in enumerate(names):
            out_path = os.path.join(out_dir, f[:-4] + '_27labelTrainIds.png')
            if os.path.exists(out_path):
                if args.resume:
                    continue
                sys.exit('FATAL: %s exists (use --resume to skip existing)' % out_path)
            arr = np.array(Image.open(os.path.join(in_dir, f)))
            src_u, src_c = np.unique(arr, return_counts=True)
            for u, c in zip(src_u.tolist(), src_c.tolist()):
                if u != IGNORE and u not in FINE_TO_COARSE:
                    unexpected_src[u] = unexpected_src.get(u, 0) + c
            out = lut[arr]
            u = np.unique(out)
            assert u.max() == IGNORE or u.max() <= 26, 'out-of-range id in %s' % f
            grand_hist += np.bincount(out.ravel(), minlength=256)
            if (out == IGNORE).all():
                all_ignore += 1
            Image.fromarray(out).save(out_path)
            total += 1
            if total % 2000 == 0:
                r = total / (time.time() - t0)
                print('  %d done  %.1f img/s' % (total, r), flush=True)

    valid = grand_hist[:27].sum()
    px = grand_hist.sum()
    print('== SUMMARY ==')
    print('converted: %d files  ->  %s' % (total, out_root))
    print('ignore(255) pixel fraction: %.4f' % (grand_hist[IGNORE] / max(px, 1)))
    print('valid-class pixel fraction: %.4f' % (valid / max(px, 1)))
    print('classes present: %d/27' % int((grand_hist[:27] > 0).sum()))
    print('all-ignore label files: %d' % all_ignore)
    print('coarse histogram (px):', {i: int(grand_hist[i]) for i in range(27) if grand_hist[i]})
    if unexpected_src:
        print('WARNING unexpected source ids (mapped to 255):', unexpected_src)
    ok = (not unexpected_src) and grand_hist[IGNORE] / max(px, 1) < 0.5
    print('RESULT: %s' % ('PASS' if ok else 'CHECK-FAILED (see above)'))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
