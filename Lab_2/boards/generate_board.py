#!/usr/bin/env python3
"""
generate_board.py  ―  Produce a printable checkerboard and ChArUco board on A4.

* Checkerboard: 9 × 6 squares, 30 mm each            (≈ 270 mm × 180 mm)
* ChArUco:      5 × 7 squares, same square size
* Output: checkerboard.png, charuco.png, calibration_boards.pdf
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ─── Parameters ────────────────────────────────────────────────────────────────
DPI        = 300          # print resolution
SQUARE_MM  = 30           # edge length of one square
CHK_NX, CHK_NY = 9, 6     # checkerboard squares (not inner corners!)
CHU_NX, CHU_NY = 5, 7     # ChArUco squares
MARGIN_PX  = 50           # white margin between boards on the PDF
OUT_DIR    = Path('.')

# ─── Helpers ───────────────────────────────────────────────────────────────────
MM_TO_PX = DPI / 25.4
mm2px = lambda mm: int(round(mm * MM_TO_PX))

def make_checker(nx, ny, sq_mm):
    sq_px = mm2px(sq_mm)
    w, h  = nx * sq_px, ny * sq_px
    img   = np.zeros((h, w), np.uint8)
    for y in range(ny):
        for x in range(nx):
            if (x + y) & 1 == 0:
                img[y*sq_px:(y+1)*sq_px, x*sq_px:(x+1)*sq_px] = 255
    return img

def make_charuco(nx, ny, sq_mm, dpi):
    # dictionary & board
    aruco = cv2.aruco
    dct   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard_create(
        squaresX      = nx,
        squaresY      = ny,
        squareLength  = sq_mm / 1000.0,         # metres, only meta-data
        markerLength  = sq_mm * 0.7 / 1000.0,   # 70 % of square edge
        dictionary    = dct
    )
    w_px, h_px = mm2px(nx * sq_mm), mm2px(ny * sq_mm)
    return board.draw((w_px, h_px))

def assemble_pdf(images, dpi, out_path):
    a4_w_px, a4_h_px = mm2px(210), mm2px(297)
    canvas = Image.new('RGB', (a4_w_px, a4_h_px), 'white')
    y = MARGIN_PX
    for img in images:
        pil = Image.fromarray(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        x   = (a4_w_px - pil.width) // 2
        canvas.paste(pil, (x, y))
        y  += pil.height + MARGIN_PX
    canvas.save(out_path, 'PDF', resolution=dpi)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    checker = make_checker(CHK_NX, CHK_NY, SQUARE_MM)
    charuco = make_charuco(CHU_NX, CHU_NY, SQUARE_MM, DPI)

    cv2.imwrite(str(OUT_DIR / 'checkerboard.png'), checker)
    cv2.imwrite(str(OUT_DIR / 'charuco.png'),     charuco)

    assemble_pdf([checker, charuco], DPI, OUT_DIR / 'calibration_boards.pdf')
    print("✅  Saved calibration_boards.pdf (and PNGs) in", OUT_DIR.resolve())

if __name__ == "__main__":
    main()