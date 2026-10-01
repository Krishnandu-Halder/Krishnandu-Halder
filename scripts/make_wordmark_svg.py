

"""
Render "KRISH" as an EXTRUDED 3D wordmark rasterized to ASCII,
and emit it as an SVG that animates on GitHub using SMIL only.

Pipeline:
    1. Draw the word with a bold TTF.
    2. Threshold the word into a mask.
    3. Extrude the mask along +Z into a surface voxel shell.
    4. Rotate / project each frame.
    5. Z-buffer splat into a character grid.
    6. Pick characters using Lambert shading of the surface normal.

Animation modes:
    rock   -- oscillates +/-11 degrees around the rest pose forever.
    once   -- one full 360-degree turn, then freezes.
    spin   -- continuous 360-degree turntable forever.
    static -- frozen frame 0, no animation.

The generated SVG uses SMIL only, so no JavaScript is required.
"""

import argparse
import html
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# WORDMARK CONFIGURATION
# ============================================================================

# Fixed wordmark.
TEXT = "KRISH"


# ============================================================================
# GEOMETRY / GRID
# ============================================================================

COLS = int(os.environ.get("WORDMARK_COLS", 50))

# Derived automatically from the rendered artwork.
ROWS = 0

# Blank rows above and below the art.
ROW_MARGIN = int(os.environ.get("WORDMARK_ROW_MARGIN", 5))

# Terminal / ASCII cell dimensions.
CELL_W = 9.0
CELL_H = 15.5


# ============================================================================
# FONT
# ============================================================================

# Futura Bold on macOS.
FONT_PATH = os.environ.get(
    "WORDMARK_FONT",
    "/System/Library/Fonts/Futura.ttc",
)

# Face inside the .ttc font collection.
FONT_INDEX = int(
    os.environ.get("WORDMARK_FONT_INDEX", 2)
)


# ============================================================================
# WORD RENDERING
# ============================================================================

# Rasterized glyph height.
MASK_H = 300

# Extra letter spacing.
TRACKING = 0.14

# Baseline-to-baseline spacing for multi-line text.
LINE_GAP = 1.20

# Extrusion depth as a fraction of glyph height.
DEPTH_FRAC = 0.34

# Slight X tilt keeps the top surface visible.
TILT_DEG = float(
    os.environ.get("WORDMARK_TILT", 4.0)
)

# Camera.
CAM_DIST = 6.0
FOCAL = 4.15

# Maximum amount of the grid used by the artwork.
FIT = 0.92


# ============================================================================
# ASCII SHADING
# ============================================================================

# Sparse/dim -> dense/bright.
RAMP = " .`:-=+*csS#%@"

# Lighting direction.
LIGHT = np.array(
    [-0.15, -0.45, -1.00],
    dtype=np.float32,
)

LIGHT = LIGHT / np.linalg.norm(LIGHT)

# Minimum ambient brightness.
AMBIENT = 0.22

# Depth fog.
FOG = 0.34
FOG_SPAN = 0.55


# ============================================================================
# SVG PALETTE
# ============================================================================

BG = "#0d1117"
BG2 = "#111722"
FRAME = "#30363d"
TITLE_TEXT = "#7d8590"
INK = "#c9d1d9"


# ============================================================================
# SVG LAYOUT
# ============================================================================

PAD = 18
TITLEBAR_H = 28


# ============================================================================
# BUILD 3D VOXEL SHELL
# ============================================================================

def build_shell():
    """
    Rasterize KRISH and return:

        P = 3D surface points
        N = surface normals
    """

    probe = TEXT.replace("\n", "")

    font_size = MASK_H

    # Shrink font until the word fits the target height.
    for _ in range(40):

        font = ImageFont.truetype(
            FONT_PATH,
            font_size,
            index=FONT_INDEX,
        )

        l, t, r, b = font.getbbox(probe)

        if b - t <= MASK_H:
            break

        font_size = int(font_size * 0.92)

    h = b - t

    # Letter spacing.
    track = int(
        round(TRACKING * font_size)
    )

    lines = TEXT.split("\n")

    line_h = int(
        round(h * LINE_GAP)
    )

    # ------------------------------------------------------------------------
    # Calculate width of each line.
    # ------------------------------------------------------------------------

    def line_w(s):
        return (
            sum(font.getlength(c) for c in s)
            + track * (len(s) - 1)
        )

    total_w = (
        int(
            round(
                max(
                    line_w(s)
                    for s in lines
                )
            )
        )
        + 8
    )

    total_h = (
        line_h * (len(lines) - 1)
        + h
        + 8
    )

    # ------------------------------------------------------------------------
    # Create mask.
    # ------------------------------------------------------------------------

    img = Image.new(
        "L",
        (total_w, total_h),
        0,
    )

    d = ImageDraw.Draw(img)

    # Draw every line.
    for li, s in enumerate(lines):

        # Center the line.
        pen = (
            4.0
            + (
                total_w
                - 8
                - line_w(s)
            )
            / 2.0
        )

        base = (
            -t
            + 4
            + li * line_h
        )

        # Draw each glyph separately so tracking is preserved.
        for ch in s:

            d.text(
                (pen, base),
                ch,
                font=font,
                fill=255,
            )

            pen += (
                font.getlength(ch)
                + track
            )

    # Convert to binary mask.
    mask = np.array(img) > 127

    # Remove empty borders.
    xs_any = np.nonzero(
        mask.any(0)
    )[0]

    ys_any = np.nonzero(
        mask.any(1)
    )[0]

    mask = mask[
        ys_any[0]:ys_any[-1] + 1,
        xs_any[0]:xs_any[-1] + 1,
    ]

    H, W = mask.shape

    # Extrusion depth.
    depth = max(
        4,
        int(
            round(
                H * DEPTH_FRAC
            )
        ),
    )

    # Coordinates of filled pixels.
    cy, cx = np.nonzero(mask)

    pts = []
    nrm = []

    # ------------------------------------------------------------------------
    # FRONT CAP
    # ------------------------------------------------------------------------

    front = np.stack(
        [
            cx,
            cy,
            np.full_like(
                cx,
                -0.6,
                dtype=float,
            ),
        ],
        1,
    )

    pts.append(front)

    nrm.append(
        np.tile(
            [0.0, 0.0, -1.0],
            (len(front), 1),
        )
    )

    # ------------------------------------------------------------------------
    # BACK CAP
    # ------------------------------------------------------------------------

    back = np.stack(
        [
            cx,
            cy,
            np.full_like(
                cx,
                depth,
                dtype=float,
            ),
        ],
        1,
    ).astype(float)

    pts.append(back)

    nrm.append(
        np.tile(
            [0.0, 0.0, 1.0],
            (len(back), 1),
        )
    )

    # ------------------------------------------------------------------------
    # SIDE WALLS
    # ------------------------------------------------------------------------

    pad = np.pad(
        mask,
        1,
    )

    empty_r = ~pad[
        1:-1,
        2:,
    ]

    empty_l = ~pad[
        1:-1,
        :-2,
    ]

    empty_d = ~pad[
        2:,
        1:-1,
    ]

    empty_u = ~pad[
        :-2,
        1:-1,
    ]

    # Find boundary pixels.
    edge = mask & (
        empty_r
        | empty_l
        | empty_d
        | empty_u
    )

    ey, ex = np.nonzero(edge)

    # Calculate side-wall normals.
    nx = (
        empty_r[ey, ex].astype(float)
        - empty_l[ey, ex].astype(float)
    )

    ny = (
        empty_d[ey, ex].astype(float)
        - empty_u[ey, ex].astype(float)
    )

    ln = np.sqrt(
        nx * nx
        + ny * ny
    )

    ln[ln == 0] = 1.0

    nx = nx / ln
    ny = ny / ln

    # Extrude the side walls through the depth.
    zsteps = np.linspace(
        0,
        depth,
        max(
            3,
            depth // 2,
        ),
    )

    for z in zsteps:

        pts.append(
            np.stack(
                [
                    ex,
                    ey,
                    np.full_like(
                        ex,
                        z,
                        dtype=float,
                    ),
                ],
                1,
            )
        )

        nrm.append(
            np.stack(
                [
                    nx,
                    ny,
                    np.zeros_like(nx),
                ],
                1,
            )
        )

    # ------------------------------------------------------------------------
    # COMBINE ALL SURFACE POINTS
    # ------------------------------------------------------------------------

    P = np.concatenate(
        pts
    ).astype(
        np.float32
    )

    N = np.concatenate(
        nrm
    ).astype(
        np.float32
    )

    # Center the object.
    P[:, 0] -= W / 2.0
    P[:, 1] -= H / 2.0
    P[:, 2] -= depth / 2.0

    # Normalize wordmark width to 1.0.
    P /= float(W)

    return P, N


# ============================================================================
# ROTATION
# ============================================================================

def rot_y(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float32,
    )


def rot_x(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float32,
    )


# ============================================================================
# PROJECT 3D OBJECT
# ============================================================================

def project(P, N, yaw):
    """
    Rotate and perspective-project the 3D surface.

    Returns:

        x
        y
        depth
        shade-index
    """

    M = (
        rot_x(
            math.radians(TILT_DEG)
        )
        @ rot_y(yaw)
    )

    p = P @ M.T
    n = N @ M.T

    # Camera sits at -Z.
    # Visible normals point toward the camera.
    vis = n[:, 2] < 0.0

    p = p[vis]
    n = n[vis]

    # Camera depth.
    z = p[:, 2] + CAM_DIST

    # Perspective.
    f = FOCAL / z

    # Lambert lighting.
    lam = n @ LIGHT

    inten = (
        AMBIENT
        + (1 - AMBIENT)
        * np.clip(
            lam,
            0,
            1,
        )
    )

    # ------------------------------------------------------------------------
    # Depth fog.
    # ------------------------------------------------------------------------

    t = np.clip(
        (z - CAM_DIST)
        / FOG_SPAN,
        -1.0,
        1.0,
    )

    inten *= (
        1.0
        - FOG
        * (t + 1.0)
        / 2.0
    )

    # Convert brightness to ASCII ramp index.
    idx = np.clip(
        (
            inten
            * (len(RAMP) - 1)
        ).round().astype(int),
        1,
        len(RAMP) - 1,
    )

    return (
        p[:, 0] * f,
        p[:, 1] * f,
        z,
        idx,
    )


# ============================================================================
# FIT ARTWORK TO ASCII GRID
# ============================================================================

def fit(projected):
    """
    Calculate scale and offsets needed to fit the wordmark
    inside the ASCII grid.
    """

    global ROWS

    xs = np.concatenate(
        [
            q[0]
            for q in projected
        ]
    )

    ys = np.concatenate(
        [
            q[1]
            for q in projected
        ]
    )

    x0 = xs.min()
    x1 = xs.max()

    y0 = ys.min()
    y1 = ys.max()

    ar = CELL_W / CELL_H

    scale = (
        FIT
        * (COLS - 1)
        / (x1 - x0)
    )

    ROWS = int(
        math.ceil(
            (y1 - y0)
            * ar
            * scale
        )
    ) + 1 + (
        2 * ROW_MARGIN
    )

    cx = (
        (COLS - 1) / 2.0
        - (x0 + x1) / 2.0
        * scale
    )

    cy = (
        (ROWS - 1) / 2.0
        - (y0 + y1) / 2.0
        * scale
        * ar
    )

    return scale, cx, cy


# ============================================================================
# RASTERIZE ONE FRAME
# ============================================================================

def rasterize(
    q,
    scale,
    cx,
    cy,
):
    """
    Z-buffer one projected 3D frame into
    an ASCII grid.
    """

    x, y, z, idx = q

    col = np.round(
        cx + x * scale
    ).astype(int)

    row = np.round(
        cy
        + y
        * scale
        * (CELL_W / CELL_H)
    ).astype(int)

    # Keep only points inside the grid.
    ok = (
        (col >= 0)
        & (col < COLS)
        & (row >= 0)
        & (row < ROWS)
    )

    col = col[ok]
    row = row[ok]
    z = z[ok]
    idx = idx[ok]

    # Empty grid.
    grid = np.zeros(
        (ROWS, COLS),
        np.int8,
    )

    # Far -> near.
    # Near pixels overwrite far pixels.
    order = np.argsort(-z)

    grid[
        row[order],
        col[order],
    ] = idx[order]

    return [
        "".join(
            RAMP[i]
            for i in r
        )
        for r in grid
    ]


# ============================================================================
# SVG GENERATION
# ============================================================================

def emit(
    frames,
    mode,
    out,
    dur,
    reveal,
):
    """
    Generate the final animated SVG.
    """

    art_w = COLS * CELL_W
    art_h = ROWS * CELL_H

    canvas_w = (
        art_w
        + PAD * 2
    )

    canvas_h = (
        TITLEBAR_H
        + art_h
        + PAD
    )

    art_top = (
        TITLEBAR_H
        + PAD * 0.3
    )

    fs = CELL_H * 0.92

    n = len(frames)

    # SVG output pieces.
    p = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_w:.0f}" '
            f'height="{canvas_h:.0f}" '
            f'viewBox="0 0 '
            f'{canvas_w:.0f} '
            f'{canvas_h:.0f}" '
            f'font-family="ui-monospace, '
            f'SFMono-Regular, Menlo, Consolas, monospace">'
        ),

        (
            '<defs>'
            '<linearGradient '
            'id="wbg" '
            'x1="0" '
            'y1="0" '
            'x2="0" '
            'y2="1">'
        ),

        (
            f'<stop offset="0" '
            f'stop-color="{BG2}"/>'
        ),

        (
            f'<stop offset="1" '
            f'stop-color="{BG}"/>'
        ),

        '</linearGradient></defs>',

        (
            f'<rect width="{canvas_w:.0f}" '
            f'height="{canvas_h:.0f}" '
            f'rx="12" '
            f'fill="url(#wbg)"/>'
        ),

        (
            f'<rect x="0.5" y="0.5" '
            f'width="{canvas_w - 1:.0f}" '
            f'height="{canvas_h - 1:.0f}" '
            f'rx="12" '
            f'fill="none" '
            f'stroke="{FRAME}" '
            f'stroke-width="1"/>'
        ),

        (
            f'<line x1="0" '
            f'y1="{TITLEBAR_H}" '
            f'x2="{canvas_w:.0f}" '
            f'y2="{TITLEBAR_H}" '
            f'stroke="{FRAME}"/>'
        ),
    ]

    # macOS-style window dots.
    for i, dot in enumerate(
        [
            "#ff5f56",
            "#ffbd2e",
            "#27c93f",
        ]
    ):
        p.append(
            f'<circle '
            f'cx="{PAD + i * 15}" '
            f'cy="{TITLEBAR_H / 2}" '
            f'r="4.5" '
            f'fill="{dot}"/>'
        )

    # Terminal title.
    p.append(
        f'<text '
        f'x="{canvas_w / 2:.0f}" '
        f'y="{TITLEBAR_H / 2 + 4:.0f}" '
        f'fill="{TITLE_TEXT}" '
        f'font-size="11.5" '
        f'text-anchor="middle">'
        f'Krishnandu-Halder@github: ~$ '
        f'./wordmark.sh --3d'
        f'</text>'
    )

    # ------------------------------------------------------------------------
    # Convert ASCII rows into SVG text.
    # ------------------------------------------------------------------------

    def frame_g(
        rows,
        extra="",
    ):

        out_rows = []

        for ry, line in enumerate(rows):

            s = line.rstrip()

            # Skip completely blank rows.
            if not s.strip():
                continue

            lead = (
                len(s)
                - len(
                    s.lstrip(" ")
                )
            )

            body = s[lead:]

            x = (
                PAD
                + lead * CELL_W
            )

            y = (
                art_top
                + ry * CELL_H
                + CELL_H * 0.78
            )

            out_rows.append(
                f'<text '
                f'xml:space="preserve" '
                f'x="{x:.1f}" '
                f'y="{y:.1f}" '
                f'font-size="{fs:.1f}" '
                f'textLength="'
                f'{len(body) * CELL_W:.1f}" '
                f'lengthAdjust="spacing">'
                f'{html.escape(body)}'
                f'</text>'
            )

        return (
            f'<g fill="{INK}"{extra}>'
            + "".join(out_rows)
            + "</g>"
        )

    # ------------------------------------------------------------------------
    # STATIC MODE
    # ------------------------------------------------------------------------

    if mode == "static":

        p.append(
            frame_g(frames[0])
        )

        p.append("</svg>")

        with open(
            out,
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(
                "".join(p)
            )

        print(
            "wrote",
            out,
        )

        return

    # ------------------------------------------------------------------------
    # INTRO WIPE
    # ------------------------------------------------------------------------

    p.append(
        f'<clipPath id="wipe">'
        f'<rect '
        f'x="{PAD}" '
        f'y="{art_top:.1f}" '
        f'height="{art_h:.1f}" '
        f'width="0">'
        f'<animate '
        f'attributeName="width" '
        f'from="0" '
        f'to="{art_w:.0f}" '
        f'begin="0s" '
        f'dur="{reveal:.2f}s" '
        f'fill="freeze"/>'
        f'</rect>'
        f'</clipPath>'
    )

    p.append(
        f'<g clip-path="url(#wipe)">'
        f'{frame_g(frames[0])}'
        f'<set '
        f'attributeName="opacity" '
        f'to="0" '
        f'begin="{reveal:.2f}s"/>'
        f'</g>'
    )

    # Moving reveal bar.
    p.append(
        f'<rect '
        f'x="{PAD}" '
        f'y="{art_top + 2:.1f}" '
        f'width="{CELL_W * 1.6:.1f}" '
        f'height="{art_h - 4:.1f}" '
        f'fill="{INK}" '
        f'opacity="0.16">'
        f'<animate '
        f'attributeName="x" '
        f'from="{PAD}" '
        f'to="{PAD + art_w:.0f}" '
        f'begin="0s" '
        f'dur="{reveal:.2f}s" '
        f'fill="freeze"/>'
        f'<set '
        f'attributeName="opacity" '
        f'to="0" '
        f'begin="{reveal:.2f}s"/>'
        f'</rect>'
    )

    # ------------------------------------------------------------------------
    # PLAY ONCE
    # ------------------------------------------------------------------------

    if mode == "once":

        step = dur / n

        for i, rows in enumerate(frames):

            begin = (
                reveal
                + i * step
            )

            sets = (
                f'<set '
                f'attributeName="opacity" '
                f'to="1" '
                f'begin="{begin:.3f}s"/>'
            )

            if i != n - 1:

                sets += (
                    f'<set '
                    f'attributeName="opacity" '
                    f'to="0" '
                    f'begin="{begin + step:.3f}s"/>'
                )

            p.append(
                frame_g(
                    rows,
                    ' opacity="0"',
                ).replace(
                    "</g>",
                    sets + "</g>",
                )
            )

    # ------------------------------------------------------------------------
    # LOOPING MODES
    # ------------------------------------------------------------------------

    else:

        for i, rows in enumerate(frames):

            if i == 0:

                vals = "1;0"

                kt = (
                    "0;"
                    f"{1 / n:.5f}"
                )

            else:

                vals = "0;1;0"

                kt = (
                    "0;"
                    f"{i / n:.5f};"
                    f"{(i + 1) / n:.5f}"
                )

            anim = (
                f'<animate '
                f'attributeName="opacity" '
                f'calcMode="discrete" '
                f'values="{vals}" '
                f'keyTimes="{kt}" '
                f'dur="{dur:.2f}s" '
                f'begin="{reveal:.2f}s" '
                f'repeatCount="indefinite"/>'
            )

            p.append(
                frame_g(
                    rows,
                    ' opacity="0"',
                ).replace(
                    "</g>",
                    anim + "</g>",
                )
            )

    # ------------------------------------------------------------------------
    # FINISH SVG
    # ------------------------------------------------------------------------

    p.append("</svg>")

    svg = "".join(p)

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(svg)

    print(
        f"wrote {out}  "
        f"{len(svg) / 1024:.1f} KB  "
        f"{n} frames  "
        f"{canvas_w:.0f}x{canvas_h:.0f}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Generate an animated 3D "
            "ASCII wordmark for KRISH."
        )
    )

    ap.add_argument(
        "--mode",
        choices=[
            "spin",
            "once",
            "rock",
            "static",
        ],
        default="rock",
    )

    ap.add_argument(
        "--out",
        default=None,
    )

    ap.add_argument(
        "--frames",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--dur",
        type=float,
        default=None,
    )

    ap.add_argument(
        "--reveal",
        type=float,
        default=1.6,
    )

    ap.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Print the first ASCII frame "
            "to stdout instead of creating SVG."
        ),
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------------
    # Build 3D wordmark.
    # ------------------------------------------------------------------------

    P, N = build_shell()

    # Rest pose.
    rest = math.radians(-13)

    # ------------------------------------------------------------------------
    # SPIN
    # ------------------------------------------------------------------------

    if args.mode == "spin":

        nf = (
            args.frames
            or 36
        )

        yaws = [
            rest
            + 2
            * math.pi
            * i
            / nf
            for i in range(nf)
        ]

        dur = (
            args.dur
            or 7.0
        )

    # ------------------------------------------------------------------------
    # ONCE
    # ------------------------------------------------------------------------

    elif args.mode == "once":

        nf = (
            args.frames
            or 32
        )

        yaws = [
            rest
            + 2
            * math.pi
            * i
            / nf
            for i in range(nf)
        ]

        # Return to rest pose.
        yaws.append(rest)

        dur = (
            args.dur
            or 3.6
        )

    # ------------------------------------------------------------------------
    # ROCK
    # ------------------------------------------------------------------------

    else:

        nf = (
            args.frames
            or 20
        )

        amp = math.radians(11)

        yaws = [
            rest
            + amp
            * math.sin(
                2
                * math.pi
                * i
                / nf
            )
            for i in range(nf)
        ]

        dur = (
            args.dur
            or 5.0
        )

    # ------------------------------------------------------------------------
    # Project every frame.
    # ------------------------------------------------------------------------

    proj = [
        project(
            P,
            N,
            yaw,
        )
        for yaw in yaws
    ]

    # ------------------------------------------------------------------------
    # Fit entire animation to grid.
    # ------------------------------------------------------------------------

    scale, cx, cy = fit(
        proj
    )

    # ------------------------------------------------------------------------
    # Rasterize frames.
    # ------------------------------------------------------------------------

    frames = [
        rasterize(
            q,
            scale,
            cx,
            cy,
        )
        for q in proj
    ]

    # ------------------------------------------------------------------------
    # ASCII preview.
    # ------------------------------------------------------------------------

    if args.preview:

        for row in frames[0]:

            print(
                row.rstrip()
            )

        return

    # ------------------------------------------------------------------------
    # Output file.
    # ------------------------------------------------------------------------

    out = (
        args.out
        or os.path.join(
            HERE,
            "..",
            f"wordmark.svg",
        )
    )

    emit(
        frames,
        args.mode,
        out,
        dur,
        args.reveal,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()