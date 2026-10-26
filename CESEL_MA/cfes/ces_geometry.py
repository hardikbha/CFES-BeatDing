"""CES cue supports: which backbone cells and which pixels each cue branch reads.

All supports derive from the CFES face template (cf.face_mask: centre cx, cy and semi-axes ax, ay as fractions of the
crop side; FF++/Lin: cx .5 cy .48 ax .30 ay .38, FairForge-2 canonical crop: cx .5 cy .578 ax .3125 ay .4297).

Normalised elliptical radius of pixel p = (x, y) (pixel centres, S = input size):
    r(p) = sqrt( ((x - cx S) / (ax S))^2 + ((y - cy S) / (ay S))^2 )         r <= 1 is the face core ellipse.

Cell masks on the backbone's G x G token grid (cell fraction = share of a cell's pixels inside the region, computed
with adaptive average pooling, so any S and G work):
    core cells  (texture)  : fraction of pixels with r <= core_r           >= core_frac   (defaults 0.78, 0.90)
    ring cells  (boundary) : fraction of pixels with ring_lo <= r <= ring_hi >= ring_frac  (defaults 0.80, 1.00, 0.30)
The two sets are disjoint by construction (a core cell has >= 90% of its pixels at r <= 0.78, hence <= 10% in the
ring); this is asserted. At S = 224, G = 16 with the FF++ template: 46 core cells and 44 ring cells (the CFEA counts,
results/cfea/2026-10-24-cfea-design.md Sec. 2); `describe()` prints the masks.

Frequency support (pixel space): the bounding box of {r <= freq_r} (default 1.0, the whole core ellipse), multiplied
by a soft elliptical window  w(p) = clamp((freq_r - r(p)) / freq_feather, 0, 1)  (1 for r <= freq_r - feather, 0 for
r >= freq_r), mean-removed per channel, and zero-padded (centred, no resampling) to a fixed P x P canvas, P = the
crop's longer side rounded up to a multiple of 16. No resampling means the spectrum keeps the generator's own
frequency content (upsampling grids are not blurred by a resize).
"""
import math

import torch
import torch.nn.functional as F


def elliptical_radius(size, cx, cy, ax, ay, device="cpu"):
    t = torch.arange(size, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.sqrt(((xx - cx * size) / (ax * size)) ** 2 + ((yy - cy * size) / (ay * size)) ** 2)


class CueGeometry:
    def __init__(self, size, grid, face, core_r=0.78, core_frac=0.90, ring=(0.80, 1.00), ring_frac=0.30,
                 freq_r=1.0, freq_feather=0.05):
        self.size, self.grid = int(size), int(grid)
        r = elliptical_radius(size, face["cx"], face["cy"], face["ax"], face["ay"])
        self.radius = r

        def frac(m):
            return F.adaptive_avg_pool2d(m.float()[None, None], grid)[0, 0]
        self.core_cells = frac(r <= core_r) >= core_frac - 1e-6
        self.ring_cells = frac((r >= ring[0]) & (r <= ring[1])) >= ring_frac - 1e-6
        if not self.core_cells.any() or not self.ring_cells.any():
            raise ValueError(f"empty cue support at size {size}, grid {grid}: core {int(self.core_cells.sum())} "
                             f"ring {int(self.ring_cells.sum())} cells; adjust ces.geometry thresholds")
        if (self.core_cells & self.ring_cells).any():
            raise ValueError("texture (core) and boundary (ring) cell masks overlap; adjust ces.geometry thresholds")
        inside = r <= freq_r
        ys, xs = torch.nonzero(inside, as_tuple=True)
        self.crop = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)    # y0, y1, x0, x1
        y0, y1, x0, x1 = self.crop
        self.window = ((freq_r - r) / freq_feather).clamp(0, 1)[y0:y1, x0:x1].contiguous()   # (h, w)
        h, w = y1 - y0, x1 - x0
        self.canvas = int(16 * math.ceil(max(h, w) / 16))
        self.params = dict(core_r=core_r, core_frac=core_frac, ring=list(ring), ring_frac=ring_frac, freq_r=freq_r,
                           freq_feather=freq_feather)

    @classmethod
    def from_cfg(cls, cfg, grid):
        g = cfg.get("ces", {}).get("geometry", {}) or {}
        return cls(cfg["data"]["input_size"], grid, cfg["cf"]["face_mask"], g.get("core_r", 0.78),
                   g.get("core_frac", 0.90), tuple(g.get("ring", (0.80, 1.00))), g.get("ring_frac", 0.30),
                   g.get("freq_r", 1.0), g.get("freq_feather", 0.05))

    def face_crop(self, x01, window=None):
        """(N,3,S,S) in [0,1] -> (N,3,P,P): windowed, per-channel mean-removed face crop on a zero canvas.
        window: the same (h,w) window already on x01's device (FreqNet keeps it as a buffer)."""
        y0, y1, x0, x1 = self.crop
        c = x01[:, :, y0:y1, x0:x1].float()
        w = (self.window if window is None else window).to(c.device)
        mu = (c * w).sum((2, 3), keepdim=True) / w.sum()
        c = (c - mu) * w
        h, wd = c.shape[-2:]
        P = self.canvas
        top, left = (P - h) // 2, (P - wd) // 2
        return F.pad(c, (left, P - wd - left, top, P - h - top))

    def describe(self):
        def art(m):
            return "\n".join("".join("#" if v else "." for v in row.tolist()) for row in m)
        return {"size": self.size, "grid": self.grid, "core_cells": int(self.core_cells.sum()),
                "ring_cells": int(self.ring_cells.sum()), "freq_crop_yyxx": list(self.crop), "freq_canvas": self.canvas,
                "params": self.params, "core_mask": art(self.core_cells), "ring_mask": art(self.ring_cells)}
