"""Label-preserving nuisance interventions on [0,1] image batches.

Background interventions (bg_blur, bg_swap) keep every pixel of the core face ellipse and only change the
background, so a fake keeps its manipulated face and a real stays real. The photometric intervention is a mild
global colour change; it touches the face, so it is listed separately and its effect on a frozen detector is
measured in scripts/validate.py before it is used for training.

The face ellipse is a template in normalised crop coordinates (configured per dataset). It must be checked on
overlays (validate.py writes them) because Lin et al.'s 380 px crops and FairForge-2's canonical 256 px crops
are aligned differently.
"""
import math

import torch
import torch.nn.functional as F

KINDS = {"bg_blur": 0, "bg_swap": 1, "photometric": 2}
BACKGROUND_KINDS = (KINDS["bg_blur"], KINDS["bg_swap"])


def ellipse(size, cx, cy, ax, ay, scale=1.0, device="cpu"):
    """Boolean ellipse; centre/axes as fractions of the crop side."""
    t = torch.arange(size, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return ((xx - cx * size) / (ax * scale * size)) ** 2 + ((yy - cy * size) / (ay * scale * size)) ** 2 <= 1


class FaceGeometry:
    """Masks derived from the face template.

    core      : (S,S) bool  pixels that background interventions never modify
    blend     : (1,1,S,S)   soft alpha, 1 inside the core, feathered to 0 at the dilated ellipse
    face_cells: (G,G) bool  patch cells fully inside the core (evidence must be stable here)
    bg_cells  : (G,G) bool  patch cells not touching the dilated ellipse (should carry no evidence)
    """

    def __init__(self, size, grid, cx, cy, ax, ay, dilate=1.15, device="cpu"):
        self.size, self.grid = size, grid
        self.core = ellipse(size, cx, cy, ax, ay, 1.0, device)
        outer = ellipse(size, cx, cy, ax, ay, dilate, device)
        t = torch.arange(size, dtype=torch.float32, device=device) + 0.5
        yy, xx = torch.meshgrid(t, t, indexing="ij")
        r = torch.sqrt(((xx - cx * size) / (ax * size)) ** 2 + ((yy - cy * size) / (ay * size)) ** 2)
        alpha = ((dilate - r) / (dilate - 1.0)).clamp(0, 1)
        alpha[self.core] = 1.0
        self.blend = alpha[None, None]
        cell = size // grid
        self.face_cells = F.avg_pool2d(self.core.float()[None, None], cell)[0, 0] > 0.999
        self.bg_cells = F.max_pool2d(outer.float()[None, None], cell)[0, 0] == 0

    @classmethod
    def from_cfg(cls, cfg, size, grid, device="cpu"):
        m = cfg["face_mask"]
        return cls(size, grid, m["cx"], m["cy"], m["ax"], m["ay"], m.get("dilate", 1.15), device)

    def to(self, device):
        for k in ("core", "blend", "face_cells", "bg_cells"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def gaussian_blur(x, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _photometric(x, p, gen):
    n = x.shape[0]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, 1, 1, generator=gen, device=x.device)
    b = u(-p["brightness"], p["brightness"])
    c = u(1 - p["contrast"], 1 + p["contrast"])
    s = u(1 - p["saturation"], 1 + p["saturation"])
    g = u(p["gamma"][0], p["gamma"][1])
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    y = (x - mean) * c + mean + b
    grey = y.mean(1, keepdim=True)
    y = (grey + (y - grey) * s).clamp(0, 1)
    return y.clamp_min(1e-6) ** g


@torch.no_grad()
def apply_intervention(x, kind, params, cf, geom, donors=None):
    """Apply one intervention kind to every image of x (same random parameters for every image when `params`
    is given, which is how a fake and its real source receive an identical intervention)."""
    if kind == KINDS["bg_blur"]:
        sigma = params["sigma"]
        bg = gaussian_blur(x, sigma)
    elif kind == KINDS["bg_swap"]:
        bg = donors
    elif kind == KINDS["photometric"]:
        g = torch.Generator(device=x.device).manual_seed(int(params["seed"]))
        return _photometric(x, cf["photometric"], g)
    else:
        raise ValueError(kind)
    a = geom.blend.to(x.dtype)
    return a * x + (1 - a) * bg


@torch.no_grad()
def sample_views(x, y, cf, geom, gen):
    """One nuisance view per image. Returns (x_cf, kind (N,) long). `gen` is a CPU torch.Generator.
    bg_swap takes the background of another REAL image of the batch; with fewer than 2 reals it uses bg_blur."""
    n, dev = x.shape[0], x.device
    names = cf["interventions"]
    probs = torch.tensor([cf["probs"][k] for k in names], dtype=torch.float32)
    pick = torch.multinomial(probs / probs.sum(), n, replacement=True, generator=gen)
    kind = torch.tensor([KINDS[names[i]] for i in pick.tolist()], device=dev)
    reals = torch.nonzero(y == 0).flatten()
    if len(reals) < 2:
        kind[kind == KINDS["bg_swap"]] = KINDS["bg_blur"]
    out = x.clone()
    lo, hi = cf["bg_blur_sigma"]
    for i in range(n):
        k = int(kind[i])
        xi = x[i:i + 1]
        if k == KINDS["bg_blur"]:
            sig = lo + (hi - lo) * float(torch.rand(1, generator=gen))
            out[i:i + 1] = apply_intervention(xi, k, {"sigma": sig}, cf, geom)
        elif k == KINDS["bg_swap"]:
            cand = reals[reals != i]
            d = cand[int(torch.randint(len(cand), (1,), generator=gen))]
            out[i:i + 1] = apply_intervention(xi, k, None, cf, geom, donors=x[d:d + 1])
        else:
            seed = int(torch.randint(2 ** 31 - 1, (1,), generator=gen))
            out[i:i + 1] = apply_intervention(xi, k, {"seed": seed}, cf, geom)
    return out, kind


@torch.no_grad()
def shared_intervention(fake, real, cf, geom, gen, donor_pool=None):
    """The same intervention, with the same parameters (and the same donor background), applied to each fake and
    to its real source, so whatever the intervention itself adds is matched across the pair."""
    names = cf["interventions"]
    probs = torch.tensor([cf["probs"][k] for k in names], dtype=torch.float32)
    f_out, r_out = fake.clone(), real.clone()
    kinds = []
    lo, hi = cf["bg_blur_sigma"]
    for i in range(fake.shape[0]):
        k = KINDS[names[int(torch.multinomial(probs / probs.sum(), 1, generator=gen))]]
        if k == KINDS["bg_swap"] and (donor_pool is None or len(donor_pool) == 0):
            k = KINDS["bg_blur"]
        if k == KINDS["bg_blur"]:
            p = {"sigma": lo + (hi - lo) * float(torch.rand(1, generator=gen))}
            donors = None
        elif k == KINDS["bg_swap"]:
            p = None
            j = int(torch.randint(len(donor_pool), (1,), generator=gen))
            donors = donor_pool[j:j + 1]
        else:
            p = {"seed": int(torch.randint(2 ** 31 - 1, (1,), generator=gen))}
            donors = None
        f_out[i:i + 1] = apply_intervention(fake[i:i + 1], k, p, cf, geom, donors)
        r_out[i:i + 1] = apply_intervention(real[i:i + 1], k, p, cf, geom, donors)
        kinds.append(k)
    return f_out, r_out, torch.tensor(kinds, device=fake.device)
