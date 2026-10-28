"""DINOv3-style PCA->RGB visualization of imagined latents, for backends with no pixel decoder.

ONE whitened 3-component basis per episode and a fixed sigmoid(2z) color map, so a patch keeps its
color as long as its feature stays put. Never per-frame PCA or per-frame min/max.
"""

import numpy as np
from PIL import Image


class EpisodePCA:
    """One fixed, whitened 3-component PCA basis shared by every frame of one episode."""

    def __init__(self):
        self.mu = None
        self.basis = None  # (D, 3), whitening 1/sigma folded in

    def fit(self, feats):
        """feats: (N, D) patch vectors pooled over the episode's reference frames."""
        X = np.asarray(feats, dtype=np.float64)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        signs = np.sign(U[np.abs(U).argmax(0), range(U.shape[1])])  # deterministic sign (svd_flip)
        n = max(len(X) - 1, 1)
        self.basis = (Vt[:3] * (signs[:3] * np.sqrt(n) / S[:3])[:, None]).T
        return self

    def rgb(self, feats, side_px=224):
        """feats: (P, D) one frame -> (side_px, side_px, 3) uint8, P a square grid."""
        z = (np.asarray(feats, dtype=np.float64) - self.mu) @ self.basis
        c = (255.0 / (1.0 + np.exp(-2.0 * z))).round().astype(np.uint8)
        g = int(round(len(feats) ** 0.5))
        im = Image.fromarray(c.reshape(g, g, 3)).resize((side_px, side_px), Image.BILINEAR)
        return np.asarray(im)


def save_pred_pca_strip(pca, cur_lat, pred_encs, goal_lat, cur_img, goal_img, path, banner=None, label=None):
    """Save [current px | current pca | imagined +1.. pca | goal pca | goal px] as one strip.

    cur_lat/goal_lat: (1, P, D) real-frame latents; pred_encs: (H, 1, P, D) mean-plan unroll.
    pca None = first strip of the episode -> fit the basis here and return it; pass it back in so
    later replans reuse it. The real-frame panels flanking the imagined ones are the color legend.
    """
    if pred_encs.dim() != 4:  # pixel-cost unroll: only its last step is filled
        return pca
    lats = [cur_lat[0], *pred_encs[:, 0], goal_lat[0]]
    lats = [x.float().cpu().numpy() for x in lats]
    if int(round(lats[0].shape[0] ** 0.5)) ** 2 != lats[0].shape[0]:  # not a square patch grid
        return pca
    if pca is None:
        pca = EpisodePCA().fit(np.concatenate(lats))
    side = 224
    panels = [pca.rgb(x, side) for x in lats]
    cur_px, goal_px = (
        np.asarray(Image.fromarray(im).resize((side, side), Image.BILINEAR)) for im in (cur_img, goal_img)
    )
    strip = np.concatenate([cur_px, panels[0], *panels[1:-1], panels[-1], goal_px], axis=1)
    (banner(strip, label) if banner else Image.fromarray(strip)).save(path)
    return pca
