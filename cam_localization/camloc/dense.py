"""v5 localizer: v3 + more keypoints, dense guided search, keyframes from different passes.

Same interface as camloc/base.py, camloc/guided.py, camloc/fast_sp.py (inherited, none of them is modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Changes vs v3, each switchable by env var for ablations:
  V5_KP=1024     SuperPoint keypoints per camera on the guided path (v3: 512). The global (v1) path keeps 512.
  V5_DENSE       dense guided search: for a projected map point, the SuperPoint dense descriptor map is sampled on a
                 1 px grid inside its window (radius <= DENSE_R) and the best position is taken (sub-pixel by a
                 parabola), accepted when similar enough and distinct from the rest of the window. Does not depend
                 on where the detector fires (lighting, blur).
                   0 = off (v3 matching), 1 = keypoint matches + dense for the map points left unmatched,
                   2 = dense only when the keypoint matches are too few (< DENSE_TRIGGER), 3 = dense only.
  V5_KF=3        keyframes whose points are projected, chosen nearest first but at most one per pass (stamps
                 > PASS_GAP_S apart), so map errors of one pass average out (v3: 2 nearest, often one pass).
"""
import os, sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lightglue.superpoint import sample_descriptors, simple_nms, top_k_keypoints
from . import fast_sp as v3, guided as v2
from .camera_model import CAMS, H, W, yaw_of
from .track_filter import wrap

MAX_KP = int(os.environ.get('V5_KP', '1024'))
DENSE = int(os.environ.get('V5_DENSE', '1'))
N_KF = int(os.environ.get('V5_KF', '3'))
PASS_GAP_S = 20.0           # keyframes closer in time than this are the same pass
DENSE_R = 8                 # px, max search radius of the dense search
DENSE_SIM = float(os.environ.get('V5_DENSE_SIM', '0.75'))
DENSE_MARGIN = float(os.environ.get('V5_DENSE_MARGIN', '0.03'))   # best minus best outside 2.5 px of it
DENSE_TRIGGER = 150         # mode 2: dense when fewer keypoint correspondences than this
_G = np.stack(np.meshgrid(np.arange(-DENSE_R, DENSE_R + 1), np.arange(-DENSE_R, DENSE_R + 1)), -1).reshape(-1, 2)
G_T = None                  # grid offsets on the GPU, (g, 2)


class Localizer(v3.Localizer):
    def __init__(self, *a, **k):
        self.n_kp_corr, self.n_dense_corr, self.n_frames_dense = [], [], 0
        super().__init__(*a, **k)
        self.n_kp_corr.clear(); self.n_dense_corr.clear(); self.n_frames_dense = 0

    # ------------------------------------------------------------ features (v3 + MAX_KP + dense maps kept)
    @torch.no_grad()
    def _features(self, half):
        names = [c for c in CAMS if c in half]
        g = np.zeros((len(CAMS), H, W), np.uint8)
        for i, c in enumerate(names):
            g[i] = v2.v1._CLAHE.apply(cv2.cvtColor(half[c], cv2.COLOR_BGR2GRAY))
        g = self._preprocess(g, names)
        sp, bb = self._fast_sp()
        x = torch.from_numpy(g).cuda().half().div_(255)[:, None].contiguous(memory_format=torch.channels_last)
        logits, dense = bb(x)
        n = len(names)
        sc = F.softmax(logits[:n].float(), 1)[:, :-1]
        b, _, h, w = sc.shape
        sc = sc.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        sc = simple_nms(sc, sp.conf.nms_radius)
        p = sp.conf.remove_borders
        if p:
            sc[:, :p] = -1; sc[:, :, :p] = -1; sc[:, -p:] = -1; sc[:, :, -p:] = -1
        dense = F.normalize(dense[:n].float(), p=2, dim=1)
        self._dense = {c: dense[i:i + 1] for i, c in enumerate(names)}
        out = {}
        for i, c in enumerate(names):
            ij = torch.nonzero(sc[i] > sp.conf.detection_threshold)
            s = sc[i][ij[:, 0], ij[:, 1]]
            ij, s = top_k_keypoints(ij, s, MAX_KP)
            k = torch.flip(ij, [1]).float()
            d = sample_descriptors(k[None], dense[i:i + 1], 8)[0].T
            k, s, d = k.cpu().numpy(), s.cpu().numpy(), d.cpu().numpy()
            ki = np.round(k).astype(int)
            keep = self.cams[c].mask[ki[:, 1].clip(0, H - 1), ki[:, 0].clip(0, W - 1)] > 0
            out[c] = dict(kps=k[keep].astype(np.float32), scores=s[keep].astype(np.float32),
                          desc=d[keep].astype(np.float16))
        return out

    def _preprocess(self, g, names):
        """hook for photometric preprocessing of the grey images (v6); identity here."""
        return g

    # ------------------------------------------------------------ keyframes: nearest, one per pass
    def _nearest_kfs(self, prior, stamp):
        allowed = self.allowed_base.copy()
        if self.self_excl_ns > 0:
            allowed &= np.abs(self.kf - stamp) > self.self_excl_ns
        d = np.hypot(*(self.kf_xy - [prior['x'], prior['y']]).T)
        ok = allowed & (d < v2.KF_MAX_DIST) & (np.abs(wrap(self.kf_yaw - prior['yaw'])) < v2.KF_MAX_DYAW)
        idx = np.where(ok)[0]
        out = []
        for i in idx[np.argsort(d[idx])]:
            if all(abs(int(self.kf[i]) - int(self.kf[j])) > PASS_GAP_S * 1e9 for j in out):
                out.append(i)
                if len(out) == N_KF:
                    break
        return np.array(out, int)

    # ------------------------------------------------------------ dense guided search
    @torch.no_grad()
    def _dense_match(self, c, uv, D, win):
        """uv (n,2) projected map points, D (n,256) map descriptors, win (n,) px -> (uv_found (m,2), index (m,))"""
        global G_T
        if G_T is None:
            G_T = torch.from_numpy(_G.astype(np.float32)).cuda()
        dm = self._dense.get(c)
        if dm is None or not len(uv):
            return np.zeros((0, 2), np.float32), np.zeros(0, int)
        n, g = len(uv), len(_G)
        uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
        r = torch.from_numpy(np.clip(win, 3, DENSE_R).astype(np.float32)).cuda()
        pos = uv_t[:, None, :] + G_T[None]                                          # (n, g, 2)
        inside = (pos[..., 0] >= 4) & (pos[..., 0] <= W - 5) & (pos[..., 1] >= 4) & (pos[..., 1] <= H - 5)
        inwin = inside & (G_T.norm(dim=1)[None] <= r[:, None])
        desc = sample_descriptors(pos.reshape(1, -1, 2), dm, 8)[0]                  # (256, n*g)
        Dm = torch.from_numpy(D).cuda().float()
        sim = torch.einsum('nc,cng->ng', Dm, desc.reshape(-1, n, g))
        sim[~inwin] = -2
        best, bi = sim.max(1)
        far = (G_T[None] - G_T[bi][:, None]).norm(dim=2) > 2.5                      # (n, g)
        second = torch.where(far, sim, torch.full_like(sim, -2)).max(1).values
        ok = (best > DENSE_SIM) & (best - second > DENSE_MARGIN)
        # sub-pixel: parabola through the neighbours on the grid (x and y separately)
        S = 2 * DENSE_R + 1
        by, bx = bi // S, bi % S
        sim3 = sim.reshape(n, S, S)
        ar = torch.arange(n, device='cuda')
        def para(m, c0, p):
            den = m - 2 * c0 + p
            return torch.where((den < -1e-6), 0.5 * (m - p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        cx = (bx > 0) & (bx < S - 1); cy = (by > 0) & (by < S - 1)
        c0 = sim3[ar, by, bx]
        dx = torch.where(cx, para(sim3[ar, by, (bx - 1).clamp(0)], c0, sim3[ar, by, (bx + 1).clamp(max=S - 1)]), 0 * c0)
        dy = torch.where(cy, para(sim3[ar, (by - 1).clamp(0), bx], c0, sim3[ar, (by + 1).clamp(max=S - 1), bx]), 0 * c0)
        found = pos[ar, bi] + torch.stack([dx, dy], 1)
        ok, found = ok.cpu().numpy(), found.cpu().numpy()
        fi = np.round(found).astype(int)
        ok &= self.cams[c].mask[fi[:, 1].clip(0, H - 1), fi[:, 0].clip(0, W - 1)] > 0
        return found[ok].astype(np.float32), np.where(ok)[0]

    # ------------------------------------------------------------ guided correction (v2 + dense + per-pass kfs)
    @torch.no_grad()
    def _guided(self, stamp, qf, prior):
        kfs = self._nearest_kfs(prior, stamp)
        if not len(kfs):
            return None
        T0 = self._T_prior(prior, kfs[0])
        per = {}
        n_kp = 0
        for c, f in qf.items():
            pts = [self.reloc.db.kf_points(c, i) for i in kfs]
            X = np.concatenate([q[1] for q in pts])
            D = np.concatenate([q[0]['desc'] for q in pts])
            if len(X) < 8:
                continue
            cam = self.cams[c]
            Tc = T0 @ cam.T_base_cam
            pc = (X - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project_cam(pc)
            vis = np.isfinite(uv[:, 0])
            if vis.sum() < 8:
                continue
            X, D, uv, depth = X[vis], D[vis], uv[vis], pc[vis, 2]
            f_px = cam.K[0, 0]
            win = np.clip(f_px * 2.5 * prior['sig_xy'] / depth + f_px * 2.5 * prior['sig_yaw'] + 3, v2.WIN_MIN, v2.WIN_MAX)
            used = np.zeros(len(X), bool)
            kp_uv, kp_X = np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)
            if DENSE != 3 and len(f['kps']) >= 8:
                uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
                k_t = torch.from_numpy(f['kps']).cuda()
                inwin = torch.cdist(uv_t, k_t) < torch.from_numpy(win.astype(np.float32)).cuda()[:, None]
                S = torch.from_numpy(D).cuda().float() @ torch.from_numpy(f['desc']).cuda().float().T
                S[~inwin] = -1
                bj = S.argmax(1); bi = S.argmax(0)
                i = torch.arange(len(X), device='cuda')
                ok = ((S[i, bj] > v2.SIM_MIN) & (bi[bj] == i)).cpu().numpy()
                bj = bj.cpu().numpy()
                if ok.sum() >= 4:
                    used = ok
                    kp_uv, kp_X = f['kps'][bj[ok]], X[ok]
            n_kp += len(kp_uv)
            per[c] = (X, D, uv, win, used, kp_uv, kp_X)
        use_dense = DENSE in (1, 3) or (DENSE == 2 and n_kp < DENSE_TRIGGER)
        corr, n_dense = {}, 0
        for c, (X, D, uv, win, used, kp_uv, kp_X) in per.items():
            U, XX = [kp_uv], [kp_X]
            if use_dense and (~used).any():
                rest = np.where(~used)[0]
                fu, j = self._dense_match(c, uv[rest], D[rest], win[rest])
                U.append(fu); XX.append(X[rest[j]]); n_dense += len(fu)
            U, XX = np.concatenate(U), np.concatenate(XX)
            if len(U) >= 4:
                corr[c] = (self.cams[c].undistort_points(U), XX)
        self.n_kp_corr.append(n_kp); self.n_dense_corr.append(n_dense)
        self.n_frames_dense += int(use_dense)
        if sum(len(v[0]) for v in corr.values()) < v2.MIN_INLIERS_GUIDED:
            return None
        Tb = T0
        for thr in (4 * v2.THR_PX, v2.THR_PX):
            r = self.reloc.residuals(Tb, corr)
            sel = {c: e < thr for c, e in r.items()}
            if sum(s.sum() for s in sel.values()) < v2.MIN_INLIERS_GUIDED:
                return None
            Tb = self._refine(Tb, corr, sel)
        r = self.reloc.residuals(Tb, corr)
        n_inl = int(sum((e < v2.THR_PX).sum() for e in r.values()))
        if n_inl < v2.MIN_INLIERS_GUIDED:
            return None
        return float(Tb[0, 3]), float(Tb[1, 3]), float(yaw_of(Tb)), n_inl

    def summary(self):
        s = super().summary()
        s.update(v5=dict(max_kp=MAX_KP, dense=DENSE, n_kf=N_KF, dense_sim=DENSE_SIM, dense_margin=DENSE_MARGIN,
                         kp_corr_median=int(np.median(self.n_kp_corr)) if self.n_kp_corr else 0,
                         dense_corr_median=int(np.median(self.n_dense_corr)) if self.n_dense_corr else 0,
                         frames_with_dense=self.n_frames_dense))
        return s
