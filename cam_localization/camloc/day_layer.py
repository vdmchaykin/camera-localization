"""Day descriptor layer on top of daylight.py: model/day/day_<cam>.npy, built by tools/build_day_layer.py from a
day-styled copy of the training run (Qwen image editing; same stamps and geometry, only the lighting changes).

Same interface as camloc/base.py ... camloc/daylight.py (inherited, none of them is modified):
    loc = Localizer(first_pose, camera_info, tf_static, track_bounds)
    x, y, yaw = loc.step(stamp_ns, images, car_state)

Each map point keeps its 3D position and gets two descriptors: the night one (features/db_<cam>.npz) and the day
one (same keyframe position, day-styled image, v6 query preprocessing). In the guided correction the similarity of
a map point to a query descriptor is max(night, day): for the keypoint matches (mutual best inside the window) and
for the dense search. Everything else is v6 (relaxed retry, drivable-area check, rate-limited global path).
  CAMLOC_DAY=1   use the day layer (0 = exactly v6)
"""
import os, sys
import numpy as np
import torch

from lightglue.superpoint import sample_descriptors
from . import daylight as v6, dense as v5, guided as v2
from .camera_model import CAMS, H, W, yaw_of
from .paths import DAY as DAY_DIR

DAY = os.environ.get('CAMLOC_DAY', '1') == '1'


class Localizer(v6.Localizer):
    def __init__(self, first_pose, camera_info, tf_static, track_bounds):
        self._day = {}
        if DAY:
            self._day = {c: np.load(os.path.join(DAY_DIR, f'day_{c}.npy'), mmap_mode='r') for c in CAMS}
        self.n_day_better = []                      # share of matched map points whose day similarity was higher
        super().__init__(first_pose, camera_info, tf_static, track_bounds)
        self.n_day_better.clear()

    def _kf_desc(self, c, i):
        d = self.reloc.db.db[c]
        a, b = d['off'][i], d['off'][i + 1]
        return np.asarray(self._day[c][a:b])

    # ------------------------------------------------------------ guided: v6 wrapper around the dual-descriptor v5 step
    def _guided(self, stamp, qf, prior):
        if not DAY:
            return super()._guided(stamp, qf, prior)
        res = self._guided_dual(stamp, qf, prior)
        if res is None and v6.RELAX and qf:
            self.n_relax += 1
            saved = v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE
            v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE = v6.RELAX_SIM_MIN, v6.RELAX_DENSE_SIM, 1
            try:
                res = self._guided_dual(stamp, qf, prior)
            finally:
                v2.SIM_MIN, v5.DENSE_SIM, v5.DENSE = saved
            self.n_relax_ok += res is not None
        if res is not None and v6.BOUNDS and not self.drivable.inside(res[0], res[1]):
            self.n_bounds_rej += 1
            return None
        return res

    @torch.no_grad()
    def _dense_match_dual(self, c, uv, Dn, Dd, win):
        """v5._dense_match with sim = max(night, day) + v6 saturation filter."""
        if v5.G_T is None:
            v5.G_T = torch.from_numpy(v5._G.astype(np.float32)).cuda()
        G_T = v5.G_T
        dm = self._dense.get(c)
        if dm is None or not len(uv):
            return np.zeros((0, 2), np.float32), np.zeros(0, int)
        n, g, R = len(uv), len(v5._G), v5.DENSE_R
        uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
        r = torch.from_numpy(np.clip(win, 3, R).astype(np.float32)).cuda()
        pos = uv_t[:, None, :] + G_T[None]
        inside = (pos[..., 0] >= 4) & (pos[..., 0] <= W - 5) & (pos[..., 1] >= 4) & (pos[..., 1] <= H - 5)
        inwin = inside & (G_T.norm(dim=1)[None] <= r[:, None])
        desc = sample_descriptors(pos.reshape(1, -1, 2), dm, 8)[0].reshape(-1, n, g)
        sim = torch.maximum(torch.einsum('nc,cng->ng', torch.from_numpy(Dn).cuda().float(), desc),
                            torch.einsum('nc,cng->ng', torch.from_numpy(Dd).cuda().float(), desc))
        sim[~inwin] = -2
        best, bi = sim.max(1)
        far = (G_T[None] - G_T[bi][:, None]).norm(dim=2) > 2.5
        second = torch.where(far, sim, torch.full_like(sim, -2)).max(1).values
        ok = (best > v5.DENSE_SIM) & (best - second > v5.DENSE_MARGIN)
        S = 2 * R + 1
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
        if v6.SAT and c in self._ok:
            ok &= self._ok[c][fi[:, 1].clip(0, H - 1), fi[:, 0].clip(0, W - 1)] > 0
        return found[ok].astype(np.float32), np.where(ok)[0]

    @torch.no_grad()
    def _guided_dual(self, stamp, qf, prior):
        """v5._guided with two descriptors per map point (similarity = max of night and day)."""
        kfs = self._nearest_kfs(prior, stamp)
        if not len(kfs):
            return None
        T0 = self._T_prior(prior, kfs[0])
        per, n_kp = {}, 0
        for c, f in qf.items():
            pts = [self.reloc.db.kf_points(c, i) for i in kfs]
            X = np.concatenate([q[1] for q in pts])
            Dn = np.concatenate([q[0]['desc'] for q in pts])
            Dd = np.concatenate([self._kf_desc(c, i) for i in kfs])
            if len(X) < 8:
                continue
            cam = self.cams[c]
            Tc = T0 @ cam.T_base_cam
            pc = (X - Tc[:3, 3]) @ Tc[:3, :3]
            uv = cam.project_cam(pc)
            vis = np.isfinite(uv[:, 0])
            if vis.sum() < 8:
                continue
            X, Dn, Dd, uv, depth = X[vis], Dn[vis], Dd[vis], uv[vis], pc[vis, 2]
            f_px = cam.K[0, 0]
            win = np.clip(f_px * 2.5 * prior['sig_xy'] / depth + f_px * 2.5 * prior['sig_yaw'] + 3, v2.WIN_MIN, v2.WIN_MAX)
            used = np.zeros(len(X), bool)
            kp_uv, kp_X = np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)
            if v5.DENSE != 3 and len(f['kps']) >= 8:
                uv_t = torch.from_numpy(uv.astype(np.float32)).cuda()
                k_t = torch.from_numpy(f['kps']).cuda()
                inwin = torch.cdist(uv_t, k_t) < torch.from_numpy(win.astype(np.float32)).cuda()[:, None]
                q = torch.from_numpy(f['desc']).cuda().float().T
                Sn = torch.from_numpy(Dn).cuda().float() @ q
                Sd = torch.from_numpy(Dd).cuda().float() @ q
                S = torch.maximum(Sn, Sd)
                S[~inwin] = -1
                bj = S.argmax(1); bi = S.argmax(0)
                i = torch.arange(len(X), device='cuda')
                ok = ((S[i, bj] > v2.SIM_MIN) & (bi[bj] == i))
                if ok.any():
                    self.n_day_better.append(float((Sd[i, bj] > Sn[i, bj])[ok].float().mean()))
                ok = ok.cpu().numpy()
                bj = bj.cpu().numpy()
                if ok.sum() >= 4:
                    used = ok
                    kp_uv, kp_X = f['kps'][bj[ok]], X[ok]
            n_kp += len(kp_uv)
            per[c] = (X, Dn, Dd, uv, win, used, kp_uv, kp_X)
        use_dense = v5.DENSE in (1, 3) or (v5.DENSE == 2 and n_kp < v5.DENSE_TRIGGER)
        corr, n_dense = {}, 0
        for c, (X, Dn, Dd, uv, win, used, kp_uv, kp_X) in per.items():
            U, XX = [kp_uv], [kp_X]
            if use_dense and (~used).any():
                rest = np.where(~used)[0]
                fu, j = self._dense_match_dual(c, uv[rest], Dn[rest], Dd[rest], win[rest])
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
        s.update(day_layer=dict(day=DAY, day_better_share_median=round(float(np.median(self.n_day_better)), 3)
                          if self.n_day_better else None))
        return s
