"""Car-state (motion) inputs read from /constructor0/state_estimation — explicit whitelist: only the fields the EKF
motion model uses (body-frame velocity and yaw rate).

INTRO slide 10: speed, accelerations, yaw rate and other motion fields are allowed as inputs; pose
(x, y, z, roll, pitch, yaw), s/n and anything derived from the map pose are ground truth and forbidden.
Checked against the data (2026-10-16):
  - vx/vy_mps are in the body frame (vx == v_mps), so they carry no heading -> allowed;
  - yaw_vel_rad is the heading of the velocity vector in `map` (corr 0.997 with yaw_rad), NOT a yaw
    rate -> forbidden, never read.
"""

ALLOWED = ['vx_mps', 'vy_mps', 'wz_radps']
