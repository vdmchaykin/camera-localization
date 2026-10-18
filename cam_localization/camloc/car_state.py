"""Allowed car-state (motion) inputs from /constructor0/state_estimation — explicit whitelist.

INTRO slide 10: speed, accelerations, yaw rate and other motion fields are allowed as inputs; pose
(x, y, z, roll, pitch, yaw), s/n and anything derived from the map pose are ground truth and forbidden.
Checked against the data (2026-10-16):
  - vx/vy/vz_mps are in the body frame (vx == v_mps), so they carry no heading -> allowed;
  - yaw_vel_rad is the heading of the velocity vector in `map` (corr 0.997 with yaw_rad), NOT a yaw
    rate -> forbidden;
  - kappa_radpm * v_mps == wz_radps -> path curvature, allowed.
Unsure fields (beta, accuracies, status) are left out until confirmed at the Q&A.
"""

ALLOWED = [
    'v_mps', 'v_raw_mps', 'vx_mps', 'vy_mps', 'vz_mps',
    'wx_radps', 'wy_radps', 'wz_radps',
    'ax_mps2', 'ay_mps2', 'az_mps2', 'ax_vel_mps2', 'ay_vel_mps2',
    'kappa_radpm', 'dbeta_radps', 'ddyaw_radps2',
    'omega_w_fl', 'omega_w_fr', 'omega_w_rl', 'omega_w_rr', 'valid_wheelsspeeds_b',
    'lambda_fl_perc', 'lambda_fr_perc', 'lambda_rl_perc', 'lambda_rr_perc',
    'alpha_fl_rad', 'alpha_fr_rad', 'alpha_rl_rad', 'alpha_rr_rad', 'diff_fr_alpha_rad', 'delta_wheel_rad',
    'gas', 'brake', 'clutch', 'gear', 'rpm', 'front_brake_pressure', 'rear_brake_pressure',
]
FORBIDDEN = ['x_m', 'y_m', 'z_m', 'roll_rad', 'pitch_rad', 'yaw_rad', 'yaw_vel_rad', 'pos_accuracy*',
             'sn_map_state.*']
UNSURE = ['beta_rad', 'vel_accuracy*', 'se_status', 'se_state', 'is_safe', 'valid_imu_b']

