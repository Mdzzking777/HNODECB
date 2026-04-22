"""
Generate DMT-KV Trajectory
Fixed parameters, minimal code
Now saves x1dot and x2dot directly from ODE (no numerical differentiation!)
"""

import numpy as np
import os

# ============================================================================
# Fixed Parameters
# ============================================================================

# Cantilever
k = 29.9
f0 = 313.57e3
wd = 2.0 * np.pi * f0
Q = 371.0
m = k / (wd**2)
c = m * wd / Q

# Contact & geometry
Estar = 15e6
eta_star = 1.3
R = 10e-9
A = 6.0e-20
a0 = 3.0e-10
beta = 5.0e11
dist = 24e-9

# Drive
Fd = 4.10e-9

# Surface (Kelvin-Voigt)
ks = 0.1
cs = 0.24e-6

# Simulation
t_end = 2e-3
nsteps = 125000
dt = t_end / nsteps

# ============================================================================
# RK4 Solver (modified to also return derivatives)
# ============================================================================

def rk4_step(f, t, X, dt):
    k1 = f(t, X)
    k2 = f(t + 0.5*dt, X + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, X + 0.5*dt*k2)
    k4 = f(t + dt, X + dt*k3)
    return X + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# DMT-KV Model
# ============================================================================

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def f_ts_from_state(x, v, y):
    """
    Unified DMT-like force with explicit x3dot closure.

        g(s) = sigmoid(beta * (s - a0))
        delta = max(a0 - s, 0)
        delta_dot = (a0 - s)_dot = -x_dot + y_dot

        F_ts(s) =
            -A*R / (6 * (g(s) * (s - a0) + a0)^2)
            + (1 - g(s)) * (4/3) * Estar * sqrt(R) * delta^(3/2)
            + eta_star * sqrt(R) * delta^(1/2) * delta_dot

    First solve x3dot explicitly from the x3 state equation, then recover the
    full F_ts at the same time point.
    """
    s = dist + x - y
    g = sigmoid(beta * (s - a0))
    denom = g * (s - a0) + a0
    denom = max(denom, 1.0e-15)
    adhesion = -(A * R) / (6.0 * (denom ** 2))
    delta = max(a0 - s, 0.0)
    f_hertz = (4.0 / 3.0) * Estar * np.sqrt(R) * (delta ** 1.5)
    kv_coeff = eta_star * np.sqrt(R) * np.sqrt(delta)

    # x3dot = (-F_ts - ks*y) / cs, with
    # F_ts = adhesion + (1-g)*f_hertz + kv_coeff*(-v + x3dot).
    ydot = (-adhesion - (1.0 - g) * f_hertz + kv_coeff * v - ks * y) / (cs + kv_coeff)
    delta_dot = ydot - v if delta > 0.0 else 0.0
    f_ts = adhesion + (1.0 - g) * f_hertz + kv_coeff * delta_dot
    return f_ts, ydot, delta_dot


def rhs_dmt_kv(t, X):
    """
    B) MOVING SURFACE (Kelvin-Voigt) + Hertz-DMT

    State: X = [x, v, y]
    Returns: dX/dt = [dxdt, dvdt, dydt]
    """
    x, v, y = X

    # Tip-sample distance
    s = dist + x - y

    # Unified DMT-like interaction force
    F_ts, ydot, _ = f_ts_from_state(x, v, y)
    dvdt = (Fd * np.cos(wd * t) - k*x - c*v + F_ts) / m
    dydt = ydot

    dxdt = v

    return np.array([dxdt, dvdt, dydt])

# ============================================================================
# Simulation
# ============================================================================

print("Simulating...")

# Time array
t = np.linspace(0, t_end, nsteps + 1)

# State arrays
x = np.zeros(nsteps + 1)
v = np.zeros(nsteps + 1)  # x1dot (velocity)
y = np.zeros(nsteps + 1)
x2dot = np.zeros(nsteps + 1)  # x2dot (acceleration)
ydot = np.zeros(nsteps + 1)   # x3dot (sample velocity)
delta_dot = np.zeros(nsteps + 1)
f_ts_hist = np.zeros(nsteps + 1)

# Initial condition
X = np.array([0.0, 0.0, 0.0])
x[0], v[0], y[0] = X

# Compute initial acceleration / force / sample velocity
derivs = rhs_dmt_kv(t[0], X)
x2dot[0] = derivs[1]  # dvdt
f_ts_hist[0], ydot[0], delta_dot[0] = f_ts_from_state(X[0], X[1], X[2])

# RK4 integration
for i in range(nsteps):
    X = rk4_step(rhs_dmt_kv, t[i], X, dt)
    x[i+1], v[i+1], y[i+1] = X

    # Compute acceleration directly from ODE (no numerical differentiation!)
    derivs = rhs_dmt_kv(t[i+1], X)
    x2dot[i+1] = derivs[1]  # dvdt = x2dot
    f_ts_hist[i+1], ydot[i+1], delta_dot[i+1] = f_ts_from_state(X[0], X[1], X[2])

# Compute tip-sample distance and Hertz-active region
s = dist + x - y
contact = (s <= a0)

print(f"Complete: {len(t)} time points")
print(f"Contact fraction: {contact.sum()/len(contact)*100:.2f}%")

# ============================================================================
# Save Data
# ============================================================================

# Save as CSV with x1dot (v) and x2dot (acceleration)
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'trajectory.csv')
data = np.column_stack([t, x, y, s, contact.astype(int), v, x2dot, ydot, delta_dot, f_ts_hist])
np.savetxt(csv_path, data, delimiter=',',
           header='time_s,x_tip_m,y_sample_m,s_tip_sample_distance_m,contact_status,x1dot_velocity,x2dot_acceleration,x3dot_sample_velocity,delta_dot_indentation_rate,fts_interaction_force',
           comments='')

print(f"Saved: {csv_path}")
print("  - Includes x1dot, x2dot, x3dot, delta_dot, and exact F_ts directly from ODE")
print("  - No numerical differentiation used!")

# ============================================================================
# Visualization
# ============================================================================

import matplotlib.pyplot as plt

# Create plots directory
plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plots_dir, exist_ok=True)

print("\nGenerating plots...")

# Convert to nm for better readability
x_nm = x * 1e9
y_nm = y * 1e9
s_nm = s * 1e9
t_us = t * 1e6  # time in microseconds

# --- Full trajectory plots ---

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Tip displacement
ax1 = axes[0]
ax1.plot(t_us, x_nm, 'b-', linewidth=0.5)
ax1.set_ylabel('Tip displacement x [nm]')
ax1.set_title('AFM DMT-KV Simulation: Full Trajectory')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 2: Sample motion
ax2 = axes[1]
ax2.plot(t_us, y_nm, 'r-', linewidth=0.5)
ax2.set_ylabel('Sample motion y [nm]')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 3: Tip-sample distance
ax3 = axes[2]
ax3.plot(t_us, s_nm, 'g-', linewidth=0.5)
ax3.axhline(y=a0 * 1e9, color='r', linestyle='-', linewidth=1, label='Hertz threshold (s=a0)')
ax3.fill_between(t_us, s_nm, a0 * 1e9, where=(s_nm <= a0 * 1e9), alpha=0.3, color='red', label='Hertz-active region')
ax3.set_ylabel('Distance s [nm]')
ax3.set_xlabel('Time [μs]')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'trajectory_full.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'trajectory_full.pdf'))
print(f"  Saved: trajectory_full.png/pdf")

# --- Zoomed view (steady-state region, ~1600-1650 μs) ---

zoom_start_us = 1600  # microseconds (steady-state region)
zoom_end_us = 1650    # ~15 oscillation cycles
zoom_idx = (t_us >= zoom_start_us) & (t_us <= zoom_end_us)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Tip displacement (zoomed)
ax1 = axes[0]
ax1.plot(t_us[zoom_idx], x_nm[zoom_idx], 'b-', linewidth=1)
ax1.set_ylabel('Tip displacement x [nm]')
ax1.set_title(f'AFM DMT-KV Simulation: Steady-State Zoomed View ({zoom_start_us}-{zoom_end_us} μs)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 2: Sample motion (zoomed)
ax2 = axes[1]
ax2.plot(t_us[zoom_idx], y_nm[zoom_idx], 'r-', linewidth=1)
ax2.set_ylabel('Sample motion y [nm]')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 3: Tip-sample distance (zoomed)
ax3 = axes[2]
ax3.plot(t_us[zoom_idx], s_nm[zoom_idx], 'g-', linewidth=1)
ax3.axhline(y=a0 * 1e9, color='r', linestyle='-', linewidth=1, label='Hertz threshold (s=a0)')
ax3.fill_between(t_us[zoom_idx], s_nm[zoom_idx], a0 * 1e9,
                  where=(s_nm[zoom_idx] <= a0 * 1e9), alpha=0.3, color='red', label='Hertz-active region')
ax3.set_ylabel('Distance s [nm]')
ax3.set_xlabel('Time [μs]')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'trajectory_zoomed.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'trajectory_zoomed.pdf'))
print(f"  Saved: trajectory_zoomed.png/pdf")

# --- Combined overlay plot (zoomed) ---

fig, ax = plt.subplots(figsize=(12, 6))

# Plot all three on same axes for comparison
ax.plot(t_us[zoom_idx], x_nm[zoom_idx], 'b-', linewidth=1, label='Tip displacement x')
ax.plot(t_us[zoom_idx], y_nm[zoom_idx], 'r-', linewidth=1, label='Sample motion y')
ax.plot(t_us[zoom_idx], s_nm[zoom_idx], 'g-', linewidth=1, label='Tip-sample distance s')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Mark equilibrium distance and Hertz threshold
ax.axhline(y=dist*1e9, color='gray', linestyle=':', linewidth=1, label=f'd = {dist*1e9:.1f} nm (equilibrium)')
ax.axhline(y=a0*1e9, color='tab:red', linestyle='-.', linewidth=1, label=f'a0 = {a0*1e9:.1f} nm')

ax.set_xlabel('Time [μs]')
ax.set_ylabel('Displacement [nm]')
ax.set_title(f'AFM DMT-KV: Tip, Sample, and Distance s ({zoom_start_us}-{zoom_end_us} μs, Steady-State)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'trajectory_overlay.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'trajectory_overlay.pdf'))
print(f"  Saved: trajectory_overlay.png/pdf")

# --- Phase-space plot: x1 vs x3 (full trajectory) ---

fig, ax = plt.subplots(figsize=(8, 6))

contact_idx = contact
non_contact_idx = ~contact
dist_nm = dist * 1e9

# Draw non-contact and contact trajectories separately for clarity
ax.plot(x_nm[non_contact_idx], y_nm[non_contact_idx],
        color='tab:blue', linewidth=0.35, alpha=0.6, label='Non-contact')
ax.plot(x_nm[contact_idx], y_nm[contact_idx],
        color='tab:red', linewidth=0.35, alpha=0.8, label='Contact')

# Hertz boundary: s = dist + x1 - x3 = a0  =>  x3 = x1 + dist - a0
x_line = np.array([x_nm.min(), x_nm.max()])
ax.plot(x_line, x_line + dist_nm - a0 * 1e9, 'k--', linewidth=1.0, label='Boundary: s = a0')

ax.scatter(x_nm[0], y_nm[0], s=20, c='k', marker='o', label='Start')
ax.scatter(x_nm[-1], y_nm[-1], s=20, c='green', marker='x', label='End')

ax.set_xlabel('x1 tip displacement [nm]')
ax.set_ylabel('x3 sample displacement [nm]')
ax.set_title('AFM DMT-KV Phase Space: x1 vs x3 (Full Trajectory)')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase_space_x1_x3_full.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'phase_space_x1_x3_full.pdf'))
print("  Saved: phase_space_x1_x3_full.png/pdf")

# --- Phase-space plot: x1 vs x3 (zoomed steady-state) ---

fig, ax = plt.subplots(figsize=(8, 6))
x_zoom = x_nm[zoom_idx]
y_zoom = y_nm[zoom_idx]
contact_zoom = contact[zoom_idx]
non_contact_zoom = ~contact_zoom

ax.plot(x_zoom[non_contact_zoom], y_zoom[non_contact_zoom],
        color='tab:blue', linewidth=0.8, alpha=0.8, label='Non-contact')
ax.plot(x_zoom[contact_zoom], y_zoom[contact_zoom],
        color='tab:red', linewidth=0.8, alpha=0.9, label='Contact')

x_zoom_line = np.array([x_zoom.min(), x_zoom.max()])
ax.plot(x_zoom_line, x_zoom_line + dist_nm - a0 * 1e9, 'k--', linewidth=1.0, label='Boundary: s = a0')

ax.set_xlabel('x1 tip displacement [nm]')
ax.set_ylabel('x3 sample displacement [nm]')
ax.set_title(f'AFM DMT-KV Phase Space: x1 vs x3 ({zoom_start_us}-{zoom_end_us} μs)')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'phase_space_x1_x3_zoomed.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'phase_space_x1_x3_zoomed.pdf'))
print("  Saved: phase_space_x1_x3_zoomed.png/pdf")

# --- Statistics ---

print("\n" + "="*60)
print("Trajectory Statistics:")
print("="*60)
print(f"  Tip displacement range: [{x_nm.min():.2f}, {x_nm.max():.2f}] nm")
print(f"  Sample motion range:    [{y_nm.min():.2f}, {y_nm.max():.2f}] nm")
print(f"  Tip-sample distance s:  [{s_nm.min():.2f}, {s_nm.max():.2f}] nm")
print(f"  Contact fraction:       {contact.sum()/len(contact)*100:.2f}%")
print(f"  Oscillation period:     {1/f0*1e6:.3f} μs")
print(f"  Number of cycles:       {t_end * f0:.0f}")

if os.environ.get("HNODECB_SHOW_PLOTS", "0") == "1":
    plt.show()
else:
    plt.close("all")
print("\nDone!")
