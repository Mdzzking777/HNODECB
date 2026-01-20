"""
Generate DMT-KV Trajectory
Fixed parameters, minimal code
Now saves x1dot and x2dot directly from ODE (no numerical differentiation!)
"""

import numpy as np

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
R = 10e-9
Fad = 2.0e-9
dist = 24e-9

# Drive
Fd = 2.05e-9

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

def rhs_dmt_kv(t, X):
    """
    B) MOVING SURFACE (Kelvin-Voigt) + Hertz-DMT

    State: X = [x, v, y]
    Returns: dX/dt = [dxdt, dvdt, dydt]
    """
    x, v, y = X

    # Separation
    s = dist + x - y

    # Contact detection
    if s <= 0:  # Contact
        delta = -s  # Indentation depth: delta = y - x - dist
        F_Hertz = (4.0/3.0) * Estar * np.sqrt(R) * (delta ** 1.5) if delta > 0 else 0.0
        dvdt = (Fd * np.cos(wd * t) - k*x - c*v + Fad - F_Hertz) / m
        dydt = (Fad - F_Hertz - ks*y) / cs
    else:  # Non-contact
        dvdt = (Fd * np.cos(wd * t) - k*x - c*v) / m
        dydt = -ks * y / cs

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

# Initial condition
X = np.array([0.0, 0.0, 0.0])
x[0], v[0], y[0] = X

# Compute initial acceleration
derivs = rhs_dmt_kv(t[0], X)
x2dot[0] = derivs[1]  # dvdt

# RK4 integration
for i in range(nsteps):
    X = rk4_step(rhs_dmt_kv, t[i], X, dt)
    x[i+1], v[i+1], y[i+1] = X

    # Compute acceleration directly from ODE (no numerical differentiation!)
    derivs = rhs_dmt_kv(t[i+1], X)
    x2dot[i+1] = derivs[1]  # dvdt = x2dot

# Compute separation and contact
s = dist + x - y
contact = (s <= 0)

print(f"Complete: {len(t)} time points")
print(f"Contact fraction: {contact.sum()/len(contact)*100:.2f}%")

# ============================================================================
# Save Data
# ============================================================================

# Save as CSV with x1dot (v) and x2dot (acceleration)
data = np.column_stack([t, x, y, s, contact.astype(int), v, x2dot])
np.savetxt('trajectory.csv', data, delimiter=',',
           header='time_s,x_tip_m,y_sample_m,s_separation_m,contact_status,x1dot_velocity,x2dot_acceleration',
           comments='')

print("Saved: trajectory.csv")
print("  - Includes x1dot (velocity) and x2dot (acceleration) directly from ODE")
print("  - No numerical differentiation used!")

# ============================================================================
# Visualization
# ============================================================================

import matplotlib.pyplot as plt
import os

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

# Plot 3: Tip-sample separation
ax3 = axes[2]
ax3.plot(t_us, s_nm, 'g-', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='-', linewidth=1, label='Contact threshold (s=0)')
ax3.fill_between(t_us, s_nm, 0, where=(s_nm <= 0), alpha=0.3, color='red', label='Contact region')
ax3.set_ylabel('Separation s [nm]')
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

# Plot 3: Tip-sample separation (zoomed)
ax3 = axes[2]
ax3.plot(t_us[zoom_idx], s_nm[zoom_idx], 'g-', linewidth=1)
ax3.axhline(y=0, color='r', linestyle='-', linewidth=1, label='Contact threshold (s=0)')
ax3.fill_between(t_us[zoom_idx], s_nm[zoom_idx], 0,
                  where=(s_nm[zoom_idx] <= 0), alpha=0.3, color='red', label='Contact region')
ax3.set_ylabel('Separation s [nm]')
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
ax.plot(t_us[zoom_idx], s_nm[zoom_idx], 'g-', linewidth=1, label='Separation s')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Mark equilibrium separation
ax.axhline(y=dist*1e9, color='gray', linestyle=':', linewidth=1, label=f'd = {dist*1e9:.1f} nm (equilibrium)')

ax.set_xlabel('Time [μs]')
ax.set_ylabel('Displacement [nm]')
ax.set_title(f'AFM DMT-KV: Tip, Sample, and Separation ({zoom_start_us}-{zoom_end_us} μs, Steady-State)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'trajectory_overlay.png'), dpi=150)
plt.savefig(os.path.join(plots_dir, 'trajectory_overlay.pdf'))
print(f"  Saved: trajectory_overlay.png/pdf")

# --- Statistics ---

print("\n" + "="*60)
print("Trajectory Statistics:")
print("="*60)
print(f"  Tip displacement range: [{x_nm.min():.2f}, {x_nm.max():.2f}] nm")
print(f"  Sample motion range:    [{y_nm.min():.2f}, {y_nm.max():.2f}] nm")
print(f"  Separation range:       [{s_nm.min():.2f}, {s_nm.max():.2f}] nm")
print(f"  Contact fraction:       {contact.sum()/len(contact)*100:.2f}%")
print(f"  Oscillation period:     {1/f0*1e6:.3f} μs")
print(f"  Number of cycles:       {t_end * f0:.0f}")

plt.show()
print("\nDone!")
