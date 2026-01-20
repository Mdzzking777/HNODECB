# Fixed parameters from Datageneration/DMT_KV/generate_trajectory_DMT_KV.py

# Cantilever
k = 29.9
f0 = 313.57e3
wd = 2.0 * pi * f0
Q = 371.0
m = k / (wd^2)
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

# Adhesion smoothing width around s=0 (meters)
adhesion_transition = 1.0e-10

# original model parameters
# order: k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs
original_parameters = Float64[k, wd, m, c, Fd, R, dist, Fad, Estar, ks, cs]

# initial conditions (x1, x2, x3)
original_u0 = [0.0, 0.0, 0.0]

# initial time / end time
initial_time_training = 0.0f0
end_time_training = 2.0e-3
