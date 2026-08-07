# AFM06a Data-Generation Configuration

AFM06a implements a fixed hard-sample DMT model and intentionally contains no
default experiment parameter values inherited from AFM04.

The contact-transition sharpness and trajectory settings are assigned:

```text
beta         = 5.0e11 1/m
dist         = etaStar
x1_start     = 0 m
x2_start     = 0 m/s
initial_time = 0 s
end_time     = 20e-3 s
data_nsteps  = 1250000
num_points   = 1250001
```

The initial tip state and 16 ns sampling interval match AFM04, while the AFM06a
simulation span is 20 ms. AFM06a has no `x3` state.

AFM06a currently assigns the natural angular frequency directly. It may still
be overridden explicitly when constructing the input object:

```text
omega0 = 11.804e3 rad/s
```

The supplied constants are stored in the AFM06a settings module as follows:

```text
C1       = -1.27462e-6
C2       = 4.63118
B1       = 1.56598
d1       = 0.0017
d2       = 2.0285
a0_norm  = 0.0132626
ybar     = 0.05585
OmegaBar = 1.002
etaStar  = 8.88249 nm
```

Here `etaStar` is a geometric length parameter. It is not the viscous
coefficient from the removed Kelvin-Voigt term. The active RHS currently uses
`omega0`, `C1`, `C2`, `B1`, `D1`, `D2`, `a0_norm`, `ybar`, `OmegaBar`, and
`etaStar`. No independent quality-factor value is currently assigned; damping
is controlled directly by `D1` and `D2` in the active RHS.

The physical DMT contact distance is derived rather than supplied separately:

```text
a0 = a0_norm*etaStar
dist = etaStar
```

The dynamic state is `x = [x1, x2]`. There is no sample-motion state, no
Kelvin-Voigt term, and no `ks` or `cs` parameter.

The second state equation is

```text
x2dot = omega0^2*etaStar*B1*OmegaBar^2*ybar*sin(omega0*t)
        - omega0^2*x1
        - Deff(s)*omega0*x2
        + bar_fts
```

The smooth contact-dependent damping and mass-scaled interaction terms are

```text
g         = sigmoid(beta*(s-a0))
Deff(s)   = g*D1 + (1-g)*D2
denom     = g*(s-a0) + a0
delta     = max(a0-s, 0)
bar_fts   = C1*omega0^2*etaStar^3/denom^2
            + (1-g)*C2*omega0^2*delta^(3/2)/sqrt(etaStar)
```

Because the revised coefficients already define the mass-scaled interaction,
the active RHS requires no numerical value of `k`, `m`, `c`, `Fd`, `A`, `R`,
or `Estar`. Only `bar_fts` is saved and used by the state-space RHS.

The API entry point is:

```python
from AFM06a.datasets import generate_afm_dmt_hard_dataset
from AFM06a.test_case_settings.afm_dmt_hard_settings.afm_dmt_hard_model_settings import (
    AFM06aHardSampleInputs,
)

settings = AFM06aHardSampleInputs(
    # All current AFM06a data-generation inputs are assigned above.
)

generate_afm_dmt_hard_dataset(settings=settings)
```

The ellipses are required user inputs; they are not AFM06a defaults.
