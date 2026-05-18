# The Nultra Operator

The Nultra Operator is the mathematical core of this project.

## Definition

```text
S_next = S + A(t) * (f(S) - S)
```

Where:

- `S` is the current state
- `f(S)` is a transformed version of that state
- `A(t)` is an aperture between `0` and `1`

This can also be read as:

```text
S_next = (1 - A(t)) * S + A(t) * f(S)
```

So the operator interpolates between:

- the identity map `S`
- the transformed map `f(S)`

## Why It Matters

The important case is:

```text
A(t) = 0
```

When the aperture is exactly zero, the state is preserved exactly:

```text
S_next = S
```

Time still advances, but the state does not mutate.

That is the **true null plateau**.

## Aperture Construction

In the broader Nultra family, the aperture usually comes from:

```text
A_base(t)  = 0.5 * (1 + sin(...))
A_gamma(t) = A_base(t)^gamma
A(t)       = 0,          if A_gamma(t) < eta
             A_gamma(t), otherwise
```

In the multiband ecosystem processor, the base motion is upgraded from a simple sine into a chaotic attractor trace. The thresholded structure remains the same:

- base motion source
- `gamma` shaping
- `eta` thresholding
- exact preservation when `A(t) = 0`

## DSP Interpretation

For audio:

- `S` is the current band-limited audio state
- `f(S)` is a transformed version of that band
- `A(t)` decides how much of the transformed state is allowed to enter

That means the processor is not merely "turning volume up and down." It is deciding when a signal is allowed to remain itself and when it is allowed to move toward a transformed identity.

## Canonical Version 1.0

The canonical baseline in this repository freezes:

- 3-band architecture
- chaotic aperture generators
- parasitic low-to-high threshold modulation
- high-band safety interlocks
- high-band diffusion morphing
- mid-band residue feedback

Future work should fork from the baseline rather than modify it in place.
