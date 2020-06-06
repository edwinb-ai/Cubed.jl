# Cubed

This a small Brownian Dynamics package that is CUDA enabled.

It is suited for hard-sphere simulations by employing a [continuous potential](https://aip.scitation.org/doi/10.1063/1.5049568).
As of right now it only computes the energy per particle (which should always be close to zero), as well as
the compressibility factor `Z`.

## Physical correctness

Although the code is now stable enough to perform extensive computations, it may not be stable enough in terms of
correctness, so some tweaking of the principal parameters like the time step should be performed at will until
it is stable enough and suited to one's own needs.
