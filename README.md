# Cubed

This a small Brownian Dynamics package that is CUDA specific. It **won't** perform on the CPU.
You must have the CUDA Development toolkit installed, otherwise it won't run at all.

It is suited for hard-sphere simulations by employing a [continuous potential](https://aip.scitation.org/doi/10.1063/1.5049568).
As of right now it only computes the energy per particle (which should always be close to zero), the compressibility factor `Z`,
as well as the mean squared displacement of a tag particle.

To solve the Langevin equation, the **Ermak-McCammon** algorithm is employed, so we are assuming an overdamped regime.
This algorithm is also sometimes known as the *Euler-Murayama* scheme.

## Physical correctness

Although the code is now stable enough to perform extensive computations, it may not be stable enough in terms of
correctness, so some tweaking of the principal parameters like the time step should be performed at will until
it is stable enough and suited to one's own needs.

## Demo

Here is a simple script to help you get started with your simulations.

```julia
# Use this if running directly from the command line
using Pkg
Pkg.activate("./")

using Cubed
using RandomNumbers.Xorshifts

# * Simulation parameters
N = 12^3 # number of particles
ϕ = 0.4f0 # filling/packing fraction
ρ = 6.0f0 * ϕ / π # reduced density
volume = N / ρ # reduced volume
Lbox = ∛volume # Cubic box length
rc = Lbox / 2.0f0 # Cut-off radius

# Create a rng
rng = Xorshift1024Star()

# * Initialize the System object
syst = System(N, Lbox, ρ, volume, rc, rng)
# * Create the Dynamic object with the desired time step
dynamic = Dynamic(0.00001f0)

# * Create the Potential object
λ = 50
b = convert(Float32, λ / (λ - 1.0f0))
a = convert(Float32, λ * b^(λ - 1.0f0))
potential = Potential(a, b, 1.4737f0, 50)

# * Simulate the full system
filenames = ["$(ϕ)_$(syst.N).csv", "msd_$(ϕ).csv"]
simulate(syst, dynamic, potential;
    thermal = 150000, cycles = 1000000, filenames = filenames)
```

`thermal` is the number of equilibration steps before computing averages in a total number
of `cycles` steps, i.e. the system will evolve with *periodic boundary conditions* during
150000 steps and then the system will evolve **without** periodic boundary conditions for
1e6 steps. This behavior cannot be modified as of right now.

The raw data obtained should be physically correct if the parameters are right, at least
here in the demo they are.
