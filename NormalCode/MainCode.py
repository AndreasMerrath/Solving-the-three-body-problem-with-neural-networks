import numpy as np
import NormalCode.kepler_solver as ks
from math import pi


def kepler_solver(tau, mij, r0, v0, gravityConstant):
    keplerConstant = mij * gravityConstant
    s0 = ks.State(r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 0.0, 0.0, 0.0)
    s = ks.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ks.kepler_step(keplerConstant, tau, s0, s)
    r = np.zeros(3)
    v = np.zeros(3)
    r[0] = s.x
    r[1] = s.y
    r[2] = s.z
    v[0] = s.xd
    v[1] = s.yd
    v[2] = s.zd
    return r, v


def do_step(tau, n, m, r, v, twoBody=False, gravityConstant=(6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
            1 / (1.496 * 10 ** 11) ** 3)):
    """
    Perform a single time step of the Sakura integrator.

    :param tau: the time-step size.
    :param n: the number of particles.
    :param m: array with particles' masses.
    :param r: 3D array with particles' positions.
    :param v: 3D array with particles' velocities.
    :return: Updated positions and velocities after the time step.
    """
    r, v = evolve_HT(tau / 2, n, m, r, v)
    r, v = evolve_HW(tau, n, m, r, v, twoBody, gravityConstant)
    r, v = evolve_HT(tau / 2, n, m, r, v)

    return r, v


def evolve_HT(tau, n, m, r, v):
    for i in range(n):
        for k in range(3):
            # Update position based on velocity and time step
            r[i][k] += v[i][k] * tau
    return r, v


def evolve_HW(tau, n, m, r, v, twoBody, gravityConstant):
    """
    Evolve the system under the potential part of the Hamiltonian.
    """
    # Allocate 3D arrays to store increments due to 2-body interactions
    dmr = np.zeros((n, 3))
    dmv = np.zeros((n, 3))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Compute the combined mass and reduced mass
                mij = m[i] + m[j]
                mu = (m[i] * m[j]) / mij
                # Initialize arrays for relative positions and velocities
                rr0 = np.zeros(3)
                vv0 = np.zeros(3)
                r0 = np.zeros(3)
                v0 = np.zeros(3)
                rr1 = np.zeros(3)
                vv1 = np.zeros(3)

                # Calculate relative positions and velocities
                for k in range(3):
                    rr0[k] = r[i][k] - r[j][k]
                    vv0[k] = v[i][k] - v[j][k]
                # Adjust initial conditions for half-tau step back in time
                for k in range(3):
                    r0[k] = rr0[k] - vv0[k] * tau / 2
                    v0[k] = vv0[k]
                # Solve the Kepler problem for the pair
                r1, v1 = kepler_solver(tau, mij, r0, v0, gravityConstant)
                # Calculate new relative positions and velocities
                for k in range(3):
                    rr1[k] = r1[k] - v1[k] * tau / 2
                    vv1[k] = v1[k]
                # Calculate the change due to the interaction
                for k in range(3):
                    dmr[i][k] += mu * (rr1[k] - rr0[k])
                    dmv[i][k] += mu * (vv1[k] - vv0[k])

    # Update positions and velocities with the changes
    for i in range(n):
        for k in range(3):
            if twoBody:
                r[i][k] += dmr[i][k]
                v[i][k] += dmv[i][k]
            else:
                r[i][k] += dmr[i][k] / m[i]
                v[i][k] += dmv[i][k] / m[i]


    return r, v

# Note: Die Funktion mach folgendes für alle Elemente:
#       mit jedem anderen Element wird die relative position und geschwindigkeit kalkuliert. Das Ergebnis ist der Vektor,
#            wenn das andere Element im Ursprung mit 0 geschwindigkeit ist und der andere Vektor darum kreist
#       Dann wird -wahrscheinlich aus numerischen Zwecken- die Position ein halten zeitschrit zurückgesetzt
#       Dann wird für ein Zeitschritt ( tau )
#
