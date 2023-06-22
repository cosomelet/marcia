import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double cysqrt(double x) nogil:
    return sqrt(x)

@cython.boundscheck(False)
@cython.wraparound(False)
def hubble_rate(double H0, double Omega_m, double Omega_b, double Omega_k, double[:] de, double[:] z):
    cdef int i, n
    cdef double Omega_r, E2
    cdef double[:] Hofz

    n = len(z)
    Hofz = np.empty(n, dtype=np.float64)

    cdef double H0_sq = (H0 / 100.)**2
    Omega_r = 4.18343 * 10**-5. / H0_sq

    cdef double[:] de_view = de
    cdef double[:] z_view = z
    cdef double[:] Hofz_view = Hofz

    for i in prange(n, nogil=True):
        E2 = Omega_r * ((1. + z_view[i])**4) + Omega_m * ((1. + z_view[i])**3) + Omega_b * ((1. + z_view[i])**3) + Omega_k * ((1. + z_view[i])**2) + (1. - Omega_m - Omega_k - Omega_b - Omega_r) * de_view[i]
        Hofz_view[i] = H0 * cysqrt(E2)

    return Hofz


@cython.boundscheck(False)
@cython.wraparound(False)
def dark_energy_f_wCDM(double w0, double wa, double z):
    return np.exp(3. * (-wa + wa / (1. + z) + (1. + w0 + wa) * np.log(1. + z)))

@cython.boundscheck(False)
@cython.wraparound(False)
def inv_hubble_rate(double H0, double Omega_m, double Omega_b, double Omega_k, double de, double z):
    cdef double[:] de_arr = de
    cdef double[:] z_arr = z
    cdef int n = z.shape[0]
    cdef double[:] inv_hubble = np.empty(n)

    for i in range(n):
        inv_hubble[i] = 1. / hubble_rate(H0, Omega_m, Omega_b, Omega_k, de_arr[i], z_arr[i])

    return inv_hubble

@cython.boundscheck(False)
@cython.wraparound(False)
def sound_horizon(double H0, double Omega_b, double Omega_m, int Obsample):
    cdef double m_nu = 0.06
    cdef double omega_nu = 0.0107 * (m_nu / 1.0)
    cdef double omega_b
    cdef double omega_cb
    cdef double rd

    if Obsample:
        omega_b = Omega_b * (H0 / 100.) ** 2.
    else:
        omega_b = 0.0217

    omega_cb = (Omega_m + Omega_b) * (H0 / 100.) ** 2 - omega_nu

    if omega_cb < 0:
        rd = -1.0
    else:
        rd = 55.154 * np.exp(-72.3 * ((omega_nu + 0.0006) ** 2)) / ((omega_cb ** 0.25351) * (omega_b ** 0.12807))

        if np.isnan(rd):
            rd = 0.0

    return rd

@cython.boundscheck(False)
@cython.wraparound(False)
def z_inp(double z):
    return np.arange(0., np.max(z) + .5, 0.01)

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate(double[:] z_inp, double[:] z, double[:] func):
    return np.interp(z, z_inp, func)