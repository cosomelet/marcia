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
