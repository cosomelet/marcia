using LinearAlgebra
using DifferentialEquations

function hubble_rate(H0, Omega_m, Omega_b, Omega_k, de, z)
    Omega_r = 4.18343e-5 / (H0 / 100.0)^2
    E2 = Omega_r * (1.0 + z)^4 + Omega_m * (1.0 + z)^3 + Omega_b * (1.0 + z)^3 + Omega_k * (1.0 + z)^2 + (1.0 - Omega_m - Omega_k - Omega_b - Omega_r) * de
    Hofz = H0 * sqrt(E2)
    return Hofz
end

function dark_energy_f_wCDM(w0, wa, z)
    return exp(3.0 * (-wa + wa / (1.0 + z) + (1.0 + w0 + wa) * log(1.0 + z)))
end

function inv_hubble_rate(H0, Omega_m, Omega_b, Omega_k, de, z)
    return 1.0 / hubble_rate(H0, Omega_m, Omega_b, Omega_k, de, z)
end

function sound_horizon(H0, Omega_b, Omega_m, Obsample)
    m_nu = 0.06  # In the units of eV
    omega_nu = 0.0107 * (m_nu / 1.0)  # This is in the units of eV. This should be equal to 6.42*10^(-4)
    if Obsample
        omega_b = Omega_b * (H0 / 100.0)^2  # 0.0217
    else
        omega_b = 0.0217
    end

    omega_cb = (Omega_m + Omega_b) * (H0 / 100.0)^2 - omega_nu
    if omega_cb < 0
        rd = -1.0
    else
        rd = 55.154 * exp(-72.3 * ((omega_nu + 0.0006)^2)) / ((omega_cb^0.25351) * (omega_b^0.12807))
        if isnan(rd)
            rd = 0.0
        end
    end
    return rd
end

function z_inp(z)
    return collect(0.0:0.01:maximum(z) + 0.5)
end

function interpolate(z_inp, z, func)
    return LinearAlgebra.interpolate((z_inp,), func, Gridded(Linear()))
end

function Ly(y, t, H0, Omega_m, Omega_b, Omega_k, de)
    return inv_hubble_rate(H0, Omega_m, Omega_b, Omega_k, de, [t])[1]
end

function transverse_distance(H0, Omega_m, Omega_b, Omega_k, de, z, clight)
    y = solve((t, y) -> Ly(y, t, H0, Omega_m, Omega_b, Omega_k, de), [0.0], z, saveat = z)
    return clight * y[1, :]
end