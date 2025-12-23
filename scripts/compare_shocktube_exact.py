#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# OpenFOAM reader
# =========================================================
def read_line_xy(path: Path):
    data = []
    with path.open("r") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            data.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.array(data, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No numeric data found in {path}")
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

# =========================================================
#  Exact Riemann solver 
# =========================================================
def _toro_exact_at_x(x, t, left_state, right_state, x0, gamma, sol_type="auto", left_biased=False):

    if t <= 0:
        raise ValueError("t must be > 0 for exact Riemann solution evaluation.")

    if left_biased:
        less = lambda a, b: a <= b
    else:
        less = lambda a, b: a < b

    rhoL, uL, pL = left_state
    rhoR, uR, pR = right_state

    gm = gamma - 1.0
    gp = gamma + 1.0
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    AL = 2.0 / (gp * rhoL)
    BL = (gm / gp) * pL
    AR = 2.0 / (gp * rhoR)
    BR = (gm / gp) * pR

    # fan (rarefaction) formulas
    WLfan_v = lambda xc: (2.0 / gp) * (aL + 0.5 * gm * uL + xc / t)
    WRfan_v = lambda xc: (2.0 / gp) * (-aR + 0.5 * gm * uR + xc / t)

    WLfan_r = lambda xc: rhoL * ((2.0 / gp) + (gm / gp / aL) * (uL - xc / t)) ** (2.0 / gm)
    WRfan_r = lambda xc: rhoR * ((2.0 / gp) - (gm / gp / aR) * (uR - xc / t)) ** (2.0 / gm)

    WLfan_p = lambda xc: pL * ((2.0 / gp) + (gm / gp / aL) * (uL - xc / t)) ** (2.0 * gamma / gm)
    WRfan_p = lambda xc: pR * ((2.0 / gp) - (gm / gp / aR) * (uR - xc / t)) ** (2.0 * gamma / gm)

    left_RW_head_speed = uL - aL
    right_RW_head_speed = uR + aR

    left_RW_head_position = x0 + left_RW_head_speed * t
    right_RW_head_position = x0 + right_RW_head_speed * t

    def build_fl_fr(sol):
        if sol == "RW-CD-SW":
            fl = lambda p: 2 * aL / gm * ((p / pL) ** (gm / (2.0 * gamma)) - 1.0)
            fr = lambda p: (p - pR) * np.sqrt(AR / (p + BR))
        elif sol == "RW-CD-RW":
            fl = lambda p: 2 * aL / gm * ((p / pL) ** (gm / (2.0 * gamma)) - 1.0)
            fr = lambda p: 2 * aR / gm * ((p / pR) ** (gm / (2.0 * gamma)) - 1.0)
        elif sol == "SW-CD-RW":
            fl = lambda p: (p - pL) * np.sqrt(AL / (p + BL))
            fr = lambda p: 2 * aR / gm * ((p / pR) ** (gm / (2.0 * gamma)) - 1.0)
        elif sol == "SW-CD-SW":
            fl = lambda p: (p - pL) * np.sqrt(AL / (p + BL))
            fr = lambda p: (p - pR) * np.sqrt(AR / (p + BR))
        else:
            raise ValueError(f"Unknown sol_type: {sol}")
        return fl, fr

    def solve_p_star(sol):
        fl, fr = build_fl_fr(sol)
        func = lambda p: fl(p) + fr(p) + (uR - uL)

        p2r = ((aL + aR - 0.5 * gm * (uR - uL)) /
               (aL / (pL ** (gm / (2.0 * gamma))) + aR / (pR ** (gm / (2.0 * gamma))))
               ) ** (2.0 * gamma / gm)
        p = max(1e-14, float(p2r))

        def dfunc(pv):
            h = 1e-8 * max(1.0, abs(pv))
            return (func(pv + h) - func(pv - h)) / (2.0 * h)

        for _ in range(200):
            f = func(p)
            df = dfunc(p)
            if df == 0.0 or not np.isfinite(df):
                break
            dp = -f / df
            lam = 1.0
            p_new = p + lam * dp
            while p_new <= 0 or not np.isfinite(p_new):
                lam *= 0.5
                if lam < 1e-10:
                    p_new = max(1e-14, p * 0.5)
                    break
                p_new = p + lam * dp
            if abs(p_new - p) < 1e-12 * max(1.0, abs(p)):
                p = p_new
                break
            p = p_new

        return p, func(p)

    if sol_type == "auto":
        candidates = ["RW-CD-SW", "RW-CD-RW", "SW-CD-RW", "SW-CD-SW"]
        best = None
        for sol in candidates:
            try:
                p_star_sol, res = solve_p_star(sol)
                score = abs(res)
                if best is None or score < best[0]:
                    best = (score, sol, p_star_sol, res)
            except Exception:
                continue
        if best is None:
            raise RuntimeError("Failed to determine sol_type automatically.")
        _, sol_type_used, p_star, residual = best
    else:
        sol_type_used = sol_type
        p_star, residual = solve_p_star(sol_type_used)

    fl, fr = build_fl_fr(sol_type_used)
    u_star = 0.5 * (uL + uR) + 0.5 * (fr(p_star) - fl(p_star))

    QL = np.sqrt((p_star + BL) / AL)
    QR = np.sqrt((p_star + BR) / AR)
    left_SW_speed = uL - QL / rhoL
    right_SW_speed = uR + QR / rhoR
    left_SW_position = x0 + left_SW_speed * t
    right_SW_position = x0 + right_SW_speed * t

    a_star_left = aL * (p_star / pL) ** (gm / (2.0 * gamma))
    a_star_right = aR * (p_star / pR) ** (gm / (2.0 * gamma))
    left_RW_tail_speed = u_star - a_star_left
    right_RW_tail_speed = u_star + a_star_right
    left_RW_tail_position = x0 + left_RW_tail_speed * t
    right_RW_tail_position = x0 + right_RW_tail_speed * t

    CW_position = x0 + u_star * t

    rho_star_left_RW = rhoL * (p_star / pL) ** (1.0 / gamma)
    rho_star_left_SW = rhoL * (p_star / pL + gm / gp) / ((gm / gp) * (p_star / pL) + 1.0)
    rho_star_right_RW = rhoR * (p_star / pR) ** (1.0 / gamma)
    rho_star_right_SW = rhoR * (p_star / pR + gm / gp) / ((gm / gp) * (p_star / pR) + 1.0)

    def sol_at(coord):
        if sol_type_used == "RW-CD-SW":
            if less(coord, left_RW_head_position):
                return rhoL, uL, pL
            if less(coord, left_RW_tail_position):
                return WLfan_r(coord - x0), WLfan_v(coord - x0), WLfan_p(coord - x0)
            if less(coord, right_SW_position):
                if less(coord, CW_position):
                    return rho_star_left_RW, u_star, p_star
                else:
                    return rho_star_right_SW, u_star, p_star
            return rhoR, uR, pR

        elif sol_type_used == "RW-CD-RW":
            if less(coord, left_RW_head_position):
                return rhoL, uL, pL
            if less(coord, left_RW_tail_position):
                return WLfan_r(coord - x0), WLfan_v(coord - x0), WLfan_p(coord - x0)
            if less(coord, right_RW_tail_position):
                if less(coord, CW_position):
                    return rho_star_left_RW, u_star, p_star
                else:
                    return rho_star_right_RW, u_star, p_star
            if less(coord, right_RW_head_position):
                return WRfan_r(coord - x0), WRfan_v(coord - x0), WRfan_p(coord - x0)
            return rhoR, uR, pR

        elif sol_type_used == "SW-CD-RW":
            if less(coord, left_SW_position):
                return rhoL, uL, pL
            if less(coord, right_RW_tail_position):
                if less(coord, CW_position):
                    return rho_star_left_SW, u_star, p_star
                else:
                    return rho_star_right_RW, u_star, p_star
            if less(coord, right_RW_head_position):
                return WRfan_r(coord - x0), WRfan_v(coord - x0), WRfan_p(coord - x0)
            return rhoR, uR, pR

        elif sol_type_used == "SW-CD-SW":
            if less(coord, left_SW_position):
                return rhoL, uL, pL
            if less(coord, right_SW_position):
                if less(coord, CW_position):
                    return rho_star_left_SW, u_star, p_star
                else:
                    return rho_star_right_SW, u_star, p_star
            return rhoR, uR, pR

        raise RuntimeError(f"Unexpected sol_type_used: {sol_type_used}")

    rho = np.zeros_like(x, dtype=float)
    u = np.zeros_like(x, dtype=float)
    p = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i] = sol_at(xi)

    meta = {
        "sol_type_used": sol_type_used,
        "p_star": float(p_star),
        "u_star": float(u_star),
        "residual": float(residual),
        "x0": float(x0),
        "t": float(t),
    }
    return rho, u, p, meta

# =========================================================
# Error norms
# =========================================================
def compute_error_norms(q_num, q_exact):
    err = np.abs(q_num - q_exact)
    L1 = np.mean(err)
    L2 = np.sqrt(np.mean(err**2))
    Linf = np.max(err)
    return L1, L2, Linf

# =========================================================
# MAIN
# =========================================================
def main():
    fF = Path("data/shocktube/fluid/line_0p007.xy")
    fM = Path("data/shocktube/multicomponent/line_0p007.xy")

    xF, TF, UF, pF = read_line_xy(fF)
    xM, TM, UM, pM = read_line_xy(fM)

    # Interpolate multicomponent onto fluid x-grid 
    if len(xF) != len(xM) or np.max(np.abs(xF - xM)) > 1e-12:
        TM_i = np.interp(xF, xM, TM)
        UM_i = np.interp(xF, xM, UM)
        pM_i = np.interp(xF, xM, pM)
        x = xF
    else:
        TM_i, UM_i, pM_i = TM, UM, pM
        x = xF

    # Output directory 
    outdir = Path("exact_comparison_plots")
    outdir.mkdir(exist_ok=True)

    # Exact-solution time 
    try:
        t_exact = float(fF.parent.name)
    except Exception:
        t_exact = 0.007

    gamma = 1.4
    R_gas = 287.0
    x0_exact = 0.0

    n_edge = max(3, int(0.05 * len(x)))
    TL = float(np.mean(TF[:n_edge]))
    uL = float(np.mean(UF[:n_edge]))
    pL = float(np.mean(pF[:n_edge]))
    rhoL = pL / (R_gas * TL)

    TR = float(np.mean(TF[-n_edge:]))
    uR = float(np.mean(UF[-n_edge:]))
    pR = float(np.mean(pF[-n_edge:]))
    rhoR = pR / (R_gas * TR)

    left_state = (rhoL, uL, pL)
    right_state = (rhoR, uR, pR)

    rhoE, uE, pE, meta = _toro_exact_at_x(
        x=x, t=t_exact,
        left_state=left_state,
        right_state=right_state,
        x0=x0_exact,
        gamma=gamma,
        sol_type="auto"
    )
    TE = pE / (rhoE * R_gas)

    # ---- Error norms (verification + consistency) ----

    # Verification: fluid vs exact 
    L1_p_F, L2_p_F, Linf_p_F = compute_error_norms(pF, pE)
    L1_T_F, L2_T_F, Linf_T_F = compute_error_norms(TF, TE)
    L1_U_F, L2_U_F, Linf_U_F = compute_error_norms(UF, np.abs(uE))

    # Verification: multicomponent vs exact
    L1_p_M, L2_p_M, Linf_p_M = compute_error_norms(pM_i, pE)
    L1_T_M, L2_T_M, Linf_T_M = compute_error_norms(TM_i, TE)
    L1_U_M, L2_U_M, Linf_U_M = compute_error_norms(UM_i, np.abs(uE))

    # Consistency: multicomponent vs fluid
    L1_p_MF, L2_p_MF, Linf_p_MF = compute_error_norms(pM_i, pF)
    L1_T_MF, L2_T_MF, Linf_T_MF = compute_error_norms(TM_i, TF)
    L1_U_MF, L2_U_MF, Linf_U_MF = compute_error_norms(UM_i, UF)

    # Aliases 
    L1_p, L2_p, Linf_p = L1_p_F, L2_p_F, Linf_p_F
    L1_T, L2_T, Linf_T = L1_T_F, L2_T_F, Linf_T_F
    L1_U, L2_U, Linf_U = L1_U_F, L2_U_F, Linf_U_F

    print("\n=== VERIFICATION: fluid vs exact solution ===")
    print(f"Pressure    : L1 = {L1_p_F:.4e}, L2 = {L2_p_F:.4e}, Linf = {Linf_p_F:.4e}")
    print(f"Temperature : L1 = {L1_T_F:.4e}, L2 = {L2_T_F:.4e}, Linf = {Linf_T_F:.4e}")
    print(f"Velocity |U|: L1 = {L1_U_F:.4e}, L2 = {L2_U_F:.4e}, Linf = {Linf_U_F:.4e}")

    print("\n=== VERIFICATION: multicomponent vs exact solution ===")
    print(f"Pressure    : L1 = {L1_p_M:.4e}, L2 = {L2_p_M:.4e}, Linf = {Linf_p_M:.4e}")
    print(f"Temperature : L1 = {L1_T_M:.4e}, L2 = {L2_T_M:.4e}, Linf = {Linf_T_M:.4e}")
    print(f"Velocity |U|: L1 = {L1_U_M:.4e}, L2 = {L2_U_M:.4e}, Linf = {Linf_U_M:.4e}")

    print("\n=== CONSISTENCY: multicomponent vs fluid solution ===")
    print(f"Pressure    : L1 = {L1_p_MF:.4e}, L2 = {L2_p_MF:.4e}, Linf = {Linf_p_MF:.4e}")
    print(f"Temperature : L1 = {L1_T_MF:.4e}, L2 = {L2_T_MF:.4e}, Linf = {Linf_T_MF:.4e}")
    print(f"Velocity |U|: L1 = {L1_U_MF:.4e}, L2 = {L2_U_MF:.4e}, Linf = {Linf_U_MF:.4e}")

    # Write all norms to a single file 
    with open(outdir / "error_norms_all.txt", "w") as f:
        f.write("VERIFICATION: fluid vs exact (absolute)\n")
        f.write(f"Pressure    : L1={L1_p_F:.6g}  L2={L2_p_F:.6g}  Linf={Linf_p_F:.6g}\n")
        f.write(f"Temperature : L1={L1_T_F:.6g}  L2={L2_T_F:.6g}  Linf={Linf_T_F:.6g}\n")
        f.write(f"Velocity |U|: L1={L1_U_F:.6g}  L2={L2_U_F:.6g}  Linf={Linf_U_F:.6g}\n\n")

        f.write("VERIFICATION: multicomponent vs exact (absolute)\n")
        f.write(f"Pressure    : L1={L1_p_M:.6g}  L2={L2_p_M:.6g}  Linf={Linf_p_M:.6g}\n")
        f.write(f"Temperature : L1={L1_T_M:.6g}  L2={L2_T_M:.6g}  Linf={Linf_T_M:.6g}\n")
        f.write(f"Velocity |U|: L1={L1_U_M:.6g}  L2={L2_U_M:.6g}  Linf={Linf_U_M:.6g}\n\n")

        f.write("CONSISTENCY: multicomponent vs fluid (absolute)\n")
        f.write(f"Pressure    : L1={L1_p_MF:.6g}  L2={L2_p_MF:.6g}  Linf={Linf_p_MF:.6g}\n")
        f.write(f"Temperature : L1={L1_T_MF:.6g}  L2={L2_T_MF:.6g}  Linf={Linf_T_MF:.6g}\n")
        f.write(f"Velocity |U|: L1={L1_U_MF:.6g}  L2={L2_U_MF:.6g}  Linf={Linf_U_MF:.6g}\n")
# ---- L1/L2/Linf combined plot ----
    labels = ["Pressure", "Temperature", "Velocity |U|"]
    L1_vals = [L1_p, L1_T, L1_U]
    L2_vals = [L2_p, L2_T, L2_U]
    Linf_vals = [Linf_p, Linf_T, Linf_U]
    xpos = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.plot(xpos, L1_vals, marker="o", label=r"$L_1$")
    plt.plot(xpos, L2_vals, marker="s", label=r"$L_2$")
    plt.plot(xpos, Linf_vals, marker="^", label=r"$L_\infty$")
    plt.xticks(xpos, labels)
    plt.yscale("log")
    plt.ylabel("Error norm")
    plt.title("Error norms vs exact solution")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "error_norms_vs_exact.png", dpi=300)
    plt.close()

    print("\n=== Toro exact overlay meta ===")
    print(meta)

    # ---- Comparison plots ----
    # T compare
    plt.figure()
    plt.plot(x, TF, label="fluid")
    plt.plot(x, TM_i, label="multicomponent", linestyle="--")
    plt.plot(x, TE, label="exact", linestyle=":")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "T_compare.png", dpi=200)
    plt.close()

    # U compare
    plt.figure()
    plt.plot(x, UF, label="fluid")
    plt.plot(x, UM_i, label="multicomponent", linestyle="--")
    plt.plot(x, np.abs(uE), label="exact", linestyle=":")
    plt.xlabel("x")
    plt.ylabel("|U|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "U_compare.png", dpi=200)
    plt.close()

    # p compare
    plt.figure()
    plt.plot(x, pF, label="fluid")
    plt.plot(x, pM_i, label="multicomponent", linestyle="--")
    plt.plot(x, pE, label="exact", linestyle=":")
    plt.xlabel("x")
    plt.ylabel("p")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "p_compare.png", dpi=200)
    plt.close()

    # ---- Absolute difference plots (fluid vs multicomponent) ----
    dT = np.abs(TF - TM_i)
    dU = np.abs(UF - UM_i)
    dp = np.abs(pF - pM_i)

    plt.figure()
    plt.plot(x, dT)
    plt.xlabel("x")
    plt.ylabel("|ΔT|")
    plt.title("Absolute temperature difference (fluid vs multicomponent)")
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_T.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, dU)
    plt.xlabel("x")
    plt.ylabel("|Δ|U||")
    plt.title("Absolute velocity magnitude difference (fluid vs multicomponent)")
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_U.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, dp)
    plt.xlabel("x")
    plt.ylabel("|Δp|")
    plt.title("Absolute pressure difference (fluid vs multicomponent)")
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_p.png", dpi=200)
    plt.close()

    # ---- Absolute differences vs exact ----
    plt.figure()
    plt.plot(x, np.abs(TF - TE), label="|T_fluid - T_exact|")
    plt.plot(x, np.abs(TM_i - TE), label="|T_multi - T_exact|", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Absolute difference")
    plt.title("Temperature error vs exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_exact_T.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, np.abs(UF - np.abs(uE)), label="||U|_fluid - |U|_exact|")
    plt.plot(x, np.abs(UM_i - np.abs(uE)), label="||U|_multi - |U|_exact|", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Absolute difference")
    plt.title("Velocity magnitude error vs exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_exact_U.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, np.abs(pF - pE), label="|p_fluid - p_exact|")
    plt.plot(x, np.abs(pM_i - pE), label="|p_multi - p_exact|", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Absolute difference")
    plt.title("Pressure error vs exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "abs_diff_exact_p.png", dpi=200)
    plt.close()

    print(f"\nSaved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
