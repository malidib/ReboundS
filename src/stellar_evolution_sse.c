/**
 * @file    stellar_evolution_sse.c
 * @brief   Analytic stellar-structure presets: R(M) and L(M) for several object classes.
 *
 * This operator updates each real particle's radius (p->r) and stores its luminosity
 * as the particle parameter "sse_L". The update is algebraic (dt is ignored), so the
 * mapping is evaluated from the particle's current mass each call.
 *
 * ---------------------------------------------------------------------------
 * BACKWARD COMPATIBILITY (drop-in behavior)
 * ---------------------------------------------------------------------------
 * If any of the legacy per-particle parameters are present:
 *   - sse_R_coeff, sse_R_exp, sse_L_coeff, sse_L_exp
 * then the operator uses the legacy power-law mapping:
 *
 *   R = sse_R_coeff * Rsun * (M/Msun)^(sse_R_exp)   (defaults: coeff=1, exp=0.8)
 *   L = sse_L_coeff * Lsun * (M/Msun)^(sse_L_exp)   (defaults: coeff=1, exp=3.5)
 *
 * exactly as in the original implementation.
 *
 * If no legacy override parameters are present, the operator uses the per-particle
 * integer selector sse_type to choose a preset object class.
 *
 * ---------------------------------------------------------------------------
 * NEW FEATURE: sse_type presets (per-particle integer stored as a double)
 * ---------------------------------------------------------------------------
 * sse_type values (per particle, interpreted as int):
 *
 *   1 : H-rich main sequence / ZAMS-like (piecewise mass-radius and mass-luminosity)
 *   2 : H-rich giant (RGB/AGB lumped; radius from a weak mass scaling; luminosity from R^2)
 *   3 : Stripped helium star (He-MS / WR-like lumped; compact and luminous power laws)
 *   4 : White dwarf (WD; Nauenberg mass-radius relation; low default luminosity)
 *   5 : Compact remnant (NS/BH; NS if M<=Mns_max else BH Schwarzschild radius)
 *
 * If sse_type is absent or equals 0, the operator falls back to the legacy default
 * power law (MS-like) using the original defaults (0.8, 3.5) unless legacy overrides
 * were provided.
 *
 * Optional per-particle multipliers (applied for sse_type 1..5 and also for type 0 fallback):
 *   sse_R_mult (double) default 1
 *   sse_L_mult (double) default 1
 *
 * ---------------------------------------------------------------------------
 * Operator-level parameters (all optional)
 * ---------------------------------------------------------------------------
 *   sse_Msun (double)   Solar mass in code units (default 1)
 *   sse_Rsun (double)   Solar radius in code units (default 1)
 *   sse_Lsun (double)   Solar luminosity in code units (default 1)
 *
 * WD / compact defaults (dimensionless factors relative to solar units):
 *   sse_Mch        (double) Chandrasekhar mass in Msun (default 1.44)
 *   sse_Rwd_coeff  (double) WD radius coefficient in Rsun (default 0.0112)
 *   sse_Lwd        (double) WD luminosity in Lsun (default 1e-3)
 *
 *   sse_Rns_factor (double) NS radius / Rsun (default 1.724e-5 ~ 12 km)
 *   sse_Rbh_factor (double) Schwarzschild radius per Msun / Rsun (default 4.246e-6 ~ 2.953 km per Msun)
 *   sse_Mns_max    (double) NS/BH boundary mass in Msun for sse_type=5 (default 3.0)
 *
 * Notes:
 * - The luminosity "sse_L" is always written (including for compact remnants, where it
 *   defaults to 0), so wind operators that use sse_L will see a consistent field.
 * - These are analytic presets meant to provide reasonable radii/luminosities without
 *   tracks; users can always override using the legacy per-particle power-law params.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "rebound.h"
#include "reboundx.h"

/* ---------- Helpers: safe pow and simple MS/giant/He prescriptions ---------- */

static double sse_pow_pos(const double x, const double a){
    /* x must be > 0 */
    return pow(x, a);
}

static double sse_ms_radius_ratio(const double m){
    /* m is M/Msun (dimensionless), ZAMS-like radius in Rsun */
    if (m <= 0.0) return 0.0;
    if (m <= 1.0){
        return sse_pow_pos(m, 0.80);
    }else if (m <= 10.0){
        return sse_pow_pos(m, 0.57);
    }else{
        /* continue smoothly past 10 Msun */
        const double R10 = sse_pow_pos(10.0, 0.57);
        return R10 * sse_pow_pos(m/10.0, 0.50);
    }
}

static double sse_ms_luminosity_ratio(const double m){
    /* m is M/Msun (dimensionless), ZAMS-like luminosity in Lsun
       Continuous piecewise law commonly used for MS scalings. */
    if (m <= 0.0) return 0.0;
    if (m < 0.43){
        return 0.23 * sse_pow_pos(m, 2.30);
    }else if (m < 2.0){
        return sse_pow_pos(m, 4.00);
    }else{
        /* Choose coefficient to match L=16 Lsun at m=2 (continuity with m^4) */
        const double coeff = 16.0 / sse_pow_pos(2.0, 3.50);  /* ~sqrt(2) */
        return coeff * sse_pow_pos(m, 3.50);
    }
}

static double sse_giant_radius_ratio(const double m){
    /* Very compact, low-dimensional "giant" radius: weak mass dependence.
       Default represents a large RGB/AGB-like radius scale. */
    if (m <= 0.0) return 0.0;
    return 100.0 * sse_pow_pos(m, -0.30);
}

static double sse_giant_luminosity_ratio_from_R(const double R_ratio){
    /* Giants: approximate L ~ (T/Tsun)^4 R^2 with a representative Teff ~ 4000 K.
       (4000/5772)^4 ~ 0.23 */
    const double Teff_factor = 0.23;
    return Teff_factor * R_ratio * R_ratio;
}

static double sse_he_radius_ratio(const double m){
    /* Stripped helium star: compact radius scaling. */
    if (m <= 0.0) return 0.0;
    return 0.20 * sse_pow_pos(m, 0.60);
}

static double sse_he_luminosity_ratio(const double m){
    /* Stripped helium star: luminous, simple power law. */
    if (m <= 0.0) return 0.0;
    return 50.0 * sse_pow_pos(m, 2.50);
}

/* ---------- Main operator ---------- */

void rebx_stellar_evolution_sse(struct reb_simulation* const sim,
                                struct rebx_operator* const operator,
                                const double dt)
{
    (void)dt; /* dt is intentionally ignored: algebraic update */

    struct rebx_extras* const rebx = sim->extras;
    const int N_real = sim->N - sim->N_var;

    /* Solar units in code units */
    double Msun = 1.0;
    double Rsun = 1.0;
    double Lsun = 1.0;

    const double* ptr = NULL;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Msun"); if (ptr) Msun = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Rsun"); if (ptr) Rsun = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Lsun"); if (ptr) Lsun = *ptr;

    /* WD / compact parameters (dimensionless, relative to solar units) */
    double Mch        = 1.44;
    double Rwd_coeff  = 0.0112;
    double Lwd        = 1e-3;

    double Rns_factor = 1.724e-5; /* ~12 km in Rsun */
    double Rbh_factor = 4.246e-6; /* Schwarzschild radius per Msun in Rsun */
    double Mns_max    = 3.0;

    ptr = rebx_get_param(rebx, operator->ap, "sse_Mch");        if (ptr) Mch        = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Rwd_coeff");  if (ptr) Rwd_coeff  = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Lwd");        if (ptr) Lwd        = *ptr;

    ptr = rebx_get_param(rebx, operator->ap, "sse_Rns_factor"); if (ptr) Rns_factor = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Rbh_factor"); if (ptr) Rbh_factor = *ptr;
    ptr = rebx_get_param(rebx, operator->ap, "sse_Mns_max");    if (ptr) Mns_max    = *ptr;

    /* Safety floors (keep radii positive to avoid pathological downstream behavior) */
    const double R_floor = 1e-12 * (Rsun > 0.0 ? Rsun : 1.0);

    for (int i = 0; i < N_real; i++){
        struct reb_particle* const p = &sim->particles[i];

        if (!(p->m > 0.0) || !isfinite(p->m) || !(Msun > 0.0) || !isfinite(Msun)){
            continue;
        }

        const double m_ratio = p->m / Msun; /* dimensionless M/Msun */

        /* ---- Check for legacy per-particle power-law override parameters ---- */
        const double* R_coeff_ptr = rebx_get_param(rebx, p->ap, "sse_R_coeff");
        const double* R_exp_ptr   = rebx_get_param(rebx, p->ap, "sse_R_exp");
        const double* L_coeff_ptr = rebx_get_param(rebx, p->ap, "sse_L_coeff");
        const double* L_exp_ptr   = rebx_get_param(rebx, p->ap, "sse_L_exp");

        const bool has_legacy_override = (R_coeff_ptr || R_exp_ptr || L_coeff_ptr || L_exp_ptr);

        double R = 0.0;
        double L = 0.0;

        if (has_legacy_override){
            /* Legacy behavior: identical to the original implementation */
            double R_coeff = 1.0;
            double R_exp   = 0.8;
            double L_coeff = 1.0;
            double L_exp   = 3.5;

            if (R_coeff_ptr) R_coeff = *R_coeff_ptr;
            if (R_exp_ptr)   R_exp   = *R_exp_ptr;
            if (L_coeff_ptr) L_coeff = *L_coeff_ptr;
            if (L_exp_ptr)   L_exp   = *L_exp_ptr;

            if (m_ratio > 0.0 && isfinite(m_ratio)){
                R = R_coeff * Rsun * pow(m_ratio, R_exp);
                L = L_coeff * Lsun * pow(m_ratio, L_exp);
            }else{
                R = 0.0;
                L = 0.0;
            }
        }else{
            /* ---- New presets via sse_type ---- */
            int sse_type = 0; /* 0 => fallback to legacy default MS-like power law */
            const double* type_ptr = rebx_get_param(rebx, p->ap, "sse_type");
            if (type_ptr){
                /* Stored as a double in REBOUNDx; interpret as int. */
                sse_type = (int)llround(*type_ptr);
            }

            double R_mult = 1.0;
            double L_mult = 1.0;
            const double* Rm_ptr = rebx_get_param(rebx, p->ap, "sse_R_mult");
            const double* Lm_ptr = rebx_get_param(rebx, p->ap, "sse_L_mult");
            if (Rm_ptr) R_mult = *Rm_ptr;
            if (Lm_ptr) L_mult = *Lm_ptr;

            double R_ratio = 0.0; /* in units of Rsun */
            double L_ratio = 0.0; /* in units of Lsun */

            switch (sse_type){
                case 1: { /* H-rich MS / ZAMS-like */
                    R_ratio = sse_ms_radius_ratio(m_ratio);
                    L_ratio = sse_ms_luminosity_ratio(m_ratio);
                } break;

                case 2: { /* H-rich giant (RGB/AGB lumped) */
                    R_ratio = sse_giant_radius_ratio(m_ratio);
                    L_ratio = sse_giant_luminosity_ratio_from_R(R_ratio);
                } break;

                case 3: { /* Stripped helium star */
                    R_ratio = sse_he_radius_ratio(m_ratio);
                    L_ratio = sse_he_luminosity_ratio(m_ratio);
                } break;

                case 4: { /* White dwarf (Nauenberg) */
                    if (Mch > 0.0 && m_ratio > 0.0){
                        const double x = m_ratio / Mch; /* M / Mch */
                        /* Nauenberg: R = R0 * sqrt( (x^-2/3) - (x^2/3) ) */
                        const double a = pow(1.0/x, 2.0/3.0);
                        const double b = pow(x,     2.0/3.0);
                        double term = a - b;
                        if (!isfinite(term) || term < 0.0) term = 0.0;
                        R_ratio = Rwd_coeff * sqrt(term); /* Rsun units */
                    }else{
                        R_ratio = 0.0;
                    }
                    L_ratio = (Lwd > 0.0 && isfinite(Lwd)) ? Lwd : 0.0;
                } break;

                case 5: { /* Compact remnant (NS/BH by mass) */
                    if (m_ratio <= Mns_max){
                        R_ratio = Rns_factor;      /* ~ constant NS radius */
                    }else{
                        R_ratio = Rbh_factor*m_ratio; /* Schwarzschild radius */
                    }
                    L_ratio = 0.0; /* intrinsic luminosity not prescribed here */
                } break;

                case 0:
                default: { /* fallback: original MS-like power-law defaults */
                    if (m_ratio > 0.0){
                        R_ratio = pow(m_ratio, 0.8);
                        L_ratio = pow(m_ratio, 3.5);
                    }else{
                        R_ratio = 0.0;
                        L_ratio = 0.0;
                    }
                } break;
            }

            /* Apply multipliers */
            if (!isfinite(R_mult) || R_mult <= 0.0) R_mult = 1.0;
            if (!isfinite(L_mult) || L_mult <  0.0) L_mult = 1.0;

            R = Rsun * R_ratio * R_mult;
            L = Lsun * L_ratio * L_mult;
        }

        /* Final sanitation */
        if (!isfinite(R) || R < R_floor) R = R_floor;
        if (!isfinite(L) || L < 0.0)     L = 0.0;

        p->r = R;
        rebx_set_param_double(rebx, &p->ap, "sse_L", L);
    }
}
