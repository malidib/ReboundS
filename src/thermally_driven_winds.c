/**
 * @file    thermally_driven_winds.c
 * @brief   Isotropic mass loss from thermally driven stellar winds.
 *
 * Default scaling (Parker‑like)
 *   dM/dt = - C_th * η *
 *           (R/R☉)^{α_R} (L/L☉)^{α_L} (M☉/M)^{α_M}  [M☉ yr⁻¹] .
 *
 * Operator‑level parameters
 * -------------------------
 *  tdw_const      (double)  – C_th, prefactor in M☉ yr⁻¹            (def: 2e‑14)
 *  tdw_Msun       (double)  – solar mass    in code units           (def: 1)
 *  tdw_Rsun       (double)  – solar radius  in code units           (def: 1)
 *  tdw_Lsun       (double)  – reference luminosity in code units    (def: 1)
 *  tdw_year       (double)  – Julian year in code‑time units        (def: 1)
 *  tdw_alpha_R    (double)  – exponent α_R                          (def: 2)
 *  tdw_alpha_L    (double)  – exponent α_L                          (def: 1.5)
 *  tdw_alpha_M    (double)  – exponent α_M                          (def: 1)
 *  tdw_max_dlnM   (double)  – max |ΔM|/M per call (safety limiter)   (def: 0.1)
 *
 * Particle‑level parameters
 * -------------------------
 *  tdw_eta        (double, req.) – heating efficiency η
 *  sse_L          (double, req.) – stellar luminosity (same units as Lsun)
 *  (stellar mass and radius are taken from the particle's m and r fields)
 *
 * Notes
 * -----
 * * Mass is removed isotropically; no linear‑momentum recoil is applied.
 * * Operator is unit‑agnostic if you set the scaling constants consistently.
 */

#include <math.h>
#include "rebound.h"
#include "reboundx.h"

/* Small helper: safe power with non‑positive base → returns 0 */
static inline double pow_safe(double x, double a){
    return (x > 0.0 && isfinite(x)) ? pow(x, a) : 0.0;
}

void rebx_thermally_driven_winds(struct reb_simulation* const sim,
                                 struct rebx_operator*  const op,
                                 const double                 dt)
{
    if (dt <= 0.0 || !isfinite(dt)) return;

    struct rebx_extras* const rx = sim->extras;
    const int N_real = sim->N - sim->N_var;                 /* ignore virtuals */

    /* Track whether any mass changed; if so, refresh SSE in lockstep. */
    int mass_changed = 0;
    struct rebx_operator* sse = rebx_get_operator(rx, "stellar_evolution_sse");

    /* ---------------- operator‑level parameters ------------------------ */
    double Msun      = 1.0;
    double Rsun      = 1.0;
    double Lsun      = 1.0;            /* reference luminosity             */
    double C0        = 2.0e-14;        /* M☉ yr⁻¹  → matches solar wind    */
    double year_len  = 1.0;

    double aR = 2.0;                   /* Parker: surface area             */
    double aL = 1.5;                   /* luminosity scaling               */
    double aM = 1.0;                   /* escape‑speed scaling             */
    double max_dlnM = 0.1;             /* safety: 10 % per call            */

    /* Retrieve overrides if present */
#define GET(name,var) do{                                \
        const double* _p = rebx_get_param(rx, op->ap, name); \
        if(_p && isfinite(*_p) && *_p > 0.0) var = *_p;  \
    } while(0)

    GET("tdw_Msun"    , Msun     );
    GET("tdw_Rsun"    , Rsun     );
    GET("tdw_Lsun"    , Lsun     );
    GET("tdw_const"   , C0       );
    GET("tdw_year"    , year_len );
    GET("tdw_alpha_R" , aR       );
    GET("tdw_alpha_L" , aL       );
    GET("tdw_alpha_M" , aM       );
    GET("tdw_max_dlnM", max_dlnM );

#undef GET

    const double* pDisCE   = rebx_get_param(rx, op->ap, "tdw_disable_in_CE");
    const double* pDisRLOF = rebx_get_param(rx, op->ap, "tdw_disable_in_RLOF");
    const int disable_in_CE   = pDisCE   ? (int)llround(*pDisCE)   : 1;
    const int disable_in_RLOF = pDisRLOF ? (int)llround(*pDisRLOF) : 1;

    /* Convert the prefactor to code‑mass per code‑time */
    const double pref = (C0 * Msun) / year_len;

    /* ---------------- iterate over real particles ---------------------- */
    for (int i = 0; i < N_real; i++){
        struct reb_particle* const p = &sim->particles[i];

        const double* eta_ptr = rebx_get_param(rx, p->ap, "tdw_eta");
        const double* L_ptr   = rebx_get_param(rx, p->ap, "sse_L");
        if (!eta_ptr || !L_ptr) continue;

        const double* inCE = rebx_get_param(rx, p->ap, "inside_CE");
        const double* rlof = rebx_get_param(rx, p->ap, "rlof_active");
        if ( (disable_in_CE   && inCE  && *inCE  > 0.5) ||
             (disable_in_RLOF && rlof && *rlof > 0.5) ){
            continue;
        }

        const double eta = *eta_ptr;
        const double L   = *L_ptr;
        const double R   = p->r;
        if (eta <= 0.0 || !isfinite(eta)) continue;
        if (L   <= 0.0 || !isfinite(L  )) continue;
        if (R   <= 0.0 || !isfinite(R  )) continue;
        if (p->m <= 0.0 || !isfinite(p->m)) continue;

        /* Dimensionless ratios and powers */
        const double RR = pow_safe(R / Rsun, aR);
        const double LL = pow_safe(L / Lsun, aL);
        const double MM = pow_safe(Msun / p->m, aM);

        /* Mass‑loss rate (code units) */
        const double mdot = -pref * eta * RR * LL * MM;

        /* Safety limiter */
        const double dM_lim = -max_dlnM * p->m;            /* negative value   */
        double dM = mdot * dt;
        if (dM < dM_lim) dM = dM_lim;

        /* Apply mass loss */
        p->m += dM;
        if (p->m < 0.0) p->m = 0.0;                       /* shouldn’t happen */

        if (dM != 0.0) mass_changed = 1;
    }

    /* Lockstep update: refresh R(M) and L(M) immediately after any mass change. */
    if (mass_changed && sse && sse->step_function){
        sse->step_function(sim, sse, 0.0);
    }

    /* Recentre on centre of mass after mass has changed */
    reb_simulation_move_to_com(sim);
}
