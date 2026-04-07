/**
 * @file    stellar_wind_mass_loss.c
 * @brief   Continuous stellar‑wind mass loss (Reimers law).
 *
 * Reimers (1975) prescription:
 *
 *   dM/dt = - 4.0e-13 η (L/L☉)(R/R☉)(M/M☉)⁻¹  M☉ yr⁻¹ .
 *
 * Operator‑level parameters
 * -------------------------
 *  swml_const  (double, optional)  – prefactor in M☉ yr⁻¹ (default 4e‑13)
 *  swml_Msun   (double, optional)  – Solar mass  in code‑mass units (default 1)
 *  swml_Rsun   (double, optional)  – Solar radius in code‑length units (default 1)
 *  swml_Lsun   (double, optional)  – Solar luminosity in code‑lum  units (default 1)
 *  swml_year   (double, optional)  – Length of one Julian year in code‑time
 *                                    units (default 1 → legacy behaviour).
 *  swml_max_dlnM (double, optional) – max |ΔM|/M per call (default 0.1)
 *
 * Particle‑level parameters
 * -------------------------
 *  swml_eta    (double, required)  – dimensionless efficiency η
 *  (stellar radius is taken from the particle's radius r; stellar
 *  luminosity is read from the ``sse_L`` parameter set by the
 *  \ref stellar_evolution_sse.c "stellar evolution" operator.)
 *
 * Notes
 * -----
 * * The operator is *unit agnostic* as long as you set the scaling
 *   parameters consistently.
 * * Mass is removed isotropically; no linear‑momentum recoil is applied.
 */

#include <math.h>
#include "rebound.h"
#include "reboundx.h"

/* ------------------------------------------------------------------------- */
/* Operator kernel                                                           */
/* ------------------------------------------------------------------------- */
void rebx_stellar_wind_mass_loss(struct reb_simulation* const sim,
                                 struct rebx_operator*  const op,
                                 const double                 dt)
{
    if (dt <= 0.0 || !isfinite(dt)) return;

    struct rebx_extras* const rx = sim->extras;
    const int N_real = sim->N - sim->N_var;          /* ignore virtuals       */

    /* Track whether any mass changed; if so, refresh SSE in lockstep. */
    int mass_changed = 0;
    struct rebx_operator* sse = rebx_get_operator(rx, "stellar_evolution_sse");

    /* --------------------- operator‑level parameters -------------------- */
    double Msun     = 1.0;    /* code‑mass units per solar mass             */
    double Rsun     = 1.0;    /* code‑length units per solar radius         */
    double Lsun     = 1.0;    /* code‑lum   units per solar luminosity      */
    double C0       = 4.0e-13;/* base prefactor [M☉ yr⁻¹]                   */
    double year_len = 1.0;    /* code‑time units per Julian year            */
    double max_dlnM = 0.1;    /* safety limiter                             */

    const double* pM   = rebx_get_param(rx, op->ap, "swml_Msun");
    const double* pR   = rebx_get_param(rx, op->ap, "swml_Rsun");
    const double* pL   = rebx_get_param(rx, op->ap, "swml_Lsun");
    const double* pC   = rebx_get_param(rx, op->ap, "swml_const");
    const double* pYr  = rebx_get_param(rx, op->ap, "swml_year");
    const double* pMax = rebx_get_param(rx, op->ap, "swml_max_dlnM");
    const double* pDisCE   = rebx_get_param(rx, op->ap, "swml_disable_in_CE");
    const double* pDisRLOF = rebx_get_param(rx, op->ap, "swml_disable_in_RLOF");

    if (pM  && isfinite(*pM ) && *pM  > 0.0) Msun     = *pM;
    if (pR  && isfinite(*pR ) && *pR  > 0.0) Rsun     = *pR;
    if (pL  && isfinite(*pL ) && *pL  > 0.0) Lsun     = *pL;
    if (pC  && isfinite(*pC ) && *pC  > 0.0) C0       = *pC;
    if (pYr  && isfinite(*pYr ) && *pYr  > 0.0) year_len = *pYr;
    if (pMax && isfinite(*pMax) && *pMax > 0.0) max_dlnM = *pMax;

    const int disable_in_CE   = pDisCE   ? (int)llround(*pDisCE)   : 1;
    const int disable_in_RLOF = pDisRLOF ? (int)llround(*pDisRLOF) : 1;

    /* Prefactor converted to *code‑mass per code‑time* */
    const double pref = (C0 * Msun) / year_len;      /* [code‑mass / code‑time] */

    /* --------------------- loop over genuine particles ------------------ */
    for (int i = 0; i < N_real; i++){
        struct reb_particle* const p = &sim->particles[i];

        /* Fetch particle parameters */
        const double* eta_ptr = rebx_get_param(rx, p->ap, "swml_eta");
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

        /* Sanity checks */
        if (!isfinite(eta) || eta <= 0.0) continue;
        if (!isfinite(L)   || L   <= 0.0) continue;
        if (!isfinite(R)   || R   <= 0.0) continue;
        if (p->m <= 0.0 || !isfinite(p->m)) continue;

        /* Reimers mass‑loss rate in code units */
        const double mdot = -pref * eta
                            * (L / Lsun)
                            * (R / Rsun)
                            * (Msun / p->m);     /* [code‑mass / code‑time] */

        double dM = mdot * dt;                   /* negative number         */
        const double dM_lim = -max_dlnM * p->m;  /* cap fractional change   */
        if (dM < dM_lim) dM = dM_lim;

        /* Prevent negative masses */
        if (p->m + dM <= 0.0) {
            p->m = 0.0;
        } else {
            p->m += dM;
        }

        if (dM != 0.0) mass_changed = 1;
    }

    /* Lockstep update: refresh R(M) and L(M) immediately after any mass change. */
    if (mass_changed && sse && sse->step_function){
        sse->step_function(sim, sse, 0.0);
    }

    /* Recentre: mass distribution changed */
    reb_simulation_move_to_com(sim);
}
