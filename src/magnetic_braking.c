/**
 * @file    magnetic_braking.c
 * @brief   Verbunt–Zwaan / Kawaler magnetic‑braking torque with saturation.
 *
 * Implements
 *
 *   dJ/dt = -K * (R/Rsun)^{1/2} * (M/Msun)^{-1/2} *
 *           { ω^3                   , ω ≤ ω_sat
 *           { ω * ω_sat^2          , ω > ω_sat
 *
 * with K specified in cgs and internally converted to code units.
 * The update for |Ω| uses closed-form, positivity‑preserving formulas in
 * both regimes (and piecewise when the step crosses the saturation threshold).
 *
 * Particle‑level parameters  (all scalars are stored as double in REBOUNDx)
 * ------------------------------------------------------------------------
 * mb_on          (double, required ≠0)  – enable braking on this particle
 * mb_convective  (double, required ≠0)  – star has a convective envelope
 * mb_omega_sat   (double, optional)     – saturation angular velocity
 * mb_tau_conv    (double, optional)     – convective turnover time
 * mb_R           (double, optional)     – radius override (in code-length units)
 * I              (double, required)     – moment of inertia
 * Omega          (reb_vec3d, required)  – spin angular‑velocity vector
 *
 * Operator‑level parameters
 * -------------------------
 * mb_K           (double, optional)     – braking constant in cgs units
 *                                             (default 2.7e47).
 * mb_Msun        (double, optional)     – solar mass  in code‑mass units
 *                                             (default 1).
 * mb_Rsun        (double, optional)     – solar radius in code‑length units
 *                                             (default 1).
 * mb_year        (double, optional)     – Julian year in code‑time units
 *                                             (default 1).
 * mb_Rossby_sat  (double, optional)     – critical Rossby number for
 *                                         saturation (default 0.1).
 *
 * Compatibility note (OmegaMag sync for tides_constant_time_lag)
 * -------------------------------------------------------------
 * The tides_constant_time_lag module uses a scalar particle parameter
 * "OmegaMag" rather than the vector "Omega". Since magnetic_braking updates
 * the vector "Omega", this operator optionally keeps "OmegaMag" synchronized
 * with |Omega| automatically:
 *
 *   - If "OmegaMag" exists, it is updated every braking step.
 *   - If "OmegaMag" does not exist but "tctl_tau" is present on the particle,
 *     "OmegaMag" is created and set to |Omega|.
 *
 * This eliminates the need for manual OmegaMag <- |Omega| synchronization in
 * user code when combining magnetic_braking with tides_constant_time_lag.
 */

#include <math.h>
#include <stdio.h>
#include "rebound.h"
#include "reboundx.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------- */
/* Internal helper: apply braking to a single particle                       */
/* ------------------------------------------------------------------------- */
static inline void apply_magnetic_brake(struct reb_particle*    const p,
                                        struct rebx_extras*     const rx,
                                        const double            K_fac,        /* K in code units */
                                        const double            dt,           /* code time step */
                                        const double            Rossby_sat,   /* critical Rossby number */
                                        const double            Msun_code,    /* solar mass in code units */
                                        const double            Rsun_code)    /* solar radius in code units */
{
    /* ----------------------- fetch flags & properties ------------------- */
    const double* mb_on  = rebx_get_param(rx, p->ap, "mb_on");
    if (!mb_on || *mb_on == 0.0) return;

    const double* conv   = rebx_get_param(rx, p->ap, "mb_convective");
    if (!conv || *conv == 0.0) return;

    const double* I_ptr = rebx_get_param(rx, p->ap, "I");
    if (!I_ptr || *I_ptr <= 0.0 || !isfinite(*I_ptr)) return;

    struct reb_vec3d* Omega = rebx_get_param_vec(rx, p->ap, "Omega");
    if (!Omega) return;

    /* Radius: optional per‑particle override, otherwise use particle radius */
    const double* Rptr = rebx_get_param(rx, p->ap, "mb_R");
    const double R = (Rptr && isfinite(*Rptr) && *Rptr > 0.0) ? *Rptr : p->r;

    const double M = p->m;

    if (!(isfinite(R) && R > 0.0)) return;
    if (!(isfinite(M) && M > 0.0)) return;
    if (!(isfinite(Msun_code) && Msun_code > 0.0)) return;
    if (!(isfinite(Rsun_code) && Rsun_code > 0.0)) return;

    const double ox = Omega->x, oy = Omega->y, oz = Omega->z;
    const double omega = sqrt(ox*ox + oy*oy + oz*oz);
    if (!isfinite(omega) || omega == 0.0) return;

    /* ------------------------- saturation threshold --------------------- */
    const double* sat_ptr = rebx_get_param(rx, p->ap, "mb_omega_sat");
    const double* tau_ptr = rebx_get_param(rx, p->ap, "mb_tau_conv");

    double omega_sat = INFINITY;  /* default: no saturation */

    if (tau_ptr && *tau_ptr > 0.0 && isfinite(*tau_ptr) &&
        isfinite(Rossby_sat) && Rossby_sat > 0.0) {
        omega_sat = 2.0*M_PI/(Rossby_sat * (*tau_ptr));
    }
    if (sat_ptr && *sat_ptr > 0.0 && isfinite(*sat_ptr)) {
        omega_sat = *sat_ptr;     /* explicit override wins */
    }

    /* --------------------------- prefactors ----------------------------- */
    /* dimensionless ratios (R/Rsun)^{1/2} and (M/Msun)^{-1/2} */
    const double Rfac = sqrt(R / Rsun_code);
    const double Mfac = 1.0 / sqrt(M / Msun_code);

    /* Core coefficient C so that:
       unsaturated: dω/dt = -C * ω^3
       saturated:   dω/dt = -C * ω_sat^2 * ω
     */
    const double C = (K_fac * Rfac * Mfac) / (*I_ptr);

    if (!(isfinite(C) && C >= 0.0)) return;           /* defensive: nothing to do */
    if (!(isfinite(dt) && dt > 0.0)) return;

    /* -------------------- closed-form magnitude update ------------------ */
    double scale = 1.0;   /* multiplicative factor applied to Ω components */

    const int has_sat   = isfinite(omega_sat);
    const int in_sat    = has_sat && (omega > omega_sat);

    if (!has_sat || !in_sat) {
        /* Entire step unsaturated: ω' = ω / sqrt(1 + 2*C*ω^2*dt) */
        const double denom = 1.0 + 2.0*C*omega*omega*dt;
        if (!(denom > 0.0 && isfinite(denom))) return;
        scale = 1.0 / sqrt(denom);
    } else {
        /* Start saturated: first decay exponentially until ω reaches ω_sat,
           then continue with unsaturated law for the remainder of the step.
         */
        const double lambda = C * omega_sat * omega_sat;   /* > 0 */
        if (!(isfinite(lambda) && lambda > 0.0)) return;

        /* Time to reach ω_sat under saturated law: ω(t) = ω0 * exp(-λ t) */
        const double t_cross = log(omega / omega_sat) / lambda;

        if (!(isfinite(t_cross) && t_cross >= 0.0)) return;

        if (dt <= t_cross) {
            /* Entire step saturated */
            scale = exp(-lambda * dt);
        } else {
            /* Saturated to ω_sat, then unsaturated for the remainder */
            const double scale_sat  = exp(-lambda * t_cross);               /* = ω_sat / ω */
            const double rem        = dt - t_cross;
            const double denom_uns  = 1.0 + 2.0*C*omega_sat*omega_sat*rem;  /* unsat piece from ω_sat */
            if (!(denom_uns > 0.0 && isfinite(denom_uns))) return;
            const double scale_uns  = 1.0 / sqrt(denom_uns);
            scale = scale_sat * scale_uns;   /* total multiplicative change */
        }
    }

    if (!(isfinite(scale) && scale > 0.0)) {
        /* Should not happen with closed-form formulas, but be safe. */
        fprintf(stderr, "[magnetic_braking] invalid scale (particle hash %u)\n", p->hash);
        return;
    }

    /* --------------------------- apply scaling -------------------------- */
    Omega->x *= scale;
    Omega->y *= scale;
    Omega->z *= scale;

    /* ------------------------------------------------------------------ */
    /* Optional convenience: keep OmegaMag synchronized for tctl users     */
    /* ------------------------------------------------------------------ */
    {
        const double omega_new = omega * scale;   /* |Omega| after braking */

        /* Update if OmegaMag already exists... */
        double* OmegaMag_ptr = rebx_get_param(rx, p->ap, "OmegaMag");
        if (OmegaMag_ptr){
            *OmegaMag_ptr = omega_new;
        } else {
            /* ...or create OmegaMag if this body appears to use tctl. */
            const double* tctl_tau = rebx_get_param(rx, p->ap, "tctl_tau");
            if (tctl_tau){
                /* Create/update OmegaMag as a scalar parameter (double). */
                rebx_set_param_double(rx, (struct rebx_node**)&p->ap, "OmegaMag", omega_new);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Operator kernel                                                           */
/* ------------------------------------------------------------------------- */
void rebx_magnetic_braking(struct reb_simulation* const sim,
                           struct rebx_operator*   const op,
                           const double                   dt)
{
    if (!(isfinite(dt) && dt > 0.0)) return;

    struct rebx_extras* const rx = sim->extras;
    const int N = sim->N;

    /* Operator‑level parameters */
    double Msun_code = 1.0;   /* solar mass in code‑mass units  */
    double Rsun_code = 1.0;   /* solar radius in code‑length units */
    double year_code = 1.0;   /* Julian year in code‑time units */

    const double* pM   = rebx_get_param(rx, op->ap, "mb_Msun");
    const double* pR   = rebx_get_param(rx, op->ap, "mb_Rsun");
    const double* pY   = rebx_get_param(rx, op->ap, "mb_year");
    const double* pK   = rebx_get_param(rx, op->ap, "mb_K");
    const double* pRo  = rebx_get_param(rx, op->ap, "mb_Rossby_sat");

    if (pM && isfinite(*pM) && *pM > 0.0) Msun_code = *pM;
    if (pR && isfinite(*pR) && *pR > 0.0) Rsun_code = *pR;
    if (pY && isfinite(*pY) && *pY > 0.0) year_code = *pY;

    const double K_cgs = (pK && isfinite(*pK) && *pK > 0.0) ? *pK : 2.7e47;

    /* Convert K from cgs to code units for:
       dJ/dt = -K_fac * (R/Rsun)^{1/2} (M/Msun)^{-1/2} ω^3.
       Dimensional analysis gives [K] = M L^2 T.
       => K_code = K_cgs / (M_unit * L_unit^2 * T_unit).
     */
    const double Msun_cgs = 1.98847e33;   /* g  */
    const double Rsun_cgs = 6.957e10;     /* cm */
    const double year_cgs = 3.15576e7;    /* s  */

    const double M_unit = Msun_cgs / Msun_code;  /* [g / code‑mass] */
    const double L_unit = Rsun_cgs / Rsun_code;  /* [cm / code‑length] */
    const double T_unit = year_cgs / year_code;  /* [s / code‑time] */

    const double K_fac = K_cgs / (M_unit * L_unit * L_unit * T_unit);

    double Rossby_sat = 0.1; /* default critical Rossby number */
    if (pRo && isfinite(*pRo) && *pRo > 0.0) Rossby_sat = *pRo;

    for (int i = 0; i < N; i++){
        apply_magnetic_brake(&sim->particles[i], rx, K_fac, dt, Rossby_sat,
                             Msun_code, Rsun_code);
    }
}
