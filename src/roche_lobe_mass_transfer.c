/* ============================================================================
 * @file    roche_lobe_mass_transfer.c
 * @brief   Roche‑Lobe Overflow (RLOF) + Common‑Envelope (CE) operator
 *          with momentum‑exact mass transfer, controlled j‑loss,
 *          optional CE reaction, adaptive sub‑stepping, and diagnostics.
 *
 *          2026‑02‑10  (drop‑in replacement; adds optional Eddington cap)
 *
 * Operator parameters (all optional unless stated)
 * ------------------------------------------------
 * rlmt_donor            (double, required)  – donor particle index (0‑based)
 * rlmt_accretor         (double, required)  – accretor particle index (0‑based)
 * rlmt_loss_fraction    (double)            – systemic loss fraction f_loss in [0,1] (default 0)
 * jloss_mode            (double)            – 0 donor wind (Jeans mode; wind mass leaves from donor)
 *                                            1 isotropic re‑emission from accretor (true re‑emission)
 *                                            2 COM‑loss (v_loss = v_CM; note: implicit donor AM loss remains)
 *                                            3 target j‑loss: add tangential Δv so that |Δv| = f_j (J/M)/r
 * jloss_factor          (double)            – scale for mode 3 (default 1.0)
 * rlmt_skip_in_CE       (double, bool)      – skip RLOF if r < R_d               (default 1)
 * rlmt_substep_max_dm   (double)            – max |ΔM|/M per sub‑step            (default 1e‑3)
 * rlmt_substep_max_dr   (double)            – max |Δr|/r per sub‑step            (default 5e‑3)
 * rlmt_min_substeps     (double)            – enforce ≥N sub‑steps               (default 3)
 * ce_profile_file       (string, optional)  – ASCII table (s, rho, cs), overrides power‑law
 * ce_kick_cfl           (double)            – |Δv| ≤ cfl × c_s per sub‑step      (default 1.0)
 * ce_reaction_on_donor  (double, bool)      – apply opposite CE kick to donor    (default 0)
 * merge_eps             (double)            – explicit merge radius; default 0.5×min(Rs) with floor
 *
 * Optional Eddington cap (operator or particle)
 * ---------------------------------------------
 * rlmt_mdot_edd         (double)            – maximum net accretion rate onto the accretor (mass/time in code units).
 *                                            If <= 0 or absent => no cap. If > 0, net accretion is capped and any
 *                                            excess is isotropically re-emitted from the accretor (no recoil).
 *                                            The particle-level (accretor) value takes precedence over operator-level.
 *
 * Particle parameters (donor unless noted)
 * ---------------------------------------
 * rlmt_Hp               (double, donor)     – pressure scale height H_P
 * rlmt_mdot0            (double, donor)     – reference mass‑loss rate \dot M_0  (>0)
 * rlmt_mdot_edd         (double, accretor)  – optional Eddington accretion cap (mass/time in code units)
 *
 * CE power‑law (operator scope; used if no table)
 * -----------------------------------------------
 * ce_rho0               (double)            – density normalization
 * ce_alpha_rho          (double)            – density slope
 * ce_cs                 (double)            – sound‑speed normalization
 * ce_alpha_cs           (double)            – sound‑speed slope
 * ce_xmin               (double)            – Coulomb cutoff x_min               (default 1e‑4)
 * ce_Qd                 (double)            – geometric drag coefficient          (default 0)
 *
 * Output diagnostic
 * -----------------
 * rlmt_last_dE          (double, op‑attr)   – accumulated ΔE_orb of the donor–accretor pair
 *                                            over the last call (Galilean invariant).
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

#ifndef RLMT_EXP_CLAMP
#define RLMT_EXP_CLAMP  80.0     /* prevents exp() overflow/underflow in Ritter law */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX2(a,b) (( (a) > (b) ) ? (a) : (b))
#define MIN2(a,b) (( (a) < (b) ) ? (a) : (b))

/* --- Optional string param getter (define REBX_ENABLE_STRING_PARAMS if available) --- */
#ifdef REBX_ENABLE_STRING_PARAMS
/* REBOUNDx provides this in newer versions; forward declare to avoid header mismatch. */
extern const char* rebx_get_param_str(struct rebx_extras* rx, struct rebx_param* ap, const char* name);
#endif

/* -------------------------- CE profile table (global) -------------------------- */
/* Kept global for simplicity; loaded at first use if ce_profile_file is set.     */
struct ce_profile {
    int     n;
    double* s;      /* r/R_d */
    double* rho;    /* density */
    double* cs;     /* sound speed */
};

static struct ce_profile ce_tab = {0, NULL, NULL, NULL};

/* ------------------------------ small helpers ------------------------------ */
static inline void cross3(const double ax, const double ay, const double az,
                          const double bx, const double by, const double bz,
                          double* rx, double* ry, double* rz){
    *rx = ay*bz - az*by;
    *ry = az*bx - ax*bz;
    *rz = ax*by - ay*bx;
}
static inline double dot3(const double ax, const double ay, const double az,
                          const double bx, const double by, const double bz){
    return ax*bx + ay*by + az*bz;
}

/* Build any unit vector perpendicular to n = (nx,ny,nz). */
static void unit_perp_to(const double nx, const double ny, const double nz,
                         double* ex, double* ey, double* ez){
    /* try cross with z‑axis first, then x‑axis */
    double cx = ny*1.0 - nz*0.0;  /* n × ez = (ny, -nx, 0) */
    double cy = nz*0.0 - nx*1.0;
    double cz = nx*0.0 - ny*0.0;
    double nrm = sqrt(cx*cx + cy*cy + cz*cz);
    if(nrm < 1e-12){
        cx = ny*0.0 - nz*0.0;     /* n × ex = (0, nz, -ny) */
        cy = nz*1.0 - nx*0.0;
        cz = nx*0.0 - ny*1.0;
        nrm = sqrt(cx*cx + cy*cy + cz*cz);
        if(nrm < 1e-12){ *ex = 1.0; *ey = 0.0; *ez = 0.0; return; }
    }
    *ex = cx/nrm; *ey = cy/nrm; *ez = cz/nrm;
}

/* Clamp helper for exponent in Ritter law */
static inline double clamp_expo(double x){
    if(x >  RLMT_EXP_CLAMP) return  RLMT_EXP_CLAMP;
    if(x < -RLMT_EXP_CLAMP) return -RLMT_EXP_CLAMP;
    return x;
}

/* ------------------------------ CE profile I/O ------------------------------ */

/* Load (s, rho, cs) ASCII table. Returns 1 on success, 0 otherwise. */
static int ce_profile_load(const char* fname){
    if(!fname) return 0;
    FILE* f = fopen(fname, "r");
    if(!f){
        fprintf(stderr, "[rlmt] Cannot open CE profile file: %s\n", fname);
        return 0;
    }
    /* free old */
    free(ce_tab.s); free(ce_tab.rho); free(ce_tab.cs);
    ce_tab.n = 0; ce_tab.s = ce_tab.rho = ce_tab.cs = NULL;

    int cap = 1024;
    ce_tab.s   = (double*)malloc((size_t)cap*sizeof(double));
    ce_tab.rho = (double*)malloc((size_t)cap*sizeof(double));
    ce_tab.cs  = (double*)malloc((size_t)cap*sizeof(double));
    if(!ce_tab.s || !ce_tab.rho || !ce_tab.cs){
        fprintf(stderr, "[rlmt] Out of memory reading CE profile.\n");
        if(ce_tab.s)  free(ce_tab.s);
        if(ce_tab.rho)free(ce_tab.rho);
        if(ce_tab.cs) free(ce_tab.cs);
        ce_tab.s = ce_tab.rho = ce_tab.cs = NULL;
        fclose(f);
        return 0;
    }

    while(1){
        double x, r, c;
        int nread = fscanf(f, "%lf %lf %lf", &x, &r, &c);
        if(nread != 3) break;
        if(x <= 0.0 || r <= 0.0 || c <= 0.0) continue;  /* require positive for log‑interp */
        if(ce_tab.n == cap){
            cap *= 2;
            double* ns  = (double*)realloc(ce_tab.s,   (size_t)cap*sizeof(double));
            double* nr  = (double*)realloc(ce_tab.rho, (size_t)cap*sizeof(double));
            double* ncs = (double*)realloc(ce_tab.cs,  (size_t)cap*sizeof(double));
            if(!ns || !nr || !ncs){
                fprintf(stderr, "[rlmt] Out of memory expanding CE profile.\n");
                if(ns)  ce_tab.s   = ns;   /* keep valid pointers for free */
                if(nr)  ce_tab.rho = nr;
                if(ncs) ce_tab.cs  = ncs;
                free(ce_tab.s); free(ce_tab.rho); free(ce_tab.cs);
                ce_tab.s = ce_tab.rho = ce_tab.cs = NULL;
                fclose(f);
                return 0;
            }
            ce_tab.s = ns; ce_tab.rho = nr; ce_tab.cs = ncs;
        }
        ce_tab.s  [ce_tab.n] = x;
        ce_tab.rho[ce_tab.n] = r;
        ce_tab.cs [ce_tab.n] = c;
        ce_tab.n++;
    }
    fclose(f);

    if(ce_tab.n < 2){
        fprintf(stderr, "[rlmt] CE profile file has fewer than 2 valid rows; ignoring.\n");
        free(ce_tab.s); free(ce_tab.rho); free(ce_tab.cs);
        ce_tab.n = 0; ce_tab.s = ce_tab.rho = ce_tab.cs = NULL;
        return 0;
    }
    return 1;
}

/* Log‑log linear interpolation (positive‑definite). */
static inline void ce_interp(double s, double* rho_out, double* cs_out){
    if(!rho_out || !cs_out) return;
    if(ce_tab.n < 2){ *rho_out = 0.0; *cs_out = 0.0; return; }
    if(s <= ce_tab.s[0]){ *rho_out = ce_tab.rho[0]; *cs_out = ce_tab.cs[0]; return; }
    if(s >= ce_tab.s[ce_tab.n-1]){ *rho_out = ce_tab.rho[ce_tab.n-1]; *cs_out = ce_tab.cs[ce_tab.n-1]; return; }

    /* binary search */
    int i = 0, j = ce_tab.n - 1;
    while(j - i > 1){
        int m = (i + j) >> 1;
        if(s < ce_tab.s[m]) j = m; else i = m;
    }
    const double t = (s - ce_tab.s[i]) / (ce_tab.s[j] - ce_tab.s[i]);
    /* positive inputs guaranteed */
    const double lrho = log(ce_tab.rho[i])*(1.0 - t) + log(ce_tab.rho[j])*t;
    const double lcs  = log(ce_tab.cs [i])*(1.0 - t) + log(ce_tab.cs [j])*t;
    *rho_out = exp(lrho);
    *cs_out  = exp(lcs);
}

/* ------------------------------ CE drag prefactor ------------------------------ */

/* Ostriker low‑Mach series piece; analytic for M<1 else handled in I_prefactor. */
static double mach_piece_sub(const double M){
    if(M < 0.0) return 0.0;
    if(M < 0.02){
        const double m2 = M*M;
        return m2*M/3. + m2*m2*M/5.;       /* ⅓ M^3 + ⅕ M^5 */
    }
    /* For M very close to 1, this diverges; caller caps it. */
    return 0.5*log((1.+M)/(1.-M)) - M;     /* analytic for 0.02 ≤ M < 1 */
}

static double I_prefactor(const double M_in, const double xmin_in){
    /* Coulomb log; xmin stands in for r_min/r_max (dimensionless) */
    double xmin = xmin_in;
    if(!(xmin > 0.0)) xmin = 1e-4;
    if(xmin < 1e-12) xmin = 1e-12;
    if(xmin > 0.5)   xmin = 0.5;
    const double coul = log(1.0/xmin);

    double M = M_in;
    if(!isfinite(M)) M = 0.0;
    if(M < 0.0) M = 0.0;

    if(M > 1.0){
        /* Supersonic: I = ln Λ + 0.5 ln(1 - M^{-2}); keep numerically safe near M→1+ */
        const double M2 = M*M;
        double arg = 1.0 - 1.0/MAX2(M2, 1e-300);
        if(arg < 1e-12) arg = 1e-12;              /* avoid log(0) */
        return coul + 0.5*log(arg);
    } else {
        /* Subsonic: cap by Coulomb log to avoid the formal divergence as M→1- */
        /* Also avoid evaluating exactly at M=1 in mach_piece_sub. */
        const double Mcap = MIN2(M, 1.0 - 1e-12);
        const double Isub = mach_piece_sub(Mcap);
        return MIN2(coul, Isub);
    }
}

/* -------------------------- robust index param reader -------------------------- */
/* Works for the common case where indices are stored as double (Python), while
   still handling int-stored params in many builds. */
static int read_index_param_any(struct reb_simulation* sim,
                                struct rebx_extras* rx,
                                struct rebx_param* ap,
                                const char* name,
                                int* out_idx){
    const void* p = (const void*)rebx_get_param(rx, ap, name);
    if(!p) return 0;

    /* Heuristic: prefer interpreting as double (common in REBOUNDx/Python),
       but fall back to int when the double interpretation looks like denormal garbage. */
    double vd = 0.0;
    memcpy(&vd, p, sizeof(double));
    if(isfinite(vd)){
        /* If vd is a "normal" number (or exactly 0), treat as double. */
        if(vd == 0.0 || fabs(vd) > 1e-200){
            const long long k = llround(vd);
            if(fabs(vd - (double)k) < 1e-6 && k >= 0 && k < (long long)sim->N){
                *out_idx = (int)k;
                return 1;
            }
        }
    }

    /* Fallback: read as int (covers some C-registered int params). */
    int vi = 0;
    memcpy(&vi, p, sizeof(int));
    if(vi >= 0 && vi < sim->N){
        *out_idx = vi;
        return 1;
    }
    return 0;
}

/* ------------------------------ diagnostics helper ------------------------------ */
/* Galilean-invariant two-body orbital energy of the donor–accretor pair. */
static inline double two_body_orbital_energy(const double G,
                                             const double m1,
                                             const double m2,
                                             const double dx,
                                             const double dy,
                                             const double dz,
                                             const double dvx,
                                             const double dvy,
                                             const double dvz){
    const double r2 = dx*dx + dy*dy + dz*dz;
    if(!(r2 > 0.0) || !(m1 > 0.0) || !(m2 > 0.0) || !isfinite(r2)) return 0.0;
    const double r = sqrt(r2);
    const double v2 = dvx*dvx + dvy*dvy + dvz*dvz;
    const double mtot = m1 + m2;
    if(!(mtot > 0.0)) return 0.0;
    const double mu = (m1*m2) / mtot;
    return 0.5*mu*v2 - G*m1*m2/r;
}

/* ========================================================================= */
void rebx_roche_lobe_mass_transfer(struct reb_simulation* const sim,
                                  struct rebx_operator*     const op,
                                  const double                   dt_req)
{
    if(!(isfinite(dt_req) && dt_req > 0.0)) return;

    struct rebx_extras* const rx = sim->extras;

    /* ---------------------- required indices ---------------------- */
    int donor_idx = -1, acc_idx = -1;
    if(!read_index_param_any(sim, rx, op->ap, "rlmt_donor",    &donor_idx) ||
       !read_index_param_any(sim, rx, op->ap, "rlmt_accretor", &acc_idx)){
        reb_simulation_error(sim,
            "[rlmt] Missing/invalid rlmt_donor/rlmt_accretor. "
            "Require 0 ≤ idx < N and donor≠accretor.");
        return;
    }
    if(donor_idx < 0 || acc_idx < 0 || donor_idx >= sim->N || acc_idx >= sim->N || donor_idx == acc_idx){
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "[rlmt] rlmt_donor/rlmt_accretor out of range or identical. N=%d, donor=%d, accretor=%d",
                 sim->N, donor_idx, acc_idx);
        reb_simulation_error(sim, buf);
        return;
    }

    struct reb_particle* d = &sim->particles[donor_idx];
    struct reb_particle* a = &sim->particles[acc_idx];

    if(!(d->m > 0.0 && a->m > 0.0)) return;  /* nothing to do if any zero/neg mass */

    /* --------------------------- operator parameters --------------------------- */
    const double* p_loss   = rebx_get_param(rx, op->ap, "rlmt_loss_fraction");
    double f_loss = p_loss ? *p_loss : 0.0;
    if(!isfinite(f_loss)) f_loss = 0.0;
    if(f_loss < 0.0) f_loss = 0.0;
    if(f_loss > 1.0) f_loss = 1.0;

    const double* p_jmode  = rebx_get_param(rx, op->ap, "jloss_mode");
    const double* p_jfac   = rebx_get_param(rx, op->ap, "jloss_factor");
    int     jloss_mode   = p_jmode ? (int) llround(*p_jmode) : 0;
    if(jloss_mode < 0 || jloss_mode > 3) jloss_mode = 0;
    double  jloss_factor = p_jfac  ? *p_jfac : 1.0;
    if(!isfinite(jloss_factor)) jloss_factor = 1.0;

    const double* p_skipCE = rebx_get_param(rx, op->ap, "rlmt_skip_in_CE");
    const int     skip_in_CE = p_skipCE ? (int) llround(*p_skipCE) : 1;

    const double* p_dmmax  = rebx_get_param(rx, op->ap, "rlmt_substep_max_dm");
    const double* p_drmax  = rebx_get_param(rx, op->ap, "rlmt_substep_max_dr");
    const double* p_nmin   = rebx_get_param(rx, op->ap, "rlmt_min_substeps");
    const double* p_cfl    = rebx_get_param(rx, op->ap, "ce_kick_cfl");
    const double* p_merge  = rebx_get_param(rx, op->ap, "merge_eps");
    const double* p_react  = rebx_get_param(rx, op->ap, "ce_reaction_on_donor");

    const double dm_max_frac = (p_dmmax && isfinite(*p_dmmax) && *p_dmmax>0.0) ? *p_dmmax : 1e-3;
    const double dr_max_frac = (p_drmax && isfinite(*p_drmax) && *p_drmax>0.0) ? *p_drmax : 5e-3;
    int          min_steps   = (p_nmin  && isfinite(*p_nmin )) ? (int) llround(*p_nmin) : 3;
    if(min_steps < 1) min_steps = 1;
    const double kick_cfl    = (p_cfl   && isfinite(*p_cfl  ) && *p_cfl>=0.0) ? *p_cfl : 1.0;
    const int    ce_react_on_donor = p_react ? (int) llround(*p_react) : 0;

    /* optional CE profile table (loaded once) */
    static int ce_loaded = 0; /* 0 = not tried, 1 = loaded OK, -1 = tried and won't retry */
#ifdef REBX_ENABLE_STRING_PARAMS
    if(ce_loaded == 0){
        const char* ce_file = rebx_get_param_str(rx, op->ap, "ce_profile_file");
        if(ce_file && ce_file[0] != '\0'){
            if(ce_profile_load(ce_file)){
                ce_loaded = 1;
            } else {
                fprintf(stderr, "[rlmt] CE table load failed; falling back to power‑law CE.\n");
                ce_loaded = -1;
            }
        } else {
            ce_loaded = -1;
        }
    }
#endif

    /* --------------------------- initial merge guard --------------------------- */
    const double dx0 = d->x - a->x;
    const double dy0 = d->y - a->y;
    const double dz0 = d->z - a->z;
    const double r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
    const double r0   = sqrt(r2_0);

    double merge_eps = p_merge ? *p_merge : 0.0;
    if(!(merge_eps > 0.0)){
        double eps_default = 0.0;
        if(d->r > 0.0 && a->r > 0.0) eps_default = 0.5 * MIN2(d->r, a->r);
        /* positive floor, e.g., 1e-6 of current separation if radii lack info */
        if(!(eps_default > 0.0)) eps_default = 1e-6 * MAX2(r0, 1.0);
        merge_eps = eps_default;
    }

    if(!(isfinite(r2_0)) || r2_0 <= merge_eps*merge_eps){
        /* Merge immediately; detach operator to avoid index drift in N>2 systems. */
        const double msum = d->m + a->m;
        if(msum > 0.0){
            d->x  = (d->m*d->x  + a->m*a->x ) / msum;
            d->y  = (d->m*d->y  + a->m*a->y ) / msum;
            d->z  = (d->m*d->z  + a->m*a->z ) / msum;
            d->vx = (d->m*d->vx + a->m*a->vx) / msum;
            d->vy = (d->m*d->vy + a->m*a->vy) / msum;
            d->vz = (d->m*d->vz + a->m*a->vz) / msum;
            d->m  = msum;
            d->r  = MAX2(d->r, a->r);
        } else {
            d->m = 0.0;
        }
        reb_simulation_remove_particle(sim, acc_idx, 1);
        rebx_remove_operator(rx, op);
        reb_simulation_move_to_com(sim);
        return;
    }

    /* --------------------------- sub‑stepping loop ---------------------------- */
    double t_left   = dt_req;
    int    steps    = 0;
    double dE_total = 0.0;  /* diagnostic (pair orbital energy change) */

    /* Cache SSE pointer once (optional) */
    struct rebx_operator* sse = rebx_get_operator(rx, "stellar_evolution_sse");

    while(t_left > 0.0 && (steps < min_steps || t_left > 1e-14*dt_req)){
        /* start with a piece that guarantees at least min_steps overall */
        double dt = t_left / (double)((min_steps - steps) > 0 ? (min_steps - steps) : 1);

        /* Separation & relative kinematics (current frame) */
        double dx = d->x - a->x;
        double dy = d->y - a->y;
        double dz = d->z - a->z;
        double r  = sqrt(dx*dx + dy*dy + dz*dz);
        if(!(r > 0.0) || !isfinite(r)) break;

        const double nx = dx / r, ny = dy / r, nz = dz / r;

        double vrelx = a->vx - d->vx;
        double vrely = a->vy - d->vy;
        double vrelz = a->vz - d->vz;
        double vrel  = sqrt(vrelx*vrelx + vrely*vrely + vrelz*vrelz);

        /* donor RLOF parameters (required) */
        const double* Hp_ptr    = rebx_get_param(rx, d->ap, "rlmt_Hp");
        const double* mdot0_ptr = rebx_get_param(rx, d->ap, "rlmt_mdot0");
        if(!Hp_ptr || !mdot0_ptr){
            reb_simulation_error(sim, "[rlmt] Donor needs rlmt_Hp and rlmt_mdot0.");
            rebx_remove_operator(rx, op);
            return;
        }
        const double Hp    = *Hp_ptr;
        const double mdot0 = *mdot0_ptr;
        if(!(Hp > 0.0 && mdot0 > 0.0) || !isfinite(Hp) || !isfinite(mdot0)){
            reb_simulation_error(sim, "[rlmt] rlmt_Hp and rlmt_mdot0 must be finite and positive.");
            rebx_remove_operator(rx, op);
            return;
        }

        /* Eggleton Roche radius (donor as primary) using instantaneous separation r. */
        const double q    = d->m / a->m;
        const double q13  = cbrt(q);
        const double q23  = q13*q13;
        const double RL   = r * (0.49*q23) / (0.6*q23 + log(1.0 + q13));

        /* Flag CE and RLOF status for stand-alone wind operators */
        const int in_CE   = (r < d->r);
        const int rlof_on = (d->r > RL);
        rebx_set_param_double(rx, &d->ap, "inside_CE",   in_CE   ? 1.0 : 0.0);
        rebx_set_param_double(rx, &a->ap, "inside_CE",   in_CE   ? 1.0 : 0.0);
        rebx_set_param_double(rx, &d->ap, "rlof_active", rlof_on ? 1.0 : 0.0);
        rebx_set_param_double(rx, &a->ap, "rlof_active", rlof_on ? 1.0 : 0.0);

        /* Ritter mass‑loss estimate for step limiting */
        double expo = clamp_expo( (d->r - RL) / Hp );
        const double mdot_est = -mdot0 * exp(expo);  /* <0 when overflowing */

        if(mdot_est != 0.0 && isfinite(mdot_est)){
            const double dt_lim = dm_max_frac * MAX2(d->m, 1e-30) / fabs(mdot_est);
            if(dt > dt_lim) dt = dt_lim;
        }
        /* safety dt based on relative motion (proxy; operator does not drift positions internally) */
        const double dt_vel = dr_max_frac * r / MAX2(vrel, 1e-15);
        if(dt > dt_vel) dt = dt_vel;

        if(dt > t_left) dt = t_left;
        if(!(dt > 0.0) || !isfinite(dt)) break;

        /* Energy before step: invariant pair orbital energy */
        const double Md0 = d->m, Ma0 = a->m;
        const double E_before = two_body_orbital_energy(sim->G, Md0, Ma0,
                                                        dx, dy, dz,
                                                        vrelx, vrely, vrelz);

        /* ===================================================================== */
        /* STEP 1 – Roche‑lobe overflow (skip if inside CE and skip_in_CE==1)    */
        /* ===================================================================== */
        if(!(skip_in_CE && r < d->r)){
            expo = clamp_expo( (d->r - RL) / Hp );
            const double mdot = -mdot0 * exp(expo);   /* <0 => donor losing mass */
            double dM   = mdot * dt;                  /* negative */
            if(!isfinite(dM)) dM = 0.0;

            /* Ensure we never cross to negative mass in a single sub-step.
               Leave a tiny positive floor to avoid division-by-zero later. */
            if(d->m + dM <= 0.0){
                dM = -(d->m - 1e-30);  /* new mass will be +1e-30 */
            }

            const double m_loss = -dM;                 /* >0: donor mass decrease */

            /* --------------------------------------------------------------- */
            /* User-controlled systemic channel                                */
            /* --------------------------------------------------------------- */
            const double m_wind_user = f_loss * m_loss;        /* user-requested systemic loss */
            const double m_acc_user  = m_loss - m_wind_user;   /* would-be net accretion absent Eddington */

            /* --------------------------------------------------------------- */
            /* Optional Eddington cap on net accretion                          */
            /* rlmt_mdot_edd may be set on the accretor particle or on op->ap.  */
            /* Any excess is isotropically re-emitted from the accretor.        */
            /* --------------------------------------------------------------- */
            double mdot_edd = 0.0;
            {
                const double* mdotedd_ptr = rebx_get_param(rx, a->ap, "rlmt_mdot_edd");
                if(!mdotedd_ptr){
                    mdotedd_ptr = rebx_get_param(rx, op->ap, "rlmt_mdot_edd");
                }
                mdot_edd = mdotedd_ptr ? *mdotedd_ptr : 0.0;
                if(!(isfinite(mdot_edd) && mdot_edd > 0.0)) mdot_edd = 0.0;
            }

            double m_acc = m_acc_user;         /* actual retained (Eddington-limited) net accretion */
            if(mdot_edd > 0.0){
                const double m_acc_max = mdot_edd * dt;
                if(isfinite(m_acc_max) && m_acc_max >= 0.0){
                    if(m_acc > m_acc_max) m_acc = m_acc_max;
                }
            }
            if(m_acc < 0.0) m_acc = 0.0;
            if(m_acc > m_acc_user) m_acc = m_acc_user;

            const double m_rej = m_acc_user - m_acc;  /* rejected (super-Eddington) mass, re-emitted from accretor */

            /* For transfer bookkeeping:
               - If jloss_mode==1, we follow true isotropic re-emission: transfer FULL m_loss to accretor first.
               - Otherwise, we transfer the non-user-wind portion m_acc_user to the accretor, and then re-emit m_rej.
               This ensures the Eddington-rejected mass is lost from the accretor (standard assumption). */
            const double m_trans = (jloss_mode == 1) ? m_loss : m_acc_user;

            /* Precompute wind velocity prescription for the user systemic wind
               (m_wind_user > 0 and jloss_mode != 1).  For jloss_mode==1 the user wind
               is implemented as accretor re-emission, and for Eddington rejection we
               always use accretor isotropic re-emission (handled later). */
            double vx_loss = 0.0, vy_loss = 0.0, vz_loss = 0.0;
            double vx_emit = 0.0, vy_emit = 0.0, vz_emit = 0.0;

            if(m_wind_user > 0.0 && jloss_mode != 1){
                if(jloss_mode == 2){
                    /* COM-loss: use the binary COM velocity at the start of the substep. */
                    const double Mtot0 = Md0 + Ma0;
                    if(Mtot0 > 0.0){
                        vx_loss = (Md0*d->vx + Ma0*a->vx) / Mtot0;
                        vy_loss = (Md0*d->vy + Ma0*a->vy) / Mtot0;
                        vz_loss = (Md0*d->vz + Ma0*a->vz) / Mtot0;
                    } else {
                        vx_loss = vy_loss = vz_loss = 0.0;
                    }
                } else if(jloss_mode == 3){
                    /* target-j: add tangential component so that specific j about donor is set */
                    double exu, eyu, ezu;
                    unit_perp_to(nx, ny, nz, &exu, &eyu, &ezu);

                    /* h = |r × v_rel|; J/M = (μ h)/Mtot */
                    double Lx, Ly, Lz;
                    cross3(dx, dy, dz, vrelx, vrely, vrelz, &Lx, &Ly, &Lz);
                    const double hmag  = sqrt(Lx*Lx + Ly*Ly + Lz*Lz) + 1e-99;
                    const double Mtot0 = Md0 + Ma0;
                    const double mu0   = (Md0*Ma0) / MAX2(Mtot0, 1e-99);
                    const double j_orb = (mu0 * hmag) / MAX2(Mtot0, 1e-99);     /* J/M */
                    const double j_target = jloss_factor * j_orb;
                    const double fac = j_target / r;  /* speed to achieve j_target at radius r */

                    vx_loss = d->vx + fac*exu;
                    vy_loss = d->vy + fac*eyu;
                    vz_loss = d->vz + fac*ezu;
                } else {
                    /* mode 0 donor wind: v_loss = v_d */
                    vx_loss = d->vx; vy_loss = d->vy; vz_loss = d->vz;
                }

                /* emission site's velocity used by the momentum bookkeeping scheme */
                if(jloss_mode == 2){
                    /* preserve legacy behaviour: v_emit = 0 (see paper discussion). */
                    vx_emit = vy_emit = vz_emit = 0.0;
                } else {
                    /* donor wind (mode 0/3): emission from donor */
                    vx_emit = d->vx; vy_emit = d->vy; vz_emit = d->vz;
                }
            }

            /* --- Mass updates (transfer stage) --- */
            const double Md1 = Md0 - m_loss;       /* donor mass after loss */
            double       Ma1 = Ma0 + m_trans;      /* accretor mass after receiving transferred mass */
            d->m = Md1;
            a->m = Ma1;

            /* --- Internal accretion: momentum-exact mixing --- */
            if(m_trans > 0.0 && Ma1 > 0.0){
                a->vx = (Ma0*a->vx + m_trans*d->vx) / Ma1;
                a->vy = (Ma0*a->vy + m_trans*d->vy) / Ma1;
                a->vz = (Ma0*a->vz + m_trans*d->vz) / Ma1;
            }

            /* --- External user wind (m_wind_user) for modes 0/2/3: momentum correction --- */
            if(m_wind_user > 0.0 && jloss_mode != 1){
                const double Mtot1 = Md1 + Ma1;
                if(Mtot1 > 0.0){
                    const double dVx = -(m_wind_user * (vx_loss - vx_emit)) / Mtot1;
                    const double dVy = -(m_wind_user * (vy_loss - vy_emit)) / Mtot1;
                    const double dVz = -(m_wind_user * (vz_loss - vz_emit)) / Mtot1;
                    d->vx += dVx; d->vy += dVy; d->vz += dVz;
                    a->vx += dVx; a->vy += dVy; a->vz += dVz;
                }
            }

            /* --- Conservative ΔL correction for transferred mass --- */
            if(m_trans > 0.0 && a->m > 0.0 && d->m > 0.0){
                /* Use separation vector as lever arm (translation-invariant), and
                   apply a pure internal torque with ΔP=0. */
                const double rrelx = a->x - d->x;
                const double rrely = a->y - d->y;
                const double rrelz = a->z - d->z;
                const double rrel2 = rrelx*rrelx + rrely*rrely + rrelz*rrelz;

                if(rrel2 > 0.0 && isfinite(rrel2)){
                    /* Work in pair CM velocity frame (invariant to uniform boosts). */
                    const double Mtot_now = d->m + a->m;
                    if(Mtot_now > 0.0){
                        const double vcmx = (d->m*d->vx + a->m*a->vx) / Mtot_now;
                        const double vcmy = (d->m*d->vy + a->m*a->vy) / Mtot_now;
                        const double vcmz = (d->m*d->vz + a->m*a->vz) / Mtot_now;

                        const double vdx = d->vx - vcmx;
                        const double vdy = d->vy - vcmy;
                        const double vdz = d->vz - vcmz;

                        /* ΔL_trans = m_trans * (r_rel × v_d) */
                        double dLx, dLy, dLz;
                        cross3(rrelx, rrely, rrelz, vdx, vdy, vdz, &dLx, &dLy, &dLz);
                        dLx *= m_trans; dLy *= m_trans; dLz *= m_trans;

                        if(isfinite(dLx) && isfinite(dLy) && isfinite(dLz)){
                            /* δv_a = -(ΔL_trans × r_rel)/(M_a |r_rel|^2)  =>  δL = -ΔL_trans */
                            const double tx = -(dLy*rrelz - dLz*rrely) / (a->m * rrel2);
                            const double ty = -(dLz*rrelx - dLx*rrelz) / (a->m * rrel2);
                            const double tz = -(dLx*rrely - dLy*rrelx) / (a->m * rrel2);

                            if(isfinite(tx) && isfinite(ty) && isfinite(tz)){
                                a->vx += tx; a->vy += ty; a->vz += tz;
                                /* Momentum conservation: M_a δv_a + M_d δv_d = 0 */
                                const double scale = a->m / d->m;
                                d->vx -= scale*tx; d->vy -= scale*ty; d->vz -= scale*tz;
                            }
                        }
                    }
                }
            }

            /* --- User wind angular-momentum removal: enforce ΔL_wind explicitly (modes 0/2/3) --- */
            if(m_wind_user > 0.0 && jloss_mode != 1 && a->m > 0.0 && d->m > 0.0){
                /* R_cm after updates (pair COM) */
                const double Mtot_now = d->m + a->m;
                const double Rcmx = (d->m*d->x + a->m*a->x) / Mtot_now;
                const double Rcmy = (d->m*d->y + a->m*a->y) / Mtot_now;
                const double Rcmz = (d->m*d->z + a->m*a->z) / Mtot_now;

                double rex, rey, rez;  /* emission point relative to COM */
                if(jloss_mode == 2){
                    /* COM-loss: treat r_emit = R_cm for the explicit wind-ΔL correction. */
                    rex = rey = rez = 0.0;
                } else {
                    /* donor wind / target-j: emission from donor */
                    rex = d->x - Rcmx; rey = d->y - Rcmy; rez = d->z - Rcmz;
                }

                /* ΔL_needed = m_wind_user * (r_emit × (v_loss - v_emit)) */
                const double dvx = vx_loss - vx_emit;
                const double dvy = vy_loss - vy_emit;
                const double dvz = vz_loss - vz_emit;

                double DLx, DLy, DLz;
                cross3(rex, rey, rez, dvx, dvy, dvz, &DLx, &DLy, &DLz);
                DLx *= m_wind_user; DLy *= m_wind_user; DLz *= m_wind_user;

                if(isfinite(DLx) && isfinite(DLy) && isfinite(DLz)){
                    /* Apply −ΔL_needed as a pure internal torque on the pair,
                       using the separation vector as the lever arm. */
                    const double rrelx = a->x - d->x;
                    const double rrely = a->y - d->y;
                    const double rrelz = a->z - d->z;
                    const double rrel2 = rrelx*rrelx + rrely*rrely + rrelz*rrelz;

                    if(rrel2 > 0.0 && isfinite(rrel2)){
                        /* δv_a = -(ΔL_needed × r_rel)/(M_a |r_rel|^2) */
                        const double tx = -(DLy*rrelz - DLz*rrely) / (a->m * rrel2);
                        const double ty = -(DLz*rrelx - DLx*rrelz) / (a->m * rrel2);
                        const double tz = -(DLx*rrely - DLy*rrelx) / (a->m * rrel2);

                        if(isfinite(tx) && isfinite(ty) && isfinite(tz)){
                            a->vx += tx; a->vy += ty; a->vz += tz;
                            const double scale = a->m / d->m;
                            d->vx -= scale*tx; d->vy -= scale*ty; d->vz -= scale*tz;
                        }
                    }
                }
            }

            /* --- Accretor-side isotropic re-emission ---
               - If jloss_mode==1: user wind is already modeled as isotropic re-emission;
                 we add the Eddington-rejected mass m_rej to that same accretor-side ejection.
               - Otherwise: only the Eddington-rejected mass is ejected from the accretor.
               No recoil is applied; momentum and orbital AM are removed implicitly by the
               accretor mass decrement at its phase-space point. */
            {
                const double m_eject_acc = (jloss_mode == 1) ? (m_wind_user + m_rej) : m_rej;
                if(m_eject_acc > 0.0){
                    a->m = a->m - m_eject_acc;  /* velocity unchanged (isotropic in accretor frame) */
                    if(a->m <= 0.0){
                        reb_simulation_error(sim, "[rlmt] Accretor mass became non-positive during isotropic re-emission (Eddington cap / re-emission).");
                        rebx_remove_operator(rx, op);
                        return;
                    }
                }
            }

            /* --- optional stellar evolution update --- */
            if(sse && sse->step_function){
                sse->step_function(sim, sse, 0.0);
            }
        } /* end RLOF */

        /* ===================================================================== */
        /* STEP 2 – Common‑Envelope dynamical friction (optional)                */
        /* ===================================================================== */
        {
            /* Recompute relative velocity after any RLOF kicks (positions unchanged). */
            const double dx_ce = d->x - a->x;
            const double dy_ce = d->y - a->y;
            const double dz_ce = d->z - a->z;
            const double r_ce  = sqrt(dx_ce*dx_ce + dy_ce*dy_ce + dz_ce*dz_ce);
            const double vrelx_ce = a->vx - d->vx;
            const double vrely_ce = a->vy - d->vy;
            const double vrelz_ce = a->vz - d->vz;
            const double vrel_ce  = sqrt(vrelx_ce*vrelx_ce + vrely_ce*vrely_ce + vrelz_ce*vrelz_ce);

            const double* rho0_ptr = rebx_get_param(rx, op->ap, "ce_rho0");
            const double* cs0_ptr  = rebx_get_param(rx, op->ap, "ce_cs");
            if(r_ce < d->r && ( (rho0_ptr && cs0_ptr) || (ce_tab.n >= 2) )){
                const double* arho_ptr = rebx_get_param(rx, op->ap, "ce_alpha_rho");
                const double* acs_ptr  = rebx_get_param(rx, op->ap, "ce_alpha_cs");
                const double* xmin_ptr = rebx_get_param(rx, op->ap, "ce_xmin");
                const double* Qd_ptr   = rebx_get_param(rx, op->ap, "ce_Qd");

                const double s = r_ce / MAX2(d->r, 1e-30);
                double rho, cs;

                if(ce_tab.n >= 2){
                    ce_interp(s, &rho, &cs);
                } else {
                    const double rho0 = *rho0_ptr;
                    const double cs0  = *cs0_ptr;
                    const double arho = arho_ptr ? *arho_ptr : 0.0;
                    const double acs  = acs_ptr  ? *acs_ptr  : 0.0;
                    rho = rho0 * pow(s, arho);
                    cs  = cs0  * pow(s, acs);
                }

                if(rho > 0.0 && cs > 0.0 && isfinite(rho) && isfinite(cs) && a->m > 0.0){
                    const double xmin = xmin_ptr ? *xmin_ptr : 1e-4;
                    const double vrel_eff = MAX2(vrel_ce, 1e-3*cs);
                    const double I = I_prefactor(vrel_eff / cs, xmin);

                    /* Ostriker drag on accretor */
                    const double fc = 4.0 * M_PI * sim->G*sim->G * a->m * rho
                                      / (vrel_eff*vrel_eff*vrel_eff) * I;

                    double dvx = -fc * vrelx_ce * dt;
                    double dvy = -fc * vrely_ce * dt;
                    double dvz = -fc * vrelz_ce * dt;

                    /* Optional geometric term */
                    if(Qd_ptr && *Qd_ptr > 0.0 && a->r > 0.0){
                        const double fc_geom = M_PI * rho * a->r * a->r * vrel_ce / a->m * (*Qd_ptr);
                        dvx += -fc_geom * vrelx_ce * dt;
                        dvy += -fc_geom * vrely_ce * dt;
                        dvz += -fc_geom * vrelz_ce * dt;
                    }

                    /* CFL‑like limiter */
                    const double dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
                    const double cap = kick_cfl * cs;
                    if(dv > cap && dv > 0.0){
                        const double f = cap / dv;
                        dvx *= f; dvy *= f; dvz *= f;
                    }

                    /* Apply to accretor; optional reaction on donor */
                    a->vx += dvx; a->vy += dvy; a->vz += dvz;
                    if(ce_react_on_donor && d->m > 0.0){
                        const double scale = a->m / d->m;
                        d->vx -= scale*dvx; d->vy -= scale*dvy; d->vz -= scale*dvz;
                    }
                }
            }
        }

        /* ===================================================================== */
        /* STEP 3 – Secondary merge guard                                        */
        /* ===================================================================== */
        {
            const double mdx = d->x - a->x;
            const double mdy = d->y - a->y;
            const double mdz = d->z - a->z;
            if(mdx*mdx + mdy*mdy + mdz*mdz <= merge_eps*merge_eps){
                const double msum = d->m + a->m;
                if(msum > 0.0){
                    d->x  = (d->m*d->x  + a->m*a->x ) / msum;
                    d->y  = (d->m*d->y  + a->m*a->y ) / msum;
                    d->z  = (d->m*d->z  + a->m*a->z ) / msum;
                    d->vx = (d->m*d->vx + a->m*a->vx) / msum;
                    d->vy = (d->m*d->vy + a->m*a->vy) / msum;
                    d->vz = (d->m*d->vz + a->m*a->vz) / msum;
                    d->m  = msum;
                    d->r  = MAX2(d->r, a->r);
                } else {
                    d->m = 0.0;
                }
                reb_simulation_remove_particle(sim, acc_idx, 1);
                rebx_remove_operator(rx, op);
                reb_simulation_move_to_com(sim);
                return;
            }
        }

        /* ===================================================================== */
        /* STEP 4 – Do NOT globally purge m<=0 particles                         */
        /* ===================================================================== */
        /* The original implementation removed *all* particles with m<=0, which
           unintentionally deletes legitimate REBOUND test particles (massless tracers).
           Here we only detach on donor/accretor invalidation, and otherwise leave the
           simulation's particle set unchanged. */
        if(!(d->m > 0.0) || !(a->m > 0.0)){
            rebx_remove_operator(rx, op);
            reb_simulation_move_to_com(sim);
            return;
        }

        /* Energy after step (diagnostic only; invariant pair orbital energy) */
        {
            const double ddx = d->x - a->x;
            const double ddy = d->y - a->y;
            const double ddz = d->z - a->z;
            const double dvx = a->vx - d->vx;
            const double dvy = a->vy - d->vy;
            const double dvz = a->vz - d->vz;
            const double E_after = two_body_orbital_energy(sim->G, d->m, a->m,
                                                           ddx, ddy, ddz,
                                                           dvx, dvy, dvz);
            dE_total += (E_after - E_before);
        }

        t_left -= dt;
        steps++;

        /* Paper-consistent behaviour: recenter after each substep. */
        reb_simulation_move_to_com(sim);

        /* Pointers remain valid because move_to_com does not reorder particles. */
    }

    /* store diagnostic */
    rebx_set_param_double(rx, &op->ap, "rlmt_last_dE", dE_total);

    /* recentre once more for safety */
    reb_simulation_move_to_com(sim);
}
