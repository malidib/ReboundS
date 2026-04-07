/* ============================================================================
 * @file    roche_lobe_mass_transfer.c
 * @brief   Binary‑star sink operator with RLOF, systemic mass‑loss,
 *          common‑envelope drag, adaptive sub‑stepping,
 *          and extensive robustness guards           (2025‑07‑10 rev B).
 * ----------------------------------------------------------------------------
 *  New operator parameters (all optional, operator scope)
 *   – jloss_mode          (int)    : 0 donor‑wind (default, legacy)
 *                                    1 isotropic re‑emission (accretor wind)
 *                                    2 Jeans / orbital mean
 *                                    3 scale‑factor × j_orb
 *   – jloss_factor        (double) : scale factor for mode = 3 (default 1.0)
 *   – rlmt_skip_in_CE     (int)    : skip RLOF when r<R_donor (default 1)
 *   – rlmt_substep_max_dm (double) : max |dM|/M per internal sub‑step (1e‑3)
 *   – rlmt_substep_max_dr (double) : max |dr|/r per internal sub‑step (5e‑3)
 *   – rlmt_min_substeps   (int)    : always take ≥ N sub‑steps   (3)
 *   – ce_profile_file     (string) : ASCII table s  rho  cs  (overrides power‑law)
 *   – ce_kick_cfl         (double) : dv_max = cfl × c_s (default 1.0)
 *   – merge_eps           (double) : explicit merge radius, otherwise
 *                                    0.5 × min(stellar radii)
 * ----------------------------------------------------------------------------
 *  Back‑compatible: If you don’t set any of the new parameters the behaviour
 *  matches the original “rlmt” module.
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

/* ---------- compile‑time helpers ----------------------------------------- */
#ifndef RLMT_EXP_CLAMP
#define RLMT_EXP_CLAMP  80.0              /* prevents exp() overflow          */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

/* ---------- simple dynamic table for CE profiles ------------------------- */
struct ce_profile {
    int     n;
    double *s, *rho, *cs;
};
static struct ce_profile ce_tab = {0, NULL, NULL};

/* Load once from ASCII (s   rho   cs) */
static void ce_profile_load(const char* fname){
    FILE* f = fopen(fname, "r");
    if(!f){
        fprintf(stderr, "[rlmt] Cannot open CE profile %s\n", fname);
        return;
    }
    int cap = 1024;
    ce_tab.s   = malloc(cap*sizeof(double));
    ce_tab.rho = malloc(cap*sizeof(double));
    ce_tab.cs  = malloc(cap*sizeof(double));
    while(!feof(f)){
        double x, r, c;
        if(fscanf(f, "%lf%lf%lf", &x, &r, &c) != 3) break;
        if(ce_tab.n == cap){
            cap *= 2;
            ce_tab.s   = realloc(ce_tab.s , cap*sizeof(double));
            ce_tab.rho = realloc(ce_tab.rho, cap*sizeof(double));
            ce_tab.cs  = realloc(ce_tab.cs , cap*sizeof(double));
        }
        ce_tab.s  [ce_tab.n] = x;
        ce_tab.rho[ce_tab.n] = r;
        ce_tab.cs [ce_tab.n] = c;
        ce_tab.n++;
    }
    fclose(f);
}

/* Log‑log linear interpolation (positive‑definite) */
static inline void ce_interp(double s, double* rho, double* cs){
    if(!ce_tab.n){ *rho = 0; *cs = 0; return; }
    if(s <= ce_tab.s[0]){ *rho = ce_tab.rho[0]; *cs = ce_tab.cs[0]; return; }
    if(s >= ce_tab.s[ce_tab.n-1]){
        *rho = ce_tab.rho[ce_tab.n-1];
        *cs  = ce_tab.cs [ce_tab.n-1];
        return;
    }
    int i = 0, j = ce_tab.n - 1;
    while(j - i > 1){
        int m = (i + j) / 2;
        if(s < ce_tab.s[m]) j = m; else i = m;
    }
    double t = (s - ce_tab.s[i]) / (ce_tab.s[j] - ce_tab.s[i]);
    *rho = exp( log(ce_tab.rho[i])*(1-t) + log(ce_tab.rho[j])*t );
    *cs  = exp( log(ce_tab.cs [i])*(1-t) + log(ce_tab.cs [j])*t );
}

/* ---------- vector helpers ---------------------------------------------- */
static inline void cross3(const double ax, const double ay, const double az,
                          const double bx, const double by, const double bz,
                          double* rx, double* ry, double* rz){
    *rx = ay*bz - az*by;
    *ry = az*bx - ax*bz;
    *rz = ax*by - ay*bx;
}

/* ---------- Ostriker drag helper ---------------------------------------- */
static double mach_piece_sub(const double M){
    if(M < 0.02){
        const double m2 = M*M;
        return m2*M/3. + m2*m2*M/5.;       /* ⅓ M³ + ⅕ M⁵ */
    }
    return 0.5*log((1.+M)/(1.-M)) - M;     /* analytic M<1 */
}
static double I_prefactor(const double M, const double xmin){
    const double coul = log(1./xmin);
    if(M > 1.0) return coul;
    return MIN(coul, mach_piece_sub(M));
}

/* ========================================================================= */
void rebx_roche_lobe_mass_transfer(struct reb_simulation* const sim,
                                   struct rebx_operator*     const op,
                                   const double                     dt_req)
{
    struct rebx_extras* const rebx = sim->extras;

/* ---------- particle handles & basic operator params -------------------- */
    int* donor_idx_ptr    = rebx_get_param(rebx, op->ap, "rlmt_donor");
    int* accretor_idx_ptr = rebx_get_param(rebx, op->ap, "rlmt_accretor");
    if(!donor_idx_ptr || !accretor_idx_ptr){
        rebx_error(rebx, "Need rlmt_donor / rlmt_accretor."); return;
    }
    int donor_idx = *donor_idx_ptr;
    int acc_idx   = *accretor_idx_ptr;
    if(donor_idx >= sim->N || acc_idx >= sim->N || donor_idx < 0 || acc_idx < 0){
        rebx_error(rebx, "rlmt_donor / rlmt_accretor out of range."); return;
    }
    struct reb_particle* d = &sim->particles[donor_idx];
    struct reb_particle* a = &sim->particles[acc_idx];

/* ---------- fetch new tuning knobs -------------------------------------- */
    const int*    jmode_ptr   = rebx_get_param(rebx, op->ap,"jloss_mode");
    const double* jfac_ptr    = rebx_get_param(rebx, op->ap,"jloss_factor");
    const int*    skipCE_ptr  = rebx_get_param(rebx, op->ap,"rlmt_skip_in_CE");
    const double* dm_max_ptr  = rebx_get_param(rebx, op->ap,"rlmt_substep_max_dm");
    const double* dr_max_ptr  = rebx_get_param(rebx, op->ap,"rlmt_substep_max_dr");
    const int*    nmin_ptr    = rebx_get_param(rebx, op->ap,"rlmt_min_substeps");
    const char**  ce_file_ptr = rebx_get_param(rebx, op->ap,"ce_profile_file");
    const double* cfl_ptr     = rebx_get_param(rebx, op->ap,"ce_kick_cfl");
    const double* merge_ptr   = rebx_get_param(rebx, op->ap,"merge_eps");

    const int    jloss_mode   = jmode_ptr ? *jmode_ptr : 0;
    const double jloss_factor = jfac_ptr  ? *jfac_ptr  : 1.0;
    const int    skip_in_CE   = skipCE_ptr? *skipCE_ptr: 1;
    const double dm_max_frac  = dm_max_ptr? *dm_max_ptr: 1e-3;
    const double dr_max_frac  = dr_max_ptr? *dr_max_ptr: 5e-3;
    const int    min_steps    = nmin_ptr  ? *nmin_ptr  : 3;
    const double kick_cfl     = cfl_ptr   ? *cfl_ptr   : 1.0;

/* ---------- load optional CE table once --------------------------------- */
    static int ce_loaded = 0;
    if(!ce_loaded && ce_file_ptr && *ce_file_ptr){
        ce_profile_load(*ce_file_ptr);
        ce_loaded = 1;
    }

/* ---------- immediate merge guard (zero or tiny separation) ------------- */
    const double dx0 = d->x - a->x;
    const double dy0 = d->y - a->y;
    const double dz0 = d->z - a->z;
    double r2_0 = dx0*dx0 + dy0*dy0 + dz0*dz0;
    double merge_eps = merge_ptr ? *merge_ptr : 0.5 * MIN(d->r, a->r);

    if(!isfinite(r2_0) || r2_0 <= merge_eps*merge_eps){
        /* merge now, avoid NaNs later */
        double msum = d->m + a->m;
        if(msum <= 0.0){
            d->m = 0.0;
        } else {
            d->x  = (d->m*d->x  + a->m*a->x ) / msum;
            d->y  = (d->m*d->y  + a->m*a->y ) / msum;
            d->z  = (d->m*d->z  + a->m*a->z ) / msum;
            d->vx = (d->m*d->vx + a->m*a->vx) / msum;
            d->vy = (d->m*d->vy + a->m*a->vy) / msum;
            d->vz = (d->m*d->vz + a->m*a->vz) / msum;
            d->m  = msum;
            d->r  = MAX(d->r, a->r);
        }
        reb_simulation_remove_particle(sim, acc_idx, 1);
        /* detach operator if binary is gone */
        if(sim->N < 2) rebx_remove_operator(rebx, op);
        reb_simulation_move_to_com(sim);
        return;
    }

/* ---------- sub‑stepping loop ------------------------------------------- */
    double t_left = dt_req;
    int    steps  = 0;
    double dE_total = 0.0;       /* accumulated energy non-conservation */

    while(t_left > 0.0 && (steps < min_steps || t_left > 1e-14*dt_req)){
        double dt = t_left / (double)( (min_steps-steps) > 0 ? (min_steps-steps) : 1 );

        /* -------- Re‑compute separation & Roche geometry ---------------- */
        double dx = d->x - a->x;
        double dy = d->y - a->y;
        double dz = d->z - a->z;
        double r  = sqrt(dx*dx + dy*dy + dz*dz);

        /* safety dt from mass‑loss rate (Ritter) */
        const double *Hp_ptr    = rebx_get_param(rebx, d->ap, "rlmt_Hp");
        const double *mdot0_ptr = rebx_get_param(rebx, d->ap, "rlmt_mdot0");
        if(!Hp_ptr || !mdot0_ptr){
            rebx_error(rebx,"Need rlmt_Hp and rlmt_mdot0 on donor."); return;
        }
        double q    = d->m / a->m;
        double q13  = cbrt(q);
        double RL   = r * (0.49*q13*q13) / (0.6*q13*q13 + log(1.+q13));
        double expo = (d->r - RL) / (*Hp_ptr);
        expo = MIN(expo, RLMT_EXP_CLAMP);
        double mdot_est = -(*mdot0_ptr) * exp(expo);
        if(mdot_est != 0.0){
            double dt_lim = dm_max_frac * MAX(d->m, 1e-30) / fabs(mdot_est);
            dt = MIN(dt, dt_lim);
        }
        /* safety dt from CE drag */
        double vrelx = a->vx - d->vx;
        double vrely = a->vy - d->vy;
        double vrelz = a->vz - d->vz;
        double vrel  = sqrt(vrelx*vrelx + vrely*vrely + vrelz*vrelz);
        double dt_vel = dr_max_frac * r / MAX(vrel, 1e-15);
        dt = MIN(dt, dt_vel);

        if(dt > t_left) dt = t_left;   /* consume the rest on last loop */

/* =======================================================================
 *  STEP 1 – Roche‑lobe overflow
 * ===================================================================== */
        if(!(skip_in_CE && r < d->r)){
            /* loss fraction */
            const double *loss_ptr = rebx_get_param(rebx, op->ap,"rlmt_loss_fraction");
            double loss_frac = loss_ptr ? *loss_ptr : 0.0;
            loss_frac = MIN(MAX(loss_frac, 0.0), 1.0);

            double mdot0 = *mdot0_ptr;
            double Hp    = *Hp_ptr;
            expo = (d->r - RL) / Hp;  expo = MIN(expo, RLMT_EXP_CLAMP);
            double mdot = -mdot0 * exp(expo);      /* <0 => loss */
            double dM   = mdot * dt;
            if(d->m + dM <= 0.0) dM = -d->m;

            const double m_loss = -dM;             /* positive   */
            const double m_acc  =  m_loss * (1.0 - loss_frac);

/* ---- specific angular momentum of escaping gas ----------------------- */
            double vx_loss, vy_loss, vz_loss;
            switch(jloss_mode){
                case 1:   /* isotropic re‑emission (accretor velocity) */
                    vx_loss = a->vx; vy_loss = a->vy; vz_loss = a->vz; break;
                case 2: { /* Jeans – centre of mass */
                    double Mtot = d->m + a->m;
                    vx_loss = (d->m*d->vx + a->m*a->vx) / Mtot;
                    vy_loss = (d->m*d->vy + a->m*a->vy) / Mtot;
                    vz_loss = (d->m*d->vz + a->m*a->vz) / Mtot;
                    break; }
                case 3: { /* scale‑factor × orbital value */
                    double Lx,Ly,Lz; cross3(dx,dy,dz, vrelx,vrely,vrelz, &Lx,&Ly,&Lz);
                    double Lmag = sqrt(Lx*Lx + Ly*Ly + Lz*Lz) + 1e-99;
                    double j_orb = Lmag / (d->m * a->m / (d->m + a->m));
                    double j_target = jloss_factor * j_orb;

                    /* Perpendicular direction via right‑hand rule e⊥ ∝ (n×v)×n */
                    double nx = dx/r, ny = dy/r, nz = dz/r;
                    double lx,ly,lz; cross3(nx,ny,nz, vrelx,vrely,vrelz, &lx,&ly,&lz); /* n×v */
                    double ex,ey,ez; cross3(lx,ly,lz, nx,ny,nz, &ex,&ey,&ez);         /* (n×v)×n */
                    double norm = sqrt(ex*ex + ey*ey + ez*ez);
                    if(norm < 1e-12){
                        /* fallback if n and v are parallel */
                        cross3(nx,ny,nz, 0,0,1, &ex,&ey,&ez);
                        norm = sqrt(ex*ex + ey*ey + ez*ez);
                        if(norm < 1e-12){
                            cross3(nx,ny,nz, 1,0,0, &ex,&ey,&ez);
                            norm = sqrt(ex*ex + ey*ey + ez*ez);
                        }
                    }
                    double vxu = ex/norm, vyu = ey/norm, vzu = ez/norm;
                    double fac = (j_target / r);
                    vx_loss = d->vx + fac*vxu;
                    vy_loss = d->vy + fac*vyu;
                    vz_loss = d->vz + fac*vzu;
                    break; }
                default:  /* donor wind – legacy behaviour */
                    vx_loss = d->vx; vy_loss = d->vy; vz_loss = d->vz;
            }

/* ---- mass update ------------------------------------------------------ */
            double Md0 = d->m, Ma0 = a->m;
            double E_before = 0.5*Md0*(d->vx*d->vx + d->vy*d->vy + d->vz*d->vz)
                             + 0.5*Ma0*(a->vx*a->vx + a->vy*a->vy + a->vz*a->vz)
                             - sim->G*Md0*Ma0/r;
            d->m -= m_loss;
            a->m += m_acc;
            double Md1 = d->m, Ma1 = a->m;

/* ---- linear momentum -------------------------------------------------- */
            /* accreted gas arrives with donor velocity */
            double px_a = Ma0*a->vx + m_acc*d->vx;
            double py_a = Ma0*a->vy + m_acc*d->vy;
            double pz_a = Ma0*a->vz + m_acc*d->vz;
            /* lost gas leaves system */
            double px_cm = Md0*d->vx + Ma0*a->vx - m_loss*vx_loss;
            double py_cm = Md0*d->vy + Ma0*a->vy - m_loss*vy_loss;
            double pz_cm = Md0*d->vz + Ma0*a->vz - m_loss*vz_loss;

            a->vx = px_a / Ma1;
            a->vy = py_a / Ma1;
            a->vz = pz_a / Ma1;
            double Vcmx = px_cm / (Md1 + Ma1);
            double Vcmy = py_cm / (Md1 + Ma1);
            double Vcmz = pz_cm / (Md1 + Ma1);
            d->vx = Vcmx - (Ma1/Md1)*(a->vx - Vcmx);
            d->vy = Vcmy - (Ma1/Md1)*(a->vy - Vcmy);
            d->vz = Vcmz - (Ma1/Md1)*(a->vz - Vcmz);

/* ---- angular‑momentum correction (legacy) ----------------------------- */
            double Rcmx = (Md1*d->x + Ma1*a->x) / (Md1 + Ma1);
            double Rcmy = (Md1*d->y + Ma1*a->y) / (Md1 + Ma1);
            double Rcmz = (Md1*d->z + Ma1*a->z) / (Md1 + Ma1);
            double rx_d = d->x - Rcmx, ry_d = d->y - Rcmy, rz_d = d->z - Rcmz;
            double rx_a = a->x - Rcmx, ry_a = a->y - Rcmy, rz_a = a->z - Rcmz;

            double Lc_x,Lc_y,Lc_z; cross3(rx_d,ry_d,rz_d, d->vx,d->vy,d->vz, &Lc_x,&Lc_y,&Lc_z);
            Lc_x *= m_acc; Lc_y *= m_acc; Lc_z *= m_acc;

            double Li_x,Li_y,Li_z; cross3(rx_a,ry_a,rz_a, d->vx,d->vy,d->vz, &Li_x,&Li_y,&Li_z);
            Li_x *= m_acc; Li_y *= m_acc; Li_z *= m_acc;

            double dLx = Lc_x - Li_x, dLy = Lc_y - Li_y, dLz = Lc_z - Li_z;
            double r2_a = rx_a*rx_a + ry_a*ry_a + rz_a*rz_a;
            double dL2  = dLx*dLx + dLy*dLy + dLz*dLz;
            if(dL2 > 0.0 && r2_a > 0.0){
                double tmpx,tmpy,tmpz; cross3(dLx,dLy,dLz, rx_a,ry_a,rz_a, &tmpx,&tmpy,&tmpz);
                double inv = 1.0 / (Ma1 * r2_a);
                double dvx = tmpx*inv, dvy = tmpy*inv, dvz = tmpz*inv;
                a->vx += dvx; a->vy += dvy; a->vz += dvz;
                d->vx -= dvx*(Ma1/Md1); d->vy -= dvy*(Ma1/Md1); d->vz -= dvz*(Ma1/Md1);
            }

            /* energy diagnostic after mass transfer & L correction */
            double E_after = 0.5*Md1*(d->vx*d->vx + d->vy*d->vy + d->vz*d->vz)
                            + 0.5*Ma1*(a->vx*a->vx + a->vy*a->vy + a->vz*a->vz)
                            - sim->G*Md1*Ma1/r;
            dE_total += (E_after - E_before);

/* ---- donor radius mass‑radius relation -------------------------------- */
            const double* R_slope_ptr = rebx_get_param(rebx, d->ap,"rlmt_R_slope");
            const double* R_refM_ptr  = rebx_get_param(rebx, d->ap,"rlmt_R_ref_mass");
            const double* R_refR_ptr  = rebx_get_param(rebx, d->ap,"rlmt_R_ref_radius");
            if(R_slope_ptr && *R_slope_ptr != 0.0){
                double alpha = *R_slope_ptr;
                double Mref  = R_refM_ptr ? *R_refM_ptr : Md0;
                double Rref  = R_refR_ptr ? *R_refR_ptr : d->r;
                d->r = Rref * pow(d->m / Mref, alpha);
            }
        } /* -------- end RLOF ------------------------------------------- */

/* =======================================================================
 *  STEP 2 – Common‑envelope drag / dynamical friction
 * ===================================================================== */
        const double *rho0_ptr = rebx_get_param(rebx, op->ap,"ce_rho0");
        const double *arho_ptr = rebx_get_param(rebx, op->ap,"ce_alpha_rho");
        const double *cs0_ptr  = rebx_get_param(rebx, op->ap,"ce_cs");
        const double *acs_ptr  = rebx_get_param(rebx, op->ap,"ce_alpha_cs");
        const double *xmin_ptr = rebx_get_param(rebx, op->ap,"ce_xmin");
        const double *Qd_ptr   = rebx_get_param(rebx, op->ap,"ce_Qd");

        if(r < d->r && (rho0_ptr || ce_tab.n)){
            double s = r / MAX(d->r, 1e-30);
            double rho, cs;

            if(ce_tab.n){
                ce_interp(s, &rho, &cs);
            } else {
                double rho0 = *rho0_ptr;
                double arho = arho_ptr ? *arho_ptr : 0.0;
                double cs0  = *cs0_ptr;
                double acs  = acs_ptr ? *acs_ptr : 0.0;
                rho = rho0 * pow(s, arho);
                cs  = cs0  * pow(s, acs);
            }
            double xmin = xmin_ptr ? *xmin_ptr : 1e-4;
            double Qd   = Qd_ptr   ? *Qd_ptr   : 0.0;

            double vfloor = 1e-3 * cs;
            double vrel_eff = MAX(vrel, vfloor);
            double I = I_prefactor(vrel_eff / cs, xmin);
            double fc = 4.0 * M_PI * sim->G*sim->G * a->m * rho /
                        (vrel_eff*vrel_eff*vrel_eff) * I;
            if(Qd > 0.0 && a->r > 0.0)
                fc += M_PI * rho * a->r*a->r * vrel * Qd / a->m;

            double dvx = -fc * vrelx * dt;
            double dvy = -fc * vrely * dt;
            double dvz = -fc * vrelz * dt;
            double dv  = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
            double cap = kick_cfl * cs;
            if(dv > cap){
                double f = cap / dv;
                dvx *= f; dvy *= f; dvz *= f;
            }
            a->vx += dvx; a->vy += dvy; a->vz += dvz;
            if(d->m > 0.0){
                double f = a->m / d->m;
                d->vx -= dvx*f; d->vy -= dvy*f; d->vz -= dvz*f;
            }
        }

/* =======================================================================
 *  STEP 3 – Merge if separation < merge_eps (fallback)
 * ===================================================================== */
        dx = d->x - a->x; dy = d->y - a->y; dz = d->z - a->z;
        if(dx*dx + dy*dy + dz*dz <= merge_eps*merge_eps){
            double msum = d->m + a->m;
            if(msum <= 0.0){
                d->m = 0.0;
            } else {
                d->x  = (d->m*d->x  + a->m*a->x ) / msum;
                d->y  = (d->m*d->y  + a->m*a->y ) / msum;
                d->z  = (d->m*d->z  + a->m*a->z ) / msum;
                d->vx = (d->m*d->vx + a->m*a->vx) / msum;
                d->vy = (d->m*d->vy + a->m*a->vy) / msum;
                d->vz = (d->m*d->vz + a->m*a->vz) / msum;
                d->m  = msum;
                d->r  = MAX(d->r, a->r);
            }
            reb_simulation_remove_particle(sim, acc_idx, 1);
            if(sim->N < 2) rebx_remove_operator(rebx, op);
            reb_simulation_move_to_com(sim);
            return;
        }

/* =======================================================================
 *  STEP 4 – purge zero‑mass particles
 * ===================================================================== */
        for(int i = sim->N-1; i >= 0; --i){
            if(sim->particles[i].m <= 0.0){
                reb_simulation_remove_particle(sim, i, 1);
                if(i == donor_idx || i == acc_idx){
                    rebx_remove_operator(rebx, op);
                    reb_simulation_move_to_com(sim);
                    return;
                }
            }
        }

        t_left -= dt;
        steps++;
    } /* ------------------- end sub‑stepping loop ----------------------- */

    rebx_set_param_double(rebx, &op->ap, "rlmt_last_dE", dE_total);

    /* final recentre on COM */
    reb_simulation_move_to_com(sim);
}
