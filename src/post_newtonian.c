/**
 * @file    post_newtonian.c
 * @brief   Post-Newtonian relativistic corrections: 2PN (PM+SS), 2.5PN (RR),
 *          with optional pre-pass particle merging.
 *
 * Implements the harmonic-coordinate point-mass equations of motion from
 * Kidder (1995, Phys. Rev. D 52, 821) for every massive pair, *omitting 1PN*.
 *
 *  – 2 PN   : conservative point-mass (PM) + spin–spin (SS) corrections
 *  – 2.5 PN : gravitational-wave radiation reaction (RR)
 *
 * Effect parameters
 * -----------------
 * c              (double, required)  – speed of light in simulation units
 * pn_2PN         (double,bool, opt.) – include 2 PN terms (PM + SS)   (default: 1)
 * pn_25PN        (double,bool, opt.) – include 2.5 PN terms           (default: 1)
 * pn_merge_dist  (double, opt.)      – if > 0, merge any pair with r <= pn_merge_dist
 *                                       (in simulation length units). If 0.0 or unset,
 *                                       only merge if distance <= 0 (exact coincidence).
 *
 * Particle parameters
 * -------------------
 * pn_spin (reb_vec3d, optional) – spin angular momentum vector (mass·L^2/T)
 *
 * Notes
 * -----
 *  • Spins must be supplied as *physical* angular momenta. For dimensionless
 *    spins χ_i, set S_i = χ_i * (G m_i^2 / c) in the simulation's unit system.
 *  • This effect updates *accelerations only*; spin precession ODEs are not
 *    included here.
 *  • Merging happens *before* PN accelerations are computed on each force call.
 *    After any merge, integrator caches are reset if the host REBOUND version
 *    provides an integrator-reset API (see pn_integrator_reset_if_available()).
 */

#include <math.h>
#include <stdio.h>
#include "rebound.h"
#include "reboundx.h"

typedef struct reb_vec3d reb_vec3d;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------- */
/* Compatibility: integrator-reset APIs                                      */
/*                                                                           */
/* Some REBOUND builds (older or mismatched at runtime) do not export         */
/* reb_simulation_integrator_reset(). If REBOUNDx is built with a hard        */
/* reference to a missing symbol, Python import fails with "undefined symbol".*/
/*                                                                           */
/* We therefore declare these symbols as WEAK and call them only if present. */
/* This keeps the shared library loadable across REBOUND versions.           */
/* ------------------------------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
extern void reb_simulation_integrator_reset(struct reb_simulation* const sim) __attribute__((weak));
extern void reb_simulation_reset_integrator(struct reb_simulation* const sim) __attribute__((weak));
#endif

static inline void pn_integrator_reset_if_available(struct reb_simulation* const sim){
#if defined(__GNUC__) || defined(__clang__)
    if (reb_simulation_integrator_reset){
        reb_simulation_integrator_reset(sim);
        return;
    }
    if (reb_simulation_reset_integrator){
        reb_simulation_reset_integrator(sim);
        return;
    }
    /* If neither symbol exists, do nothing. The merge still occurred; whether
       this is safe depends on the integrator and REBOUND version (some versions
       reset internally on remove, others may keep caches). */
#else
    /* Non-GNU/Clang builds: we cannot safely weak-link here without platform-
       specific machinery. In that case, do nothing to avoid hard-linking to a
       possibly missing symbol. */
    (void)sim;
#endif
}

/* ------------------------------------------------------------------------- */
/* Helper: merge particle j into i, conserving mass/momentum/volume and spin */
/* ------------------------------------------------------------------------- */
static inline int merge_two_particles(struct reb_simulation* const sim,
                                      const int i,
                                      const int j)
{
    struct reb_particle* const p  = sim->particles;
    struct reb_particle* const pi = &p[i];
    struct reb_particle* const pj = &p[j];

    const double mi = pi->m;
    const double mj = pj->m;
    const double m  = mi + mj;

    /* Center-of-mass position and velocity; conserve linear momentum exactly */
    if (m > 0.0){
        const double invm = 1.0/m;
        const double x  = (mi*pi->x  + mj*pj->x ) * invm;
        const double y  = (mi*pi->y  + mj*pj->y ) * invm;
        const double z  = (mi*pi->z  + mj*pj->z ) * invm;
        const double vx = (mi*pi->vx + mj*pj->vx) * invm;
        const double vy = (mi*pi->vy + mj*pj->vy) * invm;
        const double vz = (mi*pi->vz + mj*pj->vz) * invm;

        pi->x = x;  pi->y = y;  pi->z = z;
        pi->vx = vx; pi->vy = vy; pi->vz = vz;
        pi->m  = m;
    } else {
        /* If both masses are zero, keep i as-is and carry on. */
        pi->m = 0.0;
    }

    /* Conserve "volume" for radius: R_new^3 = R_i^3 + R_j^3 (if radii set) */
    if (pi->r > 0.0 || pj->r > 0.0){
        const double r3 = pi->r*pi->r*pi->r + pj->r*pj->r*pj->r;
        pi->r = (r3 > 0.0) ? cbrt(r3) : 0.0;
    }

    /* Sum PN spin vectors if present */
    struct rebx_extras* const rx = sim->extras;
    const reb_vec3d* Spi = rebx_get_param_vec(rx, pi->ap, "pn_spin");
    const reb_vec3d* Spj = rebx_get_param_vec(rx, pj->ap, "pn_spin");
    if (Spi || Spj){
        reb_vec3d Snew = (reb_vec3d){0.0, 0.0, 0.0};
        if (Spi){ Snew.x += Spi->x; Snew.y += Spi->y; Snew.z += Spi->z; }
        if (Spj){ Snew.x += Spj->x; Snew.y += Spj->y; Snew.z += Spj->z; }
        /* Store on survivor; create param if missing */
        rebx_set_param_vec3d(rx, &pi->ap, "pn_spin", Snew);
    }

    /* Remove j (keep array sorted) */
    reb_simulation_remove_particle(sim, j, 1);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Optional pre-pass: merge coincident or near-coincident pairs              */
/* If merge_dist > 0: merge when r^2 <= merge_dist^2                         */
/* Else:             merge only when r^2 <= 0 (exact coincidence)            */
/* Returns 1 if any merge occurred, 0 otherwise.                              */
/* ------------------------------------------------------------------------- */
static int merge_pairs_prepass(struct reb_simulation* const sim,
                               const double merge_dist)
{
    const int use_threshold = (merge_dist > 0.0);
    const double rcrit2 = use_threshold ? merge_dist*merge_dist : 0.0;

    int merged_any = 0;

    for (int i = 0; i < sim->N; i++){
        int j = i + 1;
        while (j < sim->N){
            const struct reb_particle* const pi = &sim->particles[i];
            const struct reb_particle* const pj = &sim->particles[j];

            const double dx = pi->x - pj->x;
            const double dy = pi->y - pj->y;
            const double dz = pi->z - pj->z;
            const double r2 = dx*dx + dy*dy + dz*dz;

            /* Merge if exactly coincident (r2 <= 0) or under threshold */
            if (r2 <= rcrit2){
                merge_two_particles(sim, i, j);
                merged_any = 1;
                /* do not increment j: new particle at index j now */
                continue;
            }
            j++;
        }
    }
    return merged_any;
}

/* ------------------------------------------------------------------------- */
/* Pair-wise relative acceleration builder                                   */
/* ------------------------------------------------------------------------- */
static inline void pn_add_pair(struct reb_simulation* const sim,
                               struct reb_particle*   const pi,
                               struct reb_particle*   const pj,
                               const double                 G,
                               const double                 c,
                               const int                    do2PN,
                               const int                    do25PN)
{
    /* PN defined for massive pairs only */
    if (!(pi->m > 0.0 && pj->m > 0.0)) return;

    /* Relative separation & velocity */
    const double dx  = pi->x  - pj->x;
    const double dy  = pi->y  - pj->y;
    const double dz  = pi->z  - pj->z;
    const double dvx = pi->vx - pj->vx;
    const double dvy = pi->vy - pj->vy;
    const double dvz = pi->vz - pj->vz;

    const double r2 = dx*dx + dy*dy + dz*dz;
    if (!(r2 > 0.0)) return;                           /* coincident or NaN */
    const double r     = sqrt(r2);
    const double invr  = 1.0/r;
    const double invr2 = 1.0/r2;

    const double v2   = dvx*dvx + dvy*dvy + dvz*dvz;
    const double rdot = (dx*dvx + dy*dvy + dz*dvz) * invr;

    /* Mass combinations */
    const double m   = pi->m + pj->m;
    const double mu  = (pi->m * pj->m) / m;           /* reduced mass */
    const double eta = mu / m;

    /* Unit separation vector n */
    const double nx = dx*invr;
    const double ny = dy*invr;
    const double nz = dz*invr;

    /* Build relative-acceleration contributions */
    double ax = 0.0, ay = 0.0, az = 0.0;

    /* --------------------------------------------------------------------- */
    /* 2 PN conservative: point‑mass (Kidder 1995 Eq. 2.2d)                  */
    /* --------------------------------------------------------------------- */
    if (do2PN){
        const double gmr    = G*m*invr;                 /* (speed)^2 */
        const double rdot2  = rdot*rdot;
        const double v4     = v2*v2;
        const double pref2  = (G*m*invr2)/(c*c*c*c);    /* (G m / r^2) / c^4 */

        double A2 = (3.0/4.0)*(12.0 + 29.0*eta)*gmr*gmr
                  + eta*(3.0 - 4.0*eta)*v4
                  + (15.0/8.0)*eta*(1.0 - 3.0*eta)*rdot2*rdot2
                  - (3.0/2.0)*eta*(3.0 - 4.0*eta)*v2*rdot2
                  - (2.0 + 25.0*eta + 2.0*eta*eta)*gmr*rdot2
                  - 0.5*eta*(13.0 - 4.0*eta)*gmr*v2;   /* missing term restored */

        const double Bv2 = -0.5*rdot*( eta*(15.0 + 4.0*eta)*v2
                                     - (4.0 + 41.0*eta + 8.0*eta*eta)*gmr
                                     - 3.0*eta*(3.0 + 2.0*eta)*rdot2 );

        ax += -pref2 * ( A2*nx + Bv2*dvx );
        ay += -pref2 * ( A2*ny + Bv2*dvy );
        az += -pref2 * ( A2*nz + Bv2*dvz );
    }

    /* --------------------------------------------------------------------- */
    /* 2 PN spin–spin (Kidder 1995 Eq. 2.2e), physical spins.                */
    /* Prefactor: 3 G/(μ c^2 r^4) times the standard SS bracket.             */
    /* --------------------------------------------------------------------- */
    if (do2PN){
        struct rebx_extras* const rx = sim->extras;
        const reb_vec3d* Spi = rebx_get_param_vec(rx, pi->ap, "pn_spin");
        const reb_vec3d* Spj = rebx_get_param_vec(rx, pj->ap, "pn_spin");
        if (Spi && Spj){
            const reb_vec3d Si = *Spi;
            const reb_vec3d Sj = *Spj;

            const double Sdot   = Si.x*Sj.x + Si.y*Sj.y + Si.z*Sj.z;
            const double nidSi  = nx*Si.x + ny*Si.y + nz*Si.z;
            const double nidSj  = nx*Sj.x + ny*Sj.y + nz*Sj.z;

            const double prefSS = (3.0*G)/(mu * c*c) * invr2 * invr2;   /* 3G/(μ c^2 r^4) */

            ax += -prefSS * ( (Sdot - 5.0*nidSi*nidSj)*nx + nidSj*Si.x + nidSi*Sj.x );
            ay += -prefSS * ( (Sdot - 5.0*nidSi*nidSj)*ny + nidSj*Si.y + nidSi*Sj.y );
            az += -prefSS * ( (Sdot - 5.0*nidSi*nidSj)*nz + nidSj*Si.z + nidSi*Sj.z );
        }
    }

    /* --------------------------------------------------------------------- */
    /* 2.5 PN radiation reaction (Kidder 1995 Eq. 2.2f)                      */
    /* --------------------------------------------------------------------- */
    if (do25PN){
        const double c2 = c*c;
        const double c4 = c2*c2;
        const double pref25 = (8.0/(5.0)) * (G*G) * m*m * eta / (c4*c) * invr2 * invr;   /* 8 G^2 m^2 η /(5 c^5 r^3) */

        const double Cn = rdot * ( 18.0*v2 + (2.0/3.0)*(G*m*invr) - 25.0*rdot*rdot );
        const double Dv =          6.0*v2 -  2.0*(G*m*invr)      - 15.0*rdot*rdot;

        ax += pref25 * ( Cn*nx - Dv*dvx );
        ay += pref25 * ( Cn*ny - Dv*dvy );
        az += pref25 * ( Cn*nz - Dv*dvz );
    }

    /* --------------------------------------------------------------------- */
    /* Map relative acceleration to body accelerations (action-reaction)     */
    /* --------------------------------------------------------------------- */
    const double fac_i = pj->m / m;
    const double fac_j = pi->m / m;

    pi->ax +=  fac_i * ax;  pj->ax -=  fac_j * ax;
    pi->ay +=  fac_i * ay;  pj->ay -=  fac_j * ay;
    pi->az +=  fac_i * az;  pj->az -=  fac_j * az;
}

/* ------------------------------------------------------------------------- */
/* Force-kernel wrapper called by REBOUNDx                                   */
/* ------------------------------------------------------------------------- */
static void rebx_calculate_post_newtonian(struct reb_simulation* const sim,
                                          struct reb_particle*   const particles, /* not used directly after prepass */
                                          const int                        N,     /* not used directly after prepass */
                                          const double                     c,
                                          const int                        do2PN,
                                          const int                        do25PN,
                                          const double                     merge_dist)
{
    (void)particles; /* silence potential warnings */
    (void)N;

    const double G = sim->G;
    if (!(isfinite(G) && isfinite(c) && c > 0.0)) return;

    /* ---- Pre-pass: merge pairs if coincident or within pn_merge_dist ---- */
    const int merged_any = merge_pairs_prepass(sim, merge_dist);

    /* If particles were removed/merged, integrator internal caches may be stale.
       Reset if the host REBOUND provides a reset routine (weak-linked). */
    if (merged_any){
        pn_integrator_reset_if_available(sim);
    }

    /* ---- PN accelerations on the updated system ---- */
    const int Nnow = sim->N;
    struct reb_particle* const pnow = sim->particles;

    for (int i = 0; i < Nnow; i++){
        for (int j = i+1; j < Nnow; j++){
            pn_add_pair(sim, &pnow[i], &pnow[j], G, c, do2PN, do25PN);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Public entry point for REBOUNDx                                           */
/* ------------------------------------------------------------------------- */
void rebx_post_newtonian(struct reb_simulation* const sim,
                         struct rebx_force*      const force,
                         struct reb_particle*    const particles,
                         const int                            N)
{
    struct rebx_extras* const rx = sim->extras;

    const double* c_ptr = rebx_get_param(rx, force->ap, "c");
    if (!(c_ptr && isfinite(*c_ptr) && *c_ptr > 0.0)){
        reb_simulation_error(sim, "post_newtonian: must supply a positive 'c' (speed of light).");
        return;
    }

    int do2PN = 1, do25PN = 1;
    double merge_dist = 0.000; /* default: off (exact coincidence only) */

    const double* d;

    d = rebx_get_param(rx, force->ap, "pn_2PN");
    if (d) do2PN = (*d != 0.0);

    d = rebx_get_param(rx, force->ap, "pn_25PN");
    if (d) do25PN = (*d != 0.0);

    /* Optional: user-set merge distance (simulation length units) */
    d = rebx_get_param(rx, force->ap, "pn_merge_dist");
    if (d && isfinite(*d) && *d >= 0.0) merge_dist = *d;

    rebx_calculate_post_newtonian(sim, particles, N, *c_ptr, do2PN, do25PN, merge_dist);
}
