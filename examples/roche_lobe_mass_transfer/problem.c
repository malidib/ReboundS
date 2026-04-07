/**
 * Roche-lobe mass transfer example
 *
 * Demonstrates the roche_lobe_mass_transfer operator which self-consistently
 * exchanges mass between binary stars when the donor overfills its Roche lobe.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

void heartbeat(struct reb_simulation* sim);

int main(int argc, char* argv[]){
    struct reb_simulation* sim = reb_simulation_create();
    sim->G = 4*M_PI*M_PI; // AU^3 / (Msun yr^2)
    sim->dt = 1e-3; // years
    sim->heartbeat = heartbeat;

    struct reb_particle star1 = {0};
    star1.m = 1.0; // accretor
    reb_simulation_add(sim, star1);

    struct reb_particle star2 = {0};
    star2.m = 0.8; // donor
    star2.r = 0.01; // AU, roughly solar radius
    struct reb_orbit o = { .a = 0.05, .e = 0.0, .inc = 0., .Omega = 0., .omega = 0., .f = 0. };
    struct reb_particle p2 = reb_particle_from_orbit(sim->G, star1, star2.m, o.a, o.e, o.inc, o.Omega, o.omega, o.f);
    p2.r = star2.r;
    reb_simulation_add(sim, p2);

    reb_simulation_move_to_com(sim);

    struct rebx_extras* rebx = rebx_attach(sim);
    struct rebx_operator* rl = rebx_load_operator(rebx, "roche_lobe_mass_transfer");
    rebx_add_operator(rebx, rl);

    rebx_set_param_int(rebx, &rl->ap, "rlmt_donor", 1);
    rebx_set_param_int(rebx, &rl->ap, "rlmt_accretor", 0);
    rebx_set_param_double(rebx, &rl->ap, "rlmt_loss_fraction", 0.2);

    rebx_set_param_double(rebx, &sim->particles[1].ap, "rlmt_Hp", 0.0005); // pressure scale height
    rebx_set_param_double(rebx, &sim->particles[1].ap, "rlmt_mdot0", 1e-5); // Msun/yr

    // Common-envelope drag parameters
    rebx_set_param_double(rebx, &rl->ap, "ce_rho0", 1e-6);
    rebx_set_param_double(rebx, &rl->ap, "ce_alpha_rho", -2.0);
    rebx_set_param_double(rebx, &rl->ap, "ce_cs", 30.0);
    rebx_set_param_double(rebx, &rl->ap, "ce_alpha_cs", 0.0);
    rebx_set_param_double(rebx, &rl->ap, "ce_xmin", 0.01);
    rebx_set_param_double(rebx, &rl->ap, "ce_Qd", 1.0);

    double tmax = 1.0; // years
    reb_simulation_integrate(sim, tmax);

    rebx_free(rebx);
    reb_simulation_free(sim);
}

void heartbeat(struct reb_simulation* sim){
    if(reb_simulation_output_check(sim, 0.05)){
        printf("t=%.3f yr m1=%.6f m2=%.6f\n", sim->t, sim->particles[0].m, sim->particles[1].m);
    }
}
