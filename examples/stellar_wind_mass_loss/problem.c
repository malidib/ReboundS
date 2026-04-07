/**
 * Stellar Wind Mass Loss Example
 *
 * Demonstrates the stellar_wind_mass_loss operator which applies
 * continuous mass loss to a star according to the Reimers law.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

void heartbeat(struct reb_simulation* sim);
double tmax = 1e5;

int main(int argc, char* argv[]){
    struct reb_simulation* sim = reb_simulation_create();
    sim->G = 4*M_PI*M_PI; // AU, yr, Msun
    sim->heartbeat = heartbeat;

    struct reb_particle star = {0};
    star.m = 1.;
    star.r = 1.0; // Rsun
    reb_simulation_add(sim, star);
    reb_simulation_move_to_com(sim);

    struct rebx_extras* rebx = rebx_attach(sim);
    struct rebx_operator* sw = rebx_load_operator(rebx, "stellar_wind_mass_loss");
    rebx_add_operator(rebx, sw);

    // set stellar parameters
    rebx_set_param_double(rebx, &sim->particles[0].ap, "swml_eta", 0.5);
    rebx_set_param_double(rebx, &sim->particles[0].ap, "swml_L", 1.0); // Lsun

    reb_simulation_integrate(sim, tmax);
    rebx_free(rebx);
    reb_simulation_free(sim);
}

void heartbeat(struct reb_simulation* sim){
    if (reb_simulation_output_check(sim, tmax/10.)){
        printf("t=%e M=%f\n", sim->t, sim->particles[0].m);
    }
}
