/**
 * Eddington Winds Example
 *
 * Demonstrates the eddington_winds operator which removes mass
 * when a star's luminosity exceeds the Eddington limit.
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
    reb_simulation_add(sim, star);
    reb_simulation_move_to_com(sim);

    struct rebx_extras* rebx = rebx_attach(sim);
    struct rebx_operator* op = rebx_load_operator(rebx, "eddington_winds");
    rebx_add_operator(rebx, op);

    // Set luminosity well above Eddington to drive mass loss
    rebx_set_param_double(rebx, &sim->particles[0].ap, "eddw_L", 1e5);

    reb_simulation_integrate(sim, tmax);
    rebx_free(rebx);
    reb_simulation_free(sim);
}

void heartbeat(struct reb_simulation* sim){
    if (reb_simulation_output_check(sim, tmax/10.)){
        printf("t=%e M=%f\n", sim->t, sim->particles[0].m);
    }
}

