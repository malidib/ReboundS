/**
 * Stellar Evolution SSE Example
 *
 * Demonstrates the simplified stellar_evolution_sse operator that updates
 * a star's radius and luminosity based on its mass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include "reboundx.h"

void heartbeat(struct reb_simulation* sim);
double tmax = 1e4;

int main(int argc, char* argv[]){
    struct reb_simulation* sim = reb_simulation_create();
    sim->G = 4*M_PI*M_PI;
    sim->heartbeat = heartbeat;

    struct reb_particle star = {0};
    star.m = 1.;
    star.r = 1.;
    reb_simulation_add(sim, star);
    reb_simulation_move_to_com(sim);

    struct rebx_extras* rebx = rebx_attach(sim);
    struct rebx_operator* sse = rebx_load_operator(rebx, "stellar_evolution_sse");
    rebx_add_operator(rebx, sse);


    for(int i=0;i<10;i++){
        reb_simulation_integrate(sim, (i+1)*tmax/10.);
        printf("t=%e R=%f L=%f\n", sim->t,
               sim->particles[0].r,
               *rebx_get_param(rebx, sim->particles[0].ap, "sse_L"));
    }

    rebx_free(rebx);
    reb_simulation_free(sim);
}

void heartbeat(struct reb_simulation* sim){
    // nothing
}
