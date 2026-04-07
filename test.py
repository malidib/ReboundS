# roche_lobe_overflow_checks_fixed.py
import math
import rebound
import reboundx

def eggleton_RL_over_a(q: float) -> float:
    q13 = q**(1.0/3.0)
    q23 = q13*q13
    return (0.49*q23)/(0.6*q23 + math.log(1.0 + q13))

def is_close(a, b, rtol=1e-10, atol=1e-12):
    return abs(a-b) <= atol + rtol*max(abs(a), abs(b))

# ---- Build a simple 2-body setup ----
sim = rebound.Simulation()
sim.G = 0
sim.integrator = "ias15"

# Donor at x=0, accretor at x=a
a_sep = 10.0
m_d = 1.0
m_a = 1.0
sim.add(m=m_d, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=1.0)     # donor (index 0)
sim.add(m=m_a, x=a_sep, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, r=1.0)  # accretor (index 1)
sim.move_to_com()

# Give a tiny relative transverse velocity to make dt_vel finite & deterministic
sim.particles[1].vy = 1e-6
sim.move_to_com()

rx = reboundx.Extras(sim)
op = rx.add_operator(rx.load_operator("roche_lobe_mass_transfer"))


# Required operator params (use floats; indices are 0-based)
rlmt.params["rlmt_donor"]     = 0  # donor is particle 0
rlmt.params["rlmt_accretor"]  = 1  # accretor is particle 1
rlmt.params["rlmt_skip_in_CE"] = 1

# RLOF controls
rlmt.params["rlmt_loss_fraction"] = 0.0
rlmt.params["jloss_mode"] =0

# Substepping
rlmt.params["rlmt_substep_max_dm"] = 1e-3
rlmt.params["rlmt_substep_max_dr"] = 5e-3
rlmt.params["rlmt_min_substeps"]   = 3

# Donor particle parameters (set on the actual donor: index 0)
Hp    = 0.1      # pressure scale height (choose not-too-small to avoid extreme exp)
mdot0 = 1e-2     # reference mass-loss rate (>0)
sim.particles[0].params["rlmt_Hp"]    = float(Hp)
sim.particles[0].params["rlmt_mdot0"] = float(mdot0)

# Make the donor slightly overflow its Roche lobe
q = sim.particles[0].m / sim.particles[1].m
RL_over_a = eggleton_RL_over_a(q)
RL_abs = RL_over_a * a_sep
sim.particles[0].r = RL_abs * 1.03  # 3% overflow

# Sanity prints
print("N =", sim.N,
      "donor=", rlmt.params["rlmt_donor"],
      "accretor=", rlmt.params["rlmt_accretor"])
print("RL =", RL_abs, "R_d =", sim.particles[0].r)

p0, p1 = sim.particles[0], sim.particles[1]
M0_i, M1_i = p0.m, p1.m
Mtot_i = M0_i + M1_i
print("masses before:", M0_i, M1_i)

# One operator step (no orbital evolution needed for this check)
sim.step()

M0_f, M1_f = p0.m, p1.m
Mtot_f = M0_f + M1_f
print("masses after :", M0_f, M1_f)

# Checks
assert M0_f < M0_i, "Donor mass should decrease."
assert M1_f > M1_i, "Accretor mass should increase."
assert is_close(Mtot_f, Mtot_i, atol=1e-12), "Total mass must be conserved for f_loss=0."
assert is_close(p0.params.get("rlof_active", 0.0), 1.0), "rlof_active flag must be 1 on donor."
assert is_close(p1.params.get("rlof_active", 0.0), 1.0), "rlof_active flag must be 1 on accretor."
assert is_close(p0.params.get("inside_CE", 0.0), 0.0), "inside_CE must be 0 (not embedded)."
assert is_close(p1.params.get("inside_CE", 0.0), 0.0), "inside_CE must be 0 (not embedded)."

print("OK: RLOF mass transfer happened and flags/cons. mass are correct.")

