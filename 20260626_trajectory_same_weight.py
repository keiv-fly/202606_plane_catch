import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
g = 9.81
m = 600.0          # aircraft mass, kg

H = 50.0           # pulley / crane top height above ground, m
h0 = 25.0          # aircraft height at catch, m
r0 = H - h0        # initial cable length, m

v0 = 20.0          # initial horizontal aircraft speed, m/s

# Spring contact zone
spring_start = 39.0     # spring contact starts at 39 m
r_max = 40.0            # hard mechanical limit, m

spring_k = 5000.0       # N/m, total spring stiffness
spring_c = 5000.0       # N*s/m, compression damping

# Smart brake targets
target_load_g_extending = 1.5

# Retraction target cycle:
# 1.0 g until r reaches 10 m
# then 0.5 g until r increases to 30 m
# then 1.0 g again until r reaches 10 m
target_retract_high_g = 1.0
target_retract_low_g = 0.5
retract_low_threshold = 10.0
retract_high_threshold = 30.0

brake_force_max = 20000.0

theta0 = 0.0
rdot0 = 0.0
thetadot0 = v0 / r0

t_max = 20.0
dt = 0.001


# -----------------------------
# Forces
# -----------------------------
def spring_force(r, rdot):
    """
    One-direction contact spring-damper.

    The spring can push, but it cannot pull.
    Positive force resists cable extension.
    """
    compression = r - spring_start

    if compression <= 0.0:
        return 0.0

    raw_force = spring_k * compression + spring_c * rdot

    return max(0.0, raw_force)


def radial_acceleration(y, resist_force):
    """
    Radial acceleration with equal-mass counterweight.

    resist_force:
        positive -> resists cable extension
        negative -> resists cable retraction
    """
    r, rdot, theta, thetadot = y

    rddot = 0.5 * (
        r * thetadot**2
        - g * (1.0 - np.cos(theta))
        - resist_force / m
    )

    return rddot


def load_factor_from_resist_force(y, resist_force):
    """
    Cable load factor T / mg for a given spring/brake force.
    """
    r, rdot, theta, thetadot = y

    rddot = radial_acceleration(y, resist_force)

    tension = m * (
        r * thetadot**2
        + g * np.cos(theta)
        - rddot
    )

    return tension / (m * g)


def smart_brake_force(y, target_load_g_retracting):
    """
    Smart variable brake.

    Extension:
        If rdot > 0, cable is extending.
        Brake tries to increase total cable tension up to 1.5 g.

    Retraction:
        If rdot < 0, cable is retracting.
        Brake tries to reduce total cable tension down to the current
        cyclic retraction target: either 1.0 g or 0.5 g.

    Sign convention:
        positive brake force resists extension
        negative brake force resists retraction
    """
    r, rdot, theta, thetadot = y

    if abs(rdot) < 1e-9:
        return 0.0

    Fs = spring_force(r, rdot)
    load_without_brake = load_factor_from_resist_force(y, Fs)

    if rdot > 0.0:
        target = target_load_g_extending

        if load_without_brake >= target:
            return 0.0

        required_brake = 2.0 * m * g * (target - load_without_brake)

        # Passive brake can only apply positive force during extension.
        return min(required_brake, brake_force_max)

    else:
        target = target_load_g_retracting

        if load_without_brake <= target:
            return 0.0

        required_brake = 2.0 * m * g * (target - load_without_brake)

        # Passive brake can only apply negative force during retraction.
        return max(required_brake, -brake_force_max)


# -----------------------------
# Equations of motion
# -----------------------------
def derivatives(y, target_load_g_retracting):
    r, rdot, theta, thetadot = y

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y, target_load_g_retracting)

    total_resist_force = Fs + Fb

    rddot = radial_acceleration(y, total_resist_force)

    thetaddot = -(
        g * np.sin(theta)
        + 2.0 * rdot * thetadot
    ) / r

    return np.array([rdot, rddot, thetadot, thetaddot])


def rk4_step(y, dt, target_load_g_retracting):
    k1 = derivatives(y, target_load_g_retracting)
    k2 = derivatives(y + 0.5 * dt * k1, target_load_g_retracting)
    k3 = derivatives(y + 0.5 * dt * k2, target_load_g_retracting)
    k4 = derivatives(y + dt * k3, target_load_g_retracting)

    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# -----------------------------
# Retraction target controller
# -----------------------------
def update_retraction_mode(r, mode):
    """
    Cyclic hysteresis controller.

    mode == "high":
        retract target is 1.0 g.
        Stay here until r <= 10 m.

    mode == "low":
        retract target is 0.5 g.
        Stay here until r >= 30 m.
    """
    if mode == "high" and r <= retract_low_threshold:
        return "low"

    if mode == "low" and r >= retract_high_threshold:
        return "high"

    return mode


def target_from_mode(mode):
    if mode == "high":
        return target_retract_high_g

    if mode == "low":
        return target_retract_low_g

    raise ValueError(f"Unknown retraction mode: {mode}")


# -----------------------------
# Simulation
# -----------------------------
y = np.array([r0, rdot0, theta0, thetadot0], dtype=float)

times = []
xs = []
hs = []
rs = []
rdots = []
thetas = []
load_factors = []
load_factors_no_brake = []
spring_forces = []
brake_forces = []
retraction_targets = []
retraction_modes = []

hit_hard_stop = False
hit_ground = False

# Start in the 1.0 g retracting-target mode.
retraction_mode = "high"

t = 0.0

while t <= t_max:
    r, rdot, theta, thetadot = y

    # Update cyclic retraction target based on current cable length.
    retraction_mode = update_retraction_mode(r, retraction_mode)
    target_load_g_retracting = target_from_mode(retraction_mode)

    x = r * np.sin(theta)
    h = H - r * np.cos(theta)

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y, target_load_g_retracting)

    load_no_brake = load_factor_from_resist_force(y, Fs)
    load_with_brake = load_factor_from_resist_force(y, Fs + Fb)

    times.append(t)
    xs.append(x)
    hs.append(h)
    rs.append(r)
    rdots.append(rdot)
    thetas.append(theta)
    load_factors.append(load_with_brake)
    load_factors_no_brake.append(load_no_brake)
    spring_forces.append(Fs)
    brake_forces.append(Fb)
    retraction_targets.append(target_load_g_retracting)
    retraction_modes.append(retraction_mode)

    if h <= 0.0:
        hit_ground = True
        break

    if r >= r_max:
        hit_hard_stop = True
        break

    y = rk4_step(y, dt, target_load_g_retracting)

    if y[0] <= 0.1:
        print("Simulation stopped: cable length became too small.")
        break

    t += dt


times = np.array(times)
xs = np.array(xs)
hs = np.array(hs)
rs = np.array(rs)
rdots = np.array(rdots)
thetas = np.array(thetas)
load_factors = np.array(load_factors)
load_factors_no_brake = np.array(load_factors_no_brake)
spring_forces = np.array(spring_forces)
brake_forces = np.array(brake_forces)
retraction_targets = np.array(retraction_targets)


# -----------------------------
# Reference paths
# -----------------------------
x_circle_initial = np.linspace(0, r0, 400)
h_circle_initial = H - np.sqrt(r0**2 - x_circle_initial**2)

x_circle_max = np.linspace(0, r_max, 400)
h_circle_max = H - np.sqrt(r_max**2 - x_circle_max**2)


# -----------------------------
# Plot path
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(xs, hs, label="Aircraft path")
plt.plot(x_circle_initial, h_circle_initial, "--", label="Fixed pendulum, initial length")
plt.plot(x_circle_max, h_circle_max, ":", label="40 m limit circle")

plt.plot([0, 0], [0, H], linewidth=3, label="Crane mast")
plt.plot([0, 15], [H, H], linewidth=3, label="15 m boom")

plt.axhline(0, linewidth=1)

plt.scatter([xs[0]], [hs[0]], s=60, label="Catch point")
plt.scatter([xs[-1]], [hs[-1]], s=60, label="End point")

plt.xlabel("Horizontal distance from pulley, m")
plt.ylabel("Height above ground, m")
plt.title("Aircraft catch path with cyclic retraction brake target")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot cable length
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, rs, label="Cable length")
plt.axhline(spring_start, linestyle="--", label="Spring contact starts at 39 m")
plt.axhline(r_max, linestyle="--", label="Hard limit at 40 m")
plt.axhline(retract_low_threshold, linestyle=":", label="Switch to 0.5 g at 10 m")
plt.axhline(retract_high_threshold, linestyle=":", label="Switch to 1.0 g at 30 m")

plt.xlabel("Time, s")
plt.ylabel("Cable length r, m")
plt.title("Cable length over time")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot retraction target
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, retraction_targets)

plt.xlabel("Time, s")
plt.ylabel("Retraction target, g")
plt.title("Cyclic retraction tension target")
plt.grid(True)
plt.show()


# -----------------------------
# Plot radial velocity
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, rdots)
plt.axhline(0, linewidth=1)

plt.xlabel("Time, s")
plt.ylabel("Radial speed rdot, m/s")
plt.title("Cable speed: positive = extending, negative = retracting")
plt.grid(True)
plt.show()


# -----------------------------
# Plot load factor
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, load_factors_no_brake, "--", label="Load without smart brake")
plt.plot(times, load_factors, label="Load with smart brake")
plt.plot(times, retraction_targets, ":", label="Retraction target")

plt.axhline(
    target_load_g_extending,
    linestyle="--",
    label=f"Extension target {target_load_g_extending:.1f} g"
)

plt.axhline(2.0, linestyle="--", label="2.0 g")
plt.axhline(3.0, linestyle="--", label="3.0 g")

plt.xlabel("Time, s")
plt.ylabel("Cable load factor, g")
plt.title("Cable load factor")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot spring force
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, spring_forces / 1000.0)
plt.axhline(0, linewidth=1)

plt.xlabel("Time, s")
plt.ylabel("Spring contact force, kN")
plt.title("One-way spring force")
plt.grid(True)
plt.show()


# -----------------------------
# Plot brake force
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, brake_forces / 1000.0)
plt.axhline(0, linewidth=1)

plt.xlabel("Time, s")
plt.ylabel("Signed brake force, kN")
plt.title("Smart brake force: positive resists extension, negative resists retraction")
plt.grid(True)
plt.show()


# -----------------------------
# Print summary
# -----------------------------
print(f"Aircraft mass: {m:.1f} kg")
print(f"Crane height: {H:.2f} m")
print(f"Starting aircraft height: {h0:.2f} m")
print(f"Initial cable length: {r0:.2f} m")
print(f"Initial speed: {v0:.2f} m/s")
print()
print(f"Spring contact starts at: {spring_start:.2f} m")
print(f"Hard cable limit: {r_max:.2f} m")
print(f"Spring stiffness k: {spring_k:.1f} N/m")
print(f"Spring damping c: {spring_c:.1f} N*s/m")
print()
print(f"Extension target load: {target_load_g_extending:.2f} g")
print(f"Retraction high target: {target_retract_high_g:.2f} g")
print(f"Retraction low target: {target_retract_low_g:.2f} g")
print(f"Switch to low target at r <= {retract_low_threshold:.2f} m")
print(f"Switch to high target at r >= {retract_high_threshold:.2f} m")
print(f"Max brake force allowed: {brake_force_max / 1000.0:.2f} kN")
print()
print(f"Simulation time: {times[-1]:.2f} s")
print(f"Max height: {hs.max():.2f} m")
print(f"Min height: {hs.min():.2f} m")
print(f"Max horizontal distance: {xs.max():.2f} m")
print(f"Min horizontal distance: {xs.min():.2f} m")
print(f"Max cable length: {rs.max():.2f} m")
print(f"Min cable length: {rs.min():.2f} m")
print(f"Max radial speed extending: {rdots.max():.2f} m/s")
print(f"Max radial speed retracting: {rdots.min():.2f} m/s")
print()
print(f"Max spring contact force: {spring_forces.max() / 1000.0:.2f} kN")
print(f"Max brake force extending: {brake_forces.max() / 1000.0:.2f} kN")
print(f"Max brake force retracting: {brake_forces.min() / 1000.0:.2f} kN")
print(f"Max absolute brake force: {np.abs(brake_forces).max() / 1000.0:.2f} kN")
print()
print(f"Max load without brake: {load_factors_no_brake.max():.2f} g")
print(f"Max load with brake: {load_factors.max():.2f} g")
print(f"Min load with brake: {load_factors.min():.2f} g")
print(f"Final height: {hs[-1]:.2f} m")
print(f"Final retraction mode: {retraction_mode}")

if hit_hard_stop:
    print()
    print("WARNING: The cable reached the hard 40 m limit.")
    print("This means the one-way spring/brake system did not stop the payout before the limit.")
    print("Increase spring_k, spring_c, brake_force_max, or allow more cable travel.")

if hit_ground:
    print()
    print("WARNING: The aircraft reached the ground in this simulation.")