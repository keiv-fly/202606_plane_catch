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
target_load_g_extending = 1.5     # target tension while cable is extending
target_load_g_retracting = 1.0    # target tension while cable is retracting
target_decrease_start_time = 8.0
target_decrease_r = 30.0
target_load_g_extending_decrement = 0.1

brake_force_max = 20000.0         # N; increase if brake saturates

theta0 = 0.0
rdot0 = 0.0
thetadot0 = v0 / r0

t_max = 240.0
dt = 0.001


# -----------------------------
# Forces
# -----------------------------
def spring_force(r, rdot):
    """
    One-direction contact spring-damper.

    Physical model:
        - The plate touches the spring only when r > spring_start.
        - The spring can push back against extension.
        - The spring cannot pull.
        - If the plate moves away and force would become negative, contact is lost.

    Sign convention:
        positive force resists cable extension.
    """
    compression = r - spring_start

    if compression <= 0.0:
        return 0.0

    raw_force = spring_k * compression + spring_c * rdot

    # Contact force cannot be negative.
    return max(0.0, raw_force)


def radial_acceleration(y, resist_force):
    """
    Radial acceleration with equal-mass counterweight.

    State:
        y = [r, rdot, theta, thetadot]

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


def brake_targets(current_target_load_g_extending):
    return current_target_load_g_extending, target_load_g_retracting


def smart_brake_force(y, current_target_load_g_extending):
    """
    Smart variable brake.

    Extension:
        If rdot > 0, cable is extending.
        Brake tries to increase total cable tension up to the extension target.

    Retraction:
        If rdot < 0, cable is retracting.
        Brake tries to reduce total cable tension down to the retraction target.

    Sign convention:
        positive brake force resists extension
        negative brake force resists retraction
    """
    r, rdot, theta, thetadot = y

    if abs(rdot) < 1e-9:
        return 0.0

    Fs = spring_force(r, rdot)
    load_without_brake = load_factor_from_resist_force(y, Fs)
    target_extending, target_retracting = brake_targets(current_target_load_g_extending)

    if rdot > 0.0:
        target = target_extending

        if load_without_brake >= target:
            return 0.0

        required_brake = 2.0 * m * g * (target - load_without_brake)

        # Passive brake can only apply positive force during extension.
        return min(required_brake, brake_force_max)

    else:
        target = target_retracting

        if load_without_brake <= target:
            return 0.0

        required_brake = 2.0 * m * g * (target - load_without_brake)

        # Passive brake can only apply negative force during retraction.
        return max(required_brake, -brake_force_max)


# -----------------------------
# Equations of motion
# -----------------------------
def derivatives(y, current_target_load_g_extending):
    r, rdot, theta, thetadot = y

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y, current_target_load_g_extending)

    total_resist_force = Fs + Fb

    rddot = radial_acceleration(y, total_resist_force)

    thetaddot = -(
        g * np.sin(theta)
        + 2.0 * rdot * thetadot
    ) / r

    return np.array([rdot, rddot, thetadot, thetaddot])


def rk4_step(y, dt, current_target_load_g_extending):
    k1 = derivatives(y, current_target_load_g_extending)
    k2 = derivatives(y + 0.5 * dt * k1, current_target_load_g_extending)
    k3 = derivatives(y + 0.5 * dt * k2, current_target_load_g_extending)
    k4 = derivatives(y + dt * k3, current_target_load_g_extending)

    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


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
target_load_g_extending_values = []
spring_forces = []
brake_forces = []

hit_hard_stop = False
hit_ground = False
current_target_load_g_extending = target_load_g_extending
can_decrease_extending_target = True
waiting_for_swing = False
last_swing_direction = np.sign(thetadot0)

t = 0.0

while t <= t_max:
    r, rdot, theta, thetadot = y

    swing_direction = np.sign(thetadot) if abs(thetadot) > 1e-9 else 0.0

    if swing_direction != 0.0:
        if (
            waiting_for_swing
            and last_swing_direction != 0.0
            and swing_direction != last_swing_direction
        ):
            can_decrease_extending_target = True
            waiting_for_swing = False

        last_swing_direction = swing_direction

    if (
        t >= target_decrease_start_time
        and r < target_decrease_r
        and can_decrease_extending_target
    ):
        current_target_load_g_extending -= target_load_g_extending_decrement
        can_decrease_extending_target = False
        waiting_for_swing = True

    x = r * np.sin(theta)
    h = H - r * np.cos(theta)

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y, current_target_load_g_extending)

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
    target_load_g_extending_values.append(current_target_load_g_extending)
    spring_forces.append(Fs)
    brake_forces.append(Fb)

    if h <= 0.0:
        hit_ground = True
        break

    if r >= r_max:
        hit_hard_stop = True
        break

    y = rk4_step(y, dt, current_target_load_g_extending)

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
target_load_g_extending_values = np.array(target_load_g_extending_values)
spring_forces = np.array(spring_forces)
brake_forces = np.array(brake_forces)


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

# Crane mast and 15 m boom
plt.plot([0, 0], [0, H], linewidth=3, label="Crane mast")
plt.plot([0, 15], [H, H], linewidth=3, label="15 m boom")

plt.axhline(0, linewidth=1)

plt.scatter([xs[0]], [hs[0]], s=60, label="Catch point")
plt.scatter([xs[-1]], [hs[-1]], s=60, label="End point")

plt.xlabel("Horizontal distance from pulley, m")
plt.ylabel("Height above ground, m")
plt.title("Aircraft catch path with counterweight, one-way spring, and smart brake")
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

plt.xlabel("Time, s")
plt.ylabel("Cable length r, m")
plt.title("Cable length over time")
plt.grid(True)
plt.legend()
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
plt.plot(
    times,
    target_load_g_extending_values,
    ":",
    label="Extension target"
)

plt.axhline(
    target_load_g_retracting,
    linestyle="--",
    label=f"Retraction target {target_load_g_retracting:.1f} g"
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
print(f"Initial target load while extending: {target_load_g_extending:.2f} g")
print(f"Target load while retracting: {target_load_g_retracting:.2f} g")
print(
    "Extension target decrease trigger: "
    f"t >= {target_decrease_start_time:.2f} s and r < {target_decrease_r:.2f} m"
)
print(f"Extension target decrease step: {target_load_g_extending_decrement:.2f} g")
print(f"Final target load while extending: {target_load_g_extending_values[-1]:.2f} g")
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

if hit_hard_stop:
    print()
    print("WARNING: The cable reached the hard 40 m limit.")
    print("This means the one-way spring/brake system did not stop the payout before the limit.")
    print("Increase spring_k, spring_c, brake_force_max, or allow more cable travel.")

if hit_ground:
    print()
    print("WARNING: The aircraft reached the ground in this simulation.")