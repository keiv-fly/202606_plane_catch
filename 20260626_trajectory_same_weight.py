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

# Spring-damper zone
spring_start = 39.0     # springs start at 39 m cable length
r_max = 40.0            # hard mechanical limit, m

spring_k = 5000.0       # N/m, total spring stiffness
spring_c = 5000.0       # N*s/m, damping during extension

# Smart brake
target_load_g = 1.5         # desired cable load while braking
brake_force_max = 20000.0   # N, change this; use np.inf for unlimited ideal brake

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
    Spring-damper force resisting cable extension.
    It starts acting only after r > spring_start.
    """
    extension = max(0.0, r - spring_start)

    if extension <= 0.0:
        return 0.0

    spring_part = spring_k * extension
    damping_part = spring_c * max(0.0, rdot)

    return spring_part + damping_part


def radial_acceleration(y, resist_force):
    """
    Radial acceleration with equal-mass counterweight.

    resist_force includes spring force + brake force.
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
    Cable load factor T / mg for a given resisting force.
    """
    r, rdot, theta, thetadot = y

    rddot = radial_acceleration(y, resist_force)

    tension = m * (
        r * thetadot**2
        + g * np.cos(theta)
        - rddot
    )

    return tension / (m * g)


def smart_brake_force(y):
    """
    Brake works only when cable is extending: rdot > 0.

    It calculates load without brake first.
    If that load is below target_load_g, it adds enough braking force
    to bring total cable load to target_load_g.
    """
    r, rdot, theta, thetadot = y

    if rdot <= 0.0:
        return 0.0

    Fs = spring_force(r, rdot)

    load_without_brake = load_factor_from_resist_force(y, Fs)

    if load_without_brake >= target_load_g:
        return 0.0

    # In this equal-mass model, extra payout resistance increases
    # aircraft cable tension by about half of that force.
    #
    # Therefore:
    # delta_load_g = Fb / (2*m*g)
    #
    # Fb = 2*m*g*(target - current)
    required_brake = 2.0 * m * g * (target_load_g - load_without_brake)

    return min(required_brake, brake_force_max)


# -----------------------------
# Equations of motion
# -----------------------------
def derivatives(y):
    r, rdot, theta, thetadot = y

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y)

    total_resist_force = Fs + Fb

    rddot = radial_acceleration(y, total_resist_force)

    thetaddot = -(
        g * np.sin(theta)
        + 2.0 * rdot * thetadot
    ) / r

    return np.array([rdot, rddot, thetadot, thetaddot])


def rk4_step(y, dt):
    k1 = derivatives(y)
    k2 = derivatives(y + 0.5 * dt * k1)
    k3 = derivatives(y + 0.5 * dt * k2)
    k4 = derivatives(y + dt * k3)

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
load_factors = []
load_factors_no_brake = []
spring_forces = []
brake_forces = []

hit_hard_stop = False
t = 0.0

while t <= t_max:
    r, rdot, theta, thetadot = y

    x = r * np.sin(theta)
    h = H - r * np.cos(theta)

    Fs = spring_force(r, rdot)
    Fb = smart_brake_force(y)

    load_no_brake = load_factor_from_resist_force(y, Fs)
    load_with_brake = load_factor_from_resist_force(y, Fs + Fb)

    times.append(t)
    xs.append(x)
    hs.append(h)
    rs.append(r)
    rdots.append(rdot)
    load_factors.append(load_with_brake)
    load_factors_no_brake.append(load_no_brake)
    spring_forces.append(Fs)
    brake_forces.append(Fb)

    # Stop if aircraft reaches ground
    if h <= 0:
        break

    # Stop if hard mechanical limit is reached.
    # This means the spring/brake system was not enough.
    if r >= r_max:
        hit_hard_stop = True
        break

    y = rk4_step(y, dt)
    t += dt


times = np.array(times)
xs = np.array(xs)
hs = np.array(hs)
rs = np.array(rs)
rdots = np.array(rdots)
load_factors = np.array(load_factors)
load_factors_no_brake = np.array(load_factors_no_brake)
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

plt.plot(xs, hs, label="Aircraft path with counterweight + spring + smart brake")
plt.plot(x_circle_initial, h_circle_initial, "--", label="Fixed pendulum, initial length")
plt.plot(x_circle_max, h_circle_max, ":", label="40 m limit circle")

# Crane mast and boom
plt.plot([0, 0], [0, H], linewidth=3, label="Crane mast")
plt.plot([0, 15], [H, H], linewidth=3, label="15 m boom")

plt.axhline(0, linewidth=1)

plt.scatter([xs[0]], [hs[0]], s=60, label="Catch point")
plt.scatter([xs[-1]], [hs[-1]], s=60, label="End point")

plt.xlabel("Horizontal distance from pulley, m")
plt.ylabel("Height above ground, m")
plt.title("Aircraft catch path with smart 1.5 g payout brake")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot cable length
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, rs, label="Cable length")
plt.axhline(spring_start, linestyle="--", label="Springs start at 39 m")
plt.axhline(r_max, linestyle="--", label="Hard limit at 40 m")

plt.xlabel("Time, s")
plt.ylabel("Cable length r, m")
plt.title("Cable length over time")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot load factor
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, load_factors_no_brake, "--", label="Load without brake")
plt.plot(times, load_factors, label="Load with smart brake")

plt.axhline(target_load_g, linestyle="--", label=f"Target {target_load_g:.1f} g")
plt.axhline(2.0, linestyle="--", label="2.0 g")
plt.axhline(3.0, linestyle="--", label="3.0 g")

plt.xlabel("Time, s")
plt.ylabel("Cable load factor, g")
plt.title("Cable load factor during catch")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot brake force
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, brake_forces / 1000.0)

plt.xlabel("Time, s")
plt.ylabel("Brake force, kN")
plt.title("Smart brake force")
plt.grid(True)
plt.show()


# -----------------------------
# Plot spring force
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, spring_forces / 1000.0)

plt.xlabel("Time, s")
plt.ylabel("Spring-damper force, kN")
plt.title("Spring-damper force")
plt.grid(True)
plt.show()


# -----------------------------
# Plot radial velocity
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, rdots)

plt.xlabel("Time, s")
plt.ylabel("Radial speed rdot, m/s")
plt.title("Cable extension speed")
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
print(f"Spring starts at: {spring_start:.2f} m")
print(f"Hard cable limit: {r_max:.2f} m")
print(f"Spring stiffness k: {spring_k:.1f} N/m")
print(f"Spring damping c: {spring_c:.1f} N*s/m")
print()
print(f"Target brake load: {target_load_g:.2f} g")
print(f"Max brake force allowed: {brake_force_max / 1000.0:.2f} kN")
print()
print(f"Simulation time: {times[-1]:.2f} s")
print(f"Max height: {hs.max():.2f} m")
print(f"Max horizontal distance: {xs.max():.2f} m")
print(f"Max cable length: {rs.max():.2f} m")
print(f"Max radial speed: {rdots.max():.2f} m/s")
print(f"Max spring-damper force: {spring_forces.max() / 1000.0:.2f} kN")
print(f"Max brake force: {brake_forces.max() / 1000.0:.2f} kN")
print(f"Max load without brake: {load_factors_no_brake.max():.2f} g")
print(f"Max load with brake: {load_factors.max():.2f} g")
print(f"Final height: {hs[-1]:.2f} m")

if hit_hard_stop:
    print()
    print("WARNING: The cable reached the hard 40 m limit.")
    print("This would create an additional shock load not shown by this smooth model.")
else:
    print()
    print("The hard 40 m limit was not reached.")