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
spring_c = 5000.0       # N*s/m, damping coefficient during extension

theta0 = 0.0
rdot0 = 0.0
thetadot0 = v0 / r0

t_max = 10.0
dt = 0.001


# -----------------------------
# State:
# y = [r, rdot, theta, thetadot]
# -----------------------------
def spring_force(r, rdot):
    """
    Spring-damper force that resists cable extension.

    It starts acting only after r > spring_start.

    spring_k:
        Total stiffness of the spring system, N/m.

    spring_c:
        Damping during extension, N*s/m.

    The damping is one-way:
        it acts only when rdot > 0, meaning the cable is still extending.
    """
    extension = max(0.0, r - spring_start)

    if extension <= 0.0:
        return 0.0

    spring_part = spring_k * extension
    damping_part = spring_c * max(0.0, rdot)

    return spring_part + damping_part


def derivatives(y):
    r, rdot, theta, thetadot = y

    Fs = spring_force(r, rdot)

    # Equal-mass counterweight plus spring-damper resisting extension.
    rddot = 0.5 * (
        r * thetadot**2
        - g * (1.0 - np.cos(theta))
        - Fs / m
    )

    thetaddot = -(g * np.sin(theta) + 2.0 * rdot * thetadot) / r

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
thetas = []
load_factors = []
spring_forces = []
hit_hard_stop = False

t = 0.0

while t <= t_max:
    r, rdot, theta, thetadot = y

    x = r * np.sin(theta)
    h = H - r * np.cos(theta)

    rddot = derivatives(y)[1]

    # Cable tension from aircraft radial equation:
    # T = m * (r * thetadot^2 + g*cos(theta) - rddot)
    tension = m * (r * thetadot**2 + g * np.cos(theta) - rddot)
    load_factor = tension / (m * g)

    Fs = spring_force(r, rdot)

    times.append(t)
    xs.append(x)
    hs.append(h)
    rs.append(r)
    rdots.append(rdot)
    thetas.append(theta)
    load_factors.append(load_factor)
    spring_forces.append(Fs)

    # Stop if aircraft reaches ground
    if h <= 0:
        break

    # Stop if hard mechanical limit is reached.
    # This means the spring-damper was not strong enough.
    # The real system would get a shock impulse here.
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
thetas = np.array(thetas)
load_factors = np.array(load_factors)
spring_forces = np.array(spring_forces)


# -----------------------------
# Reference circular paths
# -----------------------------
x_circle_initial = np.linspace(0, r0, 400)
h_circle_initial = H - np.sqrt(r0**2 - x_circle_initial**2)

x_circle_max = np.linspace(0, r_max, 400)
h_circle_max = H - np.sqrt(r_max**2 - x_circle_max**2)


# -----------------------------
# Plot path
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(xs, hs, label="Aircraft path with counterweight + spring-damper")
plt.plot(x_circle_initial, h_circle_initial, "--", label="Fixed pendulum, initial length")
plt.plot(x_circle_max, h_circle_max, ":", label="40 m limit circle")

# Crane mast and boom
plt.plot([0, 0], [0, H], linewidth=3, label="Crane mast")
plt.plot([0, 15], [H, H], linewidth=3, label="15 m boom")

# Ground
plt.axhline(0, linewidth=1)

# Points
plt.scatter([xs[0]], [hs[0]], s=60, label="Catch point")
plt.scatter([xs[-1]], [hs[-1]], s=60, label="End point")

plt.xlabel("Horizontal distance from pulley, m")
plt.ylabel("Height above ground, m")
plt.title("Aircraft catch path with spring-damper before 40 m limit")
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
# Plot load factor
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, load_factors)
plt.axhline(1.5, linestyle="--", label="1.5 g")
plt.axhline(2.0, linestyle="--", label="2.0 g")
plt.axhline(3.0, linestyle="--", label="3.0 g")

plt.xlabel("Time, s")
plt.ylabel("Cable load factor, g")
plt.title("Cable load factor during catch")
plt.grid(True)
plt.legend()
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
print(f"Simulation time: {times[-1]:.2f} s")
print(f"Max height: {hs.max():.2f} m")
print(f"Max horizontal distance: {xs.max():.2f} m")
print(f"Max cable length: {rs.max():.2f} m")
print(f"Max radial speed: {rdots.max():.2f} m/s")
print(f"Max spring-damper force: {spring_forces.max() / 1000.0:.2f} kN")
print(f"Max smooth cable load: {load_factors.max():.2f} g")
print(f"Final height: {hs[-1]:.2f} m")

if hit_hard_stop:
    print()
    print("WARNING: The cable reached the hard 40 m limit.")
    print("This would create an additional shock load not shown by this smooth model.")
    print("Increase spring_k, increase spring_c, or allow more spring travel.")
else:
    print()
    print("The hard 40 m limit was not reached.")