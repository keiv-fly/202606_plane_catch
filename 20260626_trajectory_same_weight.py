import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
g = 9.81

H = 50.0          # pulley / crane top height above ground, m
h0 = 25.0         # aircraft height at catch, m
r0 = H - h0       # initial cable length, m

r_max = 40.0      # maximum allowed pendulum/cable length, m

v0 = 20.0         # initial horizontal aircraft speed, m/s
theta0 = 0.0
rdot0 = 0.0
thetadot0 = v0 / r0

t_max = 10.0
dt = 0.002


# -----------------------------
# State:
# y = [r, rdot, theta, thetadot]
# -----------------------------
def derivatives_counterweight(y):
    """
    Variable-length pendulum with equal-mass counterweight.
    Valid only while r < r_max.
    """
    r, rdot, theta, thetadot = y

    rddot = 0.5 * (r * thetadot**2 - g * (1.0 - np.cos(theta)))
    thetaddot = -(g * np.sin(theta) + 2.0 * rdot * thetadot) / r

    return np.array([rdot, rddot, thetadot, thetaddot])


def derivatives_fixed(y):
    """
    Fixed-length pendulum after the cable reaches r_max.
    """
    r, rdot, theta, thetadot = y

    rdot = 0.0
    rddot = 0.0
    thetaddot = -g * np.sin(theta) / r

    return np.array([rdot, rddot, thetadot, thetaddot])


def rk4_step(y, dt, mode):
    if mode == "counterweight":
        f = derivatives_counterweight
    elif mode == "fixed":
        f = derivatives_fixed
    else:
        raise ValueError("Unknown mode")

    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)

    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# -----------------------------
# Simulation
# -----------------------------
y = np.array([r0, rdot0, theta0, thetadot0], dtype=float)

times = []
xs = []
hs = []
rs = []
thetas = []
load_factors = []
modes = []

t = 0.0
mode = "counterweight"

while t <= t_max:
    r, rdot, theta, thetadot = y

    # If the cable reaches the limit, lock it at r_max.
    # This ignores the shock impulse of the hard stop.
    if mode == "counterweight" and r >= r_max:
        r = r_max
        rdot = 0.0
        y = np.array([r, rdot, theta, thetadot], dtype=float)
        mode = "fixed"

    x = r * np.sin(theta)
    h = H - r * np.cos(theta)

    # Load factor calculation
    if mode == "counterweight":
        # T / mg = 1 + rddot/g
        rddot = derivatives_counterweight(y)[1]
        load_factor = 1.0 + rddot / g
    else:
        # Fixed pendulum:
        # T / mg = cos(theta) + r * thetadot^2 / g
        load_factor = np.cos(theta) + r * thetadot**2 / g

    times.append(t)
    xs.append(x)
    hs.append(h)
    rs.append(r)
    thetas.append(theta)
    load_factors.append(load_factor)
    modes.append(mode)

    # Stop if aircraft reaches ground
    if h <= 0:
        break

    y = rk4_step(y, dt, mode)
    t += dt


times = np.array(times)
xs = np.array(xs)
hs = np.array(hs)
rs = np.array(rs)
thetas = np.array(thetas)
load_factors = np.array(load_factors)
modes = np.array(modes)


# -----------------------------
# Fixed pendulum comparison paths
# -----------------------------
x_circle_start = np.linspace(0, r0, 400)
h_circle_start = H - np.sqrt(r0**2 - x_circle_start**2)

x_circle_max = np.linspace(0, r_max, 400)
h_circle_max = H - np.sqrt(r_max**2 - x_circle_max**2)


# -----------------------------
# Plot path
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(xs, hs, label="Aircraft path: counterweight, then fixed at 40 m")
plt.plot(x_circle_start, h_circle_start, "--", label="Fixed pendulum with initial 25 m length")
plt.plot(x_circle_max, h_circle_max, ":", label="Fixed pendulum with 40 m max length")

# Crane mast and 15 m boom
plt.plot([0, 0], [0, H], linewidth=3, label="Crane mast")
plt.plot([0, 15], [H, H], linewidth=3, label="15 m boom")

# Ground
plt.axhline(0, linewidth=1)

# Start and end points
plt.scatter([xs[0]], [hs[0]], s=60, label="Catch point")
plt.scatter([xs[-1]], [hs[-1]], s=60, label="End of simulation")

# Mark where cable limit is reached
limit_indices = np.where(modes == "fixed")[0]
if len(limit_indices) > 0:
    i_limit = limit_indices[0]
    plt.scatter([xs[i_limit]], [hs[i_limit]], s=80, marker="x", label="Cable reaches 40 m")

plt.xlabel("Horizontal distance from pulley, m")
plt.ylabel("Height above ground, m")
plt.title("Aircraft catch path: start height 25 m, cable limited to 40 m")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# -----------------------------
# Plot cable length
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(times, rs)
plt.axhline(r_max, linestyle="--", label="40 m cable limit")

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
print(f"Crane height: {H:.2f} m")
print(f"Starting aircraft height: {h0:.2f} m")
print(f"Initial cable length: {r0:.2f} m")
print(f"Maximum cable length: {r_max:.2f} m")
print(f"Initial speed: {v0:.2f} m/s")
print(f"Simulation time: {times[-1]:.2f} s")
print(f"Max height: {hs.max():.2f} m")
print(f"Max horizontal distance: {xs.max():.2f} m")
print(f"Max cable length reached: {rs.max():.2f} m")
print(f"Max smooth cable load: {load_factors.max():.2f} g")
print(f"Final height: {hs[-1]:.2f} m")

if len(limit_indices) > 0:
    i_limit = limit_indices[0]
    print()
    print("Cable limit reached:")
    print(f"  time: {times[i_limit]:.2f} s")
    print(f"  x: {xs[i_limit]:.2f} m")
    print(f"  height: {hs[i_limit]:.2f} m")
    print(f"  load factor just after lock: {load_factors[i_limit]:.2f} g")
else:
    print()
    print("Cable limit was not reached during the simulation.")