#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CartPoleMass:
    def __init__(
        self,
        x=0.0,
        v=0.0,
        theta=0.0,
        theta_dot=0.0,
        dt=0.01,
        L=1.0,
        g=9.81,
        m_cart=1.0,
        m_pole=1.0,
        pole_friction=0.4,
    ):
        """
        Initialize the cart-pole system with masses.

          x, v            : Cart position and velocity
          theta, theta_dot: Pole angle (0 is upright) and angular velocity (rad, rad/s)
          dt              : Simulation time step (seconds)
          L               : Pole length
          g               : Gravitational acceleration (m/s²)
          m_cart          : Mass of the cart (kg)
          m_pole          : Mass of the pole (kg)
          pole_friction   : Damping coefficient for the pole's rotation
        """
        self.x = x
        self.v = v
        self.theta = theta
        self.theta_dot = theta_dot
        self.dt = dt
        self.L = L
        self.g = g
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.pole_friction = pole_friction

    def step(self, F):
        """
        Update the system state for one time step using the applied force F on the cart.

        Equations of motion:
          - For the cart:
              ẍ = [F + m_p * sin(theta) * (L * thetȧ² + g * cos(theta))] / [m_cart + m_p * sin²(theta)]
          - For the pole (with damping):
              θ̈ = [-F*cos(theta) - m_p*L*thetȧ²*sin(theta)*cos(theta)
                    - (m_cart + m_p)*g*sin(theta) - pole_friction*thetȧ]
                    / [L*(m_cart + m_p*sin²(theta))]
        """
        # Common denominator
        denom = self.m_cart + self.m_pole * np.sin(self.theta) ** 2

        # Calculate cart acceleration
        x_ddot = (
            F
            + self.m_pole
            * np.sin(self.theta)
            * (self.L * self.theta_dot**2 + self.g * np.cos(self.theta))
        ) / denom

        # Calculate pole angular acceleration with damping
        theta_ddot = (
            -F * np.cos(self.theta)
            - self.m_pole
            * self.L
            * self.theta_dot**2
            * np.sin(self.theta)
            * np.cos(self.theta)
            - (self.m_cart + self.m_pole) * self.g * np.sin(self.theta)
            - self.pole_friction * self.theta_dot
        ) / (self.L * denom)

        # Update cart state
        self.v += x_ddot * self.dt
        self.x += self.v * self.dt

        # Update pole state
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        return self.x, self.v, self.theta, self.theta_dot


# Simulation parameters
dt = 0.01  # Time step (seconds)
T = 10.0  # Total simulation time (seconds)
num_steps = int(T / dt)

# Initialize the system:
# - Cart starts at x=0 with zero velocity.
# - Pole is slightly displaced from upright.
# - Adjust masses and friction as needed.
cp = CartPoleMass(
    x=0.0,
    v=0.0,
    theta=0.1,
    theta_dot=0.0,
    dt=dt,
    L=1.0,
    g=9.81,
    m_cart=1.0,
    m_pole=0.1,
    pole_friction=0.1,
)


# Define the force command as a function of time
def controller(x, v, theta, theta_dot):
    return 0.0


# Data storage for simulation results
time_history = []
x_history = []
v_history = []
theta_history = []
theta_dot_history = []
f_command_history = []

t = 0.0
f = 0.0
for i in range(num_steps):
    f_command_history.append(f)
    x, v, theta, theta_dot = cp.step(f)
    f = controller(x, v, theta, theta_dot)
    time_history.append(t)
    x_history.append(x)
    v_history.append(v)
    theta_history.append(theta)
    theta_dot_history.append(theta_dot)
    t += dt

# Plotting the results
plt.figure(figsize=(12, 8))

# Cart state: position, velocity, and applied force
plt.subplot(2, 1, 1)
plt.plot(time_history, x_history, label="Cart Position (x)")
plt.plot(time_history, v_history, label="Cart Velocity (v)")
plt.plot(time_history, f_command_history, "--", label="Force Command (F)")
plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity / Force")
plt.legend()
plt.title("Cart State")

# Pole state: angle and angular velocity
plt.subplot(2, 1, 2)
plt.plot(time_history, theta_history, label="Pole Angle (θ)")
plt.plot(time_history, theta_dot_history, label="Pole Angular Velocity (θ_dot)")
plt.xlabel("Time (s)")
plt.ylabel("Angle / Angular Velocity")
plt.legend()
plt.title("Pole State")

plt.tight_layout()
plt.show()

# --- Animation ---
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)
cart_width = 0.4
cart_height = 0.2
pole_length = cp.L

# Create initial objects: cart (rectangle) and pole (line)
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc="k")
ax.add_patch(cart)
(line,) = ax.plot([], [], lw=2)


def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    line.set_data([], [])
    return cart, line


def animate(i):
    if i < len(x_history):
        x = x_history[i]
        theta = theta_history[i]
    else:
        x = 0
        theta = 0
    # Update cart position (using center as reference)
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    # Compute the pole's end point from the top center of the cart
    cart_top_y = cart_height / 2
    pole_x = x + pole_length * np.sin(theta)
    pole_y = cart_top_y + pole_length * np.cos(theta)
    line.set_data([x, pole_x], [cart_top_y, pole_y])
    return cart, line


ani = animation.FuncAnimation(
    fig, animate, frames=len(x_history), init_func=init, interval=dt * 1000, blit=True
)

plt.show()
