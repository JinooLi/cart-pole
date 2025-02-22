#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cvxopt import solvers, matrix


class CartPole:
    class State:
        def __init__(self, *args):
            if len(args) == 1:
                state: np.array = args[0]
                self.x = state[0][0]
                self.v = state[1][0]
                self.theta = state[2][0]
                self.theta_dot = state[3][0]
            elif len(args) == 4:
                self.x = args[0]
                self.v = args[1]
                self.theta = args[2]
                self.theta_dot = args[3]
            else:
                raise ValueError("Invalid number of arguments for state")

        def to_np(self) -> np.ndarray:
            return np.array(
                [
                    [self.x],
                    [self.v],
                    [self.theta],
                    [self.theta_dot],
                ],
                dtype=np.float64,
            )

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
        pole_friction=1,
        f_max=15,
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
          f_max           : Maximum force that can be applied to the cart
        """
        self.state = self.State(x, v, theta, theta_dot)
        self.dt = dt
        self.L = L
        self.g = g
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.pole_friction = pole_friction
        self.f_max = f_max

    def f_x(self, state: State) -> np.ndarray:
        """f of \dot{x} = f(x)+g(x)u

        Returns:
            np.ndarray[[float], [float], [float], [float]]: f의 output (column vector)
        """
        f = np.array(
            [
                [state.v],
                [
                    (
                        self.m_pole
                        * np.sin(state.theta)
                        * (self.L * state.theta_dot**2 + self.g * np.cos(state.theta))
                    )
                    / (self.m_cart + self.m_pole * np.sin(state.theta) ** 2)
                ],
                [state.theta_dot],
                [
                    (
                        -self.m_pole
                        * self.L
                        * state.theta_dot**2
                        * np.sin(state.theta)
                        * np.cos(state.theta)
                        - (self.m_cart + self.m_pole) * self.g * np.sin(state.theta)
                        - self.pole_friction * state.theta_dot
                    )
                    / (self.L * (self.m_cart + self.m_pole * np.sin(state.theta) ** 2))
                ],
            ],
            dtype=np.float64,
        )
        return f

    def g_x(self, state: State) -> np.array:
        """g(x) of \dot{x} = f(x)+g(x)u

        Returns:
            np.array[[float], [float], [float], [float]]: g의 output (column vector)
        """
        g = np.array(
            [
                [0],
                [1 / (self.m_cart + self.m_pole * np.sin(state.theta) ** 2)],
                [0],
                [1 / (self.L * (self.m_cart + self.m_pole * np.sin(state.theta) ** 2))],
            ],
            dtype=np.float64,
        )
        return g

    def step(self, F: float) -> State:
        """
        Update the system state for one time step using the applied force F on the cart.

        Equations of motion:
          - For the cart:
              ẍ = [F + m_p * sin(theta) * (L * thetȧ² + g * cos(theta))] / [m_cart + m_p * sin²(theta)]
          - For the pole (with damping):
              θ̈ = [-F*cos(theta) - m_p*L*thetȧ²*sin(theta)*cos(theta)
                    - (m_cart + m_p)*g*sin(theta) - pole_friction*thetȧ]
                    / [L*(m_cart + m_p*sin²(theta))]

        Args:
            F (float): Force applied to the cart
        Returns:
            State: Updated state of the cart-pole system
        """
        # Common denominator
        denom = self.m_cart + self.m_pole * np.sin(self.state.theta) ** 2

        # Calculate cart acceleration
        x_ddot = (
            F
            + self.m_pole
            * np.sin(self.state.theta)
            * (self.L * self.state.theta_dot**2 + self.g * np.cos(self.state.theta))
        ) / denom

        # Calculate pole angular acceleration with damping
        theta_ddot = (
            -F * np.cos(self.state.theta)
            - self.m_pole
            * self.L
            * self.state.theta_dot**2
            * np.sin(self.state.theta)
            * np.cos(self.state.theta)
            - (self.m_cart + self.m_pole) * self.g * np.sin(self.state.theta)
            - self.pole_friction * self.state.theta_dot
        ) / (self.L * denom)

        # Update cart state
        self.state.v += x_ddot * self.dt  # 가속도 -> 속도
        self.state.x += self.state.v * self.dt  # 속도 -> 위치

        # Update pole state
        self.state.theta_dot += theta_ddot * self.dt  # 각가속도 -> 각속도
        self.state.theta += self.state.theta_dot * self.dt  # 각속도 -> 각도

        return self.state


class CLF:
    def __init__(self, cp: CartPole):
        """control lyapunov function을 정의하는 클래스.

        V(x) = x^TMx 형태로 정의함. 이때 M은 positive definite matrix

        pole의 원점은 중력방향이므로 반대 방향으로 돌리기 위해 x=0, v=0, theta=pi, theta_dot=0으로 V(x)의 원점을 설정한다.

        Args:
            cp (CartPole): CartPole 객체를 받아 초기화
        """
        self.cp = cp
        self.M = np.array(
            [
                [10, 0, 1, 0],
                [0, 0.1, 0, 0],
                [0, 0, 1000, 0],
                [0, 0, 0, 1.5],
            ],
            dtype=np.float64,
        )
        self.M = self.M.T + self.M
        self.adj_state = np.array(
            [
                [0],
                [0],
                [-np.pi],
                [0],
            ]
        )

    def V(self, state: CartPole.State) -> float:
        """Lyapunov function을 정의하는 함수

        x = x + adj_state로 변환하여 V(x) = x^TMx로 정의

        Args:
            state (State): 현재 상태
        Returns:
            float: V의 output
        """
        state: np.ndarray = state.to_np()

        adj_state = state + self.adj_state
        return adj_state.T @ self.M @ adj_state

    def dV_dx(self, state: CartPole.State) -> np.ndarray:
        """Lyapunov function의 시간미분을 정의하는 함수

        x = x + adj_state로 변환하여 dV/dx(x) = 2x^TM로 정의

        Args:
            state (State): 현재 상태
        Returns:
            np.ndarray[[float, float, float, float]]: dV_dx의 output (row vector)
        """
        state: np.ndarray = state.to_np()
        adj_state = state + self.adj_state
        return 2 * adj_state.T @ self.M


class RCBF:
    """1/h(x) = b(x)로 정의되는 장벽함수를 정의하는 클래스"""

    def __init__(self, cp: CartPole):
        self.cp = cp
        self.v_max = 10.0  # x 안전영역 최대값
        self.v_min = -10.0  # x 안전영역 최소값

    def h_x(self, state: CartPole.State) -> float:
        """state가 안전한지의 여부를 정의하는 함수 h.

        h(x) = -(x-x_max)(x-x_min)로 정의함.
        """
        return -(state.v - self.v_max) * (state.v - self.v_min)

    def dh_dx(self, state: CartPole.State) -> np.ndarray:
        """h를 x로 미분한 함수. row vector로 반환

        dh/dx(x) = -2(x-x_max) - 2(x-x_min)로 정의함.

        Args:
            state (CartPole.State): 현재 상태
        Returns:
            np.array[[float, float, float, float]]: dh_dx의 output (row vector)
        """
        return np.array(
            [
                [
                    -2 * (state.x - self.v_min) - 2 * (state.x - self.v_max),
                    0,
                    0,
                    0,
                ]
            ],
            dtype=np.float64,
        )

    def b_x(self, state: CartPole.State) -> float:
        """state가 안전한지의 여부를 정의하는 함수의 계수를 생성하는 함수\n
        b(x) = 1/h(x)로 정의함.
        Args:
            state (CartPole.State): 현재 상태

        Returns:
            float: b의 output
        """
        return 1 / (self.h_x(state))

    def db_dx(self, state: CartPole.State) -> np.ndarray:
        """b를 x로 미분한 함수 row vector로 반환
        db/dx(x) = -dh/dx(x)/h(x)^2로 정의함.


        """
        return -self.dh_dx(state) / (self.h_x(state) ** 2)


class CLBF:
    def __init__(self, cp: CartPole, clf: CLF, rcbf: RCBF):
        """CLBF를 정의하는 클래스

        Args:
            cp (CartPole): cartpole 객체
            clf (CLF): CLF 객체. control lyapunov function을 받아온다.
            rcbf (RCBF): RCBF 객체. reciprocal control barrier function을 받아온다.
        """
        self.cp = cp
        self.clf = clf
        self.rcbf = rcbf
        self.p = 20

    def alpa1(self, input) -> float:
        """class k 함수 alpha1
        alpa1(x) = coef*x로 정의함.
        """
        coef = 1
        return coef * input

    def alpa2(self, input) -> float:
        """class k 함수 alpha2
        alpa2(x) = coef*x로 정의함.
        """
        coef = 1
        return coef * input

    def getH(self, state: CartPole.State) -> np.ndarray:
        """QP문제를 풀기 위한 H를 반환하는 함수

        Args:
            state (CartPole.State): 현재 상태

        Returns:
            np.ndarray: [1x1] H
        """
        H = np.array(
            [
                [abs(state.x) + abs(state.v) + abs(state.theta) + abs(state.theta_dot)],
            ],
            dtype=np.float64,
        )
        return H

    def getQ(self, state: CartPole) -> np.ndarray:
        """QP문제를 풀기 위한 Q를 반환하는 함수.

        행렬 Q는 다음과 같이 정의한다.
        Q = [[H, 0],
             [0, p]]

        Args:
            state (CartPole): 현재 상태

        Returns:
            np.ndarray: [2x2] Q
        """
        H = self.getH(state)
        Q = np.array(
            [
                [float(H), 0],
                [0, self.p],
            ],
            dtype=np.float64,
        )

        return Q

    def condition_G(self, state: CartPole.State) -> np.ndarray:
        """부등식 제약조건 G를 반환하는 함수

        G는 다음과 같이 정의한다.
        G = [[dV_dx @ g, -1],
             [db_dx @ g, 0],
             [1, 0],
             [-1, 0]]

        위의 두 행은 CLF와 RCBF의 조건을 만족시키기 위한 제약조건이다.
        밑의 두 행은 힘의 최대값을 넘지 않도록 하는 제약조건이다.

        Args:
            state (CartPole.State): 현재 상태
        Returns:
            np.ndarray[[float], [float]]: G의 output (column vector)
        """
        dv_dx = self.clf.dV_dx(state)

        g = self.cp.g_x(state)
        db_dx = self.rcbf.db_dx(state)

        condition = np.array(
            [
                [float(dv_dx @ g), -1],
                [float(db_dx @ g), 0],
                [1, 0],
                [-1, 0],
            ],
            dtype=np.float64,
        )
        return condition

    def condition_h(self, state: CartPole.State) -> np.ndarray:
        """부등식 제약조건 h를 반환하는 함수

        h는 다음과 같이 정의한다.
        h = [[-dv_dx @ f - alpa1(v)],
             [-db_dx @ f + alpa2(h)],
             [f_max],
             [f_max],]

        위의 두 행은 CLF와 RCBF의 조건을 만족시키기 위한 제약조건이다.
        밑의 두 행은 힘의 최대값을 넘지 않도록 하는 제약조건이다.

        Args:
            state (CartPole.State): 현재 상태
        Returns:
            np.ndarray[[float], [float]]: h의 output (column vector)
        """
        v = self.clf.V(state)
        h = self.rcbf.h_x(state)
        f = self.cp.f_x(state)
        dv_dx = self.clf.dV_dx(state)
        db_dx = self.rcbf.db_dx(state)
        return np.array(
            [
                [float(-dv_dx @ f - self.alpa1(v))],
                [float(-db_dx @ f + self.alpa2(h))],
                [self.cp.f_max],
                [self.cp.f_max],
            ],
            dtype=np.float64,
        )


# Define the force command as a function of time
class Controller:
    def __init__(
        self,
        first_out: float,
        dt: float,
        ctrl_dt: float,
        cp: CartPole,
        clbf: CLBF,
    ):
        """제어기 정의

        Args:
            first_out (float): 초기 제어 출력
            dt (float): 시뮬레이션 주기
            ctrl_dt (float): 제어 주기
            cp (CartPole): Cartpole 객체
            clbf (CLBF): CLBF 객체
        """
        self.cp = cp
        self.clbf = clbf
        self.output = first_out

        if ctrl_dt / dt != int(ctrl_dt / dt):
            raise ValueError("ctrl_dt must be a multiple of dt")

        self.dt = dt
        self.ctrl_dt = ctrl_dt
        self.sum = 0

    def ctrl(
        self,
        state: CartPole.State,
        t: float,
    ) -> float:
        if t % self.ctrl_dt < 10e-6:
            return self.output
        a = 200
        b = 5
        c = 1000
        d = 10
        e = 0

        out = -(
            c * (state.theta - np.pi)
            - a * state.x
            - b * state.v
            + d * state.theta_dot
            - e * self.sum
        )

        self.sum += state.x * self.dt
        if self.sum > 200:
            self.sum = 200
        if self.sum < -200:
            self.sum = -200

        if out > 15:
            return 15
        if out < -15:
            return -15
        return out

    def clbf_ctrl(self, state: CartPole.State, t: float) -> float:
        if t % self.ctrl_dt < 10e-6:
            return self.output

        Q = self.clbf.getQ(state)
        c = np.zeros((2, 1))
        G = self.clbf.condition_G(state)
        h = self.clbf.condition_h(state)

        solution = solvers.qp(
            matrix(Q),
            matrix(c),
            matrix(G),
            matrix(h),
        )

        u = solution["x"][0]

        return float(u)


# Simulation parameters
dt = 0.01  # Time step (seconds)
ctrl_dt = dt * 10  # Controller time step (seconds)
T = 20.0  # Total simulation time (seconds)
num_steps = int(T / dt)  # Number of simulation steps

# Initialize the system:
# - Cart starts at x=0 with zero velocity.
# - Pole is slightly displaced from upright.
# - Adjust masses and friction as needed.
cp = CartPole(
    x=0.0,
    v=0.0,
    theta=np.pi + 0.2,
    theta_dot=0.0,
    dt=dt,
    L=1.0,
    g=9.81,
    m_cart=1.0,
    m_pole=0.1,
    pole_friction=0.1,
    f_max=100,
)

rcbf = RCBF(cp)
clf = CLF(cp)
clbf = CLBF(cp, clf, rcbf)
t = 0.0
f = 0.0
controller = Controller(first_out=f, dt=dt, ctrl_dt=ctrl_dt, cp=cp, clbf=clbf)
# Data storage for simulation results
time_history = []
x_history = []
v_history = []
theta_history = []
theta_dot_history = []
f_command_history = []


for i in range(num_steps):
    f_command_history.append(f)
    state: CartPole.State = cp.step(f)
    try:
        f = controller.clbf_ctrl(state, t)
    except:
        f_command_history.pop()
        break
    time_history.append(t)
    x_history.append(state.x)
    v_history.append(state.v)
    theta_history.append(state.theta)
    theta_dot_history.append(state.theta_dot)
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
plt.savefig("cartpole.png")

# --- Animation ---
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)
fig.set_size_inches(10, 4)
cart_width = 0.2
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
    pole_x = x + pole_length * np.sin(theta)
    pole_y = pole_length * np.cos(theta)
    line.set_data([x, pole_x], [0, -pole_y])
    return cart, line


ani = animation.FuncAnimation(
    fig, animate, frames=len(x_history), init_func=init, interval=dt * 1000, blit=True
)

ani.save("cartpole.mp4", writer="ffmpeg", fps=1 / dt)
