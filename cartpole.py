#!/usr/bin/env python3

import os
import time
import traceback

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from cvxopt import matrix, solvers
from scipy.linalg import solve_continuous_are

import custom_sympy as csp


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

        def copy(self):
            return CartPole.State(self.x, self.v, self.theta, self.theta_dot)

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

        def __add__(self, other):
            if isinstance(other, CartPole.State):
                return CartPole.State(
                    self.x + other.x,
                    self.v + other.v,
                    self.theta + other.theta,
                    self.theta_dot + other.theta_dot,
                )
            else:
                raise ValueError("Invalid type for addition")

        def __sub__(self, other):
            if isinstance(other, CartPole.State):
                return CartPole.State(
                    self.x - other.x,
                    self.v - other.v,
                    self.theta - other.theta,
                    self.theta_dot - other.theta_dot,
                )
            else:
                raise ValueError("Invalid type for subtraction")

        def __mul__(self, other):
            if isinstance(other, (float, int)):
                return CartPole.State(
                    self.x * other,
                    self.v * other,
                    self.theta * other,
                    self.theta_dot * other,
                )
            else:
                raise ValueError("Invalid type for multiplication")

        def __truediv__(self, other):
            if isinstance(other, (float, int)):
                return CartPole.State(
                    self.x / other,
                    self.v / other,
                    self.theta / other,
                    self.theta_dot / other,
                )
            else:
                raise ValueError("Invalid type for division")

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
        pole_friction=0,
        cart_friction=0,
        f_max=15,
    ):
        """
        Initialize the cart-pole system with masses.

        Args:
            x,              : Cart position
            v               : Cart velocity
            theta           : Pole angle (0 is downward) (rad)
            theta_dot       : Pole angular velocity (rad/s)
            dt              : Simulation time step (seconds)
            L               : Pole length
            g               : Gravitational acceleration (m/s²)
            m_cart          : Mass of the cart (kg)
            m_pole          : Mass of the pole (kg)
            pole_friction   : Damping coefficient for the pole's rotation
            cart_friction   : Damping coefficient for the cart's movement
            f_max           : Maximum force that can be applied to the cart
        """
        self.state = self.State(x, v, theta, theta_dot)
        self.dt = dt
        self.L = L
        self.g = g
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.pole_friction = pole_friction
        self.cart_friction = cart_friction
        self.f_max = f_max

        # 라그랑지안 방정식을 풀어서 운동방정식을 구하는 코드
        print("Calculating the equations of motion...")
        # 시간 정의
        t = sp.symbols("t", real=True)

        # state 변수 정의
        x = sp.Function("x")(t)
        theta = sp.Function("theta")(t)
        state = csp.make_state_vector([(x, 1), (theta, 1)], t)
        x_dot = state[1]
        theta_dot = state[3]
        x_ddot = x_dot.diff(t)
        theta_ddot = theta_dot.diff(t)

        # 차량에 대한 외력 정의
        f = sp.symbols("f", real=True)

        # 각 질량체의 x,y 좌표 정의
        cart_pos_x = x
        # cart_pos_y = 0
        pendulum_pos_x = x + self.L * sp.sin(theta)
        pendulum_pos_y = -self.L * sp.cos(theta)

        # 각 질량체의 속도의 제곱
        cart_vel_sqare = cart_pos_x.diff(t) ** 2
        pendulum_vel_sqare = pendulum_pos_x.diff(t) ** 2 + pendulum_pos_y.diff(t) ** 2

        # 시스템의 총 위치에너지와 운동에너지
        V = self.m_pole * self.g * pendulum_pos_y  # 위치에너지
        T = (
            1 / 2 * self.m_cart * cart_vel_sqare
            + 1 / 2 * self.m_pole * pendulum_vel_sqare
        )  # 운동에너지

        V = sp.simplify(V)
        T = sp.simplify(T)

        # 라그랑지안
        L = T - V

        if pole_friction != 0 or cart_friction != 0:
            print("Friction included.")
            # 마찰력 함수 정의
            friction_x = sp.Function("friction")(x_dot)
            friction_theta = sp.Function("friction")(theta_dot)
            friction_x = sp.sign(x_dot) * self.cart_friction
            friction_theta = sp.sign(theta_dot) * self.pole_friction

            # 각 좌표계에 대한 라그랑주 방정식. 마찰력과 외력을 포함한다.
            x_eq = sp.Eq(L.diff(x_dot).diff(t) - L.diff(x), -friction_x + f)
            theta_eq = sp.Eq(L.diff(theta_dot).diff(t) - L.diff(theta), -friction_theta)
        else:
            print("Friction not included.")
            # 마찰력을 제외한 각 좌표계에 대한 라그랑주 방정식
            x_eq = sp.Eq(L.diff(x_dot).diff(t) - L.diff(x), f)
            theta_eq = sp.Eq(L.diff(theta_dot).diff(t) - L.diff(theta), 0)

        # 운동방정식 풀기
        solution = sp.solve([x_eq, theta_eq], [x_ddot, theta_ddot])

        sol_x_ddot = solution[x_ddot].simplify()
        sol_theta_ddot = solution[theta_ddot].simplify()

        # f(x), g(x) 구하기
        state_dot = state.diff(t).subs(
            [(x_ddot, sol_x_ddot), (theta_ddot, sol_theta_ddot)]
        )

        f_x = state_dot.subs(f, 0)
        g_x = (state_dot - f_x) / f

        f_x = sp.simplify(f_x)
        g_x = sp.simplify(g_x)

        # lambdify하여 함수로 확장 이때 input은 [x, x_dot, theta, theta_dot] 형태로 넣어야함
        self.lambdify_f_x = sp.lambdify([[x, x_dot, theta, theta_dot]], f_x, "numpy")

        self.lambdify_g_x = sp.lambdify([[x, x_dot, theta, theta_dot]], g_x, "numpy")

        self.lambdify_x_ddot = sp.lambdify(
            [[x, x_dot, theta, theta_dot], f], sol_x_ddot, "numpy"
        )

        self.lambdify_theta_ddot = sp.lambdify(
            [[x, x_dot, theta, theta_dot], f], sol_theta_ddot, "numpy"
        )

        print("Equations of motion calculated.")

    def f_x(self, state: State) -> np.ndarray:
        """f of \dot{x} = f(x)+g(x)u

        Returns:
            np.ndarray[[float], [float], [float], [float]]: f의 output (column vector)
        """
        f = dif.f_x(
            x=state.x,
            v=state.v,
            angle=state.theta,
            omega=state.theta_dot,
            l=self.L,
            m=self.m_pole,
            M=self.m_cart,
            g=self.g,
            fric_theta=self.pole_friction,
            fric_x=0,
        )
        return f

    def g_x(self, state: State) -> np.ndarray:
        """g(x) of \dot{x} = f(x)+g(x)u

        Returns:
            np.ndarray[[float], [float], [float], [float]]: g의 output (column vector)
        """
        g = dif.g_x(
            x=state.x,
            v=state.v,
            angle=state.theta,
            omega=state.theta_dot,
            l=self.L,
            m=self.m_pole,
            M=self.m_cart,
            g=self.g,
            fric_theta=self.pole_friction,
            fric_x=0,
        )
        return g

    def x_ddot(self, state: State, F: float) -> float:
        """xddot을 구하는 함수

        Args:
            state (State): 현재 state
            F (float): cart에 가하는 힘

        Returns:
            float: x_ddot
        """
        state: np.ndarray = state.to_np().squeeze()
        x_ddot = self.lambdify_x_ddot(state, F)
        return float(x_ddot)

    def theta_ddot(self, state: State, F: float) -> float:
        """thetaddot을 구하는 함수

        Args:
            state (State): 현재 state
            F (float): cart에 가하는 힘

        Returns:
            float: theta_ddot
        """
        state: np.ndarray = state.to_np().squeeze()
        theta_ddot = self.lambdify_theta_ddot(state, F)
        return float(theta_ddot)

    def step(self, F: float) -> State:
        """
        Update the system state for one time step using the applied force F on the cart. Using Runge-Kutta 4th order method.

        Args:
            F (float): Force applied to the cart
        Returns:
            State: Updated state of the cart-pole system
        """
        # Update cart state

        # Runge-Kutta 4th order method

        cur_state = self.state
        # get k1
        x_ddot_k1 = self.x_ddot(cur_state, F)
        theta_ddot_k1 = self.theta_ddot(cur_state, F)
        x_dot_k1 = self.state.v + x_ddot_k1 * self.dt
        theta_dot_k1 = self.state.theta_dot + theta_ddot_k1 * self.dt

        # get k2
        k2_state = (
            cur_state
            + self.State(x_dot_k1, x_ddot_k1, theta_dot_k1, theta_ddot_k1) * self.dt / 2
        )
        x_ddot_k2 = self.x_ddot(k2_state, F)
        theta_ddot_k2 = self.theta_ddot(k2_state, F)
        x_dot_k2 = self.state.v + x_ddot_k2 * self.dt / 2
        theta_dot_k2 = self.state.theta_dot + theta_ddot_k2 * self.dt / 2

        # get k3
        k3_state = (
            cur_state
            + self.State(x_dot_k2, x_ddot_k2, theta_dot_k2, theta_ddot_k2) * self.dt / 2
        )
        x_ddot_k3 = self.x_ddot(k3_state, F)
        theta_ddot_k3 = self.theta_ddot(k3_state, F)
        x_dot_k3 = self.state.v + x_ddot_k3 * self.dt / 2
        theta_dot_k3 = self.state.theta_dot + theta_ddot_k3 * self.dt / 2

        # get k4
        k4_state = (
            cur_state
            + self.State(x_dot_k3, x_ddot_k3, theta_dot_k3, theta_ddot_k3) * self.dt
        )
        x_ddot_k4 = self.x_ddot(k4_state, F)
        theta_ddot_k4 = self.theta_ddot(k4_state, F)
        x_dot_k4 = self.state.v + x_ddot_k4 * self.dt
        theta_dot_k4 = self.state.theta_dot + theta_ddot_k4 * self.dt

        # Update cart state
        def update(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

        self.state.x += update(x_dot_k1, x_dot_k2, x_dot_k3, x_dot_k4)
        self.state.v += update(x_ddot_k1, x_ddot_k2, x_ddot_k3, x_ddot_k4)
        self.state.theta += update(
            theta_dot_k1, theta_dot_k2, theta_dot_k3, theta_dot_k4
        )
        self.state.theta_dot += update(
            theta_ddot_k1, theta_ddot_k2, theta_ddot_k3, theta_ddot_k4
        )

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
        A = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, cp.m_pole * cp.g / cp.m_cart, 0],
                [0, 0, 0, 1],
                [0, 0, (cp.m_cart + cp.m_pole) * cp.g / (cp.m_cart * cp.L), 0],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0],
                [1 / cp.m_cart],
                [0],
                [1 / (cp.m_cart * cp.L)],
            ],
            dtype=np.float64,
        )

        Q = np.diag([1, 1, 10, 1])

        R = np.array([[0.1]], dtype=np.float64)

        # Solve the continuous-time algebraic Riccati equation
        P = solve_continuous_are(A, B, Q, R)

        self.K = np.linalg.inv(R) @ (B.T @ P)
        print(self.K)

        self.M = P

    def adj_state(self, state: CartPole.State) -> CartPole.State:
        adj_state: CartPole.State = state.copy()
        pi2 = 2 * np.pi
        a: int = adj_state.theta // pi2

        if a != 0:
            adj_state.theta -= a * pi2

        adj_state.theta -= np.pi

        return adj_state

    def V(self, state: CartPole.State) -> float:
        """Lyapunov function을 정의하는 함수

        x = x + adj_state로 변환하여 V(x) = x^TMx로 정의

        Args:
            state (State): 현재 상태
        Returns:
            float: V의 output
        """

        state: CartPole.State = self.adj_state(state)
        return float(state.to_np().T @ self.M @ state.to_np())

    def dV_dx(self, state: CartPole.State) -> np.ndarray:
        """Lyapunov function의 시간미분을 정의하는 함수

        x = x + adj_state로 변환하여 dV/dx(x) = 2x^TM로 정의

        Args:
            state (State): 현재 상태
        Returns:
            np.ndarray[[float, float, float, float]]: dV_dx의 output (row vector)
        """
        state: CartPole.State = self.adj_state(state)
        return 2 * state.to_np().T @ self.M


class RCBF:
    """1/h(x) = b(x)로 정의되는 장벽함수를 정의하는 클래스"""

    def __init__(self, cp: CartPole):
        self.cp = cp
        self.v_max = 1.6  # v 안전영역 최대값
        self.v_min = -self.v_max  # v 안전영역 최소값
        self.delta = 0.0  # 안전영역의 폭

    def set_x_bound(self, v_max: float):
        self.v_max = v_max
        self.v_min = -v_max

    def h_x(self, state: CartPole.State) -> float:
        """state가 안전한지의 여부를 정의하는 함수 h.

        h(x) = -(x-x_max)(x-x_min)로 정의함.
        """
        return -(state.v - (self.v_max + self.delta)) * (state.v - (self.v_min - self.delta))

    def dh_dx(self, state: CartPole.State) -> np.ndarray:
        """h를 x로 미분한 함수. row vector로 반환

        dh/dx(x) = -(x-x_max) - (x-x_min)로 정의함.

        Args:
            state (CartPole.State): 현재 상태
        Returns:
            np.array[[float, float, float, float]]: dh_dx의 output (row vector)
        """
        return np.array(
            [
                [
                    0,
                    -(state.v - (self.v_min - self.delta)) - (state.v - (self.v_max + self.delta)),
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
        self.p = 10

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
                [1],
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
        # CLF의 크기에 반비례하게 p를 설정한다.
        # 이를 통해 원하는 state에 가까우면 가까울 수록 CLF의 영향력을 높인다.
        v = float(self.clf.V(state))
        print("v: ", v)
        self.p = 10000 / (10 * v + 1)
        print("p: ", self.p)

        H = self.getH(state)
        Q = np.array(
            [
                [H[0][0], 0],
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
        self.lam = 1  # swingup control
        self.u_a = 1  # swingup control의 크기
        self.linearizable_threshold = (
            5  # CLF의 V값이 이 값보다 작으면 linearizable control을 사용한다.
        )

    def check_ctrl_dt(self, t: float) -> bool:
        """현재 시각이 제어 주기의 배수인지 확인하는 함수

        Args:
            t (float): 현재 시각

        Returns:
            bool: 배수이면 True, 아니면 False
        """
        if (
            t % self.ctrl_dt < self.cp.dt
        ):  # 나머지의 오차가 시뮬레이션 주기보다 작으면 True
            return True
        return False

    def ctrl(self, state: CartPole.State, t: float) -> float:
        if self.check_ctrl_dt(t):
            self.output = -float(
                self.clbf.clf.K @ self.clbf.clf.adj_state(state).to_np()
            )
        return self.output

    def clbf_ctrl(self, state: CartPole.State, t: float) -> float:
        if self.check_ctrl_dt(t):
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

            self.output = float(u)

        return self.output

    def swingup_ctrl(self, state: CartPole.State, t: float) -> float:
        if state.theta == 0:
            return 0.3
        if self.check_ctrl_dt(t):   
            adj_state = self.clbf.clf.adj_state(state)
            lam = self.lam
            u_a = self.u_a
            E_p = (
                0.5 * adj_state.theta_dot**2 * self.cp.m_pole * self.cp.L**2
                + self.cp.m_pole * self.cp.g * self.cp.L * (np.cos(adj_state.theta) - 1)
            )
            out = -u_a * (
                E_p * np.cos(adj_state.theta) * adj_state.theta_dot + lam * adj_state.theta_dot
            )

            dh_dx = self.clbf.rcbf.dh_dx(state)
            g_x = self.cp.g_x(state)
            f_x = self.cp.f_x(state)
            h_x = self.clbf.rcbf.h_x(state)

            Q = np.array([[1]], dtype=np.float64)
            c = np.array([[-out]], dtype=np.float64)
            G = np.array([[-float(dh_dx @ g_x)]], dtype=np.float64)
            h = np.array([[float(dh_dx @ f_x + h_x)]], dtype=np.float64)

            solution = solvers.qp(
                matrix(Q),
                matrix(c),
                matrix(G),
                matrix(h),
            )

            self.output = float(solution["x"][0])

        return self.output

    def switching_ctrl(self, state: CartPole.State, t: float) -> float:
        if self.check_ctrl_dt(t):
            if self.clbf.clf.V(state) < self.linearizable_threshold:
                self.output = self.clbf_ctrl(state, t)
            else:
                self.output = self.swingup_ctrl(state, t)

        return self.output


if __name__ == "__main__":
    # Simulation parameters
    dt = 0.001  # Simulation time step (seconds)
    ctrl_dt = 0.1  # Controller time step (seconds)
    T = 30  # Total simulation time (seconds)
    num_steps = int(T / dt)  # Number of simulation steps

    # Initialize the system
    cp = CartPole(
        x=0.0,
        v=0.0,
        theta=0.0,
        theta_dot=0.0,
        dt=dt,
        L=1.0,
        g=9.81,
        m_cart=1.0,
        m_pole=0.1,
        pole_friction=0.0,
        cart_friction=0.0,
        f_max=15,
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

    maxtime = 0
    eout = 0
    eout_time = 0
    # Run the simulation
    for i in range(num_steps):
        f_command_history.append(f)
        cp.step(f)
        try:
            start = time.time()
            f = controller.switching_ctrl(cp.state, t)
            end = time.time()
            interval = end - start
            maxtime = max(maxtime, interval)  # 계산하는 데 걸린 시간의 최댓값 check
        except Exception as e:
            f_command_history.pop()
            print(f"Error occurs: {e}")
            traceback.print_exc()
            print("Simulation terminated")
            break
        if abs(cp.state.v) > controller.clbf.rcbf.v_max:  # 속도가 제한을 넘어가면 표기
            eout = f
            eout_time = t
        time_history.append(t)
        x_history.append(cp.state.x)
        v_history.append(cp.state.v)
        theta_history.append(cp.state.theta)
        theta_dot_history.append(cp.state.theta_dot)
        t += dt

    print("max time: ", maxtime)
    print("max V: ", max(v_history))
    print("min V: ", min(v_history))
    print("eout: ", eout)
    print("eout time: ", eout_time)
    print("max angle: ", max(theta_history))

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
    print("Simulation done. Results saved in 'cartpole.png'")

    # --- Animation ---
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    fig.set_size_inches(10, 4)
    cart_width = np.sqrt(cp.m_cart) * 0.2
    cart_height = np.sqrt(cp.m_cart) * 0.2
    pole_length = cp.L

    # Create initial objects: cart (rectangle) and pole (line)
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fc="k")
    ax.add_patch(cart)
    (line,) = ax.plot([], [], lw=2)

    def init():
        cart.set_xy((-cart_width / 2, -cart_height / 2))
        line.set_data([], [])
        return cart, line

    # 애니메이션을 위해 프레임 수 조절
    fps = 60
    x_fps_history = []
    theta_fps_history = []
    t = 0
    frame_interval = 1 / fps  # seconds
    x_fps_history.append(x_history[0])
    theta_fps_history.append(theta_history[0])
    for i in range(len(x_history)):
        t += dt
        if t >= frame_interval:
            x_fps_history.append(x_history[i])
            theta_fps_history.append(theta_history[i])
            t -= frame_interval

    def animate(i):
        if i < len(x_fps_history):
            x = x_fps_history[i]
            theta = theta_fps_history[i]
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
        fig,
        animate,
        frames=len(x_fps_history),
        init_func=init,
        interval=1000 * frame_interval,
        blit=True,
    )

    print("Saving animation...")
    ani.save("cartpole.mp4", writer="ffmpeg", fps=fps)
    print("Animation saved in 'cartpole.mp4'")
