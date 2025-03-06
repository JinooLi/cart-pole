#!/usr/bin/env python3
import os
import sympy as sp

# 라그랑지안 방정식을 풀어서 운동방정식을 구하는 코드

# 시간 정의
t = sp.symbols("t", real=True)

# state 변수 정의
x = sp.Function("x")(t)
theta = sp.Function("theta")(t)
x_dot = x.diff(t)
theta_dot = theta.diff(t)
x_ddot = x_dot.diff(t)
theta_ddot = theta_dot.diff(t)

# 상수 정의
m, M, l, g = sp.symbols("m M l g", real=True)
fric_x, fric_theta = sp.symbols("fric_x fric_theta", real=True)

# 마찰력 함수 정의
friction_x = sp.Function("friction")(x_dot)
friction_theta = sp.Function("friction")(theta_dot)
friction_x = 2* fric_x * sp.atan(x_dot) / sp.pi
friction_theta = 2* fric_theta * sp.atan(theta_dot) / sp.pi

# 차량에 대한 외력 정의
f = sp.symbols("f", real=True)

# 각 질량체의 x,y 좌표 정의
cart_pos_x = x
cart_pos_y = 0
pendulum_pos_x = x + l * sp.sin(theta)
pendulum_pos_y = -l * sp.cos(theta)

# 각 질량체의 속도의 제곱
cart_vel_sqare = cart_pos_x.diff(t) ** 2
pendulum_vel_sqare = pendulum_pos_x.diff(t) ** 2 + pendulum_pos_y.diff(t) ** 2

print("cart velo square:")
sp.pprint(cart_vel_sqare)
print("pendulum velo square:")
sp.pprint(pendulum_vel_sqare)

# 시스템의 총 위치에너지와 운동에너지
V = -m * g * l * sp.cos(theta)  # 위치에너지
T = 1 / 2 * M * cart_vel_sqare + 1 / 2 * m * pendulum_vel_sqare  # 운동에너지

V = sp.simplify(V)
T = sp.simplify(T)

print("T:")
sp.pprint(T)
print("V:")
sp.pprint(V)


# 라그랑지안
L = T - V

print("L:")
sp.pprint(L)

# 각 좌표계에 대한 라그랑주 방정식. 마찰력과 외력을 포함한다.
x_eq = sp.Eq(L.diff(x_dot).diff(t) - L.diff(x), -friction_x + f)
theta_eq = sp.Eq(L.diff(theta_dot).diff(t) - L.diff(theta), -friction_theta)

print("x_eq:")
sp.pprint(x_eq)
print("theta_eq:")
sp.pprint(theta_eq)

# 운동방정식 풀기
solution = sp.solve([x_eq, theta_eq], [x_ddot, theta_ddot])

sol_x_ddot = solution[x_ddot]
sol_theta_ddot = solution[theta_ddot]

# t에 대한 함수인 x, x_dot, theta, theta_dot을 각각 pos, v, angle, omega 변수로 치환
pos, v = sp.symbols("pos v", real=True)
angle, omega = sp.symbols("angle omega", real=True)

sol_x_ddot = sol_x_ddot.subs({x: pos, x_dot: v, theta: angle, theta_dot: omega})
sol_theta_ddot = sol_theta_ddot.subs({x: pos, x_dot: v, theta: angle, theta_dot: omega})

sol_x_ddot = sp.simplify(sol_x_ddot)
sol_theta_ddot = sp.simplify(sol_theta_ddot)

print("x_ddot:")
sp.pprint(sol_x_ddot)
print("theta_ddot:")
sp.pprint(sol_theta_ddot)


# 함수로 변환
file_name = "__difeq__.py"

with open(file_name, "w", encoding="utf-8") as file:
    file.write("# cartpole의 theta_ddot과 x_ddot 함수 생성\n")

    file.write("import math\n")

    file.write("def x_ddot(v, angle, omega, l, m, M, g, fric_theta, fric_x, f):\n")
    file.write(f"    return {sp.pycode(sol_x_ddot)}\n\n")

    file.write("def theta_ddot(v, angle, omega, l, m, M, g, fric_theta, fric_x, f):\n")
    file.write(f"    return {sp.pycode(sol_theta_ddot)}\n")


# 이 파일이 있는 주소를 알아낸다.
path = os.path.abspath(__file__)
# 디렉토리 주소만 가져온다.
dir_path = os.path.dirname(path)

os.system(f"black {dir_path}/{file_name}")
