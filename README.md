# Cart-Pole 시스템 제어: CLBF, QP

## 1. 시스템 설명: Cart-Pole

질량이 $m_c$인 카트가 마찰이 없는 선로 위를 움직이고, 질량이 $m_p$인 막대가 카트 상단에서 회전할 수 있다고 합시다. 상태 변수는 다음과 같습니다:

- $x$: 카트의 수평 위치  
- $v=\dot{x}$: 카트의 속도  
- $\theta$: 막대의 각도(이 코드에서는 $\theta=\pi$가 수직 위쪽)  
- $\dot{\theta}$: 막대의 각속도  

제어 입력 $F$는 카트에 가해지는 수평 힘입니다.

코드의 `CartPole.step()` 함수에 구현된 운동방정식은 아래와 같습니다:

$$
\dot{x}=\dot{x}
$$

$$
\ddot{x}=\frac{F+m_p\sin(\theta)\bigl(L\dot{\theta}^2+g\cos(\theta)\bigr)}{m_c+m_p\sin^2(\theta)}
$$

$$
\dot{\theta}=\dot{\theta}
$$

$$
\ddot{\theta}=\frac{-F\cos(\theta)-m_pL\dot{\theta}^2\sin(\theta)\cos(\theta)-(m_c+m_p)g\sin(\theta)-b\dot{\theta}}{L\bigl(m_c+m_p\sin^2(\theta)\bigr)}
$$

여기서
- $L$은 막대의 길이,
- $g$는 중력가속도,
- $b$는 막대 회전에 대한 마찰계수,
- $m_c$는 카트의 질량,
- $m_p$는 막대의 질량을 의미합니다.

코드에서 **CartPole** 클래스는 다음 두 함수를 사용해 이 시스템을 **제어-선형화(control-affine) 형태**로 표현합니다:

$$
\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})+\mathbf{g}(\mathbf{x})u
$$

- `f_x(state)`: 드리프트 항 $\mathbf{f}(\mathbf{x})$  
- `g_x(state)`: 입력 매트릭스 $\mathbf{g}(\mathbf{x})$

$$\dot{\mathbf{x}} = \begin{bmatrix}
\dot{x} \\
\ddot{x} \\
\dot{\theta} \\
\ddot{\theta}
\end{bmatrix}=
\underbrace{
\begin{bmatrix}
v \\
\displaystyle \frac{m_{\text{pole}}\sin(\theta)\Big(L\,\omega^2 + g\,\cos(\theta)\Big)}{m_{\text{cart}} + m_{\text{pole}}\sin^2(\theta)} \\
\omega \\
\displaystyle \frac{- m_{\text{pole}}L\,\omega^2\sin(\theta)\cos(\theta) - (m_{\text{cart}}+m_{\text{pole}})\,g\,\sin(\theta) - \text{(pole friction)}\,\omega}{L\Big(m_{\text{cart}} + m_{\text{pole}}\sin^2(\theta)\Big)}
\end{bmatrix}
}_{f(\mathbf{x})}
$$

$$
+
\underbrace{
\begin{bmatrix}
0 \\
\displaystyle \frac{1}{m_{\text{cart}} + m_{\text{pole}}\sin^2(\theta)} \\
0 \\
\displaystyle \frac{-\cos(\theta)}{L\Big(m_{\text{cart}} + m_{\text{pole}}\sin^2(\theta)\Big)}
\end{bmatrix}
}_{g(\mathbf{x})}
\,u
$$



## 2. 제어 리아푸노프 함수(CLF)

**Control Lyapunov Function**은 시스템의 **안정성**을 보장하기 위해 사용됩니다. 

코드의 `CLF` 클래스에서 사용되는 CLF는 다음과 같이 정의됩니다.

pole이 서있는 상황에서의 시스템을 선형근사한 선형시스템($A,B$)에 대해 LQR을 풀어서 Lyapunov function을 만들어냅니다.

$$J=\int^\infty_0(\mathbf{x}^\top Q\mathbf{x}+u^\top Ru)dt$$

이 LQR을 풀때 사용하는 continuous time Algebraic Riccati equation의 해 $P$를 통해 Lyapunov function을 만듭니다.

$$A^\top P+PA-PBR^{-1}B^\top P+Q=0$$

$$V(\mathbf{x})=\mathbf{x}^\top P\mathbf{x}$$


## 3. Reciprocal Control Barrier Function(RCBF)

**Control Barrier Function**은 **안전(safety)** 제약을 만족하도록 해 줍니다. 그중에서도 **Reciprocal CBF**는 상태가 경계에 가까워질 때 무한대로 큰 페널티를 주어, 안전 집합에서 벗어나지 않도록 합니다. 

이 함수는 다양하게 정의할 수 있겠지만, 이 코드의 `RCBF` 클래스는 다음과 같이 정의합니다.

$$
h(\mathbf{x})=-(v-v_{\max})(v-v_{\min})
$$

$$
b(\mathbf{x})=\frac{1}{h(\mathbf{x})}
$$

여기서 $v_{\min},v_{\max}$는 카트가 움직일 수 있는 안전 범위입니다. 만약 $v$가 범위를 벗어나면 $h(\mathbf{x})$가 0 또는 음수가 되어 위험해집니다.

$h(\mathbf{x})$에 대한 그래디언트 $\nabla h(\mathbf{x})$는 다음과 같습니다.

$$
\nabla h(\mathbf{x})=
\begin{bmatrix}
0&-2(v-v_{\min})-2(v-v_{\max})&0&0
\end{bmatrix}
$$

$b(\mathbf{x})=\frac{1}{h(\mathbf{x})}$에 대한 그래디언트는 다음과 같습니다.

$$
\nabla b(\mathbf{x})=-\frac{\nabla h(\mathbf{x})}{h(\mathbf{x})^2}
$$


## 4. CLF+RCBF Quadratic Program(CLBF)

**CLBF** 클래스는 CLF와 RCBF를 합쳐서, **안정성**과 **안전성**을 동시에 만족시키는 **이차 최적화(Quadratic Program, QP)** 문제를 풉니다.

### 4.1 QP 비용함수

$$
u^{*}_{1}(x):= \text{argmin}_{u,\delta}\left(\frac{1}{2}u^{\intercal}H(x)u+p \delta^{2} \right)
$$

$$
\text{subject to }L_{f}V(x)+L_{g}V(x)u \leq -\overline{\alpha}_{3}(V(x))+\delta
$$

$$
\qquad \qquad L_{f}b(x)+L_{g}b(x)u-\alpha_{3}(h(x))\leq 0
$$

코드에서는 각 시뮬레이션 스텝마다 다음의 QP를 풉니다(`clbf_ctrl` 내부):

$$
\text{argmin}_{u,\delta}
\frac{1}{2}
\begin{bmatrix}
u\\
\delta
\end{bmatrix}^\top
Q
\begin{bmatrix}
u\\
\delta
\end{bmatrix}.
$$

여기서 매트릭스 $Q$는 상태에 따라 혹은 상수로 설정됩니다. 예를 들어 코드에서는

$$
Q=
\begin{bmatrix}
H&0\\
0&p
\end{bmatrix}
$$

형태로, $H$는 input에 대한 가중치, $p$는 슬랙 변수 $\delta$에 대한 가중치입니다.

### 4.2 QP 제약조건

CLF와 RCBF 제약조건은 아래와 같이 표현될 수 있습니다:

1. **CLF 제약**  
   $$\nabla V(\mathbf{x})\mathbf{f}(\mathbf{x})+\nabla V(\mathbf{x})\mathbf{g}(\mathbf{x})u\le -\alpha_1\bigl(V(\mathbf{x})\bigr)+\delta$$  
   여기서 $\alpha_1(\cdot)$는 $V(\mathbf{x})$의 감쇠율을 설정하는 **class-$\mathcal{K}$ 함수**입니다.

   코드에 적용하기 위해 다음과 같이 정리합니다.

   $$\nabla V(\mathbf{x})\mathbf{g}(\mathbf{x})u-\delta \le -\nabla V(\mathbf{x})\mathbf{f}(\mathbf{x})-\alpha_1\bigl(V(\mathbf{x})\bigr)$$

2. **RCBF 제약**  
   $$\nabla b(\mathbf{x})\mathbf{f}(\mathbf{x})+\nabla b(\mathbf{x})\mathbf{g}(\mathbf{x})u\le \alpha_2\bigl(h(\mathbf{x})\bigr)$$  
   여기서 $\alpha_2(\cdot)$는 $h(\mathbf{x})$가 양수를 유지하도록 하는 **class-$\mathcal{K}$ 함수**입니다.

   코드에 적용하기 위해 다음과 같이 정리합니다.

   $$\nabla b(\mathbf{x})\mathbf{g}(\mathbf{x})u\le -\nabla b(\mathbf{x})\mathbf{f}(\mathbf{x})+\alpha_2\bigl(h(\mathbf{x})\bigr)$$  
3. **입력($F$) 제한**  
   $$|F|\le F_{\max}.$$

	코드에서는 `condition_G`와 `condition_h` 함수 내에서 다음과 같은 선형 부등호 형태로 만들어집니다:
	
$$G=\begin{bmatrix}\nabla V(\mathbf{x})\mathbf{g}(\mathbf{x}) &-1 \\ \nabla b(\mathbf{x})\mathbf{g}(\mathbf{x}) & 0 \\ 1 &0 \\ -1&0\end{bmatrix},\quad h=\begin{bmatrix}-\nabla V(\mathbf{x})\mathbf{f}(\mathbf{x})-\alpha_1\left(V(\mathbf{x})\right) \\ -\nabla b(\mathbf{x})\mathbf{f}(\mathbf{x})+\alpha_2\left(h(\mathbf{x})\right) \\ F_{\max} \\ F_{\max}\end{bmatrix}$$

$$G \begin{bmatrix} u \\ \delta\end{bmatrix} \leq h$$

	
그 후 `cvxopt.solvers.qp`를 호출해 입력 $u^{*}$와 슬랙 변수 $\delta^{*}$를 구합니다.



## 5. Controller 클래스

`Controller` 클래스는 두 가지 제어 방법을 제공합니다:

1. `ctrl(state,t)`: PD 혹은 PID에 가까운 단순 제어 (데모용)  
2. `clbf_ctrl(state,t)`: 위에서 설명한 **CLBF-QP** 최적화 문제를 풀어 제어입력을 계산

시뮬레이션 루프에서는 보통

```python
f = controller.clbf_ctrl(state,t)
```

로 $F$를 얻고,

```python
state = cp.step(f)
```

를 통해 카트-폴 상태를 갱신합니다.



## 6. 시뮬레이션

시뮬레이션은 $dt$ 간격으로 진행됩니다. 제어기는 $ctrl\_dt$마다(=몇 스텝마다 한 번) 위의 QP를 풀어 힘 $F$를 갱신합니다.

- `dt`: 시뮬레이션 타임스텝  
- `ctrl_dt`: 제어기 업데이트 간격  
- `T`: 총 시뮬레이션 시간  
- `num_steps`: 시뮬레이션 단계 수

각 단계마다:
1. 이전 스텝에서 구한 힘 $F$를 카트-폴에 적용  
2. `cp.step(F)`로 상태 갱신  
3. `controller.clbf_ctrl(state,t)`로 새로운 힘을 계산



## 7. 플롯과 애니메이션

코드는 다음의 결과 그래프(`cartpole.png`)를 생성, 저장합니다.

1. **카트 위치($x$), 속도($v$), 입력 힘($F$)**  
2. **폴 각도($\theta$), 각속도($\dot{\theta}$)**

그리고 `FuncAnimation`을 사용해 카트와 막대가 움직이는 애니메이션(`cartpole.mp4`)을 생성, 저장합니다.



## 8. 사용 방법

1. **의존성 설치**  
   - `numpy`  
   - `matplotlib`  
   - `cvxopt`  
   - `ffmpeg`(애니메이션 저장용)
   
   ```bash
    pip3 install numpy matplotlib cvxopt
    sudo apt-get install ffmpeg
   ```

2. **스크립트 실행**  
   ```bash
   ./cartpole.py
   ```
   실행하면 다음이 수행됩니다:
   - 시뮬레이션  
   - 결과 그래프 `cartpole.png`로 저장  
   - 애니메이션 `cartpole.mp4`로 저장

3. **파라미터 조정**  
   - `CartPole` 생성자에서 질량, 마찰계수, 막대 길이 등 시스템 파라미터 변경  
   - `dt`, `ctrl_dt`, `T` 등도 필요에 맞게 수정 가능