dynamics of particle, rigid body의 핵심적인 개념 에 대해서 설명 Meriam's Engineering Mechanics Dynamics 핵심만 요약


Meriam의 **Engineering Mechanics: Dynamics**를 시험/문제풀이 관점에서 압축하면, 핵심은 결국

**“운동을 기술하는 Kinematics” + “그 운동이 왜 생기는지 설명하는 Kinetics”**

그리고 이를 **Particle(질점)**과 **Rigid Body(강체)**에 각각 적용하는 것입니다.

## 1. 전체 구조

| 구분 | Particle | Rigid Body |
|---|---|---|
| Kinematics | 위치, 속도, 가속도 | 병진 + 회전 |
| Force–Acceleration | \(\sum \mathbf F=m\mathbf a\) | \(\sum \mathbf F=m\mathbf a_G\), \(\sum M_G=I_G\alpha\) |
| Work–Energy | \(T_1+U_{1\to2}=T_2\) | 병진 KE + 회전 KE |
| Impulse–Momentum | 선운동량 | 선운동량 + 각운동량 |

가장 중요한 차이는 **Particle은 회전을 고려하지 않지만, Rigid Body는 반드시 회전을 고려한다**는 것입니다.

---

# 2. Dynamics of Particle

질점은 물체의 크기와 회전을 무시하고 **질량이 한 점에 모여 있다고 보는 모델**입니다.

### ① Particle Kinematics

가장 기본은

\[
\mathbf v=\frac{d\mathbf r}{dt}
\]

\[
\mathbf a=\frac{d\mathbf v}{dt}
=\frac{d^2\mathbf r}{dt^2}
\]

입니다.

직선운동에서는

\[
v=\frac{dx}{dt},\qquad
a=\frac{dv}{dt}
\]

그리고 자주 쓰는 관계:

\[
a=v\frac{dv}{dx}
\]

입니다.

특히 \(a=\text{constant}\)이면

\[
v=v_0+at
\]

\[
x=x_0+v_0t+\frac12at^2
\]

\[
v^2=v_0^2+2a(x-x_0)
\]

가 됩니다.

---

### ② Curvilinear Motion

곡선운동에서는 좌표계를 잘 선택하는 것이 핵심입니다.

Cartesian 좌표:

\[
\mathbf v=\dot x\mathbf i+\dot y\mathbf j
\]

\[
\mathbf a=\ddot x\mathbf i+\ddot y\mathbf j
\]

Normal–Tangential 좌표에서는

\[
\mathbf a=a_t\mathbf e_t+a_n\mathbf e_n
\]

이며

\[
a_t=\dot v
\]

\[
a_n=\frac{v^2}{\rho}
\]

입니다.

여기서 꼭 기억할 것:

\[
\boxed{a_n=\frac{v^2}{\rho}}
\]

즉, **속도의 크기가 일정해도 방향이 변하면 가속도가 존재합니다.**

원운동이라면

\[
a_n=\frac{v^2}{r}=\omega^2r
\]

입니다.

---

### ③ Polar Coordinates

회전하면서 움직이는 물체에서는

\[
\mathbf v
=
\dot r\mathbf e_r
+
r\dot\theta\mathbf e_\theta
\]

\[
\mathbf a
=
(\ddot r-r\dot\theta^2)\mathbf e_r
+
(r\ddot\theta+2\dot r\dot\theta)\mathbf e_\theta
\]

가 중요합니다.

특히

\[
2\dot r\dot\theta
\]

는 **Coriolis-type term**입니다.

---

# 3. Particle Kinetics

여기서부터는 “왜 그렇게 움직이는가?”를 다룹니다.

가장 기본은 Newton 제2법칙:

\[
\boxed{\sum \mathbf F=m\mathbf a}
\]

입니다.

문제를 보면 먼저 **Free Body Diagram(FBD)**을 그리는 습관이 매우 중요합니다.

즉,

**Force → Acceleration → Motion**

순서입니다.

Normal–Tangential 좌표에서는

\[
\sum F_t=m\dot v
\]

\[
\sum F_n=m\frac{v^2}{\rho}
\]

가 됩니다.

곡선 운동 문제에서 매우 자주 등장합니다.

---

# 4. Work–Energy Method

시간을 구할 필요 없이 **위치와 속도 사이의 관계**를 구하고 싶다면 Newton 법칙보다 에너지법이 훨씬 편한 경우가 많습니다.

핵심 식:

\[
\boxed{T_1+U_{1\to2}=T_2}
\]

질점의 운동에너지:

\[
T=\frac12mv^2
\]

일:

\[
U_{1\to2}
=
\int_1^2 \mathbf F\cdot d\mathbf r
\]

중력 위치에너지:

\[
V_g=mgh
\]

스프링 위치에너지:

\[
V_s=\frac12kx^2
\]

보존력만 존재한다면

\[
\boxed{T_1+V_1=T_2+V_2}
\]

입니다.

문제에서

**“속도를 구하라, 이동거리가 주어졌다, 시간은 필요 없다”**

→ Energy 방법을 먼저 생각하면 좋습니다.

---

# 5. Impulse–Momentum

힘이 **짧은 시간 동안 작용**하거나 충돌 문제가 나오면 momentum이 핵심입니다.

선운동량:

\[
\mathbf p=m\mathbf v
\]

Impulse:

\[
\int_{t_1}^{t_2}\mathbf F\,dt
\]

따라서

\[
\boxed{
m\mathbf v_1+
\int_{t_1}^{t_2}\sum\mathbf F\,dt
=
m\mathbf v_2
}
\]

입니다.

외력 impulse가 없다면

\[
\boxed{
m_1\mathbf v_1+m_2\mathbf v_2
=
m_1\mathbf v_1'+m_2\mathbf v_2'
}
\]

즉 **linear momentum conservation**을 사용할 수 있습니다.

충돌에서는 coefficient of restitution:

\[
\boxed{
e=
\frac{\text{relative separation velocity}}
{\text{relative approach velocity}}
}
\]

를 사용합니다.

---

# 6. 이제 Rigid Body Dynamics

Rigid body는 입자와 달리

**물체의 질량중심 이동 + 물체 자체의 회전**

을 동시에 고려해야 합니다.

따라서 가장 중요한 개념은

\[
\boxed{
\text{Rigid Body Motion}
=
\text{Translation}
+
\text{Rotation}
}
\]

입니다.

---

# 7. Rigid Body Kinematics

강체에서 두 점 \(A,B\)가 있다면

\[
\boxed{
\mathbf v_B
=
\mathbf v_A
+
\boldsymbol\omega\times\mathbf r_{B/A}
}
\]

입니다.

이 식은 강체 운동학에서 가장 중요한 식 중 하나입니다.

가속도는

\[
\boxed{
\mathbf a_B
=
\mathbf a_A
+
\boldsymbol\alpha\times\mathbf r_{B/A}
+
\boldsymbol\omega
\times
(\boldsymbol\omega\times\mathbf r_{B/A})
}
\]

평면운동에서는 마지막 항을 보통

\[
-\omega^2\mathbf r_{B/A}
\]

형태로 생각할 수 있습니다.

즉

\[
\mathbf a_B
=
\mathbf a_A
+
\boldsymbol\alpha\times\mathbf r_{B/A}
-\omega^2\mathbf r_{B/A}
\]

입니다.

여기서

\[
\alpha r
\]

은 tangential acceleration,

\[
\omega^2r
\]

은 normal acceleration입니다.

---

# 8. Translation, Rotation, General Plane Motion

강체운동은 크게 세 종류로 이해하면 됩니다.

**Translation**

모든 점의

\[
\mathbf v_A=\mathbf v_B
\]

\[
\mathbf a_A=\mathbf a_B
\]

가 같습니다.

**Fixed-axis rotation**

한 축을 중심으로 회전합니다.

\[
v=r\omega
\]

\[
a_t=r\alpha
\]

\[
a_n=r\omega^2
\]

**General Plane Motion**

가장 중요합니다.

\[
\boxed{
\text{Translation of }G
+
\text{Rotation about }G
}
\]

로 분해합니다.

즉 자동차 바퀴, 링크, 디스크 대부분이 여기에 해당합니다.

---

# 9. Instantaneous Center of Zero Velocity

평면 강체운동에서 특정 순간에는 강체가 어떤 점 \(C\)를 중심으로 회전하는 것처럼 볼 수 있습니다.

그 점에서

\[
v_C=0
\]

이고

\[
v_A=\omega r_{A/C}
\]

\[
v_B=\omega r_{B/C}
\]

이므로

\[
\frac{v_A}{r_{A/C}}
=
\frac{v_B}{r_{B/C}}
=
\omega
\]

입니다.

따라서 **velocity 계산을 매우 빠르게 할 수 있습니다.**

하지만 중요한 주의점:

Instantaneous Center는 주로 **velocity 계산용**입니다.

일반적으로 acceleration까지

\[
a=\omega^2r
\]

만으로 처리하면 안 됩니다.

---

# 10. Rolling Without Slipping

Meriam에서 매우 중요한 유형입니다.

미끄러짐 없는 rolling이면 접촉점의 순간 속도는 0입니다.

따라서

\[
\boxed{v_G=R\omega}
\]

이고

\[
\boxed{a_{G,t}=R\alpha}
\]

입니다.

하지만 접촉점의 **속도가 0이라고 해서 가속도도 0인 것은 아닙니다.**

이 부분이 매우 흔한 함정입니다.

---

# 11. Mass Moment of Inertia

Particle의 \(m\)에 대응하는 회전운동의 핵심 물리량이

\[
\boxed{I}
\]

즉 질량관성모멘트입니다.

정의:

\[
I=\int r^2dm
\]

질량이 회전축에서 멀리 떨어질수록 \(I\)가 커집니다.

Parallel-axis theorem:

\[
\boxed{
I_O=I_G+md^2
}
\]

이것도 필수입니다.

예를 들어 중심축 기준:

solid disk

\[
I_G=\frac12mR^2
\]

thin ring

\[
I_G=mR^2
\]

slender rod 중심:

\[
I_G=\frac1{12}mL^2
\]

rod 끝:

\[
I_O=\frac13mL^2
\]

---

# 12. Rigid Body Force–Acceleration

질점에서는

\[
\sum\mathbf F=m\mathbf a
\]

였는데 rigid body에서는

\[
\boxed{
\sum\mathbf F
=
m\mathbf a_G
}
\]

와

\[
\boxed{
\sum M_G
=
I_G\alpha
}
\]

두 개를 동시에 사용합니다.

즉 아주 중요하게

\[
\boxed{
Translation:\ \sum F=ma_G
}
\]

\[
\boxed{
Rotation:\ \sum M_G=I_G\alpha
}
\]

입니다.

대부분의 rigid-body kinetics 문제는 이 두 식에서 출발합니다.

---

# 13. Rigid Body Work–Energy

강체의 운동에너지는 particle과 달리 두 부분입니다.

\[
\boxed{
T=
\frac12mv_G^2
+
\frac12I_G\omega^2
}
\]

즉

**translational kinetic energy + rotational kinetic energy**

입니다.

따라서

\[
\boxed{
T_1+U_{1\to2}=T_2
}
\]

를 사용합니다.

특히 rolling 문제에서는

\[
T=
\frac12mv_G^2
+
\frac12I_G\omega^2
\]

와

\[
v_G=R\omega
\]

를 같이 사용하면 거의 대부분 해결됩니다.

---

# 14. Rigid Body Impulse–Momentum

강체에서는 linear momentum뿐 아니라 angular momentum도 중요합니다.

Linear:

\[
\boxed{
m\mathbf v_{G1}
+
\int \sum\mathbf Fdt
=
m\mathbf v_{G2}
}
\]

Angular about \(G\):

\[
\boxed{
I_G\omega_1
+
\int\sum M_Gdt
=
I_G\omega_2
}
\]

외부 moment impulse가 없으면 angular momentum이 보존됩니다.

---

# 15. Meriam Dynamics에서 제일 중요한 사고방식

문제를 보고 바로 공식을 찾는 것보다, **어떤 방법을 사용할 것인지 선택하는 능력**이 핵심입니다.

- **시간에 따른 위치/속도/가속도 관계** → Kinematics
- **힘과 가속도를 직접 연결** → \(\sum F=ma\), \(\sum M=I\alpha\)
- **위치 ↔ 속도 관계, 시간을 몰라도 됨** → Work–Energy
- **충격, 충돌, 짧은 시간의 큰 힘** → Impulse–Momentum
- **강체의 여러 점 속도** → \(v_B=v_A+\omega\times r_{B/A}\)
- **rolling** → \(v_G=R\omega\)
- **강체 에너지** → \(\frac12mv_G^2+\frac12I_G\omega^2\)

---

## 시험 직전이라면 이 10개는 반드시 기억

\[
\boxed{\sum F=ma}
\]

\[
\boxed{a_n=\frac{v^2}{\rho}}
\]

\[
\boxed{T_1+U_{1\to2}=T_2}
\]

\[
\boxed{m v_1+\int Fdt=m v_2}
\]

\[
\boxed{v_B=v_A+\omega\times r_{B/A}}
\]

\[
\boxed{
a_B=a_A+\alpha\times r_{B/A}
+\omega\times(\omega\times r_{B/A})
}
\]

\[
\boxed{v_G=R\omega}
\]

\[
\boxed{\sum F=ma_G}
\]

\[
\boxed{\sum M_G=I_G\alpha}
\]

\[
\boxed{
T_{\text{rigid body}}
=
\frac12mv_G^2+\frac12I_G\omega^2
}
\]

한마디로 압축하면, **Particle Dynamics는 \(F=ma\)를 중심으로 운동·에너지·운동량을 보는 과목이고, Rigid Body Dynamics는 여기에 \(\omega,\alpha,I,\sum M=I\alpha\)라는 회전 개념이 추가되는 것**이라고 보면 됩니다.