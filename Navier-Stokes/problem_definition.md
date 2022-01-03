# Flow of Incompressible Fluid  

![Geometry](./geo.png)

Viscosity of fluid : $\mu = 0.02kg \cdot (m \cdot sec)^{-1}$     
Densitoy if fluid : $\rho = 1kg \cdot m^{3}$  

$$
\begin{aligned}
\tag{@\ Inlet}
u(0,y) &= u_{in}= y(H-y) \frac{4}{H^2} \\
v(0,y) &= 0
\end{aligned}
$$
* No slip Condition
* Pressure of outlet equals to zero
$$
\tag{@\ wall\ and\ cylinder\ surface}
\begin{aligned}
& p(1,y) = 0 \\
& u(x,y) = v(x,y) = 0 \ \ 
\end{aligned}
$$
* Continuity Equation (for incompressible flow)
$$
\nabla\ \cdot\ \mathbf{V} = 0
$$
* Navier-Stokes Equation
$$
\rho(\mathbf{V} \cdot \nabla)\mathbf{V} = -\nabla p + \mu \nabla^2 \mathbf{V}
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Or it can be expressed as Cauchy momentum equation
$$
\rho(\mathbf{V} \cdot \nabla)\mathbf{V} = \nabla \cdot \sigma
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
When,
$$
\sigma = -p \bar{\mathbf{I}} + \mu ( \nabla \mathbf{V}+(\nabla \mathbf{V})^T )
$$