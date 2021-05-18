---
layout: note
title: Deep Learning Notation
author: Bryan
hasmath: "true"
---

# Notation:

## Variables

- Scalars are non-bold and *italic*: *y, $\hat{y}$*
- Vectors are lower-case and **bold:** z, **w**
- Matrices are UPPER-CASE and **bold: X**
- $^T$ indicates matrix transpose: $X^{T}$
- (Parenthesized) $^{superscripts}$ specify the example: $x^{(3)}$
- Vectors are column vectors unless noted otherwise.
- n refers to number of examples; i is the index variable
- m refers to number of features; j is the index variable.

## Multiplication

- Multiplication of matrices **A** and **B**:
    - Only possible when: **A** is (n x k) and **B** is (k x m)
    - Result: (n x m)
- the **inner product** between two column vectors (same length) **x**, **y** can be written as $x^{T}y$. The result is a scalar
- the **outer product** between two column vectors (same length) **x**, **y** can be written as $xy^{T}$. The result is a matrix
- The **Hadamard** (element-wise product between two matrices **A** and **B** is written as **A** $\odot$ **B**

## Derivatives

- The **gradient** of a function ***f: $\Reals^{m} \to \Reals$*** w.r.t. input **x** is the column vector of first partial derivatives:

    $$⁍$$

The **Jacobian** of a function $f$: $\Reals^{m} \to \Reals^{n}$ w.r.t. input **x** is the matrix of first partial derivatives: 

$$
$$Jac[f] = \begin{bmatrix}
				\frac{\delta f_1}{\delta x_1} & ... & \frac{\delta f_1}{\delta x_m} 
				\\ ... & ... & ...\\ 
				\frac{\delta f_n}{\delta x_1} & ... & \frac{\delta f_n}{\delta x_m} 
			\end{bmatrix}
$$


- Note that, for n=1,
  $$
  \nabla_xf = Jac[f]^{T}
  $$
  

The **Hessian** of a function $f: \Reals^{m} \to \Reals$ w.r.t. input **x** is the matrix of second partial derivatives:
$$
H[f] = \begin{bmatrix} 
			\frac{\delta^{2}f}{\delta{x_1} \delta{x_1}} & ... & \frac{\delta^{2}f}{\delta{x_1} \delta{x_m}} 
			\\ ... & ... & ...\\ 
			\frac{\delta^{2}f}{\delta{x_m} \delta{x_1}} & ... & \frac{\delta^{2}}{\delta{x_m} \delta{x_m}} 
		\end{bmatrix}
$$


- Note that
  $$
  H[f] = Jac[\nabla_xf]
  $$
  
