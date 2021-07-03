---
layout: post
title: Homework 1
author: Bryan
hasmath: "true"
custom_css: tufte
---

# Question 3: Proofs

a) 

Let $\nabla_{x}f(x)$ represent the column vector containing all the partial derivatives of $f$ w.r.t. $x$, i.e., 
$$
\nabla_{x} = \begin{bmatrix}
				\frac{\partial{f}}{\partial{x_{1}}} \\
				... \\
				\frac{\partial{f}}{\partial{x_{n}}} \\
			 \end{bmatrix}
$$
For any two column vectors $x$, $a \in \mathbb{R}^{n}$, prove that
$$
\nabla_{x}(x^{T}a) = \nabla_{x}(a^{T}x) = a
$$
Hint: differentiate w.r.t. each element of $x$, and then gather the partial derivatives into a column vector.

**Answer:**
Because we are dealing with a column matrix for $x$, which we are taking the partial derivative in respect to, we need to be careful about the difference between taking the partial derivative 
w.r.t. $x$ and taking the partial derivative of the $i^{th}$ component of $x$ which is w.r.t. $x_{i}$ 

 taking the partial derivative of the $i^{th}$ component of $x$:
$$
\nabla_{x_{k}} (x^{T}a) = \nabla_{x_{k}} (a^{T}x) = \nabla_{x_{k}} [\underset{i=1}{\overset{n}{\sum}} x_{i} a_{i}] = \nabla_{x_{k}} [\underset{i=1}{\overset{n}{\sum}} a_{i} x_{i}] = a_{k}
$$
partial derivative w.r.t. $x$:
$$
\nabla_{x} (x^{T}a) = \nabla_{x} (a^{T}x)
	= \begin{bmatrix}
			\nabla_{x_{1}} [\underset{i=1}{\overset{n}{\sum}} x_{i} a_{i}] \\ 
			\nabla_{x_{2}} [\underset{i=1}{\overset{n}{\sum}} x_{i} a_{i}] \\
			... \\\
			\nabla_{x_{n}} [\underset{i=1}{\overset{n}{\sum}} x_{i} a_{i}] \\
		\end{bmatrix} 
		=
		\begin{bmatrix}
			\nabla_{x_{1}} [\underset{i=1}{\overset{n}{\sum}} a_{i} x_{i}] \\ 
			\nabla_{x_{2}} [\underset{i=1}{\overset{n}{\sum}} a_{i} x_{i}] \\
			... \\\
			\nabla_{x_{n}} [\underset{i=1}{\overset{n}{\sum}} a_{i} x_{i}] \\
		\end{bmatrix} 
	 
	= a
$$


b)

Prove that
$$
\nabla{x}(x^{T}Ax) = (A + A^{T})x
$$
for any column vector $x \in \mathbb{R}^{n}$ and any $n \times n$ matrix $A$ 

**Answer:**





c)

Based on the theorem above, prove that 
$$
\nabla{x}(x^{T}Ax) = 2Ax
$$
for any column vector $x \in \mathbb{R}^{n}$ and any symmetric $n \times n$ matrix $A$ 



d)

Based on the theorems above, prove that
$$
\nabla_{x}[(Ax + b)^{T}(Ax + b)] = 2A^{T}(Ax + b)
$$
for any column vector $x \in \mathbb{R}^{n}$, any symmetric $n \times n$ matrix $A$, and any constant column vector $b \in \mathbb{R}^{n}$ 