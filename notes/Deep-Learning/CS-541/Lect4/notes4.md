---
layout: post
title: Deep Learning Notes 4
author: Bryan
hasmath: "true"
custom_css: tufte
---



[TOC]



## Optimization Of ML Models

Gradient descent is guaranteed to converge to a local minimum (eventually) if the learning rate is small enough relative to the steepness of $f$.

A function $f: \mathbb{R}^{m} \rightarrow \mathbb{R}$ is Lipschitz-continuous if: 
$$
\exists L: \forall x, y \in \mathbb{R}^{m}: ||f(x) - f(y)||_{2} \leq L||x -y||_{2}
$$
$L$ is essentially an upper bound on the absolute slope of $f$.

this can guarentee a maximum slope due to the change of $x$ as it relates to $y$.  

For learning rate $\epsilon \leq \frac{1}{L}$, gradient descent will converge to a local minimum linearly, i.e., the error is O($\frac{1}{k}$) in the iterations $k$.

With linear regression, the cost function $f_{MSE}$ has a single local minimum w.r.t. the weights w:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization of ML models.png)



As long as our learning rate is small enough, we will find the optimal w.

## Optimization: What Can Go Wrong?

In general ML and DL models, optimization is usually not so simple, due to:

1. Presence of multiple local minima

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong.png)

   

2. Bad initialization of the weights w.

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong2.png)

   

3. Learning rate is too small.

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong3.png)

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong3.png)

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong3.png)

   

4. Learning rate is too large.

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong4.png)

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong5.png)

   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong6.png)



With multidimensional weight vectors, badly chosen learning rates can cause more subtle problems.

Consider the cost $f$ whose level sets are shown below:

Gradient descent guides the search along the direction of steepest decrease in $f$.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong7.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong8.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong9.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong10.png)

With multidimensional weight vectors, badly chosen learning rates can cause more subtle problems.

But what if the level sets are ellipsoids instead of spheres?

* If we are lucky, we still converge quickly.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong11.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong12.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong13.png)



* If we are unlucky, convergence is very slow.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong14.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong15.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong16.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong17.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Optimization- What Can Go Wrong18.png)



## Convexity

### Convex ML Models:

Linear regression has a loss function that is convex.

With a convex function $f$, every local minimum is also a global minimum.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Convex ML models1.png)

Convex functions are ideal for conducting gradient descent.



### Convexity in 1-D

How can we tell if a 1-D function $f$ is convex?

A: second derivative is always non-negative 

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Convex ML models2.png)



What property of $f$ ensures there is only one local minimum?

* From left to right, the slope of $f$ never decreases.
  * he derivative of the slope is always non-negative.
  * the second derivative of $f$ is always non-negative.

### Convexity In Higher Dimensions

For higher-dimensional $f$, convexity is determined by the the Hessian of $f$.

$$
H[f] = \begin{bmatrix}
	\frac{\partial^{2}f}{\partial{x_{1}}\partial{x_{1}}} & ... & \frac{\partial^{2}f}{\partial{x_{1}}\partial{x_{m}}} \\ 
	... & ... & ... \\ 
	\frac{\partial^{2}f}{\partial{x_{m}}\partial{x_{1}}} &  	... & \frac{\partial^{2}f}{\partial{x_{m}}\partial{x_{m}}} \\
\end{bmatrix}
$$
For $f: \mathbb{R}^{m} \rightarrow \mathbb{R}, \; f$ is convex if the Hessian matrix is positive semi-definite for every input $x$ 

### Positive Semi-Definite

positive semi-definite (PSD): matrix analog of being “non-negative”.

A real symmetric matrix $\textbf{A}$ is positive semi-definite (PSD) if (equivalent conditions):

* All its eigenvalues are $\geq 0$
  * In particular, if A happens to be diagonal, then A is PSD if its eigenvalues are the diagonal elements.
* For every vector $v: v^{T}Av \geq 0$ 
  * Therefore: If there exists any vector $v$ such that $v^{T}Av < 0$, then A is not PSD

### Example:

Suppose:
$$
f(x, y) = 3x^{2} + 2y^{2} -2
$$
Then the first derivatives are:
$$
\frac{\partial f}{\partial x} = 6x \; \; \; \frac{\partial f}{\partial x} = 4y 
$$
The Hessian matrix is therefore:

 
$$
H[f] = 
\begin{bmatrix}
	\frac{\partial^{2}f}{\partial{x} \partial{x}} & \frac{\partial^{2}f}{\partial{x}\partial{y}} \\ 
	\frac{\partial^{2}f}{\partial{y}\partial{x}} & \frac{\partial^{2}f}{\partial{y}\partial{y}} \\
\end{bmatrix} = 
\begin{bmatrix}
	6 & 0 \\ 
	0 & 4 \\
\end{bmatrix}
$$
Notice that $H$ for this f does not depend on $(x,y)$. 

Also, $H$ is a diagonal matrix (with 6 and 4 on the diagonal). Hence, the eigenvalues are just 6 and 4. Since they are both non-negative, then $f$ is convex.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Convex ML models3.png)



### Exercise:

Recall: if $H$ is the Hessian of $f$, then f is convex if — at every $(x,y)$ — we can show (equivalently):

* $v^{T}Av \geq 0$ for every $\textbf{v}$ 
* All eigenvalues of $H$ are non-negative.



Which of the following function(s) are convex?

1. $f(x, y) = x^{2} + y + 5$

A: is convex
$$
\begin{align*}
	\frac{\partial f}{\partial x} &= 2x \\
	\frac{\partial^{2}f}{\partial x^{2}} &= 2 \\ \\ 
	\frac{\partial f}{\partial y} &= 1 \\
	\frac{\partial^{2}f}{\partial y^{2}} &= 0 \\ \\
	H &= \begin{bmatrix}
			2 & 0 \\ 
			0 & 0 \\
		\end{bmatrix} 
\end{align*}
$$


2.  $f(x, y) = x^{4} + xy + x^{2}$

A: is not convex

note: take the second derivative in terms of both $x$ and $y$ after having taking the first derivative in both $x$ and $y$. 
$$
\begin{align*}
	\frac{\partial f}{\partial x} &= 4x^{3} + y + 2x \\
	\frac{\partial f^{2}}{\partial y \partial x} &= 1 \\
	&\text{ and/or } \\
	\frac{\partial^{2}f}{\partial x^{2}} &= 12x^{2} + 2 \\ \\ \\ 
	\frac{\partial f}{\partial y} &= x \\
	\frac{\partial f^{2}}{\partial x \partial y} &= 1 \\
	&\text{ and/or } \\
	\frac{\partial^{2}f}{\partial y^{2}} &= 0 \\ \\ \\
	H &= \begin{bmatrix}
			12x^{2} + 2 & 1 \\ 
			1 & 0 \\
		\end{bmatrix} 
\end{align*}
$$
One instance where $f(x, y) = x^{4} + xy + x^{2}$ is not PSD
$$
\begin{align*}
	H &= \begin{bmatrix}
			12x^{2} + 2 & 1 \\ 
			1 & 0 \\
		\end{bmatrix} \\ \\
	x &= 1 \\ \\
	v &= \begin{bmatrix}
			-1 \\ 
			15 \\
		\end{bmatrix} \\ \\ 
	v^{T}Hv &= -16
\end{align*}
$$


### Convexity Of Linear Regression

How do we know linear regression is a convex ML model?

First, recall that, for any matrices $\textbf{A}$,  $\textbf{B}$ that can be multiplied:

* $(\textbf{AB})^{T} = \textbf{B}^{T} \textbf{A}^{T}$ 

Next, recall the gradient and Hessian of $f_{MSE}$ (for linear regression):
$$
\begin{align*}
	f_{MSE} &= \frac{1}{2n} (X^{T}w - y)^{T}(X^{T}w - y) \\ \\
	\nabla_{w} f_{MSE} &= \frac{1}{n} X(\hat{y} - y) \\ \\
	&= \frac{1}{n} X(X^{T}w - y) \\ \\
	H &= \frac{1}{n}XX^{T}
\end{align*}
$$
For any vector $v$, we have:
$$
\begin{align} 
	v^{T}XX^{T}v &= (X^{T}v)^T(X^{T}v) \\
	&\geq 0
\end{align}
$$

### Convex ML Models:

Prominent convex models in ML include linear regression, logistic regression, softmax regression, and support vector machines (SVM).

However, models in deep learning are generally not convex.

* Much DL research is devoted to how to optimize the weights to deliver good generalization performance.

## Non-Spherical Loss Functions:

As described previously, loss functions that are non-spherical can make hill climbing via gradient descent more difficult:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Non-spherical loss functions.png)

### Curvature:

The problem is that gradient descent only considers slope (1st-order effect), i.e., how $f$ changes with $\textbf{w}$.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Non-spherical loss functions2.png)



The gradient does not consider how the slope itselfchanges with $w$ (2nd-order effect).

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Non-spherical loss functions3.png)



For linear regression with cost $f_{MSE}$,
$$
\begin{align*}
	f_{MSE} &= \frac{1}{2n} (X^{T}w - y)^{T}(X^{T}w - y) \\
\end{align*}
$$
the Hessian is:
$$
H[f](w) = \frac{1}{n}XX^{T}
$$
Hence, $H$ is constant and is proportional to the (uncentered) auto-covariance matrix of $\textbf{X}$.
$$
\mathbb{E}[(X -\mathbb{E}[X]) \; (X - \mathbb{E}[X])^{T}]
$$


To accelerate optimization of the weights, we can either:

* Alter the cost function by transforming the input data.
* Change our optimization method to account for the curvature.



## Feature Transformations

### Whitening Transformations:

Gradient descent works best when the level sets of the cost function are spherical.

note: this is similar to batch normalization ("cheap version of batch normalization")

We can “spherize” the input features using a whitening transformation, which makes the auto-covariance matrix equal the identity matrix $I$.

We compute this transformation on the training data, and then apply it to both training and testing data.

We can find a whitening transform T as follows:

* Let the auto-covariance* of our training data be $XX^{T}$.

  * Note (eigen decomposition):

  $$
  \begin{align*}
  	XX^{T}v_{1} &= \lambda_{1} v_{1}  \\
  	XX^{T}v_{2} &= \lambda_{2} v_{2}  \\
  	&... \\
  	XX^{T}v_{m} &= \lambda_{m} v_{m}  \\
  \end{align*}
  $$

* We can rewrite its eigendecomposition as:
  $$
  \begin{align*}
  	\Phi &= \text{contains all eigenvectors/values *as columns} \\ 
  	\Lambda &= \text{elements of the diagnol matrix} \\ \\
  	XX^{T}\Phi &= \Phi \Lambda
  \end{align*}
  $$

  * where $\Phi$ is the matrix of eigenvectors and Λ is the corresponding diagonal matrix of eigenvalues.

* For real-valued features, $XX^{T}$ is real and symmetric; hence, $\Phi$ is orthonormal. Also, $\Lambda$ is non-negative.

* Therefore, we can multiply both sides by $\Phi^{T}$:

$$
\begin{align*}
	XX^{T}\Phi &= \Phi \Lambda \\
	\Phi^{T} XX^{T} \Phi &= \Phi^{T} \Phi \Lambda = \Lambda \\
\end{align*}
$$

Since $\Lambda$ is diagonal and non-negative, we can easily compute $\Lambda^{-\frac{1}{2}}$

We then multiply both sides (2x) to obtain $I$ on the RHS.
$$
\begin{align*}
	\Lambda^{-\frac{1}{2}^{T}}\Phi^{T} XX^{T} \Phi \Lambda^{-\frac{1}{2}} &= \Lambda^{-\frac{1}{2}^{T}} \Lambda \Lambda^{-\frac{1}{2}} \\
	(\Lambda^{-\frac{1}{2}^{T}}\Phi^{T}X)(\Lambda^{-\frac{1}{2}^{T}}\Phi^{T}X)^{T} &= I \\
	(TX)(TX)^{T} &= I 
\end{align*}
$$
This will transform the elipsesoidal landscape into a circular landscape 





## Footnotes

