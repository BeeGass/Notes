# Logistic Regression

## Regression vs. Classification

Recall the two main supervised learning cases

* Regression: predict any real number
* Classification: choose from a finite set (e.g., {0, 1}).

So far, we have talked only about the first case.

## Binary Classification

The simplest classification problem consists of just 2 classes (binary classification), i.e., $y \; \epsilon \{ 0, 1\}$. 

One of the simplest and most common classification techniques is logistic regression.

### Using Linear Regression For Classification

During training, we penalize the linear regression model based on the MSE:
$$
\begin{align*}
	\frac{1}{2n} \underset{i=1}{\overset{n}{\sum}}(y^{(i)} - \hat{y}^{(i)})^{2}
\end{align*}
$$
Since every $y$ is either 1 or 0, why let $\hat{y}$ ever be greater than 1 or less than 0?

Why not "squash" the output to always lie in (0, 1)?

in order to do this we use the sigmoid activation function

### Sigmoid: A “Squashing” Function

A sigmoid function is an “s”-shaped, monotonically increasing and bounded function.

Here is the logistic sigmoid function $\sigma$:
$$
\sigma = \frac{1}{1+e^{-x}}
$$
![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Sigmoid.png)



## Logistic Sigmoid:

The logistic sigmoid function σ has some nice properties:

* $\sigma(-z) = 1 -\sigma(z)$

$$
\begin{align*}
	\sigma(z) &= \frac{1}{1+e^{-z}} \\ \\
	1 -\sigma(z) &= 1 - \frac{1}{1+e^{-z}} \\ \\
	&= \frac{1+e^{-z}}{1+e^{-z}} - \frac{1}{1+e^{-z}} \\ \\
	&= \frac{e^{-z}}{1+e^{-z}} \\ \\
	&= \frac{1}{\frac{1}{e^{-z}} + 1} \\ \\
    &= \frac{1}{1+e^{x}} \\ \\
    &= \sigma(-z)
\end{align*}
$$

* $\sigma^{'}(z) = \sigma(z)(1 -\sigma(z))$

$$
\begin{align*}
	\sigma(z) &= \frac{1}{1+e^{-z}} \\ \\
	\frac{\partial{\sigma}}{\partial{z}} = \sigma^{'}(z) &= -\frac{1}{(1+e^{-z})^{2}}(e^{-z} \cdot (-1)) \\ \\
	&= \frac{e^{-z}}{(1+e^{-z})^{2}} \\ \\
	&= \frac{e^{-z}}{1+e^{-z}} \cdot \frac{1}{1+e^{-z}} \\ \\
	&= \frac{1}{\frac{1}{e^{-z}} + 1} \cdot \frac{1}{1 + e^{-z}} \\ \\
    &= \frac{1}{1 + e^{z}} \cdot \frac{1}{1 + e^{-z}} \\ \\
    &= \sigma(z)(1 - \sigma(z))
\end{align*}
$$



## Logistic Regression

With logistic regression, our predictions are defined as:
$$
\hat{y} = \sigma(x^{x}w)
$$
Hence, they are forced to be in (0,1).

For classification, we can interpret the real-valued outputs as probabilities that express how confident we are in a prediction, e.g.:

Context: if we are trying to predict a smile:

* $\hat{y} = 0.95$: very confident that a face contains a smile.
* $\hat{y} = 0.58$: not very confident that a face contains a smile.

How to train? Unlike linear regression, logistic regression has no analytical (closed-form) solution.

* We can use (stochastic) gradient descent instead.
* We have to apply the chain-rule of differentiation to handle the sigmoid function.

### Gradient Descent For Logistic Regression

Let’s compute the gradient of $f_{MSE}$ for logistic regression.

For simplicity, we’ll consider a data set with just a single example so we can avoid the use of the $\sum$:
$$
\begin{align*}
	f_{MSE}(w) &= \frac{1}{2}(\hat{y} - y)^{2} \\ 
	&= \frac{1}{2}(\sigma(x^{T}w) - y)^{2} \\ 
	\nabla_{w} f_{MSE}(w) &= \nabla_{w}[\frac{1}{2}(\sigma(x^{T}w) - y)^{2}] \\
	&= x(\sigma(x^{T}w) - y) \sigma(x^{T}w)(1 - \sigma(x^{T}w))\\
    &= x(\hat{y} - y)\hat{y}(1 - \hat{y})
\end{align*}
$$

* Note: Notice the extra multiplicative terms $\hat{y}(1 - \hat{y})$ compared to the gradient for linear regression: $x(\hat{y} - y)$ 

### Attenuated Gradient

What if the weights $\textbf{w}$ are initially chosen badly, so that $\hat{y}$ is very close to 1, even though $y = 0$ (or vice-versa)?

* Then $\hat{y}(1 - \hat{y})$ is close to 0.

In this case, the gradient:
$$
\nabla_{w} f_{MSE}(w) = x(\hat{y} - y)\hat{y}(1 - \hat{y})
$$
will be very close to 0.

If the gradient is 0, then no learning will occur!

### Different Cost Function

For this reason, logistic regression is typically trained using a different cost function from $f_{MSE}$ 

One particularly well-suited cost function uses logarithms.

Logarithms and the logistic sigmoid interact well:
$$
\begin{align*}
	\frac{\partial}{\partial{w}}[log \; \sigma(x^{T}w)] &= \frac{1}{\sigma(x^{T}w)}\sigma(x^{T}w)(1 - \sigma(x^{T}w)) \\
	&= (1 - \sigma(x^{T}w)) \\
\end{align*}
$$

### Logarithm Function:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Logarithm-Function.png)



## Log Loss:

We typically use the log-loss for logistic regression:
$$
-y \; log \; \hat{y} - (1 - y) \; log(1 - \hat{y})
$$
The $y$ or $(1-y)$ “selects” which term in the expression is active, based on the ground-truth label.

### Gradient Descent For Logistic Regression With Log-Loss:

$$
\begin{align*}
	\nabla_{w}f_{log}(w) &= \nabla_{w}[-(y \; log \; \hat{y} - (1 - y) \; log(1 - \hat{y}))] \\ \\
	&= -\nabla_{w}[(y \; log \; \sigma(x^{T}w) + (1 - y) \; log(1 - \sigma(x^{T}w)))] \\ \\
	&= - (y\frac{x\sigma(x^{T}w)(1-\sigma(x^{T}w))}{\sigma(x^{T}w)} - (1 - y) \frac{x\sigma(x^{T}w)(1-\sigma(x^{T}w))}{1-\sigma(x^{T}w}) \\ \\
	&= -(yx(1-\sigma(x^{T}w)) - (1 - y)x\sigma(x^{T}w)) \\ \\
	&= -x(y - y\sigma(x^{T}w) - \sigma(x^{T}w) + \sigma(x^{T}w)) \\ \\
	&= -x(y - \sigma(x^{T}w)) \\ \\
	&= x(\hat{y} - y) \; \; \text{Same as for linear regression!}
\end{align*}
$$

### Linear Regression vs. Logistic Regression:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\logistic-regression.png)

Logistic regression is used primarily for classification even though it’s called “regression”.

Logistic regression is an instance of a generalized linear model — a linear model combined with a link function (e.g., logistic sigmoid).

* In deep learning, link functions are typically called activation functions.



## Softmax Regression (aka Multinomial Logistic Regression):

### Multi-Class Classification:

So far we have talked about classifying only 2 classes (e.g., smile versus non-smile).

* This is sometimes called binary classification.

But there are many settings in which multiple (>2) classes exist, e.g., emotion recognition, hand-written digit recognition:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\emotions.png)

6 classes (fear, anger, sadness, happiness, disgust, surprise)



![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\MNIST.png)

10 classes (0-9)



### Classification vs. Regression:

Note that, even though the hand-written digit recognition (“MNIST”) problem has classes called “0”, “1”, ..., “9”, there is no sense of “distance” between the classes.

* Misclassifying a 1 as a 2 is just as “bad” as misclassifying a 1 as a 9.

we are not trying to regress the values and determine how close they are to one another. you are either exactly right or you are not

### Multi-Class Classification(continued):

It turns out that logistic regression can easily be extended to support an arbitrary number (≥2) of classes.

* The multi-class case is called softmax regression or sometimes multinomial logistic regression.

How to represent the ground-truth $y$ and prediction $\hat{y}$?

* Instead of just a scalar $y$, we will use a vector $y$.

### Example: 2 classes

Suppose we have a dataset of 3 examples, where the ground-truth class labels are 0, 1, 0.

Then we would define our ground-truth vectors as:
$$
\begin{align*}
	y^{(1)} &= \begin{bmatrix}
					1 \\
					0 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{This "slot" is for class 0} \\
					\leftarrow \text{This "slot" is for class 1} \\
				\end{matrix} \\ \\
				
	y^{(2)} &= \begin{bmatrix}
					0 \\
					1 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{This "slot" is for class 0} \\
					\leftarrow \text{This "slot" is for class 1} \\
				\end{matrix} \\ \\
				
	y^{(3)} &= \begin{bmatrix}
					1 \\
					0 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{This "slot" is for class 0} \\
					\leftarrow \text{This "slot" is for class 1} \\
				\end{matrix} \\ 
				&\text{- Exactly 1 coordinate of each y is 1; the others are 0} \\ \\ \\
				
	\hat{y}^{(1)} &= \begin{bmatrix}
					0.93 \\
					0.07 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{Machine’s “belief” that the label is 0.} \\
					\leftarrow \text{Machine’s “belief” that the label is 1.} \\
				\end{matrix} \\ \\
				
	\hat{y}^{(2)} &= \begin{bmatrix}
					0.4 \\
					0.6 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{Machine’s “belief” that the label is 0.} \\
					\leftarrow \text{Machine’s “belief” that the label is 1.} \\
				\end{matrix} \\ \\
				
	\hat{y}^{(3)} &= \begin{bmatrix}
					0.99 \\
					0.01 \\
				\end{bmatrix} 
				\begin{matrix}
					\leftarrow \text{Machine’s “belief” that the label is 0.} \\
					\leftarrow \text{Machine’s “belief” that the label is 1.} \\
				\end{matrix} \\
	&\text{- Each coordinate of ŷ is a probability} \\
	&\text{- The sum of the coordinates in each ŷ is 1}
\end{align*}
$$


## Softmax Activation Function:

Logistic regression outputs a scalar label $\hat{y}$ representing the probability that the label is 1

* We needed just a single weight vector w, so that $\hat{y} = \sigma(x^{T}w)$.

Softmax regression outputs a $c$-vector representing the probabilities that the label is $k=1, ..., c$.

* We need $c$ different vectors of weights $w^{(1)}, ..., w^{(c)}$.
* Weight vector $w^{(k)}$ computes how much input $x$ “agrees” with class $k$.



With softmax regression, we first compute:
$$
\begin{align*}
	z_{1} &= x^{T}w^{(1)} \\
	z_{2} &= x^{T}w^{(2)} \\
	      &\text{...} \\
	z_{c} &= x^{T}w^{(c)} \\ \\
	&\text{I will refer to the z’s as “pre-activation scores”.}
\end{align*}
$$
We then normalize across all $c$ classes so that:

1. Each output $\hat{y}_{k}$ is non-negative.
2. The sum of $\hat{y}_{k}$ over all $c$ classes is 1.

## Normalization of the $\hat{y}_{k}$

1. To enforce non-negativity, we can exponentiate each $z_{k}$:
   $$
   \hat{y}_{k} = exp(z_{k})
   $$

2. To enforce that the $\hat{y}_{k}$ sum to 1, we can divide each entry by the sum:
   $$
   \hat{y}_{k} = \frac{exp(z_{k})}{\underset{k^{'}=1}{\overset{c}{\sum}}exp(z_{k^{'}})}
   $$
   ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\softmax regression diagram.png)

With softmax regression, we first compute:
$$
z_{1} = x^{T}w^{(1)}
$$
and subsequently every other node:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\softmax regression diagram2.png)
$$
\begin{align*}
	z_{1} &= x^{T}w^{(1)} \\
	z_{2} &= x^{T}w^{(2)} \\
	      &\text{...} \\
	z_{c} &= x^{T}w^{(c)} \\ \\
	
\end{align*}
$$
we then normalize across all $c$ classes. 

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\softmax regression diagram3.png)
$$
\hat{y}_{k} = P(y = k \; | \; x, w^{(1)}, ..., w^{(c)}) = \frac{exp(z_{k})}{\underset{k^{'}=1}{\overset{c}{\sum}}exp(z_{k^{'}})}
$$
This is how softmax works but how do we train it? 

We need a loss function that can support more than just 2 classes. 

Log-loss works well for logistic regression but is inherently limited to 2 classes. 

We are going to generalize to log-loss function into cross-entropy loss.

## Cross-entropy Loss

We need a loss function that can support $c \geq 2$ classes.

We will use the cross-entropy (CE) loss:
$$
f_{CE} = - \underset{i=1}{\overset{n}{\sum}} \underset{k=1}{\overset{c}{\sum}} y^{(i)}_{k} \; \text{log} \; \hat{y}^{(i)}_{k}
$$
Note that the $f_{log}$ (for logistic regression) is a special case of $f_{CE}$ (for softmax regression) for $c=2$

To see how, consider just a simple example:
$$
\begin{align*}
	f_{CE} &= - \underset{k=0}{\overset{1}{\sum}} y^{(i)}_{k} \; \text{log} \; \hat{y}^{(i)}_{k} \\ \\
	&= - y_{1} \text{log } \hat{y}_{1} - y_{0} \text{log } \hat{y}_{0} \\
	&= - y_{1} \text{log } \hat{y}_{1} - \underset{\uparrow}{(1 - y_{1})} \text{log } \underset{\uparrow}{(1 - \hat{y_{1}})} \\
	& \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \text{Recall that the sum over all coordinates} \\
	& \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \text{of each $\hat{y}$ (and each $y$) must equal 1.} \\
	& \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \text{Since there are only 2 classes, then} \\
	& \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \; \text{$\hat{y}_{0} = 1 - \hat{y}_{1}$ (and $y_{0} = 1 - y_{1}$).} \\ \\ 
	&\hat{y}^{(1)} = \begin{bmatrix}
						0.93 \\
						0.07 \\
					\end{bmatrix}
\end{align*}
$$


[12:52](https://www.youtube.com/watch?v=FVaqs3zmrv0)













































