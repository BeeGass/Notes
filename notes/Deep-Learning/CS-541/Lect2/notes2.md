---
layout: post
title: Deep Learning Notes 2
author: Bryan
hasmath: "true"
custom_css: tufte
---

# Lecture 2: Linear Regression

[TOC]



* Given our dataset D, we want to optimize **w**

We optimize **w** by choosing each "weight" $w_{j}$ to minimize the mean squared error (MSE) of our predictions

We can define the loss/cost function that we aim to minimize:
$$
\begin{align*}
	f_{MSE}(y, \hat{y}; w)&= \frac{1}{2n} \sum^{n}_{i=1}(g(\textbf{x}^{(i)};\textbf{w}) - y^{(i)})^2 \\
					   	  &= \frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2 
\end{align*}
$$

* **w** is an unconstrained real-valued vector
* to optimize for "weights" derive gradient ($\nabla$) w.r.t. **w** ($\nabla_{\textbf{w}}$), set to 0 and solve
* $f_{MSE}$ is a convex function which guarantees that the point set at 0 is a global minimum
  * determining the convexity of a function can be determined by taking the second derivative of the function and confirming that it  is positive 



## Solving For **w**:

The mean squared error (MSE)[^1]
$$
\begin{align}
	f_{MSE}(y, \hat{y}; w) = \frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2
\end{align}
$$
The gradient of the MSE[^4] w.r.t. **w**
$$
\begin{align*}
	\nabla_{\textbf{w}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{w}}[\frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2] \\ \\
	
&\text{by linearity of differention we can move the gradient inside the sum } \\
	 
&= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2] \\ \\

&\text{Apply chain rule} \\
	
&= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})] \\

&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\ \\

&\text{Now that we have solved for w, set to zero to optimize in respect to it} \\

0&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\

0&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} - \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} \\

&\text{Factor out the x} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}) \textbf{w} \\

(\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \textbf{w} \\

\textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\
	
\end{align*}
$$
MSE w.r.t. **w** in matrix notation[^5]. This is also known as linear regression:
$$
\begin{align*}
	\textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\

&\text{X is called the design matrix: } \textbf{X} = \begin{bmatrix} 
														x^{(1)} & ... & x^{(n)}								
													 \end{bmatrix} \\
&\text{y contains all the training labels: } \textbf{y} = \begin{bmatrix} 
															y^{(1)} \\
                                                            ... \\
                                                            y^{(n)} \\
													 	  \end{bmatrix} \\
&\text{In matrix notation this will look like} \\

\textbf{w} &= (\textbf{X} \textbf{X}^{T})^{-1}(\textbf{X}y)\\
	
\end{align*}
$$
Please refer to bishops section on solving for **w** for a more comprehensive proof

## Review Animations:

Linear Regression Animation:

<insert manim animation on linear regression>

Mean Square Error Animation:

<insert manim animation on MSE>

## Bias Term:

In order to account for target values ($\hat{y}$) with non-zero mean[, we can add a bias term to our model:
$$
\begin{align*}
	&\text{previously we had:} \\
	\hat{y} &= \textbf{x}^{T}\textbf{w} \\\\
	&\text{now with bias term we have:} \\
	\hat{y} &= \textbf{x}^{T}\textbf{w} + b
\end{align*}
$$
this now means that the gradient of MSE with a bias term[^2] can be computed w.r.t. **w** and b. 

* MSE without bias term:
  $$
  \begin{align*}
  	f_{MSE}(y, \hat{y}; w) = \frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2
  \end{align*}
  $$

* MSE with bias term[^2]:
  $$
  \begin{align}
  	f_{MSE}(y, \hat{y}; w) = \frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2
  \end{align}
  $$

* gradient of MSE w.r.t **w** (including b)[^7]: 
  $$
  \begin{align*}
  	\nabla_{\textbf{w}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{w}}[\frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\ \\
  	
  &\text{by linearity of differention we can move the gradient inside the sum } \\
  	 
  &= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\ \\
  
  &\text{Apply chain rule} \\
  	
  &= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})] \\
  
  &= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\ \\
  
  &\text{Now that we have solved for w, set to zero to optimize in respect to it} \\
  
  0&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\
  
  0&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} - \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)} \\
  
  \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} \\
  
  &\text{Factor out the x} \\
  
  \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}) \textbf{w} \\
  
  (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \textbf{w} \\
  
  \textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\
  	
  \end{align*}
  $$

* gradient of MSE w.r.t. b[^6]:
  $$
  \begin{align*}
  	\nabla_{\textbf{b}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{b}}[\frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\
  	 
  &= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{b}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\
  	
  &= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{b}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})] \\
  
  &= \frac{1}{n} \sum^{n}_{i=1} ((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)}) \\ \\
  
  &\text{Now that we have solved for b, set to zero to optimize in respect to it} \\
  
  0&= \frac{1}{n} \sum^{n}_{i=1} ((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)}) \\
  
  b&= \frac{1}{n} \sum^{n}_{i=1} (y^{(i)} - (\textbf{x}^{(i)^{T}}\textbf{w})) \\
  	
  \end{align*}
  $$

## Gradient Descent:

gradient descent is a hill climbing algorithm that uses the gradient (aka slope) to decide which way to "move/step" **w** to reduce the objective function (e.g. $f_{MSE}$) 

insert picture of graphical landscape>

However, how do we know how large these steps should be? This can be determined by the maginitude of the gradient which gives an indication of how far we need to go to reach the optimal **w**

<insert manim animation for gradient descent>

This may not always be true however. There are certain graphical landscapes that dont correctly indicate the direction and size of step to move toward. 

<Show examples of landscapes where this may not be true>

When performing gradient descent initialize **w** randomly. That is, the "weights" should start with random values that incrementally improve as iterations continue  

### Gradient Descent w.r.t. **w**:

$\textbf{w}_{new} = \textbf{w}_{previous} - \epsilon * \nabla_{w}$

* $\nabla_{w}$ is the gradient w.r.t. **w**
* $\epsilon$ is the learning rate. This determines the step size. This is a hyperparameter 
* $\textbf{w}_{previous}$ is the most current weight value the model is using
* $\textbf{w}_{new}$ is the new, updated, weight value  

### Gradient Descent w.r.t. b:

$\textbf{b}_{new} = \textbf{b}_{previous} - \epsilon * \nabla_{b}$

* $\nabla_{b}$ is the gradient w.r.t. **b**
* $\epsilon$ is the learning rate. This determines the step size. This is a hyperparameter 
* $\textbf{b}_{previous}$ is the most current bias value the model is using
* $\textbf{b}_{new}$ is the new, updated, bias value  

## Stochastic Gradient Descent:

With gradient descent, we only update the weights after scanning the entire training set, this is slow.

stochastic gradient descent generalizes the gradient to a smaller subset of the training set. Thus avoid overfitting to a landscape if we were to otherwise take the gradient over the entire dataset. This generalization behaves as noise that can normalize our landscape. 

<insert diagram of possible overfitting scenario>

### Pseudo Code For SGD

```pseudocode
size_of_batch = number of your choosing
num_of_epoch = number of your choosing
def stochastic_gradient_descent():
    for e in num_of_epoch:
        for batch_of_x, batch_of_y in get_batches(x, y, size_of_batch):
            perform gradient descent w.r.t. w
            perform gradient descent w.r.t. b

    return w, b
    
def get_batches(x, y, size_of_batch):
	randomly sample a mini-batch for both x and y of size: size_of_batch
	return batch_of_x, batch_of_y
```

### Python/Pseudo Code Example Of SGD

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = #hyperparameter, N size
learning_rate = #hyperparameter, N size, ex. 0.00001 
n_samples = #number of data points
n_features = #number of features
n_targets = #number of target values associated with the data points feature
n_epochs = #hyperparameter, N size
n_runs = #number of times to run SGD given same dataset (train/val/test)

for run in range(n_runs):
    #prepare data
    #split data into feature and targets
    X = np.random.normal(size=(n_samples, n_features))
    y = np.random.normal(size=(n_samples, n_targets))
    
    #initialize parameters with random values
    W = np.random.normal(size=(n_features, n_targets))
    b = np.random.normal(size=(n_targets, n_targets))
    
    # keep track of errors
    errors = []
    
    for epoch in range(n_epochs):
        # randomly shuffle the data in a way where the original set in not altered 
        permutation_indices = np.random.permutation(X.shape[0])
        permuted_X = X[permutation_indices]
        permuted_y = y[permutation_indices]
        
        for batch_index in range(0, n_samples, batch_size):
            # prepare batches
            batch_X = permuted_X[batch_index:batch_index+batch_size]
            batch_y = permuted_y[batch_index:batch_index+batch_size]
            
            # forward pass
            y_hat = np.dot(batch_X, W) + b
            
            # compute error
            error = batch_y - y_hat
            
            # update
            W += batch_X.T.dot(error) * learning_rate
            b += np.mean(batch_y - y_pred) * learning_rate

            # bookkeeping
            errors.append(np.mean(np.abs(error)))
            
         # plot run
    plt.plot(errors/np.mean(errors))

# save plots
plt.savefig('plot.png')   
```

Each batch will generalize the landscape given the data points of that batch.

### SGD - Learning Rates:

a static learning rate can result in unfavorable results. In the event where learning rate is too high then divergence can occur by "bouncing" out of local minimum area. In the event where learning rate is too small then progress can be so slow that convergence toward local minimum will take too long. 
$$
\begin{align*}
	\lim_{T \rightarrow \infin }&{\sum^{T}_{t=1} \lvert \epsilon_{t}\rvert^{2} < \infin} \\ \\
	&\text{or} \\ \\
	\lim_{T \rightarrow \infin }&{\sum^{T}_{t=1} \lvert \epsilon_{t}\rvert = \infin} \\
\end{align*}
$$

* one common learning rate "schedule"  is to multiply $\epsilon$ by c $\in$ (0, 1) every k rounds
  * this is called exponential decay
* There are many strategies around updating epsilon

## Probabilistic Machine Learning:

Probabilities provide a natural way of expressing our **uncertainty** about a particular value e.g.:

* The ground-truth y we are trying to estimate 
* Our estimate $\hat{y}$ of the ground-truth

### Frequentist Probabilities:

Example:

* Ask a large group of randomly selected people to label the face as smiling or not
* Count the number of labels for "smile" and divide by the total number of labels
* the ratio is the probability of "smile" for that face image

### Bayesian Probabilities:

Example:

* Ask one person how much she/he believes the image is smiling, quantified as a number between 0 and 1.
* The "belief" score is the probability of "smile" for that face image

## Footnotes

[^1]: mean squared error without bias

$$
\begin{align}
	f_{MSE}(y, \hat{y}; w) = \frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2
\end{align}
$$

[^2]: mean squared error with bias:

$$
\begin{align}
	f_{MSE}(y, \hat{y}; w) = \frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2
\end{align}
$$

[^3]:predicted value without and with bias term: 

$$
\begin{align*}
	&\text{previously we had:} \\
	\hat{y} &= \textbf{x}^{T}\textbf{w} \\\\
	&\text{now with bias term we have:} \\
	\hat{y} &= \textbf{x}^{T}\textbf{w} + b
\end{align*}
$$

[^4]:Derivation of MSE w.r.t. **w**

$$
\begin{align*}
	\nabla_{\textbf{w}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{w}}[\frac{1}{2n} \sum^{n}_{i=1}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2] \\ \\
	
&\text{by linearity of differention we can move the gradient inside the sum } \\
	 
&= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})^2] \\ \\

&\text{Apply chain rule} \\
	
&= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)})] \\

&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\ \\

&\text{Now that we have solved for w, set to zero to optimize in respect to it} \\

0&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\

0&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} - \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} \\

&\text{Factor out the x} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}) \textbf{w} \\

(\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \textbf{w} \\

\textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\
	
\end{align*}
$$

[^5]: MSE w.r.t. **w** in matrix notation. This is known as linear regression:

$$
\begin{align*}
	\textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\

&\text{X is called the design matrix: } \textbf{X} = \begin{bmatrix} 
														x^{(1)} & ... & x^{(n)}								
													 \end{bmatrix} \\
&\text{y contains all the training labels: } \textbf{y} = \begin{bmatrix} 
															y^{(1)} \\
                                                            ... \\
                                                            y^{(n)} \\
													 	  \end{bmatrix} \\
&\text{In matrix notation this will look like} \\

\textbf{w} &= (\textbf{X} \textbf{X}^{T})^{-1}(\textbf{X}y)\\
	
\end{align*}
$$

[^6]:MSE w.r.t. b:

$$
\begin{align*}
	\nabla_{\textbf{b}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{b}}[\frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\
	 
&= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{b}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\
	
&= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{b}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})] \\

&= \frac{1}{n} \sum^{n}_{i=1} ((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)}) \\ \\

&\text{Now that we have solved for b, set to zero to optimize in respect to it} \\

0&= \frac{1}{n} \sum^{n}_{i=1} ((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)}) \\

b&= \frac{1}{n} \sum^{n}_{i=1} (y^{(i)} - (\textbf{x}^{(i)^{T}}\textbf{w})) \\
	
\end{align*}
$$



[^7]:MSE w.r.t. w (including b):

$$
\begin{align*}
	\nabla_{\textbf{w}}f_{MSE}(y, \hat{y}; w) &= \nabla_{\textbf{w}}[\frac{1}{2n} \sum^{n}_{i=1}((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\ \\
	
&\text{by linearity of differention we can move the gradient inside the sum } \\
	 
&= \frac{1}{2n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})^2] \\ \\

&\text{Apply chain rule} \\
	
&= \frac{1}{n} \sum^{n}_{i=1} \nabla_{\textbf{w}}[((\textbf{x}^{(i)^{T}}\textbf{w} + b) - y^{(i)})] \\

&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\ \\

&\text{Now that we have solved for w, set to zero to optimize in respect to it} \\

0&= \frac{1}{n} \sum^{n}_{i=1} \textbf{x}^{(i)}(\textbf{x}^{(i)^{T}}\textbf{w} - y^{(i)}) \\

0&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} - \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} \textbf{x}^{(i)}\textbf{x}^{(i)^{T}}\textbf{w} \\

&\text{Factor out the x} \\

\sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}) \textbf{w} \\

(\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}&= \textbf{w} \\

\textbf{w} &= (\sum^{n}_{i=1} (\textbf{x}^{(i)}\textbf{x}^{(i)^{T}}))^{-1} \sum^{n}_{i=1}\textbf{x}^{(i)}y^{(i)}\\ \\
	
\end{align*}
$$







