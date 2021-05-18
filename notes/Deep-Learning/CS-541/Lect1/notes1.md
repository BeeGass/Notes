---
layout: post
title: Deep Learning Notes 1
author: Bryan
hasmath: "true"
custom_css: tufte
---

# Lecture 1: Intro and general overview of linear regression

Feed-forward networks consisting of multiple layers of neurons, each of which feeds to the next layer 

![2-layer Neural Network](notes\Deep Learning Notes\neural network-1.png)

2-layer Neural Network

Let dataset $D =  {(x^{(i)}, y^{(i)})}^{n}_{i=1}$

**x** relates to the to the matrix that contains $x_1$, ..., $x_m$

The output layer $\hat{y}$ computes the sum of the inputs multiplied by weights

$$
\hat{y} = g(x; w) = \sum^{m}_{i=1} x_{i} w_{i} = x^{T} w
$$

## What is "deep"?

In “classical” Machine Learning (e.g., SVMs, boosting, decision trees), f
is often a “shallow” function of **x**, e.g.:
$$
\hat{y} = f(x) = x^{T} w
$$
In contrast, with DL, f is the **composition** (possibly 1000s
of “layers” deep!) of many functions, e.g.:
$$
f(x) = f_n ( ... (f_2 (f_1 (x))))
$$
define the machine by a function *g* (with parameters w) whose output $\hat{y}$ is linear in its inputs:

$$
\hat{y} = g(x; w) = \sum^{m}_{i=1} x_{i} w_{i} = x^{T} w
$$
this is equivalent to a 2-layer neural network (with no activation function):

- $x_{i}$ refers to the input layer that can range from $x_{i}, ..., x_{m}$
- $w_i$ refers to the weights that $x_i$ is multipled by
- The combination of $x_i$ and $w_i$ creates the output of $\hat{y}$, the prediction
    - This prediction is compared to the target value in some way

