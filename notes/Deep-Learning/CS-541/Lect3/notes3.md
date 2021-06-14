---
layout: post
title: Deep Learning Notes 3
author: Bryan
hasmath: "true"
custom_css: tufte
---

# Lecture 3: 

[TOC]

## Quick Note:

If you want a more detailed overview of material after this lecture please read chapters 1, 2 and 5 in the Deep Learning Book

## Gradient Descent

Problem:

For the 2-layer neural network below, let m=2 (the number of input neurons or features) and $w^{(0)} = [1 \; 0]^{T}$. 

Compute the updated weight vector $w^{(1)}$ after one iteration of gradient descent using MSE loss, a single training example $(x, y) = ([2, 3]^{T}, 4)$, and learning rate $\epsilon = 0.1$. 

Attempt:
$$
\begin{align*}
	w^{(0)} &= [1 \; 0]^{T} \\ \\
	
	\textbf{w}^{(1)} &= (\textbf{X} \textbf{X}^{T})^{-1}(\textbf{X}y) \\ \\
	
	\textbf{w}^{(2)} &= \textbf{w}^{(0)} - \epsilon * \nabla_{w^{(1)}}f_{MSE}(\textbf{w}) \\ \\
	
	\textbf{w}^{(2)} &= \textbf{w}^{(0)} - \epsilon * \frac{1}{n}(\textbf{X}(\textbf{X}^{T}\textbf{w} - y)) \\ \\
	
	\textbf{w}^{(2)} &= [1 \; 0]^{T} - 0.1 * ([2, 3]([2, 3]^{T} \cdot [0 \; 1] - [4])) \\ \\
	
	\textbf{w}^{(2)} &= \begin{bmatrix} 
															1 + 0.1 * 2 * 2 \\
                                                            0 + 0.1 * 3 * 2 \\
													 	  \end{bmatrix} \\ \\
	
	\textbf{w}^{(2)} &= \begin{bmatrix} 
															1.4 \\
                                                            0.6 \\
													 	  \end{bmatrix} \\ \\
	
	
\end{align*}
$$

## Probabilistic Machine Learning

Sometimes we may be very uncertain about our prediction of the target value y from the input x.

![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob1.png)

Here you can see that the point's y value contains a lot of uncertainty due in part from the multiple points that lie at similar x values. This regression does not accurately represent the data given. 

Instead we should use a predicative distribution for our regression as shown here

![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob2.png)

it turns out that the optimal parameters for a conditional Gaussian probability model are exactly the same as for linear regression with minimal MSE

## Probabilistic Deep Learning Intro

Neural Networks can be used in various ways to make probabilistic predictions:

* For regression, estimate both the expected value and the variance of the prediction
* Model a high-dimensional distribution using a probabilistic latent variable model (LVM) - akin to factor analysis but deeper

## Definitions:

### Random Variables

A random variable (denoted sometimes as **RV** ) usually takes the form like $X$ (with sample space $\Omega$) has a value we are unsure about, maybe because 

- it is decided by some random process
- it is hidden from us



* RV's are typically written as capital letters, e.g. $X$, $Y$
* Once the value of the RV, $X$, has been "sampled", "“selected", "instantiated", or "realized" (by a random number generator, generative process, God, etc.), it takes a specific value from the sample space 
* The values the RV can take are typically written as lowercase letters, e.g., $x$, $y$.



Types of sample spaces $\Omega$:

- Finite, e.g.:
  - $\{ 0, 1 \}$
  - $\{ \text{red, blue, green} \}$
- Countable, e.g.:  
  - $\Bbb{Z}_{\geq 0}$
- Uncountable, e.g.:
  - $\Bbb{R}$



The probability that a random variable $X$ takes a particular value is determined by a:

* Probability mass function (PMF) for finite or countable sample spaces.
* Probability density function (PDF) for uncountable sample spaces.

## PMF

Example 1 (finite):

* Let RV $X$ be the outcome of rolling a 6-sided die.
* If $X$ is fair, then: 

$$
P(X = i) = \frac{1}{6} \; \forall i \in \{1, ..., 6 \}
$$

Example 2 (countable): 

* Let RV $X$ be the number of TCP/IP packets that arrive in 1 second.
* We can model the count of packets with a Poisson distribution:

$$
P(X = k) = \frac{\lambda^{k} e^{-\lambda}}{k!}
$$

where parameter $\lambda$ specifies the rate of the packet arrivals

 ![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob3.png)





* Example 1:

  * let $X$ be a uniformly-distributed RV over $\Omega = [0, 1]$.
  * Then $f_{X}(x) = 1 \; \forall x \in \Omega$

  ![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob4.png)



* Example 2:

  * Let Y be a uniformly-distributed RV over $\Omega = [\frac{1}{4},  \frac{3}{4}]$ 
  * Then $f_{Y}(y) = 2 \; \forall y \in \Omega$ 

  ![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob5.png)

  **Note that the PDF can exceed 1** due in part because the integral must be equal to 1 in all cases



* Example 3:

  * Let $Z$ be a **normally** (aka Gaussian) distributed RV with mean 1.5 (location parameter) and variance 4 (width parameter), i.e.,

  $$
  Z \sim \mathcal{N}(z; \mu = 1.5, \sigma^{2} = 4)
  $$

  *   Then

  $$
  f_{Z}(z) = \frac{1}{\sqrt{2\pi \sigma^2}} \; \text{exp}(-\frac{(z - \mu)^{2}}{2\sigma^{2}})
  $$

  ![](C:\Users\Bryan\Pictures\DL Notes Pics\lec3-Prob6.png)

### Side Note:

* In this course, we will relax notation and use “probability distribution” to mean either the PDF or PMF of a RV (as appropriate)
* As a notational shortcut, we use $P(x)$ to mean $P(X=x)$ or $f_{X}(X)$ 

## Joint Probability Distributions

* For multiple random variables $X, Y, ...,$ we can construct the joint probability distribution $P(x, y, ...)$ to mean the probability that $(X=x) \and (Y=y) \; \and \; ....$ 
  * $P(x, y, ...)$ must sum to one 
* Note that $P$ must still sum to 1 over all possible joint values $(x, y, ...)$.



Example in 2-D -- crayons:

* Let $X$ be the color (red, blue, green, white)
* Let $Y$ be the intensity (low, medium, high)

$$
\begin{align*}
	\begin{matrix}
		\text{} & \text{Red} & \text{Blue} & \text{Green} & \text{White} \\
        \text{Low} & 0.1 & 0.05 & 0.025 & 0.2 \\
        \text{Med} & 0.075 & 0.05 & 0.1 & 0 \\
        \text{High} & 0.25 & 0.05 & 0.075 & 0.025 \\
	\end{matrix}
\end{align*}
$$

Q: What is the overall probability of picking a white canyon at random?

A: 0.225



From the joint distribution we can compute the marginal distributions $P(x)$ and $P(y)$.
$$
\begin{align}
    P(x) &= \sum_{y} P(x, y) \\ \\
    P(y) &= \sum_{x} P(x, y)
\end{align}
$$

$$
\begin{align*}
	\begin{matrix}
		\text{} & \text{Red} & \text{Blue} & \text{Green} & \text{White} & P(y) \\
        \text{Low} & 0.1 & 0.05 & 0.025 & 0.2 & 0.375\\
        \text{Med} & 0.075 & 0.05 & 0.1 & 0 & 0.225\\
        \text{High} & 0.25 & 0.05 & 0.075 & 0.025 & 0.4\\
        \text{P(x)} & 0.425 & 0.15 & 0.2 & 0.225
	\end{matrix}
\end{align*}
$$
This is also called the law of total probability:

### Law Of Total Probability

* For any RVs $X$ and $Y$:
  $$
  P(x) = \sum_{y} P(x, y)
  $$

  ### Joint Probability Distributions Continued

  In machine learning, we often use joint distributions of many variables that are part of a collection, e.g.:

* Sequence $(W_{1}, W_{2}, ..., W_{T})$ of words in a sentence

  * $W_{t}$ is the $t^{th}$ RV in the sequence (representing a word that "you" picked at random from a bag of words)

* Grid $(I_{11}, ..., I_{1M}, ..., I_{N1}, ..., I_{NM})$ of the pixels in an N x M image.

## Conditional Probability Distributions

Sometimes the value of one RV is predictive of the value of another RV

Examples:

* If I know a person’s height $H$, then I have some information about their weight $W$.
* If I know how much cholesterol $C$ a person eats, then I have some information about their chance of coronary heart disease $D$ 

We can form a conditional probability distribution of RV $X$ given the value of RV $Y$:
$$
P(x \;| \;y)
$$
the bar in between $x$ and $y$ meaning "conditional on" or "given" 



Examples:

* Height given weight: $P(h \; |\; w)$
* Heart disease given cholesterol: $P(d \; |\; c)$ 

More generally, we can form a conditional probability distribution of $X_{1}, ..., X_{n}$ given the values of  $Y_{1}, ..., Y_{m}$: 
$$
P(x_{1}, ..., x_{n} \;|\; y_{1}, ..., y_{m})
$$
A conditional probability distribution is related to the joint probability distribution as follows:
$$
P(x \; | \; y) P(y) = P(x, y)
$$
It follows that:
$$
P(x \; |\; y, z)P(x \; |\; y) = P(x, y \; | \; z)
$$
More generally:
$$
P(x_{1}, ..., x_{n} \; |\; y_{1}, ..., y_{m})P(y_{1}, ..., y_{m}) = P(x_{1}, ..., x_{n}, y_{1}, ..., y_{m})
$$
And also:
$$
\begin{align*}
    P(x_{1}, ..., x_{n} \; |\; y_{1}, ..., y_{m}, z_{1}, ..., z_{p}) P(y_{1}, ..., y_{m} \; &|\; z_{1}, ..., z_{p})& \\ = P(x_{1}, ..., x_{n}, y_{1}, ..., y_{m}) \; &| \; z_{1}, ..., z_{p})
\end{align*}
$$
Note that the same joint probability can be factored in different ways, e.g.:
$$
\begin{align*}
	P(x,y,z) &= P(x, y \; | \; z)P(z) \\ \\
	&= P(x\; | \; y, z)P(y, z)
\end{align*}
$$

### Exercises:

1. $P(a, b, c, d) = P(a, c) \; * \; ?$

   A: 
   $$
   P(a,c) \; * \; P(b,d \; | \; a, c)
   $$
   
2. $P(W_{1}, W_{2}, W_{3}) = P(W_{3}\; |\; W_{1}) \; * \; ? \; * \; ?$

   A: 
   $$
   P(W_{3}\; |\; W_{1}) \; * \; P(W_{1}) \; * \; P(W_{2} \; |\; W_{1}, W_{3})
   $$
   
3. $P(X_{1}, X_{2}, X_{3}) = P(X_{1}) \; * \; ? \; * \; P(X_{3}\; | \; X_{1}, X_{2})$ 

   A:
   $$
   P(X_{1}, X_{2}, X_{3}) = P(X_{1}) \; * \; P(X_{2} \; | \; X{1}) \; * \; P(X_{3}\; | \; X_{1}, X_{2})
   $$
   
4. $P(X_{1}, ..., X_{n}) = P(X_{1}) \; * \; \underset{\text{n-1 terms}}{? \; * \; ? \; * \; ... \; * \; ?}$ 

   A:
   $$
   P(X_{1}, ..., X_{n}) = P(X_{1}) \; * \; \underset{i=2}{\overset{n}{\Pi}}P(X_{i} \; | \; X_{1}, ..., X_{i-1})
   $$

## Independence

RVs $X$ and $Y$ are independent i.f.f. $P(x,y) = P(x)P(y) \; \forall x, y,$ i.e., the joint distribution equals the product of the marginal distributions. 

Note that this implies that $P(x \; |\;  y) = P(x)$ and $P(y \; | \: x) = P(y)$ since $P(x,y) = P(x \; | \; y )P(y) = P(y \; | \; x)P(x)$ by definition of conditional probability.

* in simpler terms the above means that probability of $x$ given $y$ is simply the probability of $x$ if $y$ completely independent of $x$ and gives no further information toward the probability of $x$. The same is true for probability of $y$. 

## Conditional independence

RVs $X$ and $Y$ are conditionally independent given RV Z iff:
$$
P(x,y \; |\; z) = P(x \; | \; z)P(y \; | \; z) \; \forall x, y,z
$$
Note that this implies:
$$
P(x \; | \; y, z) = P(x \; | \; z)
$$
In words: “If I already know the value of $Z$, then knowing $Y$ tells me nothing further about $X$”

### Generalized Form:

More generally: $X_{1}, ..., X_{n}$ and $Y_{1}, ..., Y_{m}$ are conditionally independent given $Z_{1}, ..., Z_{p} iff:$
$$
\begin{align*}
    P(x_{1}, ..., x_{n}, y_{1}, ..., y_{m} \; &| \; z_{1}, ..., z_{p}) \\ = P(x_{1}, ..., x_{n} \; |\; z_{1}, ..., z_{p})P(y_{1}, ..., y_{m} \; &| \; z_{1}, ..., z_{p})
\end{align*}
$$

## Bayes' rule

It is often useful to compute $P(x \; | \; y)$ in terms of $P(y \; | \; x)$. 

* example
  * if $X$ represents a student’s skill level, and $Y$ is their test score, it’s often easier to compute $P(y \; | \; x)$. But given a student’s test score $Y$, we really want to know $P(x \; | \; y)$. 

Bayes' rule:
$$
P(x \; | \; y) = \frac{P(x, y)}{P(y)} = \frac{P(y \; | \; x)P(x)}{P(y)}
$$
We can also generalize Bayes’ rule to cases where we always condition on a tertiary variable $Z$:
$$
P(x \; | \; y, z) =  \frac{P(y \; | \; x, z)P(x \; | \; z)}{P(y \; | \; z)}
$$
It is sometimes possible — and more convenient — to work with **unnormalized** probabilities

For instance, it might suffice to know that

$[P(y^{(1)} \; | \; x), \; P(y^{(2)} \; | \; x), \; P(y^{(3)} \; | \; x)] \; \propto [3.5, 7, 0.04]$ 

rather than their exact (normalized) values.

## Probabilistic Inference

To express the conditional independence relationships between multiple RVs, it is useful to represent their dependencies in a graph.

A formal theory of probabilistic graphical models (Pearl 1998) has been devised.

* Conditional independence can be determined via the principle of **d-separation** (beyond the scope of this course).

### Probabilistic graphical models

Example 1

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Probabilistic Graph.png)

* C: whether the patients eats caviar.
* S: the patient’s sex
* H: whether the patient has high cholesterol
* A: whether the patient will have a heart attack.
* B: whether the patient has shortness of breath.



* This model implies that:

$$
\begin{align*}
	P(a, b \; | \; h,c,s) &= P(a, b \; | \; h) \text{ and} \\ 
	P(c, s \; | \; h, a, b) &= P(c, s \; | \; h)
\end{align*}
$$

* In words, “If I want to know the probability the patient will have a heart attack A, and I already know the patient has high cholesterol H, then the patient’s sex and whether she/he eats caviar C is irrelevant.”



Example 2: Markov Chain

This graph shows elements of a time series

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Probabilistic Graph (1).png)

* This chain-like model of $X_{1}, ..., X_{n}$ implies that:

$$
\begin{align*}
	P(x_{i} \; | \; x_{1}, ..., x_{i-1}) &= P(x_{i} \; | \; x_{i-1}) \text{ and} \\
	P(x_{i} \; | \; x_{i+1}, ..., x_{n}) &= P(x_{i} \; | \; x_{i+1})
\end{align*}
$$

In words, “If I want to know the value of $X_{i}$ and I already know $X_{i-1}$, then the values of any 'earlier' $X's$ are irrelevant." 



Example 3

* Given a model with multiple RVs and how they are related to each other, we can infer the values of other RVs.
* For the medical diagnosis example, suppose we knew the conditional probability distributions:

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Probabilistic Graph.png)

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\table.png)
$$
P(H=h \; | \; C=c, S=s)
$$
![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\table1.png)

Example 4

* Suppose we meet a male patient who eats caviar.
* What is the **posterior probability** that H=1, i.e., $P(H=1 \; | \; C=1, S=Ma)$? (Posterior means after observing C, S.) (given the diagram and tables from example 3)

A: $0.6$ 

* What if we also know that the patient is short of breath?

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\table2.png)

* conditional independence from the graphical model

$$
P(b \; | \; h, c, s) = P(b \; | \; h)
$$



$$
\begin{align*}
&P(H=1 \; | \; C=1, S=Ma, B=1) \\ \\
= \; &\frac{P(B=1 \; | \; H=1, C=1, S= Ma)P(H=1 \; | \; C=1, S= Ma)}{P(B=1 \; | \; C=1, S= Ma)} \text{ Bayes' Rule} \\ \\
= \; &\frac{P(B=1 \; | \; H=1)P(H=1 \; | \; C=1, S= Ma)}{P(B=1 \; | \; C=1, S= Ma)} \text{ Conditional independence} \\ \\
= \; &\frac{0.9 * 0.6}{\sum^{1}_{h=0} P(B=1, H=h \; | \; C=1, S= Ma)} \text{ Law of total probability} \\ \\
= \; &\frac{0.54}{\sum^{1}_{h=0} P(B=1 \; | \; H=h, C=1, S= Ma)P(H=h \; | \; C=1, S= Ma)} \text{ Def. of cond. prob.} \\ \\
= \; &\frac{0.54}{P(B=1 \; | \; H=h)P(H=h \; | \; C=1, S= Ma)} \text{ Conditional independence} \\ \\ 
= \; &\frac{0.54}{0.1 * 0.4 * 0.9 * 0.6} \\ \\
= \; &\frac{0.54}{0.04 * 0.54} \\ \\
\approx \; &0.93

\end{align*}
$$
Alternatively, it is often easier to work with unnormalized probabilities, i.e., values proportional to the probabilities
$$
\begin{align*}
	& \;P(H=1 \; | \; C=1, S=Ma, B=1) \\ \\
	&\propto P(B=1 \;| \;H=1)P(H=1 \; | \; C=1, S=Ma) \\ \\
	&= 0.9 * 0.6 \\ \\ \\ 
	&= P(H=0 \; | \; C=1, S=Ma, B=1) \\ \\
	&\propto P(B=1 \; | \; H= 0)P(H=0 \; | \; C=1, S=Ma) \\ \\
	&= 0.1 * 0.4 \\ \\
	
	&P(H=1 \; | \; C=1, S=Ma, B=1) = \frac{0.9 * 0.6}{0.9 * 0.6 * 0.1 * 0.4} = 0.93 \\ \\
	&P(H=0 \; | \; C=1, S=Ma, B=1) = \frac{0.1 * 0.4}{0.9 * 0.6 + 0.1 * 0.4} = 0.07 \\ \\ 
    &\text{since the probabilities must sum to 1}
\end{align*}
$$

## Maximum Likelihood Estimation (MLE)

### Parameters In Probability Distributions:

Most probabilistic models have parameters we want to estimate.

For example, the conditional probabilities for medical diagnosis are all parameters that must be learned.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\table3.png)

Most probabilistic models have parameters we want to estimate.

As another example, we might want to estimate the bias B of a coin after observing n coin flips $H_{1}, ..., H_{n}$: 

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Probabilistic Graph (2).png)
$$
\begin{align*}
	P(H_{i} = 1 \; &| \; b) = b \\
    \text{Conditional} &\text{ independence:} \\
    P(h_{i} \; | \; b, h_{1}, ..., h_{i-1}&, h_{i+1}, ...,h_{n}) =P(h_{i} \; | \; b)
\end{align*}
$$
What is a principled approach to estimating B?

Maximum likelihood estimation (MLE):

* The value of a latent variable is estimated as the one that makes the observed data as likely (probable) as possible.

the likelihood of $H_{1}, ..., H_{n}$ given B is:
$$
\begin{align*}
	&=P(h_{1}, ..., h_{n} \; | \; b) = P(h_{1} \; | \; b) \; \overset{n}{\underset{i=2}{\Pi}} P(h_{i} \; | \; b, h_{1}, ..., h_{i-1}) \\
	&=P(h_{1} \; | \; b) \overset{n}{\underset{i=2}{\Pi}} P(h_{i} \; | \; b) \; \text{ Conditional independence} \\
	&= \overset{n}{\underset{i=1}{\Pi}} P(h_{i} \; | \; b)
\end{align*}
$$
We can express the probability of each $h_{i}$ given $b$ as:
$$
\begin{align*}
	P(h_{i} \; | \; b) &= b^{h_{i}}(1-b)^{1-h_{i}} \\
	&= b \text{ if } h_{i} = 1 \text{ or} \\
	&\; \; \; \; (1-b) \text{ if } h_{i} = 0 \\
\end{align*}
$$
The exponent “chooses” the correct probability for $H_{i} = 1$ or $H_{i} = 0$.

We seek to maximize the probability of $h_{1}, ..., h_{n}$ by optimizing $b$.

It’s often easier instead to optimize the log-likelihood.
$$
\begin{align*}
	&\text{arg } \underset{b}{\text{max}}P(h_{1}, ..., h_{n} \; | \; b) = \text{arg } \underset{b}{\text{max}} \text{ log}P(h_{1}, ..., h_{n} \; | \; b)^{*} \\
&\text{assuming the probability is never exactly 0}
\end{align*}
$$

$$
\begin{align*}
	\text{log } P(h_{1}, ..., h_{n} \; | \; b) &= \text{log } \overset{n}{\underset{i=1}{\Pi}} P(h_{i} \; | \; b) \; \text{ due to conditional independence} \\ 
	&= \sum^{n}_{i=1} \text{ log}P(h_{i} \; | \; b) \\
	&= \sum^{n}_{i=1} \text{ log }b^{h_{i}}(1-b)^{1-h_{i}} \\
	&= \sum^{n}_{i=1} h_{i} \text{ log }b \; + \;(1-h_{i}) \text{ log }(1-b) \\
	n_{1} \text{ is number of heads. } \;\;\;\; \;\;\;\; &= n_{1} \text{log }b \; + \;(n-n_{1})\text{ log }(1-b)
\end{align*}
$$

We can now differentiate w.r.t. b, set to 0, and solve to obtain the MLE of B

:
$$
\begin{align*}
	\nabla_{b} [n_{1}\text{ log } b \; + \; (n -n_{1}) \; \text{log }(1-b)] &= \frac{n_{1}}{b} - \frac{(n - n_{1})}{1 - b} \\
	(1-b) n_{1} - b(n - n_{1}) &= 0 \\
	n_{1} - bn_{1} - bn \; + \;bn_{1} &= 0 \\
    n_{1} &= bn \\
    b &= \frac{n_{1}}{n} \\
    &\text{The MLE for B is the fraction of coin flips that are heads.}
	
\end{align*}
$$

## Linear-Gaussian Models

Let’s consider a different model that contains real-valued RVs (not just from a finite sample space).

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Linear-Gaussian Model.png)

* $X$ is some feature vector (e.g., face image).
* $Y$ is some outcome variable (e.g., age).
* $W$ is a vector of weights that characterize how $Y$ is related to $X$. 
* $\sigma$  expresses how uncertain we are about $Y$ after seeing $X$. 

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\Linear-Gaussian Model 1.png)

Suppose we model the relationship between X, W, σ, and Y such that:

* $Y$ is a normal/Gaussian random variable.
* The expected value of $Y$ is $x^{T}w$.
*  The variance of $Y$ is constant $(\sigma^{2})$ for all possible $x$.

If we collect a dataset $D = \{ (x^{(i)}, y^{(i)})\}^{n}_{i=1}$, what is the MLE for $W$ and $\sigma$?
$$
P(y \; | \; w, x) = \mathcal{N}(y; x^{T}w, \sigma^{2})
$$

$$
\begin{align}
	P(y \; | \; x, w, \sigma^{2}) &= \mathcal{N}(y; x^{T}w, \sigma^{2}) = \frac{1}{\sqrt{2\pi \sigma^{2}}} \text{ exp}(-\frac{(y - x^{T}w)^2}{2\sigma^{2}}) \\
	P(D \; | \; w, \sigma^{2}) &= \overset{n}{\underset{i=1}{\Pi}}P(y^{(i)} \; | \; x^{(i)}, w, \sigma^{2}) \text{ Conditional independence} \\
	\text{log }P(D \; | \; w, \sigma^{2}) &= \text{log }\overset{n}{\underset{i=1}{\Pi}}P(y^{(i)} \; | \; x^{(i)}, w, \sigma^{2}) \\
	&= \overset{n}{\underset{i=1}{\Sigma}} \text{log }P(y^{(i)} \; | \; x^{(i)}, w, \sigma^{2}) \\
	&\text{for remaining portion of the proof refer to homework 2}\\
\end{align}
$$

* MLE for **w**: 

$$
w = (\overset{n}{\underset{i=1}{\Sigma}} x^{(i)}x^{(i)^{T}})^{-1}(\overset{n}{\underset{i=1}{\Sigma}} x^{(i)}y^{(i)})
$$

This is the same solution as for linear regression, but derived as the MLE of a probabilistic model (instead of the minimum MSE).

* MLE for $\sigma^{2}$: 

$$
\sigma^{2} = \frac{1}{n} \; \overset{n}{\underset{i=1}{\Sigma}} ((x^{(i)^{T}} w) - y^{(i)})^{2}
$$

This is the sum of squared residuals of the predictions w.r.t. ground-truth.

## $L_{2}$ Regularization

### Regularization

The larger the coefficients (weights) $w$ are allowed to be, the more the neural network can overfit.

If we “encourage” the weights to be small, we can reduce overfitting

This is a form of regularization — any practice designed to improve the machine’s ability to generalize to new data.

One of the simplest and oldest regularization techniques is to penalize large weights in the cost function.

* The “unregularized” $f_{MSE}$ is:

$$
f_{MSE}(w) = \frac{1}{2n} \; \overset{n}{\underset{i=1}{\Sigma}} (y^{(i)} - \hat{y}^{(i)})^{2}
$$

* The $L_{2}$-regularized $f_{MSE}$ becomes:
  $$
  f_{MSE}(w) = \frac{1}{2n} \; \overset{n}{\underset{i=1}{\Sigma}} (y^{(i)} - \hat{y}^{(i)})^{2} + \frac{\alpha}{2n}w^{T}w
  $$

  * the points of $L_{2}$ is to ensure the weights $w$ will not grow too large 
  * the $\alpha$ term is a value used to determine how much you want to regularize the weights vs reduce the loss. 

* To help with future comprehensions think of $L_{2}$ regularization as 

$$
\text{MSE}_{L_{2}} = MSE + L_{2}
$$

This way its clear that the regularized term $\frac{\alpha}{2n}w^{T}w$ that is being added to MSE behaves as a penalty when weight values increase 



### Hyperparameter Tuning

The values we optimize when training a machine learning model - e.g., **w** and b for linear regression - are the parameters of the model.

There are also values related to the training process itself - e.g., learning rate $\epsilon$, batch size $\tilde{n}$ regularization strength $\alpha$ - which are the hyperparameters of training.

Both the parameters and hyperparameters can have a huge impact on model performance on test data.

When estimating the performance of a trained model, it is important to tune both kinds of parameters in a principled way:

* Training/validation/testing sets
* Double cross-validation

#### Training/validation/testing sets:

In an application domain with a large dataset (e.g., 100K examples), it is common to partition it into three subsets:

* Training (typically 70-80%): optimization of parameters
* Validation (typically 5-10%): tuning of hyperparameters
* Testing (typically 5-10%): evaluation of the final model

For comparison with other researchers’ methods, this partition should be fixed.



Hyperparameter tuning works as follows:

1. For each hyperparameter configuration h:
   * Train the parameters on the training set using h.
   * Evaluate the model on the validation set.
   * If performance is better than what we got with the best h so far (h* ), then save h as h*
2. Train a model with h*, and evaluate its accuracy $A$ on the testing set. (You can train either on training data, or on training + validation data).

#### Cross-validation:

When working with smaller datasets, cross-validation is commonly used so that we can use all data for training.

* Suppose we already know the best hyperparameters h* .
* We partition the data into k folds of equal sizes.
* Over k iterations, we train on $(k-1)$ folds and test on the remaining fold.
* We then compute the average accuracy over the k testing folds.

```pseudocode
# D = dataset
# k = number of folds
# h = =hyperparameter configuration

def CrossValidation(D, k, h):
    # Partition D into k folds F_{1}, ..., F_{k}
    for i in range(len(k)):
        test_var = F_{i}
        train_var = D \ F_{i}
        # Train the model on train_var using h
        acc[i] = #Evaluate NN on test
    A = Avg[acc]
    return A
        
        
```



#### Training/validation/testing sets (Continued):

Cross-validation does not measure the accuracy of any single machine.

Instead, cross-validation gives the expected accuracy of a classifier that is trained on $\frac{(k-1)}{k}$ of the data.

However, we can train another model $M$ using h* on the entire dataset, and then report $A$ as its accuracy.

Since $M$ is trained on more data than any of the crossvalidation models, its expected accuracy should be $\geq$ A.



#### Cross-Validation (continued):

But how do we find the best hyperparameters h* for each fold?

The typical approach is to use double cross-validation, i.e.:

* For each of the k “outer” folds, run cross-validation in an “inner” loop to determine the best hyperparameter configuration h* for the $k^{th}$ fold.



#### Double Cross-Validation:

```pseudocode
# D = dataset
# k = number of folds
# h = =hyperparameter configuration

def DoubleCrossValidation(D, k, H):
	# Partition D into k folds F_{1}, ..., F_{k}
	for i in range(len(k)):
		test_var = F_{i}
        train_var = D \ F_{i}
        A^{*} = # negative infinity
        For h in H:
        	A = CrossValidation(train_var, k, h)
        	if A > A^{*}:
        		A^{*} = A
        		h* = h
      	Train the model on train_var using h* accs[i] = Evaluate 			the model on test_var
    A = Avg[accs]
    return A
        
```



#### Training/validation/testing sets (Continued Again):

In contrast to (single) cross-validation, it’s not obvious how to train a model $M$ with accuracy $\geq$ $A$. 

One strategy: return an ensemble model whose output is the average of the $k$ models’ predictions…but this is rarely done.



#### Subject Independence:

In many machine learning settings, the data are not completely independent from each other - they are linked in some way.



Example:

* Predict multiple grades for each student based on their Canvas clickstream features (# logins, # forum posts, etc.).

We could partition the data into folds in different ways:

* We could randomize across all the data.
* However, if grades are correlated within each student, then one (or more) training folds can leak information about the testing fold.
* Alternatively, we can stratify across students, i.e., no student appears in more than 1 fold.
* With this partition, the cross-validation accuracy estimates the model’s performance on a subject not used for training.

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\Coding\github\Notes\notes\Deep-Learning\CS-541\pictures\table4.png)



