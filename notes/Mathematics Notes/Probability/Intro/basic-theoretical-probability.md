# Introduction:

Probability of flipping a coin and getting an outcome of heads

Denoted by:
$$
P(H) = \; ?
$$
this can be thought of as 
$$
P(H) = \frac{\# \;\text{of possibilites that meet my conditions}}{\#\; \text{of equally likely possibilities}}
$$
in this case we have two equally likely possibilities, heads and tails
$$
P(H) = \frac{\# \;\text{of possibilites that meet my conditions}}{2}
$$
additionally there is only one condition that satisfies what we are ask for; heads. 
$$
P(H) = \frac{1}{2}
$$
this event that we just went through of flipping a coin can be thought of as an experiment. For context we could run a similar random event, another experiment, by flipping the coin again.



Consider another scenario

We ran a different experiment where we roll a 6 sided die. What is the probability that we roll a one?

We know that there are 6 equally likely possibilities and that the number of possibilites that meet our condition is 1. There for:
$$
P(1) = \frac{\# \;\text{of possibilites that meet my conditions}}{\#\; \text{of equally likely possibilities}}
$$

$$
P(1) = \frac{1}{6}
$$



What would be the probability of rolling a 1 or a 6?

in this case the # of possibilities that meet my condition have changed. There are now 2 numbers where my condition is met.
$$
P(\text{1 or 6}) = \frac{2}{6} = \frac{1}{3}
$$


For the sake of understand the difference ways probability questions can be asked lets consider the scenario when a die is rolled and we roll a 2 AND a 3. 
$$
P(\text{2 and 3}) = ?
$$
This is of course impossible to do within a single roll of the dice. The rolling of the 2 or a 3 are mutually exclusive events and cant happen at the same time. 
$$
P(\text{2 and 3}) = \frac{0}{6}
$$


What is the probability of getting an even number? 
$$
P(even) = ?
$$
We know that potential rolls of a dice can result in a 1, 2, 3, 4, 5 or a 6. However of those possible rolls only 2, 4, and 6 are even. This means
$$
P(even) = \frac{3}{6} = \frac{1}{2}
$$

## Example Problem 1:

Find the probability of pulling a yellow marble from a bag with 3 yellow, 2 red, 2 green, and 1 blue. 

The event we are looking to find the probability for is
$$
P(\text{picking a single yellow marble}) = ?
$$
lets define the possible outcomes. We have the collection of all marbles that we could potentially pick, a total of 8 marbles. Lets also define the set of all these possible choices 
$$
\begin{align*}
    &Y = yellow \\
    &R = red \\
    &G = green \\
    &B = blue \\
    &\text{possible outcomes} = \{Y, Y, Y, R, R, G, G, B\}
\end{align*}
$$
another way to define "possible outcomes" is by calling it the **sample space**
$$
P(\text{picking a single yellow marble}) = \frac{3}{8}
$$

## Example Problem 2:

We have a bag with 9 red marbles, 2 blue marbles and 3 green marbles in it. What is the probability of randomly selecting a non-Blue marble from the bag?
$$
\begin{align*}
    &R = red \\
    &G = green \\
    &B = blue \\
    &\text{sample space} = \{R, R, R, R, R, R, R, R, R, B, B, G, G, G\}
\end{align*}
$$

$$
P(\text{non-blue marbles}) = \frac{12}{14} = \frac{6}{7}
$$

## Example Problem 3:

The circumference of a circle is 36$\pi$. Contained in that circle is a smaller circle with area 16$\pi$. A point is selected at random from inside the larger circle. What is the probability that the point also lies in the smaller circle?

![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\prob.png)

Although there are an infinite number of points that can be drawn within the larger circle we can still find the likelihood by simply finding the ratio between the the smaller and larger AREAs of these circles
$$
C = 2\pi r
$$

$$
\begin{align}
    r &= \frac{C}{2\pi} \\
    r &= \frac{36\pi}{2\pi} \\
    r &= 18
\end{align}
$$

$$
\begin{align}
    A &= \pi r^{2} \\
    A &= \pi 18^{2} \\ 
    A &= 324\pi \\ 
\end{align}
$$

$$
P(\text{Point falls within the smaller circle}) = \frac{16\pi}{324\pi} = \frac{4}{81}
$$

## Intuitive Sense Of Probabilities

Consider a problem where we need to find the probability, that given a roll of a dice, of rolling a number less than or equal to ($\leq$) 2. 
$$
\text{Sample space}\; = \{1, 2, 3, 4, 5, 6 \}
$$

$$
P(rolling \leq 2) = \frac{2}{6} = \frac{1}{3}
$$



Consider another scenario where we need to find the probability, that given a roll of a dice, of rolling a number greater than or equal to ($\geq$) 3. 
$$
\text{Sample space}\; = \{1, 2, 3, 4, 5, 6 \}
$$

$$
P(rolling \geq 3) = \frac{4}{6} = \frac{2}{3}
$$

