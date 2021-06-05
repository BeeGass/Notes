# Basic Set Operations:

* **Set**  

  * set is a collection of distinct objects.

    * They can contain other sets, numbers, names of cars, farm animals, e.g.:

    $$
    \begin{align}
    	\textbf{X} &= \{ 3, 12, 5, 13 \} \\
    	\textbf{Y} &= \{ 14, 15, 6, 3 \} \\
    \end{align}
    $$

    

  * There are operations we can perform on sets. One of which is the **intersection** of 2 different sets, e.g.:
    $$
    X \; \underset{\text{"and"}}{\cap} \; Y = \{ 3 \} \\ \\ \\
    X \; \cap \; Y \text{, will generate a third new set}
    $$
    
  * Another operation we can perform on sets is **union**. e.g.:
    $$
    X \; \underset{\text{"or"}}{\cup} \; Y = \{ 3, 5, 6, 12, 13, 14, 15 \} \\ \\ \\
    
    X \; \cup \; Y \text{, will also generate a third new set}
    $$
    ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\Union (3).png)

  * Example 2:
    $$
    \begin{align*}
    	A &= \{ 11, 4, 12, 7 \} \\
    	B &= \{ 13, 4, 12, 10, 3 \} \\ \\ \\
    	C &= \text{A} \; \cap \; \text{B} = \{ 4, 12 \} \\ \\ \\
    	D &= \text{A} \; \cup \; \text{B} = \{ 11, 4, 12, 7, 13, 10, 3 \}
    \end{align*}
    $$
    

  * Example 3:
    $$
    \begin{align*}
    	\text{A}_{1} &=  \{ 5, 3, 17, 12, 19 \} \\
    	\text{B}_{1} &= \{ 17, 19, 6 \} \\ \\ \\
    	&\text{"B subtracted from A"} \\
    	&\text{"Relative compliment of set B in A"} \\
    	\text{A}_{1} \; \backslash \; \text{B}_{1} = \text{A}_{1} \; - \; \text{B}_{1} &= \{ 5, 3, 12 \} \\ \\ \\
    	\text{B}_{1} \; \backslash \; \text{A}_{1} = \text{B}_{1} \; - \; \text{A}_{1} &= \{ 6 \} \\ \\ \\
    	
    	\text{A}_{1} \; \backslash \; \text{A}_{1} = \text{A}_{1} \; - \; \text{A}_{1} &= \overset{\text{"empty set"}}{\{ \; \}} = \overset{\text{null set}}{\varnothing}
    	
    \end{align*}
    $$
    

  * Example 4: Universe

    * The universe set contains all possible samples within the "universal" sample space.

    * Within the universal set there can be subsets

      ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\Universe.png)

      ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\Universe (1).png)

      * When referring to the:

      $$
      \underset{\text{set of all things in U that arent in A}}{A^{'}} = \text{U} - \text{A} = \text{U} \; \backslash \; \text{A}
      $$

      

      

      

      ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\Universe (2).png)

      * when referring to the: 
        $$
        \underset{\text{set of all things in U that are in A}}{\text{A}} = \text{A} \; - \; \text{U} = \text{A} \; \backslash \; \text{U}
        $$

      

      

      

      * $\mathbb{R}$ = real $\#'s$ 

      * $\mathbb{Q}$ = rational

      * $\mathbb{Z}$ = integers

        ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\C.png)

        

        

        ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\within.png)
        $$
        \begin{align*}
        	\text{C} &= \{ -5, 0, 7 \} \\
        	-5 &\underset{\text{"membership" or "within"}}{\in} \text{C} \\
        	0 &\in \text{C} \\
        	7 &\in \text{C} \\ \\ \\
        	
        	-8 &\notin \text{C} \\
        	53 &\notin \text{C} \\
        	42 &\notin \text{C} \\
        	
        \end{align*}
        $$
        

        

        ![](C:\Users\Bryan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Desktop\C_prime (1).png)
        $$
        \begin{align*}
        	\text{C}^{'} &= \text{U} - \text{C} \\
        	-5 &\notin \text{C}^{'} \\
        	0 &\notin \text{C}^{'} \\ \\ \\
        	53 &\in \text{C}^{'} \\
        	42 &\in \text{C}^{'} \\
        \end{align*}
        $$
        

  * Example 5:
    $$
    \begin{align*}
    	\text{A} &= \{ 1, 3, 5, 7, 18\} \\
    	\text{B} &= \{ 1, 7, 18\} \\
    	\text{C} &= \{ 18, 7, 1, 19\} \\ \\
    	
    	&\text{B is subset A because B contains 1, 7 and 18 which are numbers found in A} \\
    	\text{B} &\underset{\text{"subset"}}{\subseteq} \text{A} \\ \\ 
    	
    	&\text{if you want to specify that B is a subset of A but A is not a subset of B} \\
    	\text{B} &\underset{\text{"strict subset" or "proper subset"}}{\subsetneq} \text{A} \\ \\
    	
    	\text{A} \subseteq \text{A} &= \text{true} \\
    	\text{A} \subsetneq \text{A} &= \text{false} \\
    \end{align*}
    $$

    * $\text{A} \subseteq \text{A} = true$ but $\text{A} \subsetneq \text{A} = false$ because A is a subset of itself which would violate the strict subset rule 

    $$
    \begin{align*}
    	\text{B} &\subseteq \text{C} \;? \\
    	\text{B} &\subseteq \text{C} = true
    \end{align*}
    $$

    $$
    \begin{align*}
    	\text{C} &\subseteq \text{A} \;? \\
    	\text{C} &\subseteq \text{A} = false
    \end{align*}
    $$

    * We can also reverse the subset symbol notation to show **supersets**

      * a superset can be thought of as set $A$ having all the same elements if not more elements as set $B$. It is important to note that $A$ CAN contain exactly the same elements as $B$ 
        $$
        \text{A} \supseteq \text{B}
        $$

      * a strict super set specifically specifies that the contents of $A$ have elements that match the contents of $B$ as well as additional elements
        $$
        \text{A} \supsetneq \text{B}
        $$

