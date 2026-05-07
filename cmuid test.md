### **Task 2 Solution**

**a) Fill the missing value**
We can use the **arithmetic mean** of the available values in the second attribute (column 2) to fill the missing data[cite: 6]. 
*   Values: $2.54, 4.98, 0.33$
*   Mean $= \frac{2.54 + 4.98 + 0.33}{3} = \frac{7.85}{3} \approx \mathbf{2.62}$

The complete data set $X$ is:
$$X = \begin{bmatrix} 1.11 & 2.54 & 5.23 \\ 0.73 & 4.98 & 0.21 \\ -0.93 & 0.33 & 2.65 \\ 2.75 & 2.62 & 3.14 \end{bmatrix}$$

**b) Calculate correlations and decide on elimination**
Let the attributes be $A_1, A_2, A_3$.

**1. Pearson Correlation ($r$):** Evaluates linear dependence[cite: 7].
*   $\mu = [0.915, \quad 2.618, \quad 2.808]$
*   $r_{1,2} \approx \mathbf{0.44}$
*   $r_{1,3} \approx \mathbf{0.20}$
*   $r_{2,3} \approx \mathbf{-0.51}$

**2. Spearman Correlation ($r_s$):** Evaluates rank correlation ($r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$)[cite: 7].
*   Ranks $A_1$: $[3, 2, 1, 4]$
*   Ranks $A_2$: $[2, 4, 1, 3]$
*   Ranks $A_3$: $[4, 1, 2, 3]$
*   $r_{s(1,2)} = 1 - \frac{6(1^2 + (-2)^2 + 0^2 + 1^2)}{4(15)} = 1 - \frac{36}{60} = \mathbf{0.40}$
*   $r_{s(1,3)} = 1 - \frac{6((-1)^2 + 1^2 + (-1)^2 + 1^2)}{60} = 1 - \frac{24}{60} = \mathbf{0.60}$
*   $r_{s(2,3)} = 1 - \frac{6((-2)^2 + 3^2 + (-1)^2 + 0^2)}{60} = 1 - \frac{84}{60} = \mathbf{-0.40}$

**Decision:** **None of the attributes should be eliminated.** Elimination is usually reserved for highly redundant features (e.g., $|r| > 0.85$ or $0.90$)[cite: 6]. All correlations here are weak to moderate, meaning each attribute provides independent, valuable information.

**c) Normalize using z-scoring & Logarithmic scaling issue**
Z-scoring standardizes features using $z = \frac{x - \mu}{\sigma}$[cite: 7]. 
*   Population standard deviations: $\sigma = [1.308, \quad 1.645, \quad 1.785]$
*   Normalized Matrix $Z$:
    $$Z = \begin{bmatrix} 
    \frac{1.11 - 0.915}{1.308} & \frac{2.54 - 2.618}{1.645} & \frac{5.23 - 2.808}{1.785} \\ 
    \frac{0.73 - 0.915}{1.308} & \frac{4.98 - 2.618}{1.645} & \frac{0.21 - 2.808}{1.785} \\ 
    \frac{-0.93 - 0.915}{1.308} & \frac{0.33 - 2.618}{1.645} & \frac{2.65 - 2.808}{1.785} \\ 
    \frac{2.75 - 0.915}{1.308} & \frac{2.62 - 2.618}{1.645} & \frac{3.14 - 2.808}{1.785} 
    \end{bmatrix} \approx \mathbf{\begin{bmatrix} 
    0.15 & -0.05 & 1.36 \\ 
    -0.14 & 1.44 & -1.46 \\ 
    -1.41 & -1.39 & -0.09 \\ 
    1.40 & 0.00 & 0.19 
    \end{bmatrix}}$$

**Why logarithmic scaling causes problems here:**
Logarithmic scaling ($x' = \log(x)$)[cite: 7] cannot be applied because the first attribute contains a **negative value** ($-0.93$). The logarithm of a negative number is undefined in real numbers, which would cause a mathematical error during transformation.