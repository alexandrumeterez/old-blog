---
layout: distill
title: Neural Tangent Kernel 
description: notes on NTK and related material
giscus_comments: false
date: 2023-02-26

authors:
  - name: Alexandru Meterez
    url: "https://alexandrumeterez.github.io"
    affiliations:
      name: D-INFK, ETH Zurich

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Functional Gradient Descent
  - name: Functionals
    subsections:
    - name: Reproducing Kernel Hilbert Space (RKHS)
    - name: Inner product and norm
    - name: Reproducing property
    - name: Evaluation functional 
    - name: Functional derivative
    - name: Chain rule 
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
---

## Functional Gradient Descent
For examples and plots see [here](https://simple-complexities.github.io/optimization/functional/gradient/descent/2020/03/04/functional-gradient-descent.html).

Suppose we want to learn a function $f(x)$ using gradient descent. One example would be to parameterize $f$ as a linear function using weights $w$: $f(x) = w^\top x$. 

In order to learn $w$ we take a loss function, i.e. MSE: 

$$
L(w) = \sum_{i=1}^n (y_i - w^\top x_i)^2 + \| w \|^2 \\
\nabla L(w) = \sum_{i=1}^n -2 x_i (y_i - w^\top x_i) + 2w \\
= \sum_{i=1}^n (-2x_iy_i + 2w^\top \| x_i \|^2) + 2w
$$. 

Then we do gradient descent steps on $w$: 

$$
w = w - \alpha \nabla L(w)
$$

Using functional gradient descent, this can be generalized to any function $f$: 

$$
L(f) = \sum_{i=1}^n (y_i - f(x_i)) + \| f \|^2 \\
f_{t+1} = f_t - \alpha \nabla L(f)
$$

This has 2 advantages:
- Some loss functions are non-convex in parameter space but can be convex in functional space
- In NTK, when width $\rightarrow \infty$, the weights become almost static between GD steps; however we can still study the function trajectory during GD in functional space

***

## Functionals
Some basic functional notions are needed to understand this post.

### Reproducing Kernel Hilbert Space (RKHS)
Denote RKHS of (fixed) kernel $k$ by $\mathcal{H_k}$, $k(\cdot, \cdot)$ is a kernel function and $K_{ij} = k(x_i, x_j)$. Then:

$$
f \in \mathcal{H_k} \implies f(\cdot) = \sum_{i=1}^n \beta_i k(x_i, \cdot), \beta_i \in \mathbb{R}
$$

In other words, $f$ is in RKHS if it can be written as a weighted sum of kernel functions evaluated over $n$ points. Note that $f$ is completely determined by the $\beta_i$ and $x_i$.

### Inner product and norm
Let $f, g \in \mathcal{H_k}$. Then:

$$
f \cdot g = \sum_{i=1}^{n_f} \sum_{j=1}^{n_g} \alpha_i \beta_j k(x_i, x_j) = \alpha K^\top \beta \\
\| f \|^2 = f \cdot f = \alpha K^\top \beta
$$

### Reproducing property

$$
f \cdot k(x, \cdot) = \sum_{i=1}^n \beta_i [k(x_i, \cdot) k(x, \cdot)] = \sum_{i=1}^n \beta_i k(x_i, x) = f
$$

### Evaluation functional

$$
E_x[f] = f(x)
$$

### Functional derivative
From the definition of the derivative, the functional derivative is the coefficient of the linear term in the Taylor expansion of the functional.

Example 1 - for the functional $E_x[f] = f(x)$:

$$
E_x[f + df] = f(x) + df(x) \\
= E_x[f] + df(x) \\
= E_x[f] + k(x, \cdot) \cdot df \\
\implies \nabla E_x[f] = k(x, \cdot)
$$

Example 2 - for the functional $E[f] = \| f \|^2$

$$
E[f + df] = (f + df) (f + df) \\
= \| f \| + 2 f \cdot df + \| df \|^2 \\
= E[f] + 2 f \cdot df + \| df \|^2 \\ 
\implies \nabla E[f] = 2f
$$

### Chain rule
Let $E[f]$ be a functional and $g : \mathbb{R} \to \mathbb{R}$. Then:

$$
\nabla g(E[f]) = \nabla E[f] g'(E[f])
$$