# The Most Important Algorithms of All Time

While this repository focuses primarily on data structures and the algorithims surrounding these, the most important algorithims behind many applications are much closer to applied mathematics. Here are a few articles on listing these most important algorithims:

- https://lemire.me/blog/2010/07/05/the-five-most-important-algorithms/
- https://press.princeton.edu/ideas/nicholas-higham-on-the-top-10-algorithms-in-applied-mathematics (specifically for this, the author used indexes in the *Princeton Companion to Applied Mathematics* to get these algorithims)
- https://onlinegrad.syracuse.edu/blog/twentieth-century-algorithms/
- The Best of the 20th Century: Editors Name Top 10  Algorithms [SIAM News, Volume 33, Number 4]
- https://www.andrew.cmu.edu/course/15-355/misc/Top%20Ten%20Algorithms.html
- https://pi.math.cornell.edu/~ajt/presentations/TopTenAlgorithms.pdf
- http://www.koutschan.de/misc/algorithms.php
- https://www.computer.org/csdl/magazine/cs/2000/01/c1022/13rRUxBJhBm

Here's a summary written by Sonnet 4-5:

A comprehensive catalog of algorithms that shaped science, engineering, and computation in the 20th and 21st centuries.

---

## Table of Contents

1. [Numerical Linear Algebra](#1-numerical-linear-algebra)
2. [Optimization Algorithms](#2-optimization-algorithms)
3. [Sorting and Searching](#3-sorting-and-searching)
4. [Signal Processing and Transforms](#4-signal-processing-and-transforms)
5. [Graph and Network Algorithms](#5-graph-and-network-algorithms)
6. [Probabilistic and Statistical Methods](#6-probabilistic-and-statistical-methods)
7. [Cryptography](#7-cryptography)
8. [Machine Learning and AI](#8-machine-learning-and-ai)
9. [Computational Geometry and Physics](#9-computational-geometry-and-physics)
10. [Data Compression and Encoding](#10-data-compression-and-encoding)

---

## 1. Numerical Linear Algebra

### 1.1 QR Algorithm (1959-1961)

**Purpose:** Compute eigenvalues and eigenvectors of matrices

**Mathematical Explanation:**
Given a matrix $A$, the QR algorithm iteratively decomposes it:
```
Initialize: A₀ = A
For k = 1, 2, 3, ...
    Aₖ = QₖRₖ         (QR factorization)
    Aₖ₊₁ = RₖQₖ       (form next iterate)
End
```
Where $Q$ is orthogonal/unitary and $R$ is upper triangular. The sequence $\{A_k\}$ converges to a diagonal or block-diagonal matrix revealing eigenvalues.

**Key Properties:**
- Each $A_{k+1}$ is similar to $A_k$, preserving eigenvalues
- Diagonal entries converge to eigenvalues
- For symmetric matrices, convergence is to diagonal form

**Applications:**
- Structural engineering (resonant frequencies of bridges, buildings)
- Quantum mechanics (energy levels)
- Principal Component Analysis (PCA)
- Stability analysis of dynamical systems
- Google PageRank computations

---

### 1.2 Singular Value Decomposition (SVD)
**Era:** 1960s, refined continuously

**Purpose:** Decompose any matrix into orthogonal rotations and scalings

**Mathematical Explanation:**
Any $m \times n$ matrix $A$ can be factored as:
$$A = U\Sigma V^T$$

Where:
- $U$ is $m \times m$ orthogonal (left singular vectors)
- $\Sigma$ is $m \times n$ diagonal with $\sigma_1 \geq \sigma_2 \geq ... \geq 0$ (singular values)
- $V$ is $n \times n$ orthogonal (right singular vectors)

**Low-Rank Approximation:**
Best rank-$k$ approximation: $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$

**Applications:**
- Image compression and denoising
- Recommender systems (Netflix Prize)
- Latent Semantic Analysis (LSA)
- Pseudoinverse computation for least squares
- Data visualization and dimensionality reduction
- Numerical weather prediction
- Signal processing

---

### 1.3 Matrix Decompositions (LU, Cholesky, QR)
**Era:** Formalized by Alston Householder (1951)

**Purpose:** Factor matrices into simpler forms for efficient computation

**Mathematical Explanation:**

**LU Decomposition:**
$$A = LU$$
- $L$ is lower triangular with 1s on diagonal
- $U$ is upper triangular
- Enables efficient solving of $Ax = b$ via forward/backward substitution

**Cholesky Decomposition** (for positive definite $A$):
$$A = LL^T$$
- $L$ is lower triangular with positive diagonal
- Requires only $n^3/6$ operations (half of LU)

**QR Decomposition:**
$$A = QR$$
- $Q$ is orthogonal ($Q^TQ = I$)
- $R$ is upper triangular
- Numerically stable for least squares

**Applications:**
- Solving linear systems $Ax = b$
- Computing determinants and inverses
- Least squares regression
- Kalman filtering
- Computer graphics transformations
- Finite element analysis

---

### 1.4 Krylov Subspace Iteration Methods (1950)
**Developers:** Magnus Hestenes, Eduard Stiefel, Cornelius Lanczos

**Purpose:** Solve large sparse linear systems iteratively

**Mathematical Explanation:**
For solving $Ax = b$, construct the Krylov subspace:
$$\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2r_0, ..., A^{k-1}r_0\}$$
where $r_0 = b - Ax_0$ is the initial residual.

**Conjugate Gradient (CG)** for symmetric positive definite $A$:
```
r₀ = b - Ax₀
p₀ = r₀
For k = 0, 1, 2, ...
    αₖ = (rₖᵀrₖ)/(pₖᵀApₖ)
    xₖ₊₁ = xₖ + αₖpₖ
    rₖ₊₁ = rₖ - αₖApₖ
    βₖ = (rₖ₊₁ᵀrₖ₊₁)/(rₖᵀrₖ)
    pₖ₊₁ = rₖ₊₁ + βₖpₖ
End
```

**Variants:**
- GMRES (Generalized Minimal Residual) for non-symmetric systems
- Bi-CGSTAB (Bi-Conjugate Gradient Stabilized)
- MINRES for symmetric indefinite systems

**Complexity:** $O(N)$ or $O(N \log N)$ vs. $O(N^3)$ for direct methods

**Applications:**
- Computational fluid dynamics (CFD)
- Electromagnetic simulations
- Structural mechanics
- Image reconstruction
- Machine learning (large-scale optimization)

---

## 2. Optimization Algorithms

### 2.1 Simplex Method (1947)
**Developer:** George Dantzig

**Purpose:** Solve linear programming problems

**Mathematical Explanation:**
Maximize (or minimize): $c^Tx$  
Subject to: $Ax \leq b$, $x \geq 0$

**Key Insight:** Optimal solution occurs at a vertex of the feasible polytope

**Algorithm:**
1. Start at a vertex (basic feasible solution)
2. Move along edges to adjacent vertices with improving objective
3. Stop when no improving neighbor exists (optimality)

**Complexity:** 
- Worst case: Exponential
- Average case: Polynomial (in practice, very efficient)

**Applications:**
- Resource allocation and scheduling
- Transportation and logistics optimization
- Manufacturing production planning
- Portfolio optimization in finance
- Dating app matching
- Military logistics (WWII origins)
- Diet planning and nutrition optimization

---

### 2.2 Newton's Method (1669)
**Developer:** Isaac Newton (method of fluxions)

**Purpose:** Find roots of equations; optimize functions

**Mathematical Explanation:**

**For root finding** $f(x) = 0$:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**For optimization** (finding critical points):
$$x_{n+1} = x_n - [H(x_n)]^{-1}\nabla f(x_n)$$
where $H$ is the Hessian matrix of second derivatives.

**Convergence:** Quadratic near the solution

**Quasi-Newton Methods** (BFGS, L-BFGS):
Approximate the Hessian to avoid expensive computation:
$$x_{k+1} = x_k - \alpha_k B_k^{-1} \nabla f(x_k)$$

**Applications:**
- Nonlinear equation solving
- Machine learning optimization
- Scientific computing
- Computer graphics (ray tracing)
- Game physics engines
- Economic modeling

---

### 2.3 Gradient Descent (Modern formalization: 1847)
**Developer:** Augustin-Louis Cauchy

**Purpose:** Find local minima of functions

**Mathematical Explanation:**
$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

Where $\alpha_k$ is the learning rate/step size.

**Variants:**
- **Stochastic Gradient Descent (SGD):** Use random subsets
  $$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta; x^{(i)}, y^{(i)})$$
  
- **Momentum:** Accumulate velocity
  $$v_t = \beta v_{t-1} + \nabla f(x_t)$$
  $$x_{t+1} = x_t - \alpha v_t$$

- **Adam:** Adaptive learning rates with momentum
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f(x_t)$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla f(x_t))^2$$

**Applications:**
- **Deep learning** (training neural networks)
- Logistic regression
- Support vector machines
- Image reconstruction
- Signal processing
- Robotics control

---

### 2.4 Dynamic Programming (1950s)
**Developer:** Richard Bellman

**Purpose:** Solve optimization problems by breaking into overlapping subproblems

**Mathematical Explanation:**
**Bellman Principle of Optimality:**
An optimal policy has the property that whatever the initial state and decision, remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

**General form:**
$$V(s) = \max_{a} \left\{R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right\}$$

**Classic Example - Fibonacci:**
```
Recursive (exponential): fib(n) = fib(n-1) + fib(n-2)
DP (linear): Store computed values, reuse
```

**Applications:**
- Shortest path algorithms (Bellman-Ford)
- Sequence alignment (bioinformatics)
- Resource allocation
- Economics and finance
- Robotics path planning
- Operations research

---

## 3. Sorting and Searching

### 3.1 Quicksort (1962)
**Developer:** Tony Hoare

**Purpose:** Efficient in-place sorting

**Mathematical Explanation:**
**Divide and conquer approach:**
```
quicksort(A, low, high):
    if low < high:
        pivot_index = partition(A, low, high)
        quicksort(A, low, pivot_index - 1)
        quicksort(A, pivot_index + 1, high)

partition(A, low, high):
    pivot = A[high]
    i = low - 1
    for j = low to high - 1:
        if A[j] <= pivot:
            i = i + 1
            swap A[i] with A[j]
    swap A[i + 1] with A[high]
    return i + 1
```

**Complexity:**
- Average: $O(n \log n)$
- Worst case: $O(n^2)$ (rare with good pivot selection)
- Space: $O(\log n)$ stack space

**Applications:**
- Database query processing
- Numerical computation libraries
- Operating system schedulers
- Graphics rendering pipelines
- General-purpose sorting

---

### 3.2 Merge Sort (1945)
**Developer:** John von Neumann

**Purpose:** Stable sorting with guaranteed $O(n \log n)$ performance

**Mathematical Explanation:**
```
mergesort(A):
    if length(A) <= 1:
        return A
    mid = length(A) / 2
    left = mergesort(A[0:mid])
    right = mergesort(A[mid:end])
    return merge(left, right)

merge(L, R):
    result = []
    while L and R not empty:
        if L[0] <= R[0]:
            append L[0] to result
            remove L[0]
        else:
            append R[0] to result
            remove R[0]
    append remaining elements
    return result
```

**Complexity:** Always $O(n \log n)$ time, $O(n)$ space

**Applications:**
- External sorting (too large for memory)
- Sorting linked lists
- Counting inversions
- Parallel sorting algorithms
- Database systems

---

### 3.3 Binary Search (Ancient concept, formalized 1946)
**Purpose:** Efficiently search sorted arrays

**Mathematical Explanation:**
```
binary_search(A, target, low, high):
    if low > high:
        return NOT_FOUND
    mid = (low + high) / 2
    if A[mid] == target:
        return mid
    else if A[mid] > target:
        return binary_search(A, target, low, mid - 1)
    else:
        return binary_search(A, target, mid + 1, high)
```

**Complexity:** $O(\log n)$ time, $O(1)$ or $O(\log n)$ space

**Recurrence relation:** $T(n) = T(n/2) + O(1)$

**Applications:**
- Database indexing
- Dictionary lookups
- Finding boundaries in sorted data
- Numerical root finding
- Version control systems (git bisect)

---

### 3.4 Heap Sort (1964)
**Developer:** J.W.J. Williams

**Purpose:** In-place sorting using heap data structure

**Mathematical Explanation:**
A heap is a complete binary tree satisfying the heap property:
- Max-heap: Parent ≥ children
- Array representation: parent at $i$, children at $2i+1$ and $2i+2$

```
heapsort(A):
    build_max_heap(A)
    for i = length(A) - 1 down to 1:
        swap A[0] with A[i]
        heap_size = heap_size - 1
        max_heapify(A, 0)
```

**Complexity:** $O(n \log n)$ time, $O(1)$ space

**Applications:**
- Priority queues
- Event-driven simulations
- Job scheduling in operating systems
- Huffman coding
- Graph algorithms (Dijkstra, Prim)

---

## 4. Signal Processing and Transforms

### 4.1 Fast Fourier Transform (FFT) (1965)
**Developers:** James Cooley and John Tukey (rediscovered from Gauss, 1805)

**Purpose:** Efficiently compute Discrete Fourier Transform

**Mathematical Explanation:**
**Discrete Fourier Transform (DFT):**
$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}, \quad k = 0, ..., N-1$$

**Complexity:** Naive DFT is $O(N^2)$

**FFT Algorithm (Cooley-Tukey):**
Divide and conquer: split into even and odd indices
$$X_k = \sum_{m=0}^{N/2-1} x_{2m} e^{-2\pi i k(2m)/N} + \sum_{m=0}^{N/2-1} x_{2m+1} e^{-2\pi i k(2m+1)/N}$$

$$X_k = E_k + e^{-2\pi ik/N} O_k$$

**Complexity:** $O(N \log N)$

**Matrix View:**
The DFT matrix $F$ has special structure allowing factorization:
$$F_N = P_N(I_2 \otimes F_{N/2})T_N$$
with sparse matrices $P$ (permutation) and $T$ (twiddle factors).

**Applications:**
- **Audio processing:** MP3, speech recognition
- **Image processing:** JPEG compression
- **Telecommunications:** OFDM (WiFi, 4G/5G)
- **Solving PDEs:** Spectral methods
- **Nuclear test detection** (original 1960s application)
- **Convolution:** Via convolution theorem $f * g = \mathcal{F}^{-1}[\mathcal{F}[f] \cdot \mathcal{F}[g]]$
- **Signal analysis:** Identifying frequency components

---

### 4.2 Wavelet Transform (1980s)
**Purpose:** Multi-resolution signal analysis

**Mathematical Explanation:**
$$W(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f(t) \psi^*\left(\frac{t-b}{a}\right) dt$$

Where:
- $a$ is scale parameter
- $b$ is translation parameter
- $\psi$ is the mother wavelet

**Discrete Wavelet Transform (DWT):**
$$c_{j,k} = \sum_n x[n] \phi_{j,k}[n], \quad d_{j,k} = \sum_n x[n] \psi_{j,k}[n]$$

**Applications:**
- JPEG 2000 image compression
- Signal denoising
- Edge detection
- Seismic data analysis
- Biomedical signal processing (EEG, ECG)

---

## 5. Graph and Network Algorithms

### 5.1 Dijkstra's Algorithm (1959)
**Developer:** Edsger Dijkstra

**Purpose:** Find shortest path in weighted graph with non-negative edges

**Mathematical Explanation:**
Maintain distance estimates $d[v]$ and a priority queue:
```
dijkstra(G, source):
    for each vertex v in G:
        dist[v] = ∞
        prev[v] = undefined
    dist[source] = 0
    Q = priority_queue(all vertices)
    
    while Q not empty:
        u = Q.extract_min()
        for each neighbor v of u:
            alt = dist[u] + length(u, v)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                Q.decrease_key(v, alt)
```

**Complexity:** 
- With binary heap: $O((V + E) \log V)$
- With Fibonacci heap: $O(V \log V + E)$

**Correctness:** Greedy choice property - once a vertex is finalized, its distance is optimal.

**Applications:**
- GPS navigation and routing
- Network routing protocols (OSPF)
- Social network analysis
- Robotics path planning
- Package delivery optimization (UPS, Amazon)
- Phone call routing

---

### 5.2 PageRank (1998)
**Developers:** Larry Page and Sergey Brin

**Purpose:** Rank web pages by importance

**Mathematical Explanation:**
PageRank models web surfing as a random walk. The rank of page $i$ is:
$$PR(i) = \frac{1-d}{N} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)}$$

Where:
- $d$ is damping factor (typically 0.85)
- $N$ is total number of pages
- $M(i)$ is set of pages linking to $i$
- $L(j)$ is number of outbound links from page $j$

**Matrix Formulation:**
$$\mathbf{PR} = \left(\frac{1-d}{N}\mathbf{1} \mathbf{1}^T + d \mathbf{M}\right)\mathbf{PR}$$

This is an eigenvector problem: $\mathbf{PR}$ is the principal eigenvector of the Google matrix.

**Solution:** Power iteration
```
PR₀ = [1/N, 1/N, ..., 1/N]
repeat:
    PRₖ₊₁ = (1-d)/N * 1 + d * M * PRₖ
until convergence
```

**Applications:**
- Web search ranking (Google)
- Academic citation analysis
- Social network influence
- Recommendation systems
- Protein interaction networks

---

### 5.3 Bellman-Ford Algorithm (1958)
**Purpose:** Shortest path with negative edge weights

**Mathematical Explanation:**
```
bellman_ford(G, source):
    for each vertex v:
        dist[v] = ∞
    dist[source] = 0
    
    for i = 1 to |V| - 1:
        for each edge (u, v) with weight w:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    // Check for negative cycles
    for each edge (u, v) with weight w:
        if dist[u] + w < dist[v]:
            return "Negative cycle detected"
```

**Complexity:** $O(VE)$

**Applications:**
- Currency arbitrage detection
- Network routing with costs
- Negative cycle detection

---

### 5.4 Maximum Flow (Ford-Fulkerson, 1956)
**Purpose:** Find maximum flow through a network

**Mathematical Explanation:**
Given flow network $G = (V, E)$ with capacities $c(u,v)$:

**Max-Flow Min-Cut Theorem:**
$$\max_{f} |f| = \min_{(S,T)} c(S, T)$$

**Ford-Fulkerson Algorithm:**
```
while there exists augmenting path p from s to t:
    cf = min{c(u,v) - f(u,v) : (u,v) in p}
    for each edge (u,v) in p:
        f(u,v) += cf
        f(v,u) -= cf
```

**Applications:**
- Network capacity planning
- Bipartite matching
- Image segmentation
- Airline scheduling
- Supply chain optimization

---

### 5.5 A* Search Algorithm (1968)
**Developers:** Peter Hart, Nils Nilsson, Bertram Raphael

**Purpose:** Heuristic pathfinding

**Mathematical Explanation:**
$$f(n) = g(n) + h(n)$$

Where:
- $g(n)$ = cost from start to node $n$
- $h(n)$ = heuristic estimate from $n$ to goal
- $f(n)$ = total estimated cost

**Admissibility:** If $h(n)$ never overestimates (admissible heuristic), A* finds optimal path.

**Algorithm:** Dijkstra's with heuristic priority

**Applications:**
- Video game pathfinding
- Robotics navigation
- Map applications
- Puzzle solving
- Motion planning

---

## 6. Probabilistic and Statistical Methods

### 6.1 Metropolis Algorithm / Monte Carlo Method (1946)
**Developers:** John von Neumann, Stan Ulam, Nick Metropolis

**Purpose:** Solve problems via random sampling

**Mathematical Explanation:**
Estimate expectation $E[f(X)]$ where $X \sim p(x)$:
$$E[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i), \quad x_i \sim p(x)$$

**Metropolis-Hastings Algorithm:**
Generate samples from $\pi(x)$:
```
Initialize x₀
for t = 1 to N:
    Propose: x* ~ q(x*|xₜ₋₁)
    Compute acceptance ratio:
        α = min(1, π(x*)q(xₜ₋₁|x*) / (π(xₜ₋₁)q(x*|xₜ₋₁)))
    With probability α:
        xₜ = x*
    Else:
        xₜ = xₜ₋₁
```

**Applications:**
- Statistical physics (Ising model, protein folding)
- Bayesian inference (MCMC)
- Computational finance (option pricing)
- Particle transport (nuclear reactor design)
- Integration in high dimensions
- Machine learning (sampling from posteriors)

---

### 6.2 Expectation-Maximization (EM) Algorithm (1977)
**Developers:** Arthur Dempster, Nan Laird, Donald Rubin

**Purpose:** Maximum likelihood with latent variables

**Mathematical Explanation:**
Maximize $\log p(X|\theta)$ when $Z$ (latent) unobserved:

**E-step:** Compute expected log-likelihood
$$Q(\theta|\theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$$

**M-step:** Maximize
$$\theta^{(t+1)} = \arg\max_\theta Q(\theta|\theta^{(t)})$$

**Convergence:** Guaranteed to increase likelihood at each step.

**Applications:**
- Gaussian Mixture Models (clustering)
- Hidden Markov Models (speech recognition)
- Missing data imputation
- Image segmentation
- Collaborative filtering

---

### 6.3 Kalman Filter (1960)
**Developer:** Rudolf Kálmán

**Purpose:** Optimal state estimation with noisy measurements

**Mathematical Explanation:**
**System model:**
$$x_t = Ax_{t-1} + Bu_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$z_t = Hx_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

**Predict:**
$$\hat{x}_{t|t-1} = A\hat{x}_{t-1|t-1} + Bu_t$$
$$P_{t|t-1} = AP_{t-1|t-1}A^T + Q$$

**Update:**
$$K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}$$ (Kalman gain)
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$$
$$P_{t|t} = (I - K_tH)P_{t|t-1}$$

**Applications:**
- GPS navigation
- Spacecraft guidance (Apollo missions)
- Autonomous vehicles
- Weather forecasting
- Economics (GDP estimation)
- Virtual reality (position tracking)
- Robotics (sensor fusion)

---

### 6.4 Viterbi Algorithm (1967)
**Developer:** Andrew Viterbi

**Purpose:** Find most likely sequence of hidden states in HMM

**Mathematical Explanation:**
Given HMM with states $S$, observations $O$, find:
$$\arg\max_{q_1,...,q_T} P(q_1,...,q_T | O_1,...,O_T)$$

**Dynamic programming:**
$$\delta_t(i) = \max_{q_1,...,q_{t-1}} P(q_1,...,q_{t-1}, q_t=i, O_1,...,O_t)$$
$$\delta_t(j) = \max_i[\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(O_t)$$

**Applications:**
- Speech recognition
- Part-of-speech tagging (NLP)
- Gene finding in DNA
- Error correction in communication
- Gesture recognition

---

## 7. Cryptography

### 7.1 RSA Algorithm (1977)
**Developers:** Ron Rivest, Adi Shamir, Leonard Adleman

**Purpose:** Public-key cryptography

**Mathematical Explanation:**
**Key generation:**
1. Choose large primes $p, q$
2. Compute $n = pq$ and $\phi(n) = (p-1)(q-1)$
3. Choose $e$ with $\gcd(e, \phi(n)) = 1$
4. Compute $d \equiv e^{-1} \pmod{\phi(n)}$
5. Public key: $(e, n)$, Private key: $(d, n)$

**Encryption:** $c = m^e \mod n$  
**Decryption:** $m = c^d \mod n$

**Security:** Based on difficulty of factoring large integers

**Applications:**
- Secure web communication (HTTPS/TLS)
- Digital signatures
- Secure email (PGP)
- Cryptocurrency
- Software licensing
- VPN authentication

---

### 7.2 Diffie-Hellman Key Exchange (1976)
**Developers:** Whitfield Diffie, Martin Hellman

**Purpose:** Secure key establishment over public channel

**Mathematical Explanation:**
Public: prime $p$, generator $g$

**Protocol:**
1. Alice chooses secret $a$, sends $A = g^a \mod p$
2. Bob chooses secret $b$, sends $B = g^b \mod p$
3. Alice computes $s = B^a \mod p$
4. Bob computes $s = A^b \mod p$
5. Shared secret: $s = g^{ab} \mod p$

**Security:** Based on discrete logarithm problem

**Applications:**
- TLS/SSL handshake
- IPsec VPNs
- Secure messaging (Signal, WhatsApp)
- SSH protocol
- Blockchain protocols

---

## 8. Machine Learning and AI

### 8.1 Backpropagation (1986)
**Developers:** David Rumelhart, Geoffrey Hinton, Ronald Williams

**Purpose:** Train neural networks via gradient descent

**Mathematical Explanation:**
Given neural network with layers $l = 1,...,L$:

**Forward pass:** Compute activations
$$a^{(l)} = \sigma(W^{(l)}a^{(l-1)} + b^{(l)})$$

**Backward pass:** Compute gradients using chain rule
$$\delta^{(L)} = \nabla_{a^{(L)}}L \odot \sigma'(z^{(L)})$$
$$\delta^{(l)} = ((W^{(l+1)})^T\delta^{(l+1)}) \odot \sigma'(z^{(l)})$$
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T$$

**Update:**
$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

**Applications:**
- Deep learning (all modern neural networks)
- Computer vision (CNNs)
- Natural language processing (transformers)
- Speech recognition
- Game playing (AlphaGo)
- Autonomous driving

---

### 8.2 Q-Learning (1989)
**Developer:** Chris Watkins

**Purpose:** Model-free reinforcement learning

**Mathematical Explanation:**
Learn action-value function $Q(s,a)$:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Where:
- $s, a$: current state, action
- $r$: reward
- $s'$: next state
- $\alpha$: learning rate
- $\gamma$: discount factor

**Convergence:** Under conditions, $Q \to Q^*$ (optimal)

**Applications:**
- Game playing (Atari, board games)
- Robotics control
- Resource allocation
- Autonomous vehicles
- Recommendation systems

---

### 8.3 k-Means Clustering (1957)
**Purpose:** Partition data into k clusters

**Mathematical Explanation:**
Minimize within-cluster variance:
$$\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} ||x - \mu_i||^2$$

**Lloyd's Algorithm:**
```
Initialize k centroids randomly
repeat:
    Assign each point to nearest centroid
    Update centroids: μᵢ = mean of points in cluster i
until convergence
```

**Complexity:** $O(nkdi)$ where $i$ is iterations

**Applications:**
- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection
- Feature learning

---

### 8.4 Support Vector Machines (1995)
**Developers:** Corinna Cortes, Vladimir Vapnik

**Purpose:** Classification with maximum margin

**Mathematical Explanation:**
Find hyperplane $w^Tx + b = 0$ that maximizes margin:
$$\min_{w,b} \frac{1}{2}||w||^2$$
$$\text{subject to } y_i(w^Tx_i + b) \geq 1, \quad \forall i$$

**Dual form:** 
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

**Kernel trick:** $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ for nonlinear boundaries

**Applications:**
- Text classification
- Image recognition
- Bioinformatics (protein classification)
- Handwriting recognition
- Face detection

---

## 9. Computational Geometry and Physics

### 9.1 Fast Multipole Method (1987)
**Developers:** Leslie Greengard, Vladimir Rokhlin

**Purpose:** Compute N-body interactions in $O(N)$ time

**Mathematical Explanation:**
For gravitational/electrostatic potential:
$$\phi(x) = \sum_{i=1}^N \frac{q_i}{|x - y_i|}$$

Naive computation: $O(N^2)$

**Key idea:** Group distant particles, use multipole expansion:
$$\frac{1}{|x-y|} = \sum_{l=0}^\infty \sum_{m=-l}^l M_l^m(y) \frac{Y_l^m(\hat{x})}{r^{l+1}}$$

**Hierarchical approach:**
1. Build octree/quadtree
2. Compute multipole moments
3. Translate between levels
4. Compute interactions

**Complexity:** $O(N)$ or $O(N \log N)$

**Applications:**
- Astrophysics (galaxy simulations)
- Molecular dynamics
- Protein folding simulations
- Electromagnetic scattering
- Acoustics
- Computer graphics (radiosity)

---

### 9.2 Finite Element Method (FEM) (1940s-1960s)
**Purpose:** Solve PDEs on complex domains

**Mathematical Explanation:**
Approximate solution in finite-dimensional space:
$$u(x) \approx \sum_{i=1}^n u_i \phi_i(x)$$

**Weak formulation** of PDE $-\nabla \cdot (a\nabla u) = f$:
$$\int_\Omega a\nabla u \cdot \nabla v \, dx = \int_\Omega fv \, dx$$

**Discretization leads to linear system:** $Ku = F$

**Applications:**
- Structural mechanics
- Heat transfer
- Fluid dynamics
- Electromagnetics
- Biomechanics
- Geophysics

---

## 10. Data Compression and Encoding

### 10.1 JPEG Compression (1992)
**Purpose:** Lossy image compression

**Mathematical Explanation:**
**Pipeline:**
1. **Color space transform:** RGB → YCbCr
2. **Block splitting:** 8×8 blocks
3. **DCT:** For each block
   $$F(u,v) = \sum_{x=0}^7 \sum_{y=0}^7 f(x,y) \cos\frac{(2x+1)u\pi}{16} \cos\frac{(2y+1)v\pi}{16}$$
4. **Quantization:** $F_Q(u,v) = \text{round}(F(u,v)/Q(u,v))$
5. **Entropy coding:** Huffman or arithmetic coding

**Compression ratio:** Typically 10:1 to 20:1

**Applications:**
- Digital photography
- Web images
- Medical imaging
- Satellite imagery
- Video compression (MPEG basis)

---

### 10.2 Huffman Coding (1952)
**Developer:** David Huffman

**Purpose:** Optimal prefix-free encoding

**Mathematical Explanation:**
Given symbol frequencies, build binary tree:
1. Create leaf for each symbol
2. Repeatedly merge two lowest-frequency nodes
3. Assign 0/1 to edges
4. Code = path from root to leaf

**Optimality:** Minimizes expected code length:
$$L = \sum_{i=1}^n p_i l_i$$

**Applications:**
- JPEG, PNG compression
- ZIP files
- Network protocols
- Fax transmission

---

### 10.3 LZ77/LZ78 (Lempel-Ziv) Compression (1977-1978)
**Developers:** Abraham Lempel, Jacob Ziv

**Purpose:** Dictionary-based lossless compression

**Mathematical Explanation:**
**LZ77:** Sliding window, replace with (offset, length, next char)
**LZ78:** Build dictionary dynamically

**Compression:** 
$$\text{Compressed} = \{(p, l, c)\}$$ 
where $p$ = position, $l$ = length, $c$ = next character

**Applications:**
- GIF images
- ZIP/GZIP files
- PNG compression
- Network protocols

---

## 11. Number Theory and Algebra

### 11.1 Euclidean Algorithm (300 BC)
**Developer:** Euclid

**Purpose:** Compute greatest common divisor (GCD)

**Mathematical Explanation:**
$$\gcd(a, b) = \gcd(b, a \mod b), \quad \gcd(a, 0) = a$$

```
gcd(a, b):
    while b ≠ 0:
        temp = b
        b = a mod b
        a = temp
    return a
```

**Complexity:** $O(\log \min(a,b))$

**Extended Euclidean:** Also finds $x, y$ such that $ax + by = \gcd(a,b)$

**Applications:**
- Fraction simplification
- Modular arithmetic (RSA)
- Cryptography
- Number theory
- Polynomial GCD

---

### 11.2 Integer Relation Detection (PSLQ, 1977)
**Developers:** Helaman Ferguson, Rodney Forcade

**Purpose:** Find integer relations among real numbers

**Mathematical Explanation:**
Given $x_1, ..., x_n \in \mathbb{R}$, find integers $a_1, ..., a_n$ (not all zero) such that:
$$a_1x_1 + a_2x_2 + ... + a_nx_n = 0$$

**PSLQ Algorithm:** Iterative method using orthogonal projections

**Applications:**
- Experimental mathematics
- Identifying constants (e.g., $\pi$, $e$ relations)
- Quantum field theory
- Discovering formulas (Bailey-Borwein-Plouffe for $\pi$)

---

## 12. String and Sequence Algorithms

### 12.1 Boyer-Moore String Search (1977)
**Purpose:** Fast substring search

**Mathematical Explanation:**
Use two heuristics:
1. **Bad character:** Skip based on rightmost occurrence
2. **Good suffix:** Skip based on matched suffix

**Preprocessing:** Build skip tables in $O(m + |\Sigma|)$ time

**Search complexity:** 
- Best case: $O(n/m)$
- Worst case: $O(n \cdot m)$
- Average: Sublinear in $n$

**Applications:**
- Text editors (search function)
- Bioinformatics (DNA sequence matching)
- Network intrusion detection
- Plagiarism detection

---

### 12.2 Knuth-Morris-Pratt (KMP) (1977)
**Purpose:** Linear-time string matching

**Mathematical Explanation:**
Build failure function $\pi[i]$ = length of longest proper prefix of pattern[0..i] that is also suffix

```
kmp_search(text, pattern):
    π = compute_failure_function(pattern)
    q = 0  // matched length
    for i = 0 to length(text) - 1:
        while q > 0 and pattern[q] ≠ text[i]:
            q = π[q - 1]
        if pattern[q] == text[i]:
            q = q + 1
        if q == length(pattern):
            return i - q + 1  // match found
    return NOT_FOUND
```

**Complexity:** $O(n + m)$

**Applications:**
- Text processing
- DNA sequence analysis
- Compiler design
- Network packet inspection

---

## 13. Hashing

### 13.1 Hash Tables (1953)
**Purpose:** Fast key-value lookup

**Mathematical Explanation:**
Map key to index via hash function:
$$h: K \to \{0, 1, ..., m-1\}$$

**Collision resolution:**
- **Chaining:** Store linked list at each bucket
- **Open addressing:** Probe sequence
  - Linear: $(h(k) + i) \mod m$
  - Quadratic: $(h(k) + c_1i + c_2i^2) \mod m$
  - Double hashing: $(h_1(k) + i \cdot h_2(k)) \mod m$

**Load factor:** $\alpha = n/m$

**Expected complexity:** $O(1)$ for search/insert/delete when $\alpha$ constant

**Applications:**
- Databases (indexing)
- Compilers (symbol tables)
- Caches
- Deduplication
- Password storage (with cryptographic hashing)
- Blockchain (Merkle trees)

---

## 14. Specialized Algorithms

### 14.1 RANSAC (1981)
**Developers:** Martin Fischler, Robert Bolles

**Purpose:** Robust parameter estimation with outliers

**Mathematical Explanation:**
```
best_model = null
best_inliers = 0

for i = 1 to N iterations:
    sample = randomly select min points
    model = fit model to sample
    inliers = count points within threshold of model
    if inliers > best_inliers:
        best_inliers = inliers
        best_model = refit model to all inliers
        
return best_model
```

**Probability of success:**
$$P = 1 - (1 - w^s)^N$$
where $w$ = inlier ratio, $s$ = sample size

**Applications:**
- Computer vision (homography estimation)
- 3D reconstruction
- Camera calibration
- Object recognition
- Autonomous driving

---

### 14.2 Union-Find (Disjoint Set) (1964)
**Purpose:** Track disjoint set partitions

**Mathematical Explanation:**
**Operations:**
- **Find(x):** Determine which set $x$ belongs to
- **Union(x, y):** Merge sets containing $x$ and $y$

**Optimizations:**
- **Path compression:** Point directly to root during Find
- **Union by rank:** Attach smaller tree to larger

```
find(x):
    if parent[x] ≠ x:
        parent[x] = find(parent[x])  // path compression
    return parent[x]

union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rank[rootX] < rank[rootY]:
        parent[rootX] = rootY
    else if rank[rootX] > rank[rootY]:
        parent[rootY] = rootX
    else:
        parent[rootY] = rootX
        rank[rootX]++
```

**Complexity:** Amortized $O(\alpha(n))$ where $\alpha$ is inverse Ackermann (effectively constant)

**Applications:**
- Kruskal's minimum spanning tree
- Connected components
- Image segmentation
- Network connectivity
- Percolation theory

---

## 15. Algorithms in Context: Historical Impact

### Timeline of Algorithmic Breakthroughs

**Pre-1900:**
- 300 BC: Euclidean Algorithm
- 1805: FFT (discovered by Gauss, forgotten)
- 1847: Gradient Descent (Cauchy)

**1940s-1950s:**
- 1945: Merge Sort (von Neumann)
- 1946: Monte Carlo (von Neumann, Ulam, Metropolis)
- 1947: Simplex (Dantzig)
- 1950: Krylov Methods (Hestenes, Stiefel, Lanczos)
- 1951: Matrix Decompositions (Householder)
- 1952: Huffman Coding
- 1956: Ford-Fulkerson
- 1957: Fortran Compiler (Backus)
- 1958: Bellman-Ford
- 1959: Dijkstra, QR Algorithm

**1960s:**
- 1960: Kalman Filter
- 1962: Quicksort (Hoare)
- 1964: Heap Sort, Union-Find
- 1965: FFT rediscovered (Cooley-Tukey)
- 1967: Viterbi Algorithm
- 1968: A* Search

**1970s:**
- 1976: Diffie-Hellman
- 1977: RSA, LZ77, Boyer-Moore, KMP, EM Algorithm, Integer Relations
- 1978: LZ78

**1980s:**
- 1981: RANSAC
- 1986: Backpropagation
- 1987: Fast Multipole Method
- 1989: Q-Learning

**1990s:**
- 1992: JPEG
- 1995: SVM
- 1998: PageRank

**2000s-Present:**
- Deep learning revolution
- Transformer architecture (2017)
- AlphaGo (2016), AlphaFold (2020)

---