# 7.1 Introduction

### 1. 概念引入与背景 (Introduction)

许多物理问题涉及多个独立但相互关联的组件。例如：

- **电网**中的电流和电压。
- **机械系统**中的各个质量块。
- **化学系统**中的元素或化合物。
- **生物系统**中的物种。

这些问题通常表现为包含两个或更多微分方程的**方程组 (System)**。一个关键结论是：这些系统总能被写成**一阶微分方程 (First-order differential equations)** 的形式。

#### 变量表示法

在研究中，我们通常设定：

- $t$ 为**自变量 (Independent variable)**。
- $x_1, x_2, x_3, \dots, x_n$ 为**因变量 (Dependent variables)**，它们都是关于 $t$ 的函数。
- **导数符号**：对 $t$ 的求导记为 $\frac{dx_1}{dt}$ 或 $x_1'$。在某些处理中，也会在函数上方加点表示，如 $\dot{x}$。

------

### 2. 物理模型推导 (Physical Models)

#### A. 弹簧-质量块系统 (Mechanical System)

考虑一个在无摩擦表面上的双质量、三弹簧系统（见图 7.1.1）。

- 质量分别为 $m_1, m_2$。
- 弹簧系数分别为 $k_1, k_2, k_3$。
- 外部驱动力为 $F_1(t), F_2(t)$。

**数学推导：** 根据牛顿第二定律（受力平衡），我们可以得到描述位移 $x_1$ 和 $x_2$ 的二阶方程组：

1. **对于 $m_1$**： $$m_1 \frac{d^2x_1}{dt^2} = k_2(x_2 - x_1) - k_1x_1 + F_1(t)$$ 整理得： $$m_1 \frac{d^2x_1}{dt^2} = -(k_1 + k_2)x_1 + k_2x_2 + F_1(t) \quad (1)$$
2. **对于 $m_2$**： $$m_2 \frac{d^2x_2}{dt^2} = -k_3x_2 - k_2(x_2 - x_1) + F_2(t)$$ 整理得： $$m_2 \frac{d^2x_2}{dt^2} = k_2x_1 - (k_2 + k_3)x_2 + F_2(t) \quad (2)$$

#### B. 并联 LRC 电路 (Electrical System)

考虑一个包含电感 $L$、电容 $C$ 和电阻 $R$ 的并联电路（见图 7.1.2）。 设 $V$ 为电容两端的电压降，$I$ 为通过电感的电流。该系统的描述方程为： $$\frac{dI}{dt} = \frac{V}{L}$$ $$\frac{dV}{dt} = -\frac{I}{C} - \frac{V}{RC}$$ 这是一个天然的**一阶方程组**。

------

### 3. 高阶方程向一阶方程组的转换 (Transformation)

这是本章最重要的技巧之一。高阶方程之所以要转换成一阶方程组，是因为大多数生成数值近似解的代码（如第 8 章所述）都是为一阶方程组编写的。

#### 转换逻辑与步骤

对于一个任意的 $n$ 阶微分方程： $$y^{(n)} = F(t, y, y', y'', \dots, y^{(n-1)}) \quad (7)$$

我们通过引入 $n$ 个新变量 $x_1, x_2, \dots, x_n$ 来进行转换：

1. 令 $x_1 = y$。
2. 令 $x_2 = y'$，由此推得 $x_1' = x_2$。
3. 令 $x_3 = y''$，由此推得 $x_2' = x_3$。
4. 依此类推，直到 $x_n = y^{(n-1)}$，则 $x_{n-1}' = x_n$。
5. 最后，原方程 (7) 变为 $x_n' = F(t, x_1, x_2, \dots, x_n)$。

#### **实例分析 1 (Example 1)**

**问题**：将 $u'' + \frac{1}{8}u' + u = 0$ 转换为一阶方程组。

**解题步骤**：

1. **定义变量**：设 $x_1 = u$，$x_2 = u'$。

2. 建立一阶导数关系

   ：

   - 由定义直接得出：$x_1' = u' = x_2$。
   - 由原方程得出：$u'' = -u - \frac{1}{8}u'$。因为 $u'' = x_2'$，代入变量得：$x_2' = -x_1 - \frac{1}{8}x_2$。

3. **最终形式**： $$\begin{cases} x_1' = x_2 \ x_2' = -x_1 - \frac{1}{8}x_2 \end{cases}$$

------

### 4. 一般方程组与解的定义 (General Systems and Solutions)

一阶方程组的最一般形式为： $$\begin{cases} x_1' = F_1(t, x_1, x_2, \dots, x_n) \ \dots \ x_n' = F_n(t, x_1, x_2, \dots, x_n) \end{cases} \quad (11)$$

- **解 (Solution)**：在区间 $I: \alpha < t < \beta$ 上，解由 $n$ 个可微函数组成：$x_1 = \phi_1(t), x_2 = \phi_2(t), \dots, x_n = \phi_n(t)$。
- **初值问题 (Initial Value Problem)**：在方程组的基础上给出 $n$ 个初值条件 $x_i(t_0) = x_i^0$。
- **几何解释**：解可以看作 $n$ 维空间中的一条参数曲线（轨迹/路径）。当 $n=2$ 时，轨迹位于 $x_1x_2$ 平面内。

------

### 5. 存在性与唯一性定理 (Existence and Uniqueness)

#### 定理 7.1.1 (针对一般/非线性系统)

若函数 $F_1, \dots, F_n$ 及其对所有因变量的偏导数 $\frac{\partial F_i}{\partial x_j}$ 在区域 $R$ 内连续，且初值点在 $R$ 内，则在区间 $|t - t_0| < h$ 内存在唯一解。

- **备注**：定理未对 $F_i$ 关于 $t$ 的偏导数做要求；$h$ 可能很小。

#### 定理 7.1.2 (针对线性系统)

如果方程组是**线性**的，即形如： $$x_i' = p_{i1}(t)x_1 + \dots + p_{in}(t)x_n + g_i(t) \quad (14)$$ 只要系数 $p_{ij}$ 和函数 $g_i$ 在开区间 $I$ 上连续，则对于该区间内的任意 $t_0$ 和任意初值，在**整个区间 $I$** 上存在唯一解。

- **关键区别**：线性系统的解存在于整个连续区间内，而不仅仅是一个小的邻域；且初值是完全任意的。

#### 术语定义：

- **齐次 (Homogeneous)**：若所有 $g_i(t) = 0$，则称系统为齐次；否则为非齐次。

------



# 7.2 Matrices

### 1. 矩阵的基本定义与背景

- **历史背景**：矩阵的性质最早由英国代数学家 **阿瑟·凯莱 (Arthur Cayley)** 在 1858 年的一篇论文中深入探讨，而“矩阵”一词则由他的好友 **詹姆斯·西尔维斯特 (James Sylvester)** 于 1850 年引入。凯莱在 1849 年至 1863 年从事法律工作期间完成了许多杰出的数学工作，随后成为剑桥大学数学教授。
- **定义**：一个矩阵 $\mathbf{A}$ 是由数字或元素组成的矩形阵列，排列成 $m$ 行和 $n$ 列。 $$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \ a_{21} & a_{22} & \cdots & a_{2n} \ \vdots & \vdots & & \vdots \ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} \quad (1)$$ 我们称 $\mathbf{A}$ 为 **$m \times n$ 矩阵**。
- **元素表示**：位于第 $i$ 行和第 $j$ 列的元素记为 $a_{ij}$，第一个下标标识行，第二个下标标识列。矩阵也可简记为 $(a_{ij})$。本节允许元素为**复数**。

------

### 2. 矩阵的相关变换

对于矩阵 $\mathbf{A} = (a_{ij})$，有以下重要概念：

1. **转置 (Transpose)**：记作 $\mathbf{A}^T$，通过交换 $\mathbf{A}$ 的行和列得到。即若 $\mathbf{A} = (a_{ij})$，则 $\mathbf{A}^T = (a_{ji})$。
2. **共轭 (Conjugate)**：记作 $\bar{\mathbf{A}}$，将 $\mathbf{A}$ 中的每个元素 $a_{ij}$ 替换为其复共轭 $\bar{a}_{ij}$。
3. **伴随 (Adjoint)**：记作 $\mathbf{A}^*$，定义为共轭矩阵的转置，即 $\mathbf{A}^* = \bar{\mathbf{A}}^T$。

**示例**： 设 $\mathbf{A} = \begin{pmatrix} 3 & 2-i \ 4+3i & -5+2i \end{pmatrix}$，则：

- $\mathbf{A}^T = \begin{pmatrix} 3 & 4+3i \ 2-i & -5+2i \end{pmatrix}$ （交换行列）。
- $\bar{\mathbf{A}} = \begin{pmatrix} 3 & 2+i \ 4-3i & -5-2i \end{pmatrix}$ （虚部取反）。
- $\mathbf{A}^* = \begin{pmatrix} 3 & 4-3i \ 2+i & -5-2i \end{pmatrix}$ （对 $\bar{\mathbf{A}}$ 进行转置）。

------

### 3. 特殊矩阵与向量

- **方阵 (Square Matrix)**：行数与列数相等（$m=n$）的矩阵。$n$ 行 $n$ 列的方阵称为 $n$ 阶方阵。
- **向量 (Vector/Column Vector)**：视为 $n \times 1$ 矩阵。通常用加粗小写字母表示，如 $\mathbf{x}, \mathbf{y}, \boldsymbol{\xi}$。
- **行向量**：$n \times 1$ 列向量的转置 $\mathbf{x}^T$ 是一个 $1 \times n$ 的行向量。

------

### 4. 矩阵的代数性质

1. **相等 (Equality)**：若两个 $m \times n$ 矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 的所有对应元素都相等（即对所有 $i, j$ 都有 $a_{ij} = b_{ij}$），则称它们相等。

2. **零矩阵 (Zero)**：所有元素均为零的矩阵（或向量），记作 $\mathbf{0}$。

3. 加法 (Addition)

   ：两个 $m \times n$ 矩阵相加是将对应元素相加。满足：

   - **交换律**：$\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$。
   - **结合律**：$\mathbf{A} + (\mathbf{B} + \mathbf{C}) = (\mathbf{A} + \mathbf{B}) + \mathbf{C}$。

4. 数乘 (Multiplication by a Number)

   ：标量 $\alpha$ 与矩阵 $\mathbf{A}$ 相乘，定义为 $\alpha \mathbf{A} = (\alpha a_{ij})$，即每个元素都乘以 $\alpha$。满足分配律：

   - $\alpha(\mathbf{A} + \mathbf{B}) = \alpha \mathbf{A} + \alpha \mathbf{B}$。
   - $(\alpha + \beta)\mathbf{A} = \alpha \mathbf{A} + \beta \mathbf{A}$。

5. **减法 (Subtraction)**：定义为 $\mathbf{A} - \mathbf{B} = \mathbf{A} + (-1)\mathbf{B}$，即对应元素相减。

------

### 5. 矩阵乘法

- **定义条件**：只有当第一个矩阵 $\mathbf{A}$ 的**列数**等于第二个矩阵 $\mathbf{B}$ 的**行数**时，乘积 $\mathbf{AB}$ 才有定义。

- **计算规则**：若 $\mathbf{A}$ 是 $m \times n$，$\mathbf{B}$ 是 $n \times r$，则 $\mathbf{C} = \mathbf{AB}$ 是 $m \times r$ 矩阵。其元素 $c_{ij}$ 计算公式为： $$c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj} \quad (9)$$

- 乘法性质

  ：

  - **结合律**：$(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$。
  - **分配律**：$\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$。
  - **不可交换性**：一般情况下，$\mathbf{AB} \neq \mathbf{BA}$。

------

### 6. 向量乘法 (Vector Products)

对于具有 $n$ 个分量的向量 $\mathbf{x}$ 和 $\mathbf{y}$：

1. **点积 (Dot Product)**：记作 $\mathbf{x}^T \mathbf{y}$。 $$\mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i \quad (13)$$ 满足性质：$\mathbf{x}^T \mathbf{y} = \mathbf{y}^T \mathbf{x}$；$\mathbf{x}^T(\mathbf{y} + \mathbf{z}) = \mathbf{x}^T \mathbf{y} + \mathbf{x}^T \mathbf{z}$；$(\alpha \mathbf{x})^T \mathbf{y} = \alpha(\mathbf{x}^T \mathbf{y}) = \mathbf{x}^T(\alpha \mathbf{y})$。
2. **标量积/内积 (Scalar or Inner Product)**：记作 $(\mathbf{x}, \mathbf{y})$。 $$(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} x_i \bar{y}_i \quad (15)$$ 与点积的关系：$(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \bar{\mathbf{y}}$。
3. **长度/模 (Length/Magnitude)**：记作 $|\mathbf{x}|$，定义为 $(\mathbf{x}, \mathbf{x})^{1/2} = (\sum |x_i|^2)^{1/2}$。只有零向量长度为 0。
4. **正交 (Orthogonal)**：若 $(\mathbf{x}, \mathbf{y}) = 0$，则称两向量正交。

------

### 7. 单位矩阵与逆矩阵

- **单位矩阵 (Identity Matrix)**：主对角线全为 1，其余为 0 的方阵 $\mathbf{I}$。满足 $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$。

- **逆矩阵 (Inverse)**：若存在 $\mathbf{B}$ 使得 $\mathbf{AB} = \mathbf{I}$ 且 $\mathbf{BA} = \mathbf{I}$，则 $\mathbf{A}$ 是 **非奇异的 (Nonsingular)** 或 **可逆的**。记作 $\mathbf{B} = \mathbf{A}^{-1}$。

- **行列式与可逆性**：$\mathbf{A}$ 可逆的充分必要条件是其行列式 $\det \mathbf{A} \neq 0$。

- 代数余子式计算法

  ：

  - **余子式 $M_{ij}$**：划去第 $i$ 行第 $j$ 列后的行列式。
  - **代数余子式 $C_{ij}$**：$C_{ij} = (-1)^{i+j}M_{ij}$。
  - **逆矩阵元素**：$b_{ij} = \frac{C_{ji}}{\det \mathbf{A}}$ （注意下标互换）。

------

### 8. 高斯消元法 (Gaussian Elimination)

计算 $\mathbf{A}^{-1}$ 的更有效方法是**初等行变换**：

1. 交换两行。
2. 一行乘以非零标量。
3. 将一行的倍数加到另一行。

#### **示例 2 分析：求 $\mathbf{A}$ 的逆**

$$\mathbf{A} = \begin{pmatrix} 1 & -1 & -1 \ 3 & -1 & 2 \ 2 & 2 & 3 \end{pmatrix}$$

**推导步骤**：

1. **构造增广矩阵 $(\mathbf{A} | \mathbf{I})$**。
2. **第一列消元**：用 $-3 \times R_1 + R_2$ 和 $-2 \times R_1 + R_3$ 使得 $a_{21}=0, a_{31}=0$。
3. **第二列归一化与消元**：将 $R_2$ 乘以 $1/2$ 得到主元 1，再利用 $R_2$ 消除 $a_{12}$ 和 $a_{32}$。
4. **第三列处理**：将 $R_3$ 乘以 $-1/5$ 得到主元 1，再消除 $a_{13}$ 和 $a_{23}$。
5. **最终结果**：当左侧变为 $\mathbf{I}$ 时，右侧即为 $\mathbf{A}^{-1}$。

------

### 9. 矩阵函数 (Matrix Functions)

若矩阵元素是变量 $t$ 的函数，记作 $\mathbf{A}(t) = (a_{ij}(t))$。

- **导数**：$\frac{d\mathbf{A}}{dt} = \left( \frac{da_{ij}}{dt} \right)$，即对每个元素求导。

- **积分**：$\int_a^b \mathbf{A}(t)dt = \left( \int_a^b a_{ij}(t)dt \right)$。

- 运算规则

  ：

  1. $\frac{d}{dt}(\mathbf{CA}) = \mathbf{C}\frac{d\mathbf{A}}{dt}$ （$\mathbf{C}$ 为常数矩阵）。
  2. $\frac{d}{dt}(\mathbf{A} + \mathbf{B}) = \frac{d\mathbf{A}}{dt} + \frac{d\mathbf{B}}{dt}$。
  3. **乘积法则**：$\frac{d}{dt}(\mathbf{AB}) = \mathbf{A}\frac{d\mathbf{B}}{dt} + \frac{d\mathbf{A}}{dt}\mathbf{B}$ （注意不可交换顺序）。

**特别提醒**：矩阵的某些操作（如求平方）**不能**通过对每个元素单独操作完成。

------



# 7.3 Systems of Linear Algebraic Equations;Linear Independence,Eigenvalues,Eigenvectors

### 1. 线性代数方程组 (Systems of Linear Algebraic Equations)

一个包含 $n$ 个变量的 $n$ 阶线性方程组可以表示为： $$a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n = b_1$$ $$\dots$$ $$a_{n1}x_1 + a_{n2}x_2 + \dots + a_{nn}x_n = b_n \quad (1)$$ 其**矩阵形式**为： $$\mathbf{Ax} = \mathbf{b} \quad (2)$$ 其中 $\mathbf{A}$ 是 $n \times n$ 系数矩阵，$\mathbf{b}$ 是已知向量，$\mathbf{x}$ 是待求向量。

#### **分类与性质：**

- **齐次 (Homogeneous)**：若 $\mathbf{b} = \mathbf{0}$。

- **非齐次 (Nonhomogeneous)**：若 $\mathbf{b} \neq \mathbf{0}$。

- **非奇异 (Nonsingular)**：若 $\det \mathbf{A} \neq 0$，则方程组有唯一解 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$。对于齐次方程组，这意味着只有**平凡解 (Trivial solution)** $\mathbf{x} = \mathbf{0}$。

- 奇异 (Singular)

  ：若 $\det \mathbf{A} = 0$，则解可能不存在，或存在但不唯一。此时：

  - **齐次系统**：有无穷多非零解。
  - **非齐次系统**：仅当向量 $\mathbf{b}$ 满足条件 $(\mathbf{b}, \mathbf{y}) = 0$ 时有解，其中 $\mathbf{y}$ 是伴随矩阵方程 $\mathbf{A}^*\mathbf{y} = \mathbf{0}$ 的所有解。
  - **通解结构**：$\mathbf{x} = \mathbf{x}^{(0)} + \boldsymbol{\xi}$，其中 $\mathbf{x}^{(0)}$ 是特解，$\boldsymbol{\xi}$ 是齐次方程的通解。

#### **求解方法：增广矩阵 (Augmented Matrix)**

通过构造 $(\mathbf{A} | \mathbf{b})$ 并进行初等行变换，将 $\mathbf{A}$ 转化为**上三角矩阵 (Upper triangular matrix)**。

------

### 2. 实例分析 1 & 2 (Example 1 & 2)

#### **实例 1：唯一解情况**

**方程组**： $$\begin{cases} x_1 - 2x_2 + 3x_3 = 7 \ -x_1 + x_2 - 2x_3 = -5 \ 2x_1 - x_2 - x_3 = 4 \end{cases}$$ **解题逻辑**：

1. **构造增广矩阵**：从系数和常数项得到矩阵。

2. 行变换过程

   ：

   - 将第一行加到第二行；将第一行的 $-2$ 倍加到第三行，使第一列下方归零。
   - 将第二行乘以 $-1$。
   - 将第二行的 $-3$ 倍加到第三行，使第二列下方归零。
   - 最后将第三行除以 $-4$，得到上三角形式。

3. **回代 (Back-substitution)**：从最后一行得 $x_3=1$，代入前一行得 $x_2=-1$，再代入第一行得 $x_1=2$。解是唯一的，说明矩阵非奇异。

#### **实例 2：奇异与一致性条件**

若将实例 1 第三行 $x_3$ 的系数改为 3，常数项改为 $b_3$。行变换后第三行变为 $0 = b_1 + 3b_2 + b_3$。

- **存在解的条件**：$b_1 + 3b_2 + b_3 = 0$。
- **解的形式**：若满足条件，令 $x_3 = \alpha$（任意常数），通过回代可得 $\mathbf{x} = \alpha \begin{pmatrix} -1 \ 1 \ 1 \end{pmatrix} + \begin{pmatrix} -4 \ -3 \ 0 \end{pmatrix}$。

------

### 3. 线性相关与线性无关 (Linear Dependence and Independence)

**定义**：对于向量集合 $\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k)}$，若存在不全为零的常数 $c_i$ 使得： $$c_1\mathbf{x}^{(1)} + \dots + c_k\mathbf{x}^{(k)} = \mathbf{0} \quad (17)$$ 则称这些向量**线性相关 (Linearly dependent)**。若只有当所有 $c_i=0$ 时等式才成立，则称**线性无关 (Linearly independent)**。

**行列式判别法**：对于 $n$ 个 $n$ 维向量，构造矩阵 $\mathbf{X}$（各向量为列）：

- 若 $\det \mathbf{X} \neq 0$，则线性无关。
- 若 $\det \mathbf{X} = 0$，则线性相关。

**向量函数**：若向量是关于 $t$ 的函数 $\mathbf{x}^{(i)}(t)$，在区间 $\alpha < t < \beta$ 上线性相关的定义是：存在不全为零的常数使得在**整个区间**上线性组合恒为零。

------

### 4. 特征值与特征向量 (Eigenvalues and Eigenvectors)

**基本方程**：寻找非零向量 $\mathbf{x}$ 和标量 $\lambda$ 使得： $$\mathbf{Ax} = \lambda \mathbf{x} \quad \text{或} \quad (\mathbf{A} - \lambda \mathbf{I})\mathbf{x} = \mathbf{0} \quad (25, 26)$$ **特征方程 (Characteristic Equation)**： $$\det(\mathbf{A} - \lambda \mathbf{I}) = 0 \quad (27)$$ 这是一个关于 $\lambda$ 的 $n$ 次多项式。其根为**特征值**，对应的非零解 $\mathbf{x}$ 为**特征向量**。

#### **重要概念**：

- **代数重数 (Algebraic Multiplicity, $m$)**：特征根在特征方程中出现的次数。
- **几何重数 (Geometric Multiplicity, $q$)**：与该特征值对应的线性无关特征向量的数量。
- **性质**：$1 \leq q \leq m$。若所有特征值均不同（单根），则所有特征向量线性无关。

------

### 5. 实例分析 4 & 5 (Example 4 & 5)

#### **实例 4：2x2 矩阵**

$\mathbf{A} = \begin{pmatrix} 3 & -1 \ 4 & -2 \end{pmatrix}$

1. **特征方程**：$\det \begin{pmatrix} 3-\lambda & -1 \ 4 & -2-\lambda \end{pmatrix} = \lambda^2 - \lambda - 2 = (\lambda-2)(\lambda+1) = 0$。

2. **特征值**：$\lambda_1 = 2, \lambda_2 = -1$。

3. 求特征向量

   ：

   - $\lambda = 2$ 时：方程变为 $x_1 - x_2 = 0$，取 $\mathbf{x}^{(1)} = \begin{pmatrix} 1 \ 1 \end{pmatrix}$。
   - $\lambda = -1$ 时：方程变为 $4x_1 - x_2 = 0$，取 $\mathbf{x}^{(2)} = \begin{pmatrix} 1 \ 4 \end{pmatrix}$。

#### **实例 5：3x3 实对称矩阵**

1. **特征值**：通过特征方程得到 $\lambda_1 = 2, \lambda_2 = -1, \lambda_3 = -1$。

2. 特征向量

   ：

   - $\lambda = 2$（单根）对应一个特征向量 $\begin{pmatrix} 1 & 1 & 1 \end{pmatrix}^T$。
   - $\lambda = -1$（二重根）对应两个线性无关的特征向量，满足 $x_1+x_2+x_3=0$。

------

### 6. 埃尔米特矩阵 (Hermitian Matrices)

若 $\mathbf{A}^* = \mathbf{A}$（对于实矩阵即为对称矩阵 $\mathbf{A}^T = \mathbf{A}$），其具有以下 4 条关键性质：

1. 所有特征值均为**实数**。
2. 始终存在 $n$ 个线性无关的特征向量（即 $q=m$）。
3. 不同特征值对应的特征向量彼此**正交**。
4. 对于重根特征值，总可以选出彼此正交的特征向量。

**备注**：特征向量在乘以非零常数后仍是特征向量，通常为了方便计算会对其进行**标准化 (Normalized)**。

------



# 7.4 Basic Theory of Systems of First-Order Linear Equations

### 一、 一阶线性方程组的一般形式

#### 1. 标量形式与向量形式的转换

一个由 $n$ 个一阶线性方程组成的系统可以表示为： $$x'*i = p*{i1}(t)x_1 + \dots + p_{in}(t)x_n + g_i(t), \quad i=1, \dots, n \quad$$ 为了简化表达和计算，我们引入向量和矩阵：

- **解向量** $\mathbf{x}(t)$：分量为 $x_1(t), \dots, x_n(t)$。
- **系数矩阵** $\mathbf{P}(t)$：元素为 $p_{ij}(t)$ 的 $n \times n$ 矩阵。
- **非齐次项向量** $\mathbf{g}(t)$：分量为 $g_1(t), \dots, g_n(t)$。

由此，系统可以写成简洁的**矩阵方程**： $$\mathbf{x}' = \mathbf{P}(t)\mathbf{x} + \mathbf{g}(t) \quad \text{--- (2)} \quad$$

#### 2. 连续性假设与存在唯一性

**前提条件：** 我们假设 $\mathbf{P}$ 和 $\mathbf{g}$ 在区间 $\alpha < t < \beta$ 内是**连续的**。这意味着矩阵和向量中的每一个标量函数（$p_{ij}$ 和 $g_i$）在该区间内都必须连续。 **结论：** 根据定理 7.1.2，这一条件足以保证在给定区间内方程 (2) 存在解。

#### 3. 齐次方程 (Homogeneous Equation)

当 $\mathbf{g}(t) = \mathbf{0}$ 时，方程变为： $$\mathbf{x}' = \mathbf{P}(t)\mathbf{x} \quad \text{--- (3)} \quad$$ 这是我们讨论基本理论的核心起点。

------

### 二、 叠加原理 (Theorem 7.4.1)

#### 1. 定理陈述

**定理 7.4.1：** 如果向量函数 $\mathbf{x}^{(1)}$ 和 $\mathbf{x}^{(2)}$ 是齐次系统 (3) 的解，那么它们的线性组合 $c_1\mathbf{x}^{(1)} + c_2\mathbf{x}^{(2)}$ 也是该系统的解，其中 $c_1, c_2$ 为任意常数。

#### 2. 数学证明推导

我们要验证 $\mathbf{x} = c_1\mathbf{x}^{(1)} + c_2\mathbf{x}^{(2)}$ 是否满足 $\mathbf{x}' = \mathbf{P}\mathbf{x}$：

1. **左侧求导：** 利用导数的线性性质， $$\frac{d}{dt}(c_1\mathbf{x}^{(1)} + c_2\mathbf{x}^{(2)}) = c_1\frac{d\mathbf{x}^{(1)}}{dt} + c_2\frac{d\mathbf{x}^{(2)}}{dt} \quad$$
2. **代入已知条件：** 因为 $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}$ 是解，所以 $\mathbf{x}^{(1)'} = \mathbf{P}\mathbf{x}^{(1)}$ 且 $\mathbf{x}^{(2)'} = \mathbf{P}\mathbf{x}^{(2)}$。
3. **合并：** $$c_1(\mathbf{P}\mathbf{x}^{(1)}) + c_2(\mathbf{P}\mathbf{x}^{(2)}) = \mathbf{P}(c_1\mathbf{x}^{(1)} + c_2\mathbf{x}^{(2)}) \quad$$ *此处利用了矩阵乘法的分配律。*
4. **结论：** 左右两侧相等，证毕。该原理可推广至 $k$ 个解的线性组合。

------

### 三、 朗斯基行列式与基本解组

#### 1. 基本定义

设 $\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}$ 是系统 (3) 的 $n$ 个解。我们构造一个**解矩阵** $\mathbf{X}(t)$，其列向量由这些解组成： $$\mathbf{X}(t) = \begin{pmatrix} x_{11}(t) & \dots & x_{1n}(t) \ \vdots & & \vdots \ x_{n1}(t) & \dots & x_{nn}(t) \end{pmatrix} \quad$$ **朗斯基行列式 (Wronskian)** 定义为该矩阵的行列式： $$W[\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}](https://notebooklm.google.com/notebook/t) = \det \mathbf{X}(t) \quad$$

#### 2. 线性无关性与通解 (Theorem 7.4.2)

**定理 7.4.2：** 如果 $n$ 个解在区间内每一点都线性无关（即 $W \neq 0$），那么系统 (3) 的**每一个**解都可以唯一地表示为这些解的线性组合： $$\mathbf{x} = c_1\mathbf{x}^{(1)}(t) + \dots + c_n\mathbf{x}^{(n)}(t) \quad$$ 这种集合被称为**基本解组 (Fundamental Set of Solutions)**。

**证明逻辑：**

1. 选取初始点 $t_0$，设已知解在此时的值为 $\mathbf{x}(t_0) = \mathbf{y}$。
2. 我们需要求常数 $c_i$ 使得 $\sum c_i \mathbf{x}^{(i)}(t_0) = \mathbf{y}$。这等价于线性方程组 $\mathbf{X}(t_0)\mathbf{c} = \mathbf{y}$。
3. 由于 $W(t_0) = \det \mathbf{X}(t_0) \neq 0$，该方程组有唯一解 $\mathbf{c}$。
4. 根据解的唯一性定理，这个线性组合在整个区间内就是该解。

------

### 四、 阿贝尔定理 (Abel's Theorem 7.4.3)

#### 1. 定理内容

**定理 7.4.3：** 在区间内，朗斯基行列式 $W(t)$ 要么**恒等于零**，要么**处处不为零**。这意味着我们只需要在某一个方便的点检查 $W$ 的值，就能判断整组解是否线性无关。

#### 2. 阿贝尔公式 (Abel's Formula) 的推导

朗斯基行列式满足以下一阶微分方程： $$\frac{dW}{dt} = [p_{11}(t) + p_{22}(t) + \dots + p_{nn}(t)]W \quad \text{--- (14)} \quad$$ 方括号内的项是矩阵 $\mathbf{P}(t)$ 的**迹 (Trace)**。解这个方程：

1. **分离变量并积分：** $$\int \frac{1}{W} dW = \int \text{tr}(\mathbf{P}(t)) dt$$
2. **得到公式：** $$W(t) = c \exp\left(\int \sum p_{ii}(t) dt\right) \quad \text{--- (15)} \quad$$ 由于指数函数永远不为零，因此 $W(t)$ 的值完全取决于常数 $c$ 是否为零。

------

### 五、 基本解组的存在性 (Theorem 7.4.4)

齐次系统 (3) 是否总是有基本解组？**答案是肯定的**。 **推导方法：**

1. 选取 $n$ 个特殊的初始向量 $\mathbf{e}^{(1)} = (1, 0, \dots, 0)^T, \dots, \mathbf{e}^{(n)} = (0, 0, \dots, 1)^T$。
2. 根据存在唯一性定理，存在解 $\mathbf{x}^{(i)}$ 满足 $\mathbf{x}^{(i)}(t_0) = \mathbf{e}^{(i)}$。
3. 此时 $W(t_0) = \det(\mathbf{I}) = 1$。
4. 因为 $W(t_0) \neq 0$，这组解构成了基本解组。

------

### 六、 复数值解的处理 (Theorem 7.4.5)

**定理 7.4.5：** 如果 $\mathbf{P}$ 是实值矩阵，且 $\mathbf{x} = \mathbf{u}(t) + i\mathbf{v}(t)$ 是复数值解，那么其实部 $\mathbf{u}(t)$ 和虚部 $\mathbf{v}(t)$ **各自**也是系统的解。

**推导过程：**

1. 将 $\mathbf{x}$ 代入方程：$(\mathbf{u} + i\mathbf{v})' = \mathbf{P}(\mathbf{u} + i\mathbf{v})$。

2. 展开得：$\mathbf{u}' + i\mathbf{v}' = \mathbf{P}\mathbf{u} + i\mathbf{P}\mathbf{v}$。

3. 分离实虚部：

    由于 $\mathbf{P}$ 是实数的，我们可以写出：

   - 实部：$\mathbf{u}' = \mathbf{P}\mathbf{u}$
   - 虚部：$\mathbf{v}' = \mathbf{P}\mathbf{v}$

4. 由此证明 $\mathbf{u}, \mathbf{v}$ 都是解。

------

### 七、 线性代数逻辑专题 (基于习题)

这些内容涉及到了矩阵理论在微分方程中的深度应用：

#### 1. 厄米特矩阵 (Hermitian Matrix) 的性质 (Problem 21, 27, 28)

- **性质 A：** $(\mathbf{Ax}, \mathbf{y}) = (\mathbf{x}, \mathbf{A}^* \mathbf{y})$。如果 $\mathbf{A}$ 是实对称的，则 $\mathbf{A}^* = \mathbf{A}^T$。
- 特征值是实数：
  - 设 $\mathbf{Ax} = \lambda \mathbf{x}$，则 $(\mathbf{Ax}, \mathbf{x}) = (\lambda \mathbf{x}, \mathbf{x}) = \lambda (\mathbf{x}, \mathbf{x})$。
  - 又因为 $(\mathbf{Ax}, \mathbf{x}) = (\mathbf{x}, \mathbf{Ax}) = (\mathbf{x}, \lambda \mathbf{x}) = \bar{\lambda}(\mathbf{x}, \mathbf{x})$。
  - 所以 $\lambda = \bar{\lambda}$，特征值必为实数。
- **正交性：** 不同特征值对应的特征向量彼此**正交**（即内积为 0）。

#### 2. 特征向量的线性无关性 (Problem 29)

**定理：** 如果 $\lambda_1 \neq \lambda_2$，则对应的特征向量 $\mathbf{\xi}^{(1)}, \mathbf{\xi}^{(2)}$ 线性无关。 **逻辑推导：**

1. 假设 $c_1\mathbf{\xi}^{(1)} + c_2\mathbf{\xi}^{(2)} = \mathbf{0}$。
2. 左乘矩阵 $\mathbf{A}$：$c_1\lambda_1\mathbf{\xi}^{(1)} + c_2\lambda_2\mathbf{\xi}^{(2)} = \mathbf{0}$。
3. 通过消元法，可以证明 $c_1 = c_2 = 0$。

#### 3. 奇异矩阵与解的存在性 (Problem 22-25)

当 $\det \mathbf{A} = 0$ 时，方程 $\mathbf{Ax} = \mathbf{b}$：

- 若有解 $\mathbf{x}^{(0)}$，且 $\mathbf{A\xi} = \mathbf{0}$，则 $\mathbf{x} = \mathbf{x}^{(0)} + \alpha\mathbf{\xi}$ 也是解。
- **相容性条件：** 要使方程有解，向量 $\mathbf{b}$ 必须与 $\mathbf{A}^* \mathbf{y} = \mathbf{0}$ 的所有解 $\mathbf{y}$ 正交，即 $(\mathbf{b}, \mathbf{y}) = 0$。

------

### 总结要点

1. $n$ 个线性无关的解构成**基本解组**。
2. 此类解组在连续性条件下**始终存在**。
3. 任何解都可以通过基本解组的**线性组合**来表示（通解）。

导师提示：在处理具体的微分方程组时，优先寻找 $n$ 个特征值，并检查其 Wronskian 是否非零，这是构建通解的标准流程。如果你对某个推导步骤有具体的代数疑问，请随时提出。

# 7.5 Homogeneous Linear Systems with Constant Coefficients

### 一、 核心概念与系统定义

#### 1. 常系数齐次线性方程组的通用形式

我们主要研究具有以下形式的系统： $$\mathbf{x}' = \mathbf{Ax} \quad \text{--- (1)}$$ 其中 $\mathbf{A}$ 是一个 $n \times n$ 的常数矩阵。除非特别说明，我们假设 $\mathbf{A}$ 的所有元素均为实数。

#### 2. 一维情形 ($n=1$) 的对比

当 $n=1$ 时，系统退化为单个一阶方程： $$\frac{dx}{dt} = ax \quad \text{--- (2)}$$ 其解为 $x(t) = ce^{at}$。

- **平衡点 (Equilibrium Solution)：** 当 $a \neq 0$ 时，$x=0$ 是唯一的临界点（即平衡解）。
- 稳定性判定：
  - 如果 $a < 0$，所有非平凡解在 $t \to \infty$ 时趋于 $x(t) = 0$，此时称 $x(t) = 0$ 是**渐近稳定的 (asymptotically stable)**。
  - 如果 $a > 0$，除平衡解外的所有解随 $t$ 增加而远离 $x=0$，此时称其为**不稳定的 (unstable)**。

#### 3. 平衡解与相平面分析 (Phase Plane Analysis)

对于 $n$ 维系统，平衡解通过求解代数方程 $\mathbf{Ax} = \mathbf{0}$ 获得。通常假设 $\det \mathbf{A} \neq 0$，因此 $\mathbf{x} = \mathbf{0}$ 是唯一的平衡解。

- **相平面 (Phase Plane)：** 特指 $n=2$ 的情形，即 $x_1x_2$ 平面。
- **方向场 (Direction Field)：** 在大量点处评估 $\mathbf{Ax}$ 并绘制切向量，用于定性理解解的行为。
- **轨迹 (Trajectories)：** 在图中绘制出的解曲线。
- **相图 (Phase Portrait)：** 展示系统代表性轨迹样本的图表，能够提供二维系统的全局直观信息。

------

### 二、 数学推导：寻找一般解

为了求解方程 (1)，我们尝试寻找形式为指数函数乘以向量的解： $$\mathbf{x} = \mathbf{\xi} e^{rt} \quad \text{--- (7)}$$ 其中指数 $r$ 和向量 $\mathbf{\xi}$ 是待确定的常数。

**推导步骤：**

1. 将假设解 $\mathbf{x} = \mathbf{\xi} e^{rt}$ 代入原方程 $\mathbf{x}' = \mathbf{Ax}$。
2. 计算左侧导数：$\mathbf{x}' = r \mathbf{\xi} e^{rt}$。
3. 代入得：$r \mathbf{\xi} e^{rt} = \mathbf{A} \mathbf{\xi} e^{rt}$。
4. 由于 $e^{rt}$ 永远不为零，我们可以消去该标量因子，得到： $$\mathbf{A\xi} = r\mathbf{\xi} \quad \text{或} \quad (\mathbf{A} - r\mathbf{I})\mathbf{\xi} = \mathbf{0} \quad \text{--- (8)}$$ 其中 $\mathbf{I}$ 是 $n \times n$ 单位矩阵。

**结论：** 这表明，要使 $\mathbf{x} = \mathbf{\xi} e^{rt}$ 成为方程 (1) 的解，$r$ 必须是矩阵 $\mathbf{A}$ 的**特征值 (eigenvalue)**，而 $\mathbf{\xi}$ 必须是对应的**特征向量 (eigenvector)**。

------

### 三、 详尽例题分析

#### 例题 1：对角矩阵情形

求解系统：$\mathbf{x}' = \begin{pmatrix} 2 & 0 \ 0 & -3 \end{pmatrix} \mathbf{x} \quad \text{--- (3)}$

- **线性代数逻辑：** 该系数矩阵是**对角矩阵**。这意味着变量 $x_1$ 和 $x_2$ 的方程是解耦的。
- 求解步骤：
  1. 写成标量形式：$x_1' = 2x_1$ 且 $x_2' = -3x_2$。
  2. 分别积分得：$x_1 = c_1 e^{2t}$，$x_2 = c_2 e^{-3t}$。
  3. 写回向量形式： $$\mathbf{x} = c_1 \begin{pmatrix} 1 \ 0 \end{pmatrix} e^{2t} + c_2 \begin{pmatrix} 0 \ 1 \end{pmatrix} e^{-3t} \quad \text{--- (4)}$$
- **朗斯基行列式 (Wronskian) 验证：** $$W[\mathbf{x}^{(1)}, \mathbf{x}^{(2)}](https://notebooklm.google.com/notebook/t) = \det \begin{pmatrix} e^{2t} & 0 \ 0 & e^{-3t} \end{pmatrix} = e^{-t} \neq 0$$ 因此，这两个解构成基本解集。

#### 例题 2：实数、异号特征值（鞍点）

求解系统：$\mathbf{x}' = \begin{pmatrix} 1 & 1 \ 4 & 1 \end{pmatrix} \mathbf{x} \quad \text{--- (9)}$

- **步骤 1：寻找特征值** 计算 $\det(\mathbf{A} - r\mathbf{I}) = 0$： $$\det \begin{pmatrix} 1-r & 1 \ 4 & 1-r \end{pmatrix} = (1-r)^2 - 4 = r^2 - 2r - 3 = (r-3)(r+1) = 0$$ 解得 $r_1 = 3, r_2 = -1$。
- 步骤 2：寻找特征向量
  - 对于 $r_1 = 3$：求解 $\begin{pmatrix} -2 & 1 \ 4 & -2 \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} 0 \ 0 \end{pmatrix}$。得到 $-2\xi_1 + \xi_2 = 0 \Rightarrow \xi_2 = 2\xi_1$。取 $\mathbf{\xi}^{(1)} = \begin{pmatrix} 1 \ 2 \end{pmatrix}$。
  - 对于 $r_2 = -1$：求解 $\begin{pmatrix} 2 & 1 \ 4 & 2 \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} 0 \ 0 \end{pmatrix}$。得到 $2\xi_1 + \xi_2 = 0 \Rightarrow \xi_2 = -2\xi_1$。取 $\mathbf{\xi}^{(2)} = \begin{pmatrix} 1 \ -2 \end{pmatrix}$。
- **步骤 3：写出通解** $$\mathbf{x} = c_1 \begin{pmatrix} 1 \ 2 \end{pmatrix} e^{3t} + c_2 \begin{pmatrix} 1 \ -2 \end{pmatrix} e^{-t} \quad \text{--- (17)}$$
- 几何意义（鞍点 Saddle Point）：
  - 由于特征值一正一负，当 $t \to \infty$ 时，$e^{3t}$ 项占主导，解沿特征向量 $\mathbf{\xi}^{(1)}$ 方向无限发散。
  - 当 $t \to -\infty$ 时，$e^{-t}$ 项占主导，解沿特征向量 $\mathbf{\xi}^{(2)}$ 方向发散。
  - 原点是不稳定的。

#### 例题 3：实数、同号特征值（节点）

求解系统：$\mathbf{x}' = \begin{pmatrix} -3 & \sqrt{2} \ \sqrt{2} & -2 \end{pmatrix} \mathbf{x} \quad \text{--- (18)}$

- **步骤 1：特征值计算** $\det \begin{pmatrix} -3-r & \sqrt{2} \ \sqrt{2} & -2-r \end{pmatrix} = r^2 + 5r + 4 = (r+1)(r+4) = 0$。 得 $r_1 = -1, r_2 = -4$。
- 步骤 2：特征向量计算
  - 对于 $r_1 = -1$，得到 $\mathbf{\xi}^{(1)} = \begin{pmatrix} 1 \ \sqrt{2} \end{pmatrix}$。
  - 对于 $r_2 = -4$，得到 $\mathbf{\xi}^{(2)} = \begin{pmatrix} -\sqrt{2} \ 1 \end{pmatrix}$。
- **步骤 3：通解与稳定性分析** $$\mathbf{x} = c_1 \begin{pmatrix} 1 \ \sqrt{2} \end{pmatrix} e^{-t} + c_2 \begin{pmatrix} -\sqrt{2} \ 1 \end{pmatrix} e^{-4t} \quad \text{--- (25)}$$
- 几何意义（节点 Node）：
  - 两个特征值均为负，所有解在 $t \to \infty$ 时都趋于原点。
  - 由于 $e^{-4t}$ 比 $e^{-t}$ 衰减得快，当 $t \to \infty$ 时，$e^{-t}$ 项（较慢的一项）占主导，因此轨迹切于直线 $x_2 = \sqrt{2}x_1$ 进入原点。
  - 原点是**渐近稳定**的。

------

### 四、 一般 $n \times n$ 系统理论

对于 $n \times n$ 矩阵 $\mathbf{A}$，存在以下三种特征值可能性：

1. **特征值均为实数且互不相同。**
2. **存在复共轭特征值对。**
3. **存在重复的特征值（实数或复数）。**

#### 1. 实数且互不相同的情形

如果 $n$ 个特征值 $r_1, \dots, r_n$ 互不相同，则存在 $n$ 个线性无关的特征向量 $\mathbf{\xi}^{(1)}, \dots, \mathbf{\xi}^{(n)}$。 对应的基本解集为： $$\mathbf{x}^{(i)}(t) = \mathbf{\xi}^{(i)} e^{r_it}, \quad i=1, \dots, n \quad \text{--- (27)}$$

**线性无关性证明（朗斯基行列式推导）：** $$W[\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}](https://notebooklm.google.com/notebook/t) = e^{(r_1 + \dots + r_n)t} \det(\mathbf{\xi}^{(1)}, \dots, \mathbf{\xi}^{(n)}) \quad \text{--- (28)}$$

- 指数项永远不为零。
- 由于特征向量线性无关，其构成的行列式不为零。
- 结论：这 $n$ 个解构成通解 $\mathbf{x} = c_1 \mathbf{\xi}^{(1)} e^{r_1t} + \dots + c_n \mathbf{\xi}^{(n)} e^{r_nt}$。

#### 2. 特殊情形：实对称矩阵 (Real Symmetric Matrix)

如果 $\mathbf{A}$ 是实对称矩阵，即使存在重复特征值，也总能找到 $n$ 个线性无关（且正交）的特征向量。

**例题 4 ($3 \times 3$ 对称矩阵)：** $\mathbf{x}' = \begin{pmatrix} 0 & 1 & 1 \ 1 & 0 & 1 \ 1 & 1 & 0 \end{pmatrix} \mathbf{x} \quad \text{--- (30)}$

- 特征值为 $r_1 = 2, r_2 = -1, r_3 = -1$。
- 即使 $r = -1$ 是二重根（代数重数为2），对称矩阵保证了我们能找到两个对应的特征向量 $\mathbf{\xi}^{(2)}, \mathbf{\xi}^{(3)}$。
- **解的行为：** 若初始点在 $\mathbf{\xi}^{(2)}$ 和 $\mathbf{\xi}^{(3)}$ 构成的平面上（即 $c_1=0$），则解趋于原点；否则，由于 $e^{2t}$ 的存在，解将变得无界。

------

### 五、 补充：重要定理与练习要点

#### 1. 朗斯基行列式的性质 (Problems 9, 10)

- 系统 (1) 的两个基本解集的朗斯基行列式最多相差一个常数因子。
- 对于二阶方程 $y'' + py' + qy = 0$ 转换成的系统，系统的朗斯基行列式与原方程的朗斯基行列式成正比：$W[y_1, y_2] = cW[\mathbf{x}^{(1)}, \mathbf{x}^{(2)}]$。

#### 2. 非齐次方程的结构 (Problem 11)

非齐次方程 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x} + \mathbf{g}(t)$ 的通解等于其对应齐次方程的通解 $\mathbf{x}_c$ 与该方程的一个特解 $\mathbf{x}_p$ 之和。

#### 3. 线性相关性的唯一性证明 (Problems 14, 15)

如果解向量 $\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(m)}$ 在区间内的某一点 $t_0$ 线性相关，那么它们在整个区间内都线性相关。

- **证明逻辑：** 构造 $\mathbf{z}(t) = \sum c_i \mathbf{x}^{(i)}(t)$。如果在 $t_0$ 处 $\mathbf{z}(t_0) = \mathbf{0}$，根据唯一性定理，在整个区间内 $\mathbf{z}(t) \equiv \mathbf{0}$。

------

**导师总结：** 求解常系数线性系统的关键在于将微分方程问题转化为矩阵的**特征值问题**。

1. **实数异号特征值** $\rightarrow$ 鞍点（不稳定）。
2. **实数同号负特征值** $\rightarrow$ 节点（渐近稳定）。
3. **实数同号正特征值** $\rightarrow$ 节点（不稳定）。
4. **对称矩阵** 即使有重复特征值也通常能获得完整解集。



# 7.6 Complex-Valued Eigenvalues

### 第一部分：基础理论与核心定义

#### 1. 齐次线性方程组 (Homogeneous Systems)

我们考虑具有常系数的 $n$ 个线性齐次方程组： $$\mathbf{x}' = \mathbf{Ax} \quad \text{--- (1)}$$ 其中 $\mathbf{A}$ 是一个实数值系数矩阵。

#### 2. 特征值与特征向量 (Eigenvalues and Eigenvectors)

为了求解方程 (1)，我们寻找形式为 $\mathbf{x} = \mathbf{\xi} e^{rt}$ 的解。根据推导，这要求：

- $r$ 必须是矩阵 $\mathbf{A}$ 的**特征值**。
- $\mathbf{\xi}$ 是对应的**特征向量**。

特征值 $r_1, \dots, r_n$ 是通过求解**特征方程**（Characteristic Equation）得到的： $$\det(\mathbf{A} - r\mathbf{I}) = 0 \quad \text{--- (2)}$$ 对应的特征向量是满足以下线性代数方程组的非零向量： $$(\mathbf{A} - r\mathbf{I})\mathbf{\xi} = \mathbf{0} \quad \text{--- (3)}$$

#### 3. 复共轭特性 (Complex Conjugate Pairs)

**定理：** 如果矩阵 $\mathbf{A}$ 是实数值的，那么特征方程 (2) 的系数也是实数。因此，任何复数值特征值必须以共轭对的形式出现。

- 如果 $r_1 = \lambda + i\mu$ 是特征值（其中 $\lambda, \mu$ 是实数），那么 $r_2 = \lambda - i\mu$ 也必然是特征值。
- 相应地，如果 $\mathbf{\xi}^{(1)}$ 是对应于 $r_1$ 的特征向量，那么其共轭向量 $\mathbf{\xi}^{(2)} = \bar{\mathbf{\xi}}^{(1)}$ 就是对应于 $r_2 = \bar{r}_1$ 的特征向量。

------

### 第二部分：详尽示例分析 (Example 1)

**问题：** 寻找系统 $\mathbf{x}' = \begin{pmatrix} -1/2 & 1 \ -1 & -1/2 \end{pmatrix} \mathbf{x}$ 的实数值基本解集，并绘制相位图。

#### 步骤 1：寻找特征值

构造特征方程： $$\det\begin{pmatrix} -1/2 - r & 1 \ -1 & -1/2 - r \end{pmatrix} = (-1/2 - r)^2 - (1)(-1) = r^2 + r + \frac{5}{4} = 0 \quad \text{--- (7)}$$ 利用求根公式，得到： $$r = \frac{-1 \pm \sqrt{1^2 - 4(5/4)}}{2} = -\frac{1}{2} \pm i$$ 所以，$r_1 = -\frac{1}{2} + i$，$r_2 = -\frac{1}{2} - i$。

#### 步骤 2：寻找特征向量

将 $r_1 = -1/2 + i$ 代入 $(\mathbf{A} - r\mathbf{I})\mathbf{\xi} = \mathbf{0}$： $$\begin{pmatrix} -1/2 - (-1/2 + i) & 1 \ -1 & -1/2 - (-1/2 + i) \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} -i & 1 \ -1 & -i \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} 0 \ 0 \end{pmatrix}$$ 解得 $-i\xi_1 + \xi_2 = 0 \Rightarrow \xi_2 = i\xi_1$。取 $\xi_1 = 1$，得 $\mathbf{\xi}^{(1)} = \begin{pmatrix} 1 \ i \end{pmatrix}$。 同理，$\mathbf{\xi}^{(2)} = \begin{pmatrix} 1 \ -i \end{pmatrix}$。

#### 步骤 3：构造实值解

复数解为 $\mathbf{x}^{(1)}(t) = \begin{pmatrix} 1 \ i \end{pmatrix} e^{(-1/2 + i)t}$。 利用欧拉公式 $e^{it} = \cos t + i \sin t$ 进行展开： $$\mathbf{x}^{(1)}(t) = e^{-t/2} \begin{pmatrix} 1 \ i \end{pmatrix} (\cos t + i \sin t) = e^{-t/2} \begin{pmatrix} \cos t + i \sin t \ i \cos t - \sin t \end{pmatrix}$$ 拆分为实部 $\mathbf{u}(t)$ 和虚部 $\mathbf{v}(t)$： $$\mathbf{u}(t) = e^{-t/2} \begin{pmatrix} \cos t \ -\sin t \end{pmatrix}, \quad \mathbf{v}(t) = e^{-t/2} \begin{pmatrix} \sin t \ \cos t \end{pmatrix}$$ 注意：资料中提到的 $\mathbf{u}(t)$ 在公式 (11) 中略有不同，这是因为选择了不同的特征向量形式，但本质是一致的。

#### 步骤 4：验证线性无关性 (Wronskian)

计算 Wronskian 行列式： $$W[\mathbf{u}, \mathbf{v}](https://notebooklm.google.com/notebook/t) = \det \begin{pmatrix} e^{-t/2}\cos t & e^{-t/2}\sin t \ -e^{-t/2}\sin t & e^{-t/2}\cos t \end{pmatrix} = e^{-t}(\cos^2 t + \sin^2 t) = e^{-t}$$ 由于 $e^{-t}$ 永远不为零，因此 $\mathbf{u}$ 和 $\mathbf{v}$ 构成基本解集。

------

### 第三部分：通解推导与几何意义

#### 1. 一般实值解公式

设特征值为 $r_1 = \lambda + i\mu$，特征向量为 $\mathbf{\xi}^{(1)} = \mathbf{a} + i\mathbf{b}$。 对应的两个实值解为： $$\mathbf{u}(t) = e^{\lambda t} (\mathbf{a} \cos \mu t - \mathbf{b} \sin \mu t) \quad \text{--- (17)}$$ $$\mathbf{v}(t) = e^{\lambda t} (\mathbf{a} \sin \mu t + \mathbf{b} \cos \mu t)$$

#### 2. 相位轨迹与平衡点分类

根据特征值的实部 $\lambda$ 决定轨迹行为：

- **$\lambda < 0$：** 轨迹向原点螺旋靠近，原点称为**渐近稳定螺旋点 (Asymptotically Stable Spiral Point)**。
- **$\lambda > 0$：** 轨迹远离原点螺旋发散，原点是**不稳定螺旋点**。
- **$\lambda = 0$：** 轨迹是围绕原点的封闭曲线，原点称为**中心点 (Center)**，它是稳定的，但不是渐近稳定的。

#### 3. 确定旋转方向

为了判断是顺时针还是逆时针旋转，只需检查某一点的切向量 $\mathbf{x}'$。 例如在点 $(0, 1)^T$ 处，计算 $\mathbf{Ax}$。如果 $x_1$ 分量为正，则从第二象限进入第一象限，运动方向为顺时针。

------

### 第四部分：参数研究与分岔 (Example 2)

**系统：** $\mathbf{x}' = \begin{pmatrix} \alpha & 2 \ -2 & 0 \end{pmatrix} \mathbf{x}$。

#### 1. 特征值随 $\alpha$ 的变化

特征方程为 $r^2 - \alpha r + 4 = 0$，解得： $$r = \frac{\alpha \pm \sqrt{\alpha^2 - 16}}{2} \quad \text{--- (21)}$$

#### 2. 分岔值 (Bifurcation Values)

分岔值是指系统性质发生质变的参数值：

- **$\alpha = -4$ 和 $\alpha = 4$：** 特征值由实数变为复数（或反之）。此时为节点与螺旋点的转换。
- **$\alpha = 0$：** 特征值实部为零（纯虚数）。此时轨迹从向内螺旋变为向外螺旋，原点从稳定螺旋点变为中心点，再变为不稳定螺旋点。

------

### 第五部分：高阶应用 - 双质量三弹簧系统 (Example 3)

资料详细分析了一个 $4 \times 4$ 系统： $$\mathbf{y}' = \begin{pmatrix} 0 & 0 & 1 & 0 \ 0 & 0 & 0 & 1 \ -2 & 3/2 & 0 & 0 \ 4/3 & -3 & 0 & 0 \end{pmatrix} \mathbf{y}$$

#### 1. 物理意义

$y_1, y_2$ 是位置，$y_3, y_4$ 是速度。

#### 2. 特征值与基本模式 (Fundamental Modes)

计算特征方程得到 $r^4 + 5r^2 + 4 = (r^2+1)(r^2+4)=0$。 特征值为 $r = \pm i, \pm 2i$。

- **模式 1 (频率 1, 周期 $2\pi$)：** 两个质量同向移动。$y_2 = \frac{2}{3}y_1$。
- **模式 2 (频率 2, 周期 $\pi$)：** 两个质量反向移动。$y_2 = -\frac{4}{3}y_1$。

#### 3. 轨迹分析

虽然在二维投影图（如 $y_1-y_3$ 平面）中轨迹可能看起来相交，但在四维相空间中，由于**解的唯一性定理**，轨迹绝不会自交。

------

### 备注与补充

- **电路系统：** 资料中的问题 24 和 25 提到了 $RLC$ 电路系统。当特征值为复数或重复时，电流 $I(t)$ 和电压 $V(t)$ 仍会趋于 0，只要实部为负。
- **双罐系统：** 问题 22 给出了一个处理盐量偏差的初始值问题，涉及矩阵计算和稳定性分析。



# 7.7 Fundamental Matrices

### 一、 基本矩阵 (Fundamental Matrices) 的定义与性质

在处理线性微分方程组 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x}$ 时，解的结构可以通过**基本矩阵**来清晰地展示。

#### 1.1 定义

假设 $\mathbf{x}^{(1)}(t), \dots, \mathbf{x}^{(n)}(t)$ 是方程 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x}$ 在区间 $\alpha < t < \beta$ 上的一个**基本解组（Fundamental Set of Solutions）**。

我们将这些解向量作为列构造一个矩阵 $\mathbf{\Psi}(t)$： $$\mathbf{\Psi}(t) = \begin{pmatrix} \mathbf{x}^{(1)}(t) & \cdots & \mathbf{x}^{(n)}(t) \end{pmatrix} = \begin{pmatrix} x_{11}(t) & \cdots & x_{1n}(t) \ \vdots & \ddots & \vdots \ x_{n1}(t) & \cdots & x_{nn}(t) \end{pmatrix} \quad$$ 这个矩阵 $\mathbf{\Psi}(t)$ 就被称为该系统的**基本矩阵（Fundamental Matrix）**。

- **重要备注：** 由于其各列是线性无关的向量，基本矩阵 $\mathbf{\Psi}(t)$ 是**非奇异的（Nonsingular）**，这意味着它的行列式不为零，逆矩阵 $\mathbf{\Psi}^{-1}$ 存在。

#### 1.2 矩阵微分方程

由于基本矩阵的每一列都是原方程的解，我们可以推导出基本矩阵本身满足一个矩阵形式的微分方程： $$\mathbf{\Psi}' = \mathbf{P}(t)\mathbf{\Psi} \quad$$ **推导逻辑：** 将 $\mathbf{\Psi}$ 的每一列代入 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x}$，对比等式两端即可确认这一结论。

------

### 二、 利用基本矩阵表示解

基本矩阵极大简化了初值问题（IVP）的表达方式。

#### 2.1 通解的矩阵形式

方程组的通解可以写作基本解的线性组合： $$\mathbf{x} = c_1\mathbf{x}^{(1)}(t) + \cdots + c_n\mathbf{x}^{(n)}(t) \quad$$ 使用基本矩阵 $\mathbf{\Psi}(t)$，这可以简写为： $$\mathbf{x} = \mathbf{\Psi}(t)\mathbf{c} \quad$$ 其中 $\mathbf{c}$ 是一个包含任意常数 $c_1, \dots, c_n$ 的常向量。

#### 2.2 初值问题的求解

考虑初值条件 $\mathbf{x}(t_0) = \mathbf{x}_0$。

1. 代入通解公式：$\mathbf{\Psi}(t_0)\mathbf{c} = \mathbf{x}_0$。
2. 利用 $\mathbf{\Psi}$ 的非奇异性求出 $\mathbf{c}$：$\mathbf{c} = \mathbf{\Psi}^{-1}(t_0)\mathbf{x}_0$。
3. 代回原式得到初值问题的唯一解： $$\mathbf{x} = \mathbf{\Psi}(t)\mathbf{\Psi}^{-1}(t_0)\mathbf{x}_0 \quad$$

- **教学提示：** 虽然公式 (10) 在理论上非常优美，但在实际计算中，我们通常通过**行化简（Row Reduction）**来解方程组 $\mathbf{\Psi}(t_0)\mathbf{c} = \mathbf{x}_0$ 得到 $\mathbf{c}$，而不是直接计算逆矩阵。

------

### 三、 特殊基本矩阵 $\mathbf{\Phi}(t)$

有时我们需要一个更特殊的矩阵，记作 $\mathbf{\Phi}(t)$，它不仅满足微分方程，还满足特定的初始条件。

#### 3.1 定义与性质

$\mathbf{\Phi}(t)$ 的各列向量 $\mathbf{x}^{(j)}(t)$ 满足在 $t_0$ 点的初始条件为单位向量 $\mathbf{e}^{(j)}$： $$\mathbf{x}^{(j)}(t_0) = \mathbf{e}^{(j)} \quad$$ 这意味着在 $t = t_0$ 时，该矩阵变为单位矩阵： $$\mathbf{\Phi}(t_0) = \mathbf{I} \quad$$

#### 3.2 优势

使用 $\mathbf{\Phi}(t)$ 时，初值问题的解变得极其简单： $$\mathbf{x} = \mathbf{\Phi}(t)\mathbf{x}_0 \quad$$ 这是因为 $\mathbf{\Phi}^{-1}(t_0) = \mathbf{I}^{-1} = \mathbf{I}$。 **联系公式：** $\mathbf{\Phi}(t) = \mathbf{\Psi}(t)\mathbf{\Psi}^{-1}(t_0)$。

------

### 四、 矩阵指数 (The Matrix $\exp(\mathbf{A}t)$)

当我们面对常系数矩阵 $\mathbf{A}$ 时，解的形式可以类比标量微分方程 $x' = ax$。

#### 4.1 定义

类比标量指数函数的幂级数展开 $e^{at} = 1 + \sum_{n=1}^{\infty} \frac{a^n t^n}{n!}$，我们定义矩阵指数为： $$\exp(\mathbf{A}t) = \mathbf{I} + \sum_{n=1}^{\infty} \frac{\mathbf{A}^n t^n}{n!} = \mathbf{I} + \mathbf{A}t + \frac{\mathbf{A}^2 t^2}{2!} + \cdots + \frac{\mathbf{A}^n t^n}{n!} + \cdots \quad$$

#### 4.2 微分性质推导

对上述级数逐项求导： $$\frac{d}{dt} \exp(\mathbf{A}t) = \sum_{n=1}^{\infty} \frac{\mathbf{A}^n t^{n-1}}{(n-1)!}$$ 提取出一个 $\mathbf{A}$： $$\frac{d}{dt} \exp(\mathbf{A}t) = \mathbf{A} \left( \mathbf{I} + \mathbf{A}t + \frac{\mathbf{A}^2 t^2}{2!} + \cdots \right) = \mathbf{A} \exp(\mathbf{A}t) \quad$$ 此外，当 $t=0$ 时，$\exp(\mathbf{A} \cdot 0) = \mathbf{I}$。

#### 4.3 结论

由唯一性定理可知，$\exp(\mathbf{A}t)$ 正是满足初始条件 $\mathbf{\Phi}(0) = \mathbf{I}$ 的那个唯一的基本矩阵。 因此，初值问题 $\mathbf{x}' = \mathbf{A}\mathbf{x}, \mathbf{x}(0) = \mathbf{x}_0$ 的解可以简练地写作： $$\mathbf{x} = \exp(\mathbf{A}t)\mathbf{x}_0 \quad$$

------

### 五、 对角化 (Diagonalization) 与相似变换

为了简化计算，我们可以通过坐标变换将“耦合”的方程组解开。

#### 5.1 变换原理

如果矩阵 $\mathbf{A}$ 有 $n$ 个线性无关的特征向量 $\mathbf{\xi}^{(1)}, \dots, \mathbf{\xi}^{(n)}$，我们可以构造**变换矩阵 $\mathbf{T}$**，其各列即为这些特征向量： $$\mathbf{T} = \begin{pmatrix} \mathbf{\xi}^{(1)} & \cdots & \mathbf{\xi}^{(n)} \end{pmatrix} \quad$$ 通过**相似变换**，我们可以得到对角矩阵 $\mathbf{D}$： $$\mathbf{T}^{-1}\mathbf{A}\mathbf{T} = \mathbf{D} = \text{diag}(\lambda_1, \dots, \lambda_n) \quad$$

#### 5.2 求解步骤

1. 令 $\mathbf{x} = \mathbf{Ty}$。
2. 原方程 $\mathbf{x}' = \mathbf{Ax}$ 变为 $\mathbf{Ty}' = \mathbf{ATy}$。
3. 两端乘以 $\mathbf{T}^{-1}$ 得到解耦后的系统：$\mathbf{y}' = \mathbf{Dy}$。
4. 该系统的基本矩阵是 $\mathbf{Q}(t) = \exp(\mathbf{D}t)$，即对角线上为 $e^{\lambda_i t}$ 的对角矩阵。
5. 回到原坐标系，得到原系统的基本矩阵： $$\mathbf{\Psi} = \mathbf{T}\mathbf{Q} \quad$$

------

### 六、 实例深度分析

#### 例 1 & 2：计算基本矩阵 $\mathbf{\Psi}$ 与 $\mathbf{\Phi}$

考虑系统 $\mathbf{x}' = \begin{pmatrix} 1 & 1 \ 4 & 1 \end{pmatrix} \mathbf{x}$。

1. **特征值与特征向量：** 已知特征对为 $r_1=3, \mathbf{\xi}^{(1)}=\begin{pmatrix} 1 \ 2 \end{pmatrix}$ 和 $r_2=-1, \mathbf{\xi}^{(2)}=\begin{pmatrix} 1 \ -2 \end{pmatrix}$。
2. **构造 $\mathbf{\Psi}(t)$：** $$\mathbf{x}^{(1)}(t) = \begin{pmatrix} e^{3t} \ 2e^{3t} \end{pmatrix}, \quad \mathbf{x}^{(2)}(t) = \begin{pmatrix} e^{-t} \ -2e^{-t} \end{pmatrix} \implies \mathbf{\Psi}(t) = \begin{pmatrix} e^{3t} & e^{-t} \ 2e^{3t} & -2e^{-t} \end{pmatrix} \quad$$
3. **计算 $\mathbf{\Phi}(t)$ (满足 $\mathbf{\Phi}(0)=\mathbf{I}$)：** 通过求解 $\mathbf{\Phi}(t) = \mathbf{\Psi}(t)\mathbf{\Psi}^{-1}(0)$，或寻找特定系数 $c_i$ 使得初始值为单位向量： $$\mathbf{\Phi}(t) = \begin{pmatrix} \frac{1}{2}e^{3t} + \frac{1}{2}e^{-t} & \frac{1}{4}e^{3t} - \frac{1}{4}e^{-t} \ e^{3t} - e^{-t} & \frac{1}{2}e^{3t} + \frac{1}{2}e^{-t} \end{pmatrix} \quad$$

#### 例 3 & 4：对角化应用

使用相同的矩阵 $\mathbf{A} = \begin{pmatrix} 1 & 1 \ 4 & 1 \end{pmatrix}$。

1. **变换矩阵 $\mathbf{T}$：** $\mathbf{T} = \begin{pmatrix} 1 & 1 \ 2 & -2 \end{pmatrix}$。
2. **对角矩阵 $\mathbf{D}$：** $\mathbf{T}^{-1}\mathbf{AT} = \begin{pmatrix} 3 & 0 \ 0 & -1 \end{pmatrix}$。
3. **计算 $\exp(\mathbf{D}t)$：** 直接对角线元素指数化 $\begin{pmatrix} e^{3t} & 0 \ 0 & e^{-t} \end{pmatrix}$。
4. **合成 $\mathbf{\Psi}(t)$：** $\mathbf{T}\exp(\mathbf{D}t) = \begin{pmatrix} 1 & 1 \ 2 & -2 \end{pmatrix} \begin{pmatrix} e^{3t} & 0 \ 0 & e^{-t} \end{pmatrix} = \begin{pmatrix} e^{3t} & e^{-t} \ 2e^{3t} & -2e^{-t} \end{pmatrix}$。 *这证明了两种方法得到的结果完全一致！*

------

### 七、 补充：物理模型应用（弹簧-质量系统）

在习题部分，教材还展示了如何将这些理论应用于物理系统：

- **单质量系统：** $mu'' + ku = 0$。通过引入 $x_1 = u, x_2 = u'$，可以转化为一阶系统： $$\mathbf{x}' = \begin{pmatrix} 0 & 1 \ -k/m & 0 \end{pmatrix} \mathbf{x}$$ 该系统的特征值与系统的**固有频率（Natural Frequency）**直接相关。
- **多质量系统：** 涉及两个质量和三个弹簧的系统。通过 $\mathbf{x} = \mathbf{\xi} e^{rt}$，可以得到 $(\mathbf{A} - r^2 \mathbf{I})\mathbf{\xi} = \mathbf{0}$，这里 $r^2$ 是矩阵 $\mathbf{A}$ 的特征值。

### 总结检查

- **线性无关性：** 来源中提到，如果特征值不同，则对应的特征向量线性无关。
- **Hermitian 矩阵：** 如果 $\mathbf{A}$ 是 Hermitian 矩阵，则 $\mathbf{T}^{-1} = \mathbf{T}^*$（共轭转置），这简化了逆矩阵的计算。
- **不可对角化情况：** 如果线性无关的特征向量不足 $n$ 个，则矩阵不可对角化。





# 7.8 Repeated Eigenvalues

### 1. 核心概念 breakdown：具有重复特征值的线性齐次系统

我们研究的是具有常系数的线性齐次微分方程组： $$\mathbf{x}' = \mathbf{A}\mathbf{x} \tag{1}$$ 其中 $\mathbf{A}$ 是一个 $n \times n$ 的常数矩阵。

#### 1.1 特征值的重数 (Multiplicity of Eigenvalues)

当求解特征方程 $\det(\mathbf{A} - r\mathbf{I}) = 0$ 时，如果某个根 $r = \rho$ 出现了 $m$ 次，我们称 $\rho$ 为具有 **代数重数 (Algebraic Multiplicity)** $m$ 的特征值,。

对于代数重数为 $m \geq 2$ 的特征值，存在两种情况：

1. **几何重数等于代数重数**：存在 $m$ 个线性无关的特征向量 $\boldsymbol{\xi}^{(1)}, \dots, \boldsymbol{\xi}^{(m)}$。在这种情况下，我们可以直接构造 $m$ 个线性无关的解：$\mathbf{x}^{(i)}(t) = \boldsymbol{\xi}^{(i)} e^{\rho t}$。这种情况常见于 **Hermitian 矩阵**（或实对称矩阵）。
2. **几何重数小于代数重数**：线性无关的特征向量少于 $m$ 个。此时，形式为 $\boldsymbol{\xi} e^{\rho t}$ 的解不足以构成通解，我们需要寻找其他形式的解,,。

------

### 2. 深入例题分析：特征值与特征向量的计算

我们通过 **例题 1** 来观察特征向量不足的情况。

**题目：** 求矩阵 $\mathbf{A} = \begin{pmatrix} 1 & -1 \ 1 & 3 \end{pmatrix}$ 的特征值和特征向量。

**步骤 1：建立特征方程** 我们需要求解 $\det(\mathbf{A} - r\mathbf{I}) = 0$： $$\det \begin{pmatrix} 1-r & -1 \ 1 & 3-r \end{pmatrix} = (1-r)(3-r) - (-1)(1) = r^2 - 4r + 4 = 0 \tag{4}$$ 解得 $(r-2)^2 = 0$，因此特征值 $r_1 = r_2 = 2$。这是一个代数重数为 2 的特征值。

**步骤 2：寻找特征向量** 将 $r = 2$ 代入 $(\mathbf{A} - r\mathbf{I})\boldsymbol{\xi} = \mathbf{0}$： $$\begin{pmatrix} 1-2 & -1 \ 1 & 3-2 \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} -1 & -1 \ 1 & 1 \end{pmatrix} \begin{pmatrix} \xi_1 \ \xi_2 \end{pmatrix} = \begin{pmatrix} 0 \ 0 \end{pmatrix} \tag{5}$$ 这给出了单一条件 $\xi_1 + \xi_2 = 0$。因此，特征向量为： $$\boldsymbol{\xi}^{(1)} = \begin{pmatrix} 1 \ -1 \end{pmatrix} \tag{6}$$ **逻辑说明：** 尽管特征值是二重的，但我们只能找到一个线性无关的特征向量。这说明该矩阵的几何重数为 1，小于代数重数 2。

------

### 3. 数学推导：构造第二个独立解

在 **例题 2** 中，我们需要求解 $\mathbf{x}' = \begin{pmatrix} 1 & -1 \ 1 & 3 \end{pmatrix} \mathbf{x}$。

我们已经知道一个解是 $\mathbf{x}^{(1)}(t) = \begin{pmatrix} 1 \ -1 \end{pmatrix} e^{2t}$。借鉴二阶常系数线性方程的经验（当有重根时，第二个解通常包含 $t e^{rt}$ 项），我们尝试构造第二个解,。

#### 3.1 尝试形式 $\mathbf{x} = \boldsymbol{\xi} t e^{2t}$

如果代入原方程 $\mathbf{x}' = \mathbf{A}\mathbf{x}$： $$\frac{d}{dt}(\boldsymbol{\xi} t e^{2t}) = \mathbf{A}(\boldsymbol{\xi} t e^{2t})$$ $$2\boldsymbol{\xi} t e^{2t} + \boldsymbol{\xi} e^{2t} = \mathbf{A}\boldsymbol{\xi} t e^{2t} \tag{11}$$ 为了让此式对所有 $t$ 成立，必须使 $e^{2t}$ 的系数相等，即 $\boldsymbol{\xi} = \mathbf{0}$,。这说明单靠 $\boldsymbol{\xi} t e^{2t}$ 无法构成解。

#### 3.2 正确形式：广义特征向量法

我们假设第二个解的形式为： $$\mathbf{x} = \boldsymbol{\xi} t e^{2t} + \boldsymbol{\eta} e^{2t} \tag{13}$$ 代入原方程： $$2\boldsymbol{\xi} t e^{2t} + (\boldsymbol{\xi} + 2\boldsymbol{\eta}) e^{2t} = \mathbf{A}(\boldsymbol{\xi} t e^{2t} + \boldsymbol{\eta} e^{2t}) \tag{14}$$ 比较 $t e^{2t}$ 和 $e^{2t}$ 的系数，得到两个基本条件：

1. $(\mathbf{A} - 2\mathbf{I})\boldsymbol{\xi} = \mathbf{0}$ （说明 $\boldsymbol{\xi}$ 是特征向量）
2. $(\mathbf{A} - 2\mathbf{I})\boldsymbol{\eta} = \boldsymbol{\xi}$ （用于求 $\boldsymbol{\eta}$）

**推导求解 $\boldsymbol{\eta}$：** 已知 $\boldsymbol{\xi} = \begin{pmatrix} 1 \ -1 \end{pmatrix}$，代入条件 2 得到增广矩阵： $$\begin{pmatrix} -1 & -1 & | & 1 \ 1 & 1 & | & -1 \end{pmatrix} \tag{10}$$ 对应的方程为 $-\eta_1 - \eta_2 = 1$。 若令 $\eta_1 = k$（任意常数），则 $\eta_2 = -k - 1$。写成向量形式： $$\boldsymbol{\eta} = \begin{pmatrix} 0 \ -1 \end{pmatrix} + k \begin{pmatrix} 1 \ -1 \end{pmatrix} \tag{17}$$ 在构造第二个独立解时，我们可以取 $k=0$，因为含 $k$ 的项只是第一个解的倍数。

因此，第二个独立解为： $$\mathbf{x}^{(2)}(t) = \begin{pmatrix} 1 \ -1 \end{pmatrix} t e^{2t} + \begin{pmatrix} 0 \ -1 \end{pmatrix} e^{2t} \tag{19}$$

这里的 $\boldsymbol{\eta}$ 被称为矩阵 $\mathbf{A}$ 对应于特征值 2 的 **广义特征向量 (Generalized Eigenvector)**。它满足 $(\mathbf{A} - \rho \mathbf{I})^2 \boldsymbol{\eta} = \mathbf{0}$。

------

### 4. 相图分析与稳定性：非正则节点 (Improper Node)

对于上述系统，其通解为： $$\mathbf{x} = c_1 \begin{pmatrix} 1 \ -1 \end{pmatrix} e^{2t} + c_2 \left[ \begin{pmatrix} 1 \ -1 \end{pmatrix} t e^{2t} + \begin{pmatrix} 0 \ -1 \end{pmatrix} e^{2t} \right] \tag{20}$$

- **极限行为：** 随着 $t \to -\infty$，$\mathbf{x} \to \mathbf{0}$。随着 $t \to \infty$，解是无界的,。
- **轨迹方向：** 当 $t \to \pm \infty$ 时，轨迹的斜率 $\frac{x_2}{x_1}$ 趋向于 $-1$，即趋向于特征向量所确定的直线 $x_2 = -x_1$。
- **定义：** 这种具有相等特征值且只有一个独立特征向量的 $2 \times 2$ 系统，其原点被称为 **非正则节点 (Improper Node)**。
- **稳定性：** 如果特征值为负，轨迹向内靠拢，系统渐近稳定；如果特征值为正（如本例），系统是不稳定的。

------

### 5. 高级结构：基本矩阵与 Jordan 标准型

#### 5.1 基本矩阵 (Fundamental Matrices)

我们可以将独立解排列成列来构造基本矩阵 $\boldsymbol{\Psi}(t)$： $$\boldsymbol{\Psi}(t) = e^{2t} \begin{pmatrix} 1 & t \ -1 & -1-t \end{pmatrix} \tag{25}$$ 满足 $\boldsymbol{\Phi}(0) = \mathbf{I}$ 的特殊基本矩阵为 $\boldsymbol{\Phi}(t) = \exp(\mathbf{A}t)$。通过公式 $\boldsymbol{\Phi}(t) = \boldsymbol{\Psi}(t)\boldsymbol{\Psi}^{-1}(0)$ 计算得到： $$\boldsymbol{\Phi}(t) = e^{2t} \begin{pmatrix} 1-t & -t \ t & 1+t \end{pmatrix} \tag{27}$$

#### 5.2 Jordan 标准型 (Jordan Forms)

如果矩阵因特征值重复而缺乏足够的特征向量，它不能被对角化，但可以转换为 **Jordan 标准型** $\mathbf{J}$。 利用变换矩阵 $\mathbf{T}$（第一列为特征向量 $\boldsymbol{\xi}$，第二列为广义特征向量 $\boldsymbol{\eta}$），有 $\mathbf{T}^{-1}\mathbf{AT} = \mathbf{J}$。 对于本例： 
$$
\mathbf{T} = \begin{pmatrix} 1 & 0 \\ -1 & -1 \end{pmatrix} \implies \mathbf{J} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix} \tag{28, 29}
$$
在 Jordan 标准型中，主对角线上是特征值，上方相邻位置出现 1,。

------

### 6. 备注与补充信息

- **解的结构差异：** 与二阶单变量方程不同，在方程组中，第二个解的 $e^{\rho t}$ 项通常不是第一个解的简单倍数，因此必须保留。
- **计算复杂性：** 对于 $n \geq 3$ 的系统，可能存在更高重数的特征值和更复杂的广义特征向量结构，通常需要计算机软件辅助,。
- **物理意义：** 在电路模型（如 RLC 电路）中，当 $L = 4R^2C$ 时，会出现特征值相等的情况。
- **数值不确定性：** 在实际物理建模中，由于测量误差，很难判断两个特征值是完全相等还是仅仅非常接近。

**补充：** 关于方程 $(\mathbf{A} - 2\mathbf{I})\boldsymbol{\eta} = \boldsymbol{\xi}$ 的可解性，一个重要的数学事实是：该方程总是有解的，因为 $\boldsymbol{\xi}$ 与 $\mathbf{A}^*$（伴随矩阵）的特征向量正交。



# 7.9 Nonhomogeneous Linear Systems

### 1. 非齐次线性系统的定义与通解结构

#### 1.1 基本定义

非齐次线性一阶微分方程组的标准形式如下： $$\mathbf{x}' = \mathbf{P}(t)\mathbf{x} + \mathbf{g}(t) \tag{1}$$ 其中：

- $\mathbf{P}(t)$ 是一个 $n \times n$ 的矩阵函数。
- $\mathbf{g}(t)$ 是一个 $n \times 1$ 的非零向量函数（称为非齐次项或强制函数）。
- 假设 $\mathbf{P}(t)$ 和 $\mathbf{g}(t)$ 在区间 $\alpha < t < \beta$ 上连续。

#### 1.2 通解定理

该系统的通解可以表示为： $$\mathbf{x} = c_1\mathbf{x}^{(1)}(t) + \cdots + c_n\mathbf{x}^{(n)}(t) + \mathbf{v}(t) \tag{2}$$ **详细解释：**

- **齐次部分**：$c_1\mathbf{x}^{(1)}(t) + \cdots + c_n\mathbf{x}^{(n)}(t)$ 是对应齐次系统 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x}$ 的通解。
- **特解部分**：$\mathbf{v}(t)$ 是原非齐次系统 (1) 的任意一个特定解。 这与我们在高阶线性标量微分方程中学到的结构完全一致。

------

### 2. 方法一：对角化法 (Diagonalization)

当系数矩阵 $\mathbf{A}$ 是常数矩阵且**可对角化**时，可以使用此方法。

#### 2.1 数学推导过程

假设我们要解 $\mathbf{x}' = \mathbf{Ax} + \mathbf{g}(t)$。

1. **构造变换矩阵**：令 $\mathbf{T}$ 为由 $\mathbf{A}$ 的特征向量 $\boldsymbol{\xi}^{(1)}, \dots, \boldsymbol{\xi}^{(n)}$ 作为列构成的矩阵。
2. **变量代换**：定义新变量 $\mathbf{y}$，使得 $\mathbf{x} = \mathbf{Ty}$。
3. **代入方程**： $$\mathbf{Ty}' = \mathbf{ATy} + \mathbf{g}(t)$$ 这里我们将 $\mathbf{x}'$ 替换为 $(\mathbf{Ty})' = \mathbf{Ty}'$。
4. **左乘反矩阵**：方程两边同时左乘 $\mathbf{T}^{-1}$： $$\mathbf{y}' = (\mathbf{T}^{-1}\mathbf{AT})\mathbf{y} + \mathbf{T}^{-1}\mathbf{g}(t) \tag{5}$$
5. **解耦方程组**： $$\mathbf{y}' = \mathbf{Dy} + \mathbf{h}(t)$$ 其中 $\mathbf{D} = \text{diag}(r_1, \dots, r_n)$ 是特征值构成的对角矩阵，$\mathbf{h}(t) = \mathbf{T}^{-1}\mathbf{g}(t)$。此时，系统变成了 $n$ 个相互独立的标量一阶方程： $$y_j' = r_j y_j + h_j(t), \quad j=1, \dots, n \tag{6}$$

#### 2.2 标量方程的解

根据一阶线性方程的求解公式，每个 $y_j$ 的解为： $$y_j(t) = e^{r_j t} \int^t e^{-r_j s} h_j(s) ds + c_j e^{r_j t} \tag{7}$$ 最后通过 $\mathbf{x} = \mathbf{Ty}$ 映射回原始变量。

------

### 3. 例题 1 详析：对角化法的应用

**题目**：解 $\mathbf{x}' = \begin{pmatrix} -2 & 1 \ 1 & -2 \end{pmatrix} \mathbf{x} + \begin{pmatrix} 2e^{-t} \ 3t \end{pmatrix}$。

**步骤 1：求解特征值与特征向量** 计算 $\det(\mathbf{A}-r\mathbf{I}) = 0$ 得到：

- $r_1 = -3$，对应特征向量 $\boldsymbol{\xi}^{(1)} = \begin{pmatrix} 1 \ -1 \end{pmatrix}$。
- $r_2 = -1$，对应特征向量 $\boldsymbol{\xi}^{(2)} = \begin{pmatrix} 1 \ 1 \end{pmatrix}$。

**步骤 2：标准化与构造 $\mathbf{T}$** 由于 $\mathbf{A}$ 是实对称矩阵，特征向量正交。为了简化计算，我们将向量长度标准化为 1： $|\boldsymbol{\xi}^{(1)}| = \sqrt{1^2+(-1)^2} = \sqrt{2}$。 因此 $\mathbf{T} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \ -1 & 1 \end{pmatrix}$。 **逻辑备注**：标准化后，$\mathbf{T}$ 是正交矩阵，故 $\mathbf{T}^{-1} = \mathbf{T}^T = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \ 1 & 1 \end{pmatrix}$。

**步骤 3：求解 $\mathbf{y}$ 系数** 计算 $\mathbf{h}(t) = \mathbf{T}^{-1}\mathbf{g}(t)$： $$\mathbf{h}(t) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \ 1 & 1 \end{pmatrix} \begin{pmatrix} 2e^{-t} \ 3t \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 2e^{-t} - 3t \ 2e^{-t} + 3t \end{pmatrix} \tag{12}$$

**步骤 4：解标量方程** 解 $y_1' + 3y_1 = \frac{2e^{-t} - 3t}{\sqrt{2}}$ 和 $y_2' + y_2 = \frac{2e^{-t} + 3t}{\sqrt{2}}$。 计算得出：

- $y_1 = \frac{\sqrt{2}}{2} e^{-t} - \frac{3\sqrt{2}}{2} (\frac{t}{3} - \frac{1}{9}) + c_1 e^{-3t}$
- $y_2 = \sqrt{2} te^{-t} + \frac{3\sqrt{2}}{2} (t-1) + c_2 e^{-t}$

**步骤 5：回归原变量 $\mathbf{x}$** 利用 $\mathbf{x} = \mathbf{Ty}$ 合并结果，得到通解： $$\mathbf{x} = k_1 \begin{pmatrix} 1 \ -1 \end{pmatrix} e^{-3t} + k_2 \begin{pmatrix} 1 \ 1 \end{pmatrix} e^{-t} + \frac{1}{2} \begin{pmatrix} 1 \ -1 \end{pmatrix} e^{-t} + \begin{pmatrix} 1 \ 1 \end{pmatrix} te^{-t} + \begin{pmatrix} 1 \ 2 \end{pmatrix} t - \frac{1}{3} \begin{pmatrix} 4 \ 5 \end{pmatrix} \tag{15}$$

------

### 4. 补充说明：不可对角化的情况 (Jordan Form)

如果 $\mathbf{A}$ 存在重复特征值且特征向量不足，则无法对角化。此时需使用 **Jordan 标准型 $\mathbf{J}$**。

- 变换后方程变为 $\mathbf{y}' = \mathbf{Jy} + \mathbf{h}(t)$。
- 由于 $\mathbf{J}$ 在对角线上方可能有 1，方程组不会完全解耦，但可以从最后一个方程 $y_n$ 开始**递归向上依次求解**。

------

### 5. 方法二：待定系数法 (Undetermined Coefficients)

此方法适用于系数矩阵 $\mathbf{P}$ 为常数矩阵 $\mathbf{A}$，且 $\mathbf{g}(t)$ 为多项式、指数或三角函数的组合。

#### 5.1 假设形式的规则

基本原则与单变量方程类似，但有一个重要区别：

- **重复根处理**：若 $\mathbf{g}(t) = \mathbf{u} e^{\lambda t}$，且 $\lambda$ 是特征方程的单根，则必须假设特解形式为 $\mathbf{v}(t) = \mathbf{a} t e^{\lambda t} + \mathbf{b} e^{\lambda t}$，而不仅仅是 $\mathbf{a} t e^{\lambda t}$。

#### 5.2 例题 2 详析

**题目**：用待定系数法解例题 1 的系统。 $\mathbf{g}(t) = \begin{pmatrix} 2 \ 0 \end{pmatrix} e^{-t} + \begin{pmatrix} 0 \ 3 \end{pmatrix} t$。

1. **观察特征值**：$r = -1$ 是矩阵 $\mathbf{A}$ 的特征值。

2. **设定假设形式**： $$\mathbf{v}(t) = \mathbf{a} t e^{-t} + \mathbf{b} e^{-t} + \mathbf{c} t + \mathbf{d} \tag{18}$$

3. 代入原方程并对比系数

   ：得到一系列向量方程：

   - $\mathbf{Aa} = -\mathbf{a}$ （说明 $\mathbf{a}$ 是特征向量）
   - $\mathbf{Ab} = \mathbf{a} - \mathbf{b} - \begin{pmatrix} 2 \ 0 \end{pmatrix}$
   - $\mathbf{Ac} = -\begin{pmatrix} 0 \ 3 \end{pmatrix}$
   - $\mathbf{Ad} = \mathbf{c}$

4. **求解向量**：通过解这些线性方程组，求得 $\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}$ 的具体值。

------

### 6. 方法三：参数变值法 (Variation of Parameters)

这是最通用的方法，适用于 $\mathbf{P}(t)$ 不是常数矩阵的情况。

#### 6.1 公式推导

1. 设齐次系统的基本矩阵为 $\mathbf{\Psi}(t)$，齐次解为 $\mathbf{x} = \mathbf{\Psi}(t)\mathbf{c}$。
2. 将常数向量 $\mathbf{c}$ 替换为待求向量函数 $\mathbf{u}(t)$，设 $\mathbf{x} = \mathbf{\Psi}(t)\mathbf{u}(t)$。
3. 求导：$\mathbf{x}' = \mathbf{\Psi}'(t)\mathbf{u}(t) + \mathbf{\Psi}(t)\mathbf{u}'(t)$。
4. 代入原方程 $\mathbf{x}' = \mathbf{P}(t)\mathbf{x} + \mathbf{g}(t)$： $$\mathbf{P\Psi u} + \mathbf{\Psi u}' = \mathbf{P\Psi u} + \mathbf{g}(t)$$ （利用了 $\mathbf{\Psi}' = \mathbf{P\Psi}$ 的性质）
5. 简化得到：$\mathbf{\Psi}(t)\mathbf{u}'(t) = \mathbf{g}(t) \tag{26}$。
6. 解得 $\mathbf{u}'(t) = \mathbf{\Psi}^{-1}(t)\mathbf{g}(t)$。
7. 积分得到通解公式： $$\mathbf{x} = \mathbf{\Psi}(t)\mathbf{c} + \mathbf{\Psi}(t) \int_{t_1}^t \mathbf{\Psi}^{-1}(s)\mathbf{g}(s) ds \tag{29}$$

**重要提示**：在实际计算中，通常通过**行化简（Row Reduction）**求解 $\mathbf{\Psi u}' = \mathbf{g}$，而不是直接去计算复杂的逆矩阵 $\mathbf{\Psi}^{-1}$。

------

### 7. 方法四：拉普拉斯变换法 (Laplace Transforms)

对于常系数系统，拉普拉斯变换非常有效，尤其是处理不连续的强制函数时。

#### 7.1 核心公式

对 $\mathbf{x}' = \mathbf{Ax} + \mathbf{g}(t)$ 两边取变换： $$s\mathbf{X}(s) - \mathbf{x}(0) = \mathbf{AX}(s) + \mathbf{G}(s) \tag{41}$$ 整理得： $$(s\mathbf{I} - \mathbf{A})\mathbf{X}(s) = \mathbf{x}(0) + \mathbf{G}(s)$$ 若初值为 $\mathbf{0}$，则： $$\mathbf{X}(s) = (s\mathbf{I} - \mathbf{A})^{-1} \mathbf{G}(s) \tag{44}$$ 其中 $(s\mathbf{I} - \mathbf{A})^{-1}$ 被称为**传递矩阵 (Transfer Matrix)**。

------

### 8. 总结与方法对比

| 方法             | 优点                      | 缺点                                             |
| ---------------- | ------------------------- | ------------------------------------------------ |
| **对角化法**     | 对对称/Hermitian矩阵极快  | 需要求特征向量及逆矩阵                           |
| **待定系数法**   | 无需积分，计算直接        | 仅限特定 $\mathbf{g}(t)$，且可能需解多组代数方程 |
| **参数变值法**   | 最通用，适用于变系数矩阵  | 计算复杂度最高，涉及积分和变系数方程组           |
| **拉普拉斯变换** | 极适合处理不连续/冲击函数 | 涉及矩阵求逆和复杂的逆变换                       |

------

### 9. 附录：关于 Jordan 块的幂（基于习题 21）

源码中提到了一个重要的矩阵性质。设 $3 \times 3$ 的 Jordan 块 $\mathbf{J} = \begin{pmatrix} \lambda & 1 & 0 \ 0 & \lambda & 1 \ 0 & 0 & \lambda \end{pmatrix}$。 通过归纳法可以证明，其 $n$ 次幂为： $$\mathbf{J}^n = \begin{pmatrix} \lambda^n & n\lambda^{n-1} & \frac{n(n-1)}{2}\lambda^{n-2} \ 0 & \lambda^n & n\lambda^{n-1} \ 0 & 0 & \lambda^n \end{pmatrix}$$ 这在计算矩阵指数 $\exp(\mathbf{J}t)$ 时非常关键，它是求解具有重复特征值系统的核心工具。