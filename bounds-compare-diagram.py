from scipy.special import binom
import cvxpy as cp

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")

def calculate_c_j_lb(Lambda, alpha):
    C_J_LB = []
    for j in range(1, Lambda + 1):
        C_J_LB.append(
            binom(Lambda, alpha + j) / (binom(Lambda, alpha) * binom(Lambda, j))
        )
    return C_J_LB

def calculate_c_j_ub(Lambda, alpha):
    C_J_UB = []
    for j in range(1, Lambda + 1):
        C_J_UB.append(binom(Lambda - alpha, j) / (binom(Lambda, j) * (binom(j + alpha, j) - 1)))
    return C_J_UB

def calculate_d_j(Lambda, alpha):
    D_J = []
    for j in range(1, Lambda + 1):
        D_J.append((binom(Lambda, alpha) - binom(Lambda - j, alpha)) / (alpha * binom(Lambda, alpha)))
    return D_J

def achievable_bounds(Lambda, alpha, r):
    #---#
    # Calculating original MADC achievable bound following the approach in https://arxiv.org/pdf/2206.12851
    #---#

    D_J = calculate_d_j(Lambda, alpha)
    C_J_UB = calculate_c_j_ub(Lambda, alpha)
    C_J_LB = calculate_c_j_lb(Lambda, alpha)

    xs_a_madc = []
    for _ in range(Lambda):
        xs_a_madc.append(cp.Variable())

    constraints = [x >= 0 for x in xs_a_madc]
    constraints.append(sum(xs_a_madc) <= 1)
    constraints.append(sum(xs_a_madc) >= 1)
    constraints.append(sum(xs_a_madc[i] * (i + 1) for i in range(Lambda)) <= r)

    # As stated in https://arxiv.org/pdf/2206.12851,
    # a linear program aimed at minimizing the *converse* bound, rather than the *achievable* bound is solved
    objective = cp.Minimize(0.5 * cp.sum([(C_J_LB[i] + D_J[i]) * xs_a_madc[i] for i in range(0, Lambda)]))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # The derived solutions of the linear program are used to calculate the *achievable* bound
    L = sum([C_J_UB[i] * xs_a_madc[i].value for i in range(0, Lambda)])
    D = sum([D_J[i] * xs_a_madc[i].value for i in range(0, Lambda)])

    achievable_madc = max(L, D)

    # ---#
    # Calculating improved bounds
    # ---#

    xs_a_better = []
    for _ in range(Lambda):
        xs_a_better.append(cp.Variable())

    M = cp.Variable()

    # We solve a linear program aimed for minimizing a linear program
    # corresponding to the definition of the achievable bound instead
    constraints = [x >= 0 for x in xs_a_better]
    constraints.append(sum(xs_a_better) <= 1)
    constraints.append(sum(xs_a_better) >= 1)
    constraints.append(sum(xs_a_better[i] * (i + 1) for i in range(Lambda)) <= r)
    constraints.append(M >= cp.sum([C_J_UB[i] * xs_a_better[i] for i in range(0, Lambda)]))
    constraints.append(M >= cp.sum([D_J[i] * xs_a_better[i] for i in range(0, Lambda)]))

    objective = cp.Minimize(M)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return achievable_madc, prob.value

def converse_bounds(Lambda, alpha, r):
    # ---#
    # Calculating original MADC converse bound following the approach in https://arxiv.org/pdf/2206.12851
    # ---#
    C_J_LB = calculate_c_j_lb(Lambda, alpha)
    D_J = calculate_d_j(Lambda, alpha)

    xs_a_madc = []
    for _ in range(Lambda):
        xs_a_madc.append(cp.Variable())

    constraints = [x >= 0 for x in xs_a_madc]
    constraints.append(sum(xs_a_madc) <= 1)
    constraints.append(sum(xs_a_madc) >= 1)
    constraints.append(sum(xs_a_madc[i] * (i + 1) for i in range(Lambda)) <= r)

    # The converse bound is estimated using linear program coefficients which are a mean of two values
    objective = cp.Minimize(0.5 * cp.sum([(C_J_LB[i] + D_J[i]) * xs_a_madc[i] for i in range(0, Lambda)]))

    prob = cp.Problem(objective, constraints)
    prob.solve()
    converse_madc = prob.value

    # ---#
    # Calculating improved bounds
    # ---#

    xs_a_better = []
    for _ in range(Lambda):
        xs_a_better.append(cp.Variable())
    M = cp.Variable()

    # Again, our approach improves this result
    # by minimizing a linear program corresponding to the definition of the converse bound.
    constraints = [x >= 0 for x in xs_a_better]
    constraints.append(sum(xs_a_better) <= 1)
    constraints.append(sum(xs_a_better) >= 1)
    constraints.append(sum(xs_a_better[i] * (i + 1) for i in range(Lambda)) <= r)
    constraints.append(M >= cp.sum([C_J_LB[i] * xs_a_better[i] for i in range(0, Lambda)]))
    constraints.append(M >= cp.sum([D_J[i] * xs_a_better[i] for i in range(0, Lambda)]))

    objective = cp.Minimize(M)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return converse_madc, prob.value

achievable_madcs = []
achievable_betters = []
converse_madcs = []
converse_betters = []

iters = 15
Lambda_global = 15
r_global = 2
for a in range(1, iters):
    achievable_madc, achievable_better = achievable_bounds(Lambda_global, a, r_global)
    converse_madc, converse_better = converse_bounds(Lambda_global, a, r_global)

    achievable_madcs.append(achievable_madc)
    achievable_betters.append(achievable_better)
    converse_madcs.append(converse_madc)
    converse_betters.append(converse_better)

import matplotlib.pyplot as plt

range_begin = 2
fig, ax = plt.subplots(figsize = (7, 4))
plt.plot(range(range_begin, iters), achievable_madcs[range_begin - 1:],
         color = "#0000ff", label="MADC Achievable", marker = 'o')
plt.plot(range(range_begin, iters), converse_madcs[range_begin - 1:],
         color = "#0000ff", label="MADC Converse", marker = 'o')
plt.plot(range(range_begin, iters), achievable_betters[range_begin - 1:],
         color = "#ff0000", label="Improved Achievable", marker = 'o')
plt.plot(range(range_begin, iters), converse_betters[range_begin - 1:],
         color = "#ff0000", label="Improved Converse", marker = 'o')

plt.ylim(bottom = 0)
plt.xlabel("α")
plt.xticks(range(range_begin, iters))
plt.ylabel("Max-link communication load")
plt.title("Comparison of bounds, Λ = 15, r = 2")
plt.legend()
plt.grid(True)
plt.savefig('bound-comparison.png')

plt.show()