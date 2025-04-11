import cvxpy as cp
import numpy as np


m1 = 1 
m2 = 0.5 
k1 = 1 
k2 = 1 
c0 = 2 # coef

# Definir matrizes do sistema
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-(k1 + k2) / m1, k2 / m1, -c0 / m1, 0],
              [k2 / m2, -k2 / m2, 0, -c0 / m2]])

B = np.array([[0], [0], [1 / m1], [0]])
C = np.array([[0, 1, 0, 0]])
D = np.array([[0]])  

# Definir variáveis de decisão
P = cp.Variable((4, 4), symmetric=True) # matriz P
mu = cp.Variable() 

# Construção da LMI
LMI = cp.bmat([
    [A.T @ P + P @ A + C.T @ C, P @ B + C.T @ D],
    [B.T @ P + D.T @ C, D.T @ D - mu]
])

# restrições
constraints = [
    P >> 0,  
    LMI << 0  
]

# função objetivo
objective = cp.Minimize(mu)

# resolver o problema de otimização
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

#  resultados
if problem.status == cp.OPTIMAL:
    print("Solução encontrada:")
    mu_opt = mu.value 
    P_opt = P.value  
    Hinf_norm = np.sqrt(mu_opt)  # Norma H_infinito
    print("mu_opt:", mu_opt)
    print("P_opt:\n", P_opt)
    print("Hinf_norm:", Hinf_norm)
else:
    print("Erro ao resolver o problema.")
    print("Status:", problem.status)
