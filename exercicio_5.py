import cvxpy as cp
import numpy as np


k = 1  
f = 1  

# Definir as matrizes do sistema
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-k, k, -f, f],
              [k, -k, f, -f]])

B1 = np.array([[0], [0], [0], [1]])  
B2 = np.array([[0], [0], [0], [1]])  

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0]])

D = np.array([[0], [0], [1]]) 

# Definir variáveis de decisão
W = cp.Variable((4, 4), symmetric=True) 
Z = cp.Variable((1, 4)) 
mu = cp.Variable() 

# Construção das LMIs
LMI1 = W >> 0

LMI2 = cp.bmat([
    [A @ W + W @ A.T + B2 @ Z + Z.T @ B2.T, W @ C.T + Z.T @ D.T, B1],
    [C @ W + D @ Z, -np.eye(C.shape[0]), np.zeros((C.shape[0], B1.shape[1]))],
    [B1.T, np.zeros((B1.shape[1], C.shape[0])), -mu * np.eye(B1.shape[1])]
]) << 0   

constraints = [LMI1, LMI2]

# Definir função objetivo
objective = cp.Minimize(mu)

# Resolver o problema
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=True)

#  resultados
if prob.status == cp.OPTIMAL:
    print("Solução encontrada:")
    mu_opt = mu.value
    W_opt = W.value
    Z_opt = Z.value
    K = np.linalg.solve(W_opt, Z_opt.T).T  
    Hinf_norm = np.sqrt(mu_opt) 
    
    print("mu_opt:", mu_opt)
    print("W_opt:\n", W_opt)
    print("Z_opt:\n", Z_opt)
    print("K:\n", K)
    print("Hinf_norm:", Hinf_norm)
else:
    print("Erro ao resolver o problema.")
    print("Status:", prob.status)
