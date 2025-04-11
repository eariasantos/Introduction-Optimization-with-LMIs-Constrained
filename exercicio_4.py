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
X = cp.Variable((3, 3), symmetric=True)  
W = cp.Variable((4, 4), symmetric=True) 
Z = cp.Variable((1, 4)) 
rho = cp.Variable()  


# Construção das LMIs
LMI1 = cp.bmat([
    [X, C @ W + D @ Z],
    [(C @ W + D @ Z).T, W]
]) >>  0              
LMI2 = rho >= cp.trace(X) 

LMI3 = cp.bmat([
    [A @ W + B2 @ Z + W @ A.T + Z.T @ B2.T, B1],
    [B1.T, -np.eye(1)]
]) << 0    

# restrições
constraints = [
    X >> 0, 
    W >> 0, 
    LMI1, LMI2, LMI3
]

# Definir função objetivo
objective = cp.Minimize(rho)

# resolver o problema de otimização
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

# resultados 
if problem.status == cp.OPTIMAL:
    print("Solução encontrada:")
    rho_opt = rho.value
    X_opt = X.value
    W_opt = W.value
    Z_opt = Z.value
    K = np.linalg.solve(W_opt, Z_opt.T).T  # K = Z_opt / W_opt 
    H2_norm = np.sqrt(rho_opt) # norma H2
    
    print("rho_opt:", rho_opt)
    print("X_opt:\n", X_opt)
    print("W_opt:\n", W_opt)
    print("Z_opt:\n", Z_opt)
    print("K:\n", K)
    print("H2_norm:", H2_norm)

