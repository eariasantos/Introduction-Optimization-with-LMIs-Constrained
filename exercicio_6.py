import cvxpy as cp
import numpy as np

# Dados básicos
k1, k2 = 0.09, 0.4
f1, f2 = 0.0038, 0.04

A_1 = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [-k1, k1, -f1, f1],
               [k1, -k1, f1, -f1]])
B1_1 = np.array([[0], [0], [0], [1]])
B2_1 = np.array([[0], [0], [0], [1]])
C_1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
D_1 = np.array([[0], [0], [1]])


A_2 = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [-k1, k1, -f2, f2],
               [k1, -k1, f2, -f2]])
B1_2 = np.array([[0], [0], [0], [1]])
B2_2 = np.array([[0], [0], [0], [1]])
C_2 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
D_2 = np.array([[0], [0], [1]])

A_3 = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [-k2, k2, -f1, f1],
               [k2, -k2, f1, -f1]])
B1_3 = np.array([[0], [0], [0], [1]])
B2_3 = np.array([[0], [0], [0], [1]])
C_3 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
D_3 = np.array([[0], [0], [1]])

A_4 = np.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [-k2, k2, -f2, f2],
               [k2, -k2, f2, -f2]])
B1_4 = np.array([[0], [0], [0], [1]])
B2_4 = np.array([[0], [0], [0], [1]])
C_4 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]])
D_4 = np.array([[0], [0], [1]])

# Variáveis
X = cp.Variable((4, 4), symmetric=True)
W = cp.Variable((4, 4), symmetric=True)
Z = cp.Variable((1, 4))
rho = cp.Variable()
W_I = cp.Variable((1, 1), symmetric=True)  


# Construção das LMIs

LMI_1 = cp.bmat([[X, B1_1], [B1_1.T, W_I]]) >> 0
LMI_2 = cp.bmat([[X, B1_2], [B1_2.T, W_I]]) >> 0
LMI_3 = cp.bmat([[X, B1_3], [B1_3.T, W_I]]) >> 0
LMI_4 = cp.bmat([[X, B1_4], [B1_4.T, W_I]]) >> 0

LMI_5 = rho >= cp.trace(X) 

LMI_6 = cp.bmat([
    [A_1 @ W + B2_1 @ Z + W @ A_1.T + Z.T @ B2_1.T, B1_1],
    [B1_1.T, -np.eye(1)]
]) << 0  
LMI_7 = cp.bmat([
    [A_2 @ W + B2_2 @ Z + W @ A_2.T + Z.T @ B2_2.T, B1_2],
    [B1_2.T, -np.eye(1)]
]) << 0  
LMI_8 = cp.bmat([
    [A_3 @ W + B2_3 @ Z + W @ A_3.T + Z.T @ B2_3.T, B1_3],
    [B1_3.T, -np.eye(1)]
]) << 0  
LMI_9 = cp.bmat([
    [A_4 @ W + B2_4 @ Z + W @ A_4.T + Z.T @ B2_4.T, B1_4],
    [B1_4.T, -np.eye(1)]
]) << 0  

# restrições
constraints = [
    X >> 0, 
    W >> 0, 
    LMI_1, LMI_2, LMI_3,
    LMI_4, LMI_5, LMI_6,
    LMI_7, LMI_8, LMI_9
]

# Definir função objetivo
objective = cp.Minimize(rho)

# resolver otimização
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

