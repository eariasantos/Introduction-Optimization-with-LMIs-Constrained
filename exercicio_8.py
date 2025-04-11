import cvxpy as cp
import numpy as np


A_1 = np.array([[5.3, 5.0],
                [-5.3, -12.7]])
A_2 = np.array([[-13.0, 20.9],
                [-0.6, -14.2]])
B_1 = np.array([[-0.2],
               [-1.2]])
B_2 = np.array([[0.0],
               [0.1]])

# Variáveis
X = cp.Variable((2, 2))  
Z = cp.Variable((1, 2))  
W_1 = cp.Variable((2, 2), symmetric=True) 
W_2 = cp.Variable((2, 2), symmetric=True)  
s = 1.0  

# Construção das LMIs 

# LMI 1 
LMI_1 = cp.bmat([
    [A_1 @ X + X.T @ A_1.T + B_1 @ Z + Z.T @ B_1.T, W_1 - X.T + s * (A_1 @ X + B_1 @ Z)],
    [W_1 - X + s * (X.T @ A_1.T + Z.T @ B_1.T), -s * (X + X.T)]
])

# LMI 2 
LMI_2 = cp.bmat([
    [A_2 @ X + X.T @ A_2.T + B_2 @ Z + Z.T @ B_2.T, W_2 - X.T + s * (A_2 @ X + B_2 @ Z)],
    [W_2 - X + s * (X.T @ A_2.T + Z.T @ B_2.T), -s * (X + X.T)]
])

# Restrições
constraints = [
    W_1 >> 0,  
    W_2 >> 0,  
    LMI_1 << 0,  
    LMI_2 << 0   
]

# Definir função objetivo 
objective = cp.Minimize(0)

# Resolver otimização
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

# Resultados
if problem.status == cp.OPTIMAL:
    print("Solução encontrada:")
    X_opt = X.value
    W1_opt = W_1.value
    W2_opt = W_2.value
    Z_opt = Z.value
    K = np.linalg.solve(X_opt, Z_opt.T).T  # K = Z_opt / X_opt
    print("X_opt:\n", X_opt)
    print("W1_opt:\n", W1_opt)
    print("W2_opt:\n", W2_opt)
    print("Z_opt:\n", Z_opt)
    print("Controlador K:\n", K)
else:
    print("Deu ruim:", problem.status)