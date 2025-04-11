import cvxpy as cp
import numpy as np

# Definição das matrizes do problema
M0 = np.array([[-2.4, -0.6, -1.7, 3.1],
               [0.7, -2.1, -2.6, -3.6],
               [0.5, 2.4, -5.0, -1.6],
               [-0.6, 2.9, -2.0, -0.6]])

M1 = np.array([[ 1.1, -0.6, -0.3, -0.1],
               [-0.8,  0.2, -1.1,  2.8],
               [-1.9,  0.8, -1.1,  2.0],
               [-2.4, -3.1, -3.7, -0.1]])

M2 = np.array([[ 0.9,  3.4,  1.7,  1.5],
               [-3.4, -1.4,  1.3,  1.4],
               [ 1.1,  2.0, -1.5, -3.4],
               [-0.4,  0.5,  2.3,  1.5]])

M3 = np.array([[-1.0, -1.4, -0.7, -0.7],
               [ 2.1,  0.6, -0.1, -2.1],
               [ 0.4, -1.4,  1.3,  0.7],
               [ 1.5,  0.9,  0.4, -0.5]])

n = 4
'''
rho = 1

A_0 = M0 + rho * M1 
A_1 = M0 + rho * M2
A_2 = M0 + rho * M3

#  Problema de factibilidade 1:

P = cp.Variable((n, n), symmetric=True) 
X1 = cp.Variable((n, n))  
X2 = cp.Variable((n, n))

# Construindo LMIs

LMI_1 = cp.bmat([[X1 @ A_0 + A_0.T @ X1.T,  P - X1 + A_0.T @ X2.T],
               [P + X2 @ A_0 - X1.T, -X2 - X2.T]])

LMI_2 = cp.bmat(([[X1 @ A_1 + A_1.T @ X1.T,  P - X1 + A_1.T @ X2.T],
               [P + X2 @ A_1 - X1.T, -X2 - X2.T]]))

LMI_3 = cp.bmat(([[X1 @ A_2 + A_2.T @ X1.T,  P - X1 + A_2.T @ X2.T],
               [P + X2 @ A_2 - X1.T, -X2 - X2.T]])) 


# Restrições
constraints_1 = [
    P >> 0,  # P deve ser positiva definida
    LMI_1 << 0,  
    LMI_2 << 0,  
    LMI_3 << 0   
]

# Resolver o problema
problem = cp.Problem(cp.Minimize(0), constraints_1) # Usamos 0 porque é só factibilidade
problem.solve(solver=cp.MOSEK, verbose=True)

# Verificar e exibir o resultado
if problem.status == cp.OPTIMAL:
    print(f"Para rho = {rho}, o sistema é estável!")
else:
    print(f"Para rho = {rho}, o sistema NÃO é estável.")

'''

#  Problema de factibilidade 2:

rho_1 = 1

A_0 = M0 + rho_1 * M1 
A_1 = M0 + rho_1 * M2
A_2 = M0 + rho_1 * M3


# Definindo as variáveis 
P_1 = cp.Variable((n, n), symmetric=True) 
P_2 = cp.Variable((n, n), symmetric=True)  
P_3 = cp.Variable((n, n), symmetric=True)  



# Primeira condição
LMI_4 = A_0.T @ P_1 + P_1 @ A_0 
LMI_5 = A_1.T @ P_2 + P_2 @ A_1  
LMI_6 = A_2.T @ P_3 + P_3 @ A_2  

# Segunda condição
LMI_7 = A_0.T @ P_2 + P_2 @ A_0 + A_1.T @ P_1 + P_1 @ A_1  # i=1, j=2
LMI_8 = A_0.T @ P_3 + P_3 @ A_0 + A_2.T @ P_1 + P_1 @ A_2  # i=1, j=3
LMI_9 = A_1.T @ P_3 + P_3 @ A_1 + A_2.T @ P_2 + P_2 @ A_2  # i=2, j=3

# Restrições
constraints = [
    P_1 >> 0,  
    P_2 >> 0,  
    P_3 >> 0,  
    LMI_4 << 0,  
    LMI_5 << 0,  
    LMI_6 << 0,  
    LMI_7 << 0,  
    LMI_8 << 0,  
    LMI_9 << 0   
]

# Resolver o problema
problem = cp.Problem(cp.Minimize(0), constraints)  
problem.solve(solver=cp.MOSEK, verbose=True)

# Verificar e exibir o resultado
if problem.status == cp.OPTIMAL:
    print(f"Para rho_1 = {rho_1}, o sistema é estável!")
else:
    print(f"Para rho_1 = {rho_1}, o sistema NÃO é estável.")


