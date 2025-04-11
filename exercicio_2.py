import cvxpy as cp 
import numpy as np  

m1 = 1  
m2 = 0.5 
k1 = 1 
k2 = 1 
c0 = 2 # coef

# Definir as matrizes do sistema 

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-(k1 + k2) / m1, k2 / m1, -c0 / m1, 0],
              [k2 / m2, -k2 / m2, 0, -c0 / m2]])

B = np.array([[0], [0], [1 / m1], [0]])

C = np.array([[0, 1, 0, 0]])  

W = cp.Variable((4, 4), symmetric=True)   
rho = cp.Variable()  

# restrições
constraints = [
    W >> 0,  
    A @ W + W @ A.T + B @ B.T << 0,  
    rho >= cp.trace(C @ W @ C.T)  
]

# função objetivo
objective = cp.Minimize(rho)

# resolver o problema de otimização
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True) 

# resultados
if problem.status == cp.OPTIMAL:
    print("Solução encontrada!")
    rho_opt = rho.value  
    W_opt = W.value  
    H2_norm = np.sqrt(rho_opt)  
    
    print("rho_opt:", rho_opt)
    print("Matriz W ótima:\n", W_opt)
    print("Norma H2:", H2_norm)
else:
    print("Erro ao resolver o problema.")
    print("Status:", problem.status)
