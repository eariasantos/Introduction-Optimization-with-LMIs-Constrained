import cvxpy as cp
import numpy as np


A = np.array([[0, 1], [-2, -3]])
Q = np.array([[1, 0], [0, 1]])

# Definir a variável P 
P = cp.Variable((2, 2), symmetric=True)
'''
# Definir epsilon para garantir positividade estrita
epsilon = 0.1
I = np.eye(2)

#  restrições 
constraints = [P >> epsilon * I, A.T @ P + P @ A + Q << -epsilon * I]
'''
# Definição do problema 
constraints = [P >> 0, A.T @ P + P @ A + Q << 0]

# função objetivo
objective = cp.Minimize(cp.trace(P))

# Resolver o problema
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

# Verificar e exibir o resultado
if problem.status == cp.OPTIMAL:
    P_opt = P.value
    Tr_P = np.trace(P_opt)
    print("Matriz P ótima:")
    print(P_opt)
    print(f"Traço ótimo de P: {Tr_P}")
    
    # Verificar a LMI explicitamente
    M = A.T @ P_opt + P_opt @ A + Q
    print("A^T P + P A + Q:")
    print(M)
else:
    print("Erro ao resolver o problema!")
