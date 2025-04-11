clc; clear all; close all; rng('default');

%% Definir as matrizes conhecidas
% Limpar memória do YALMIP
yalmip('clear'); % isso é feito para evitar problemas com variáveis antigas

%definir as matrizes conhecidas:
A = [0 1; -2 -3]; 
Q = [1 0; 0 1];

%{ 
Definir a variável P.
No YALMIP 'sdpvar' é o comando usado para criar variáveis de decisão.
Ele permite definir variáveis escalares, vetores e matrizes.
%}
P = sdpvar(2, 2, 'symmetric'); % P é uma matriz simétrica 2×2 (porque A é 2×2).


%% Definir as restrições (LMIs)
%{ 
as restrições: 1. P > 0 ; 2. A' + PA + Q < 0
No YALMIP, Constraints é uma variável usada para armazenar as restrições de
um problema. As restrições são declaradas usando operadores.
%}

%Constraints = [P >= 0, A'*P + P*A + Q <= 0];

%% Definir epsilon e restrições ajustadas
%{
Para garantir estrita positividade (P>0), alguns solvers aceitam 
P≥ϵI (com ϵ pequeno),e ajustar se necessário.
%}
epsilon = 0.1; % Vamos adicionar uma margem pequena (ϵ) para forçar a estrictez
%M = A'*P + P*A + Q
Constraints = [P >= epsilon*eye(2), A'*P + P*A + Q <= -epsilon*eye(2)];

%% Definir o objetivo
%O objetivo é minimizar o traço de P.

Objective = trace(P);

%{
Configurar opções do solver
Especifica o MOSEK como solver 
e Mostra informações do solver enquanto ele roda
%}

options = sdpsettings('solver', 'mosek', 'verbose', 1);

%{
 Resolver o problema
Isso tenta encontrar P que minimiza Tr(P) sujeito às restrições. 
%}
sol = optimize(Constraints, Objective, options); % a função optimize do YALMIP.

% Verificar e exibir o resultado
if sol.problem == 0
    P_opt = value(P);
    Tr_P = trace(P_opt);
    disp('Matriz P ótima:');
    disp(P_opt);
    disp(['Traço ótimo de P: ', num2str(Tr_P)]);
    % Verificar a LMI explicitamente
    M = A'*P_opt + P_opt*A + Q;
    disp('A^T P + P A + Q:');
    disp(M);
else
    disp('Erro ao resolver o problema!');
    disp(sol.info);
end
