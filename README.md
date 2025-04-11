
# Otimização Convexa com LMIs (YALMIP/CVXPY)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)](https://www.mathworks.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

---

## 📋 Pré-requisitos

### Para MATLAB/YALMIP
- [MATLAB](https://www.mathworks.com/) instalado (versão R2020a ou superior)
- **YALMIP** instalado:

```matlab
% No MATLAB:
addpath(genpath('caminho/para/yalmip'));
savepath;
```

**Solvers recomendados**:
- [MOSEK](https://www.mosek.com/) (licença acadêmica gratuita)
- [SeDuMi](https://github.com/sqlp/sedumi) (open-source)
- [SDPT3](https://github.com/sqlp/sdpt3) (alternativa gratuita)

---

### Para Python/CVXPY
- **Python 3.8+** instalado.

Instale as bibliotecas necessárias:

```bash
pip install cvxpy numpy scipy
```

**(Opcional)** Instale o solver MOSEK:

```bash
pip install mosek
```

> Para o MOSEK, é necessário obter uma [licença acadêmica gratuita](https://www.mosek.com/products/academic-licenses/).

---

## 🚀 Começando

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/otimizacao-lmis.git
cd otimizacao-lmis
```

### Executando no MATLAB

```matlab
% Execute os exemplos:
run('matlab/exercicio_1.m');
```

### Executando no Python

```bash
python python/exercicio_1.py
```

