import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def main() -> None:
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    b = np.array([1, 0, 2])
    c = 3

    f = lambda x: (1/2) * np.array(x).T @ A @ np.array(x) + b.T @ np.array(x) + c
    initVector = np.array([0, 0, 0], dtype = float)
    errorTol = 1e-6
    maxIter = 100
    conjugateGradient(f, initVector, errorTol, maxIter)

def conjugateGradient(f, initVector: np.ndarray, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    currVector = initVector
    g_k = gradient(f, currVector)
    s_k = -g_k
    gamma = None
    
    for k in range(maxIter + 1):
        lambdaa = exactLineSearch(f, currVector, s_k)
        table.append([k] + [format_values(x) for x in [currVector[0], currVector[1], currVector[2], g_k, s_k, lambdaa, gamma, f(currVector)]])
        
        if np.linalg.norm(g_k) < errorTol:
            break

        currVector = (currVector) + (lambdaa * s_k)
        new_g_k = gradient(f, currVector)
        gamma = (new_g_k.T @ new_g_k) / (g_k.T @ g_k)
        s_k = -new_g_k + (gamma * s_k)
        g_k = new_g_k

    df = pd.DataFrame(table, columns = ["k", "x1_k", "x2_k", "x3_k", "g_k", "s_k", "lambda", "gamma", "f(x)"])

    if df.iloc[-1, 0] > 100:
        separator = pd.DataFrame(["..."] * df.shape[1]).T
        separator.columns = df.columns
        df_summary = pd.concat([df.head(20), separator, df.tail(20)])
        print(df_summary.to_string(index = None))
    else:
        print(df.to_string(index = None))

    print()
    print(f"Initial vector: ({df.iloc[0, 1]}, {df.iloc[0, 2]}, {df.iloc[0, 3]})")
    print(f"Optimal solution ({df.iloc[-1, 0]} iterations, {errorTol} error tolerance): ({df.iloc[-1, 1]}, {df.iloc[-1, 2]}, {df.iloc[-1, 3]})")
    print(f"f({df.iloc[-1, 1]}, {df.iloc[-1, 2]}, {df.iloc[-1, 3]}) = {df.iloc[-1, 7]}")
    print(f"Norm: {df.iloc[-1, 6]}")

def gradient(f, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    diff = np.eye(len(x)) * epsilon
    return np.array([(f(x + h) - f(x - h)) / (2 * epsilon) for h in diff])

def exactLineSearch(f, x: np.ndarray, d: np.ndarray) -> float:
    minf = lambda alpha: f(x + (alpha * d))
    return minimize_scalar(minf).x

def format_values(value, decimal = 6) -> str:
    if isinstance(value, (np.ndarray, list)):
        if value.ndim == 1 or isinstance(value, list):
            return "[" + ", ".join([f"{float(val):.{decimal}f}" for val in value]) + "]"
        elif value.ndim == 2:
            if value.shape == (2, 2):
                return (f"[[{float(value[0,0]):.{decimal}f}, {float(value[0,1]):.{decimal}f}] "
                       f"[{float(value[1,0]):.{decimal}f}, {float(value[1,1]):.{decimal}f}]]")
            else:
                rows = []
                for row in value:
                    row_str = ", ".join([f"{float(val):.{decimal}f}" for val in row])
                    rows.append(f"[{row_str}]")
                return "[" + ", ".join(rows) + "]"
    elif isinstance(value, (int, float)):
        return f"{float(value):.{decimal}f}"
    elif value is None:
        return "None"
    return str(value)
    
if __name__ == "__main__":
    main()