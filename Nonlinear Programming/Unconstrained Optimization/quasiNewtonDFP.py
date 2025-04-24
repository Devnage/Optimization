import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def main() -> None:
    f = lambda x1, x2: (7*x1**2) + (2*x1*x2) + (x2**2) + (x1**4) + (x2**4)
    initVector = np.array([-4343, 34590], dtype=float)
    errorTol = 1e-3
    maxIter = 10

    quasiNewtonDFP(f, initVector, errorTol, maxIter)

def quasiNewtonDFP(f, initVector: np.ndarray, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    currVector = initVector
    H_k = np.identity(len(initVector))
    g_k = gradient(f, currVector)

    for k in range(maxIter + 1):
        norm = np.linalg.norm(g_k)
        alpha = exactLineSearch(f, currVector, H_k, g_k)

        table.append([k] + [format_values(x) for x in [currVector[0], currVector[1], g_k, H_k, alpha, norm, f(*currVector)]])
        
        if norm < errorTol:
            break

        newVector = currVector - (alpha * (H_k @ g_k))
        new_g_k = gradient(f, newVector)
        diffVector = newVector - currVector
        diffGrad = new_g_k - g_k

        A_k = np.outer(diffVector, diffVector) / (diffVector.T @ diffGrad)
        C_k = (np.outer(-H_k @ diffGrad, diffGrad) @ H_k.T) / (diffGrad.T @ (H_k.T @ diffGrad))

        H_k = H_k + A_k + C_k
        g_k = new_g_k
        currVector = newVector

    df = pd.DataFrame(table, columns=["k", "x1_k", "x2_k", "del_k", "H_k", "alpha_k", "error", "f(x)"])

    if df.iloc[-1, 0] > 100:
        separator = pd.DataFrame(["..."] * df.shape[1]).T
        separator.columns = df.columns
        df_summary = pd.concat([df.head(20), separator, df.tail(20)])
        print(df_summary.to_string(index=False))
    else:
        print(df.to_string(index=False))

    print()
    print(f"Initial vector: ({df.iloc[0, 1]}, {df.iloc[0, 2]})")
    print(f"Optimal solution ({df.iloc[-1, 0]} iterations, {errorTol} error tolerance): ({df.iloc[-1, 1]}, {df.iloc[-1, 2]})")
    print(f"f({df.iloc[-1, 1]}, {df.iloc[-1, 2]}) = {df.iloc[-1, 7]}")
    print(f"Norm: {df.iloc[-1, 6]}")

def gradient(f, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    diff = np.eye(len(x)) * epsilon
    return np.array([(f(*x + h) - f(*x - h)) / (2 * epsilon) for h in diff])

def exactLineSearch(f, x:np.ndarray, H: np.ndarray, grad: np.ndarray) -> float:
    minf = lambda alpha: f(*x - (alpha * (H @ grad)))
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