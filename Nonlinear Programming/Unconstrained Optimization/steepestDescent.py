import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def main() -> None:
    f = lambda x, y: 10*(y - x**2)**2 + (x - 1)**2
    initVector = np.array([-10, 20], dtype = float)
    errorTol = 1e-6
    maxIter = 1000
    steepestDescent(f, initVector, errorTol, maxIter)

def steepestDescent(f, initVector: np.ndarray, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    currVector = initVector
    
    for k in range(maxIter + 1):
        g_k = gradient(f, currVector)
        norm = np.linalg.norm(g_k)
        d_k = -g_k
        #d_k = -(g_k / norm)
        alpha = exactLineSearch(f, currVector, d_k)
        table.append([k] + [format_values(x) for x in [currVector[0], currVector[1], d_k[0], d_k[1], norm, alpha, f(*currVector)]])
        
        if norm < errorTol:
            break

        currVector = (currVector) + (alpha * d_k)

    df = pd.DataFrame(table, columns = ["k", "x_k", "y_k", "d1_k", "d2_k", "||d||_2", "alpha_k", "f(x)"])

    if df.iloc[-1, 0] > 100:
        separator = pd.DataFrame(["..."] * df.shape[1]).T
        separator.columns = df.columns
        df_summary = pd.concat([df.head(20), separator, df.tail(20)])
        print(df_summary.to_string(index = None))
    else:
        print(df.to_string(index = None))

    print()
    print(f"Initial vector: ({df.iloc[0, 1]}, {df.iloc[0, 2]})")
    print(f"Optimal solution ({df.iloc[-1, 0]} iterations, {errorTol} error tolerance): ({df.iloc[-1, 1]}, {df.iloc[-1, 2]})")
    print(f"f({df.iloc[-1, 1]}, {df.iloc[-1, 2]}) = {df.iloc[-1, 7]}")
    print(f"Norm: {df.iloc[-1, 5]}")

def gradient(f, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    diff = np.eye(len(x)) * epsilon
    return np.array([(f(*x + h) - f(*x - h)) / (2 * epsilon) for h in diff])

def exactLineSearch(f, x: np.ndarray, d: np.ndarray) -> float:
    minf = lambda alpha: f(*x + (alpha * d))
    return minimize_scalar(minf, bounds=(-1e6, 1e6), method='bounded').x

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