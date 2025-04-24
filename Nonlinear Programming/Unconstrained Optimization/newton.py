import numpy as np
import pandas as pd

def main() -> None:
    f = lambda x1, x2: (7*x1**2) + (2*x1*x2) + (x2**2) + (x1**4) + (x2**4)
    initVector = np.array([-1, 1], dtype=float)
    errorTol = 1e-3
    maxIter = 100

    newton(f, initVector, errorTol, maxIter)

def newton(f, initVector: np.ndarray, errorTol: float = 1e-6, maxIter: int = 100) -> None:
    table = []
    currVector = initVector

    for k in range(maxIter + 1):
        g_k = gradient(f, currVector)
        H = hessian(f, currVector)
        H_inv = np.linalg.inv(H)
        norm = np.linalg.norm(g_k)
        s_k = -H_inv @ g_k

        table.append([k] + [format_values(x) for x in [currVector[0], currVector[1], g_k, H, H_inv, s_k, norm, f(*currVector)]])
        
        if norm < errorTol:
            break
        currVector = currVector + s_k

    df = pd.DataFrame(table, columns=["k", "x1_k", "x2_k", "del_k", "H_k", "Hinv_k", "s_k", "error", "f(x)"])

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
    print(f"f({df.iloc[-1, 1]}, {df.iloc[-1, 2]}) = {df.iloc[-1, 8]}")
    print(f"Norm: {df.iloc[-1, 7]}")

def gradient(f, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    diff = np.eye(len(x)) * epsilon
    return np.array([(f(*x + h) - f(*x - h)) / (2 * epsilon) for h in diff])

def hessian(f, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    n = len(x)
    hf = np.zeros((n, n))
    I = np.eye(n) * epsilon

    for i in range(n):  
        for j in range(n):
            d1 = f(*x + I[i] + I[j])
            d2 = f(*x + I[i] - I[j])
            d3 = f(*x - I[i] + I[j])
            d4 = f(*x - I[i] - I[j])

            hf[i, j] = (d1 - d2 - d3 + d4) / (4*epsilon**2)

    return hf

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
