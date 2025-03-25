import sympy as sp


def make_symbolic_matrix(row: int, col: int, prefix: str) -> sp.Matrix:
    """get symbolic matrix

    Args:
        row (int): row size
        col (int): column size
        prefix (str): prefix for symbol

    Returns:
        sp.Matrix: symbolic matrix
    """
    return sp.Matrix(
        [
            [sp.symbols(f"{prefix}_{i}{j}", real=True) for j in range(col)]
            for i in range(row)
        ]
    )


vars: dict = {}


def make_state_vector(tuples: list, dif_symbol: sp.Symbol) -> sp.Matrix:
    """symbolic 변수와 그것의 n차 미분을 포함하는 state vector(sp.Matrix)를 만든다.
    input으로는 (변수명, n)의 tuple list와 미분할 기호를 받는다.

    ex) make_state_matrix([(x, 2), (y, 3)], t)

    -> [[x], [dx/dt], [d^2x/dt^2], [y], [dy/dt], [d^2y/dt^2], [d^3y/dt^3], [d^4y/dt^4]]

    Args:
        *tuples: (sympy.FunctionClass, int) tuple list
        dif_symbol: 이 기호로 미분한다.
    Returns:
        sp.Matrix: column vector of symbolic variables
    """

    # check input type
    tuple_list = []
    for t in tuples:
        if not isinstance(t, tuple):
            raise TypeError(f"Expected tuple, but got {type(t).__name__}: {t}")
        if not isinstance(t[0], sp.Function):
            raise TypeError(
                f"Expected sympy.Function, but got {type(t[0]).__name__}: {t[0]}"
            )
        if not isinstance(t[1], int):
            raise TypeError(f"Expected int, but got {type(t[1]).__name__}: {t[1]}")
        tuple_list.append(t)
    if not isinstance(dif_symbol, sp.Symbol):
        raise TypeError(
            f"Expected sympy.Symbol, but got {type(dif_symbol).__name__}: {dif_symbol}"
        )

    # make state matrix
    states = []
    vardict = {}
    for t in tuple_list:
        states.append(t[0])
        vardict[t[0]] = sp.symbols(f"{t[0].name}", real=True)
        d_string = "d"
        for _ in range(t[1]):
            states.append(states[-1].diff(dif_symbol))
            vardict[states[-1]] = sp.symbols(f"{t[0].name}{d_string}ot", real=True)
            d_string += "d"
        vardict[states[-1].diff(dif_symbol)] = sp.symbols(
            f"{t[0].name}{d_string}ot", real=True
        )

    # save variable dictionary
    for _ in range(len(vardict)):
        key, value = vardict.popitem()
        vars[key] = value

    return sp.Matrix(states)


def make_state_comfy_to_see(func: sp.Function) -> sp.Function:
    """state 변수들을 d/dt 대신 dot으로 표현하도록 한다.

    Args:
        func (sp.Function): state 변수들을 사용하는 symbolic function

    Returns:
        sp.Function: state 변수들을 dot으로 표현한 symbolic function
    """
    out = func
    for key, value in vars.items():
        out = out.subs(key, value)
    return out


def show_state_vars():
    """state 변수들을 보여준다."""
    for key, value in vars.items():
        sp.pprint(sp.Eq(key, value))

def make_11matrix_to_scalar(matrix: sp.Matrix) -> sp.Symbol:
    """1x1 matrix를 scalar로 변환한다.

    Args:
        matrix (sp.Matrix): 1x1 matrix

    Returns:
        sp.Matrix: scalar
    """
    if matrix.shape != (1, 1):
        raise ValueError("Expected 1x1 matrix, but got {matrix.shape}")
    out = matrix[0, 0]
    out = sp.simplify(out)
    return out


# test
if __name__ == "__main__":
    A = make_symbolic_matrix(3, 3, "a")
    sp.pprint(A)

    t = sp.symbols("t")
    x = sp.Function("x")(t)
    y = sp.Function("y")(t)

    state_vector = make_state_vector([(x, 0), (y, 0)], t)

    sp.pprint(state_vector)
    show_state_vars()
    sp.pprint(make_state_comfy_to_see(state_vector))
    sp.pprint(make_11matrix_to_scalar(state_vector.T @ state_vector))
