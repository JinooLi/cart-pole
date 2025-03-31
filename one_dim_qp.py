import numpy as np


def one_qp(desired: float, const_lhs: np.ndarray, const_rhs: np.ndarray) -> float:
    """const_lhs @ x <= const_rhs 을 만족하며 desired에 가장 가까운 x를 찾는다.

    Args:
        desired (float):  목표 값
        const_lhs (np.ndarray): column vector
        const_rhs (np.ndarray): column vector
    """
    # check input type
    if not isinstance(desired, (int, float)):
        raise TypeError(
            f"Expected int or float, but got {type(desired).__name__}: {desired}"
        )
    if not isinstance(const_lhs, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, but got {type(const_lhs).__name__}: {const_lhs}"
        )
    if not isinstance(const_rhs, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, but got {type(const_rhs).__name__}: {const_rhs}"
        )
    if desired == np.nan:
        raise ValueError("desired must be a number")
    if const_lhs.shape[1] != 1:
        raise ValueError(f"Expected 1, but got {const_lhs.shape[1]}")
    if const_rhs.shape[1] != 1:
        raise ValueError(f"Expected 1, but got {const_rhs.shape[1]}")
    if const_lhs.shape[0] != const_rhs.shape[0]:
        raise ValueError("rhs and lhs Constraints must have the same shape")

    print("==== start ====")

    out = desired
    delta = 1e-8
    length = const_lhs.shape[0]
    collide = False
    for i in range(length):
        comp = const_rhs[i, 0] / const_lhs[i, 0]
        if const_lhs[i, 0] < 0:
            print(f"{comp} <= u")
        else:
            print(f"u <= {comp}")
        if const_lhs[i, 0] * out > const_rhs[i, 0]:
            collide = True
            out = comp

    if collide:
        print("collide")

    for i in range(length):
        if const_lhs[i, 0] * out - delta > const_rhs[i, 0]:
            raise ValueError("No solution")
    
    print(f"Optimal: {out}")
    print("==== done ====")

    return out


if __name__ == "__main__":
    desired = 0.0
    const_lhs = np.array([[1.0, 2.0]])
    const_rhs = np.array([[3.0, 4.0]])
    print(const_lhs.shape)
    print(const_rhs.ndim)
    out = one_qp(desired, const_lhs, const_rhs)
    print(out)
