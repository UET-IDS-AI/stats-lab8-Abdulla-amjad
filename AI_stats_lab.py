import numpy as np

def joint_cdf_unit_square(x, y):
    if x <= 0 or y <= 0:
        return 0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:
        return 1

def rectangle_probability(x1, x2, y1, y2):
    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )

def marginal_fx_unit_square(x):
    return 1 if 0 < x < 1 else 0

def marginal_fy_unit_square(y):
    return 1 if 0 < y < 1 else 0

def joint_pmf_heads(x, y):
    pmf = {
        (0, 0): 1/4,
        (0, 1): 1/4,
        (0, 2): 0,
        (1, 0): 0,
        (1, 1): 1/4,
        (1, 2): 1/4,
    }
    return pmf.get((x, y), 0)

def marginal_px_heads(x):
    return sum(joint_pmf_heads(x, y) for y in [0, 1, 2])

def marginal_py_heads(y):
    return sum(joint_pmf_heads(x, y) for x in [0, 1])

def check_independence_heads():
    for x in [0, 1]:
        for y in [0, 1, 2]:
            if joint_pmf_heads(x, y) != marginal_px_heads(x) * marginal_py_heads(y):
                return False
    return True
