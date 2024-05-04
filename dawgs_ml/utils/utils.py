from dawgs_ml.dataframe import dataframe as df


def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))


def step_activation(x):
    if x > 0:
        return 1
    else:
        return 0


def cartesian_product(*arrays):
    if len(arrays) == 1:
        return [[item] for item in arrays[0]]  # base case, single list
    else:  # recursive case
        result = []
        remaining = cartesian_product(*arrays[1:])
        for item in arrays[0]:
            for r in remaining:
                result.append([item] + r)
        return result
