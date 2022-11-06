import numpy as np


def smooth(window, input):
    result = []
    i = 0
    input_abs = np.absolute(input)
    while i < len(input) - window:
        result.append(np.mean(input_abs[i:i+window]))
        i += 1
    return result

