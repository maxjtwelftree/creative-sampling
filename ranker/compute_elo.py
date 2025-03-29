import numpy as np

class Elo:
    def __init__(self):
        pass
    def rank(self, p1, p2, winner):
        k = 32
        if winner == 0:
            s_a = 1
            s_b = 0
        else:
            s_a = 0
            s_b = 1
        e_a = 1/(1 + 10**((p2 - p1)/400))
        e_b = 1/(1 + 10**((p1 - p2)/400))

        p1_new = p1 + k*(s_a - e_a)
        p2_new = p2 + k*(s_b - e_b)

        return [p1_new, p2_new]