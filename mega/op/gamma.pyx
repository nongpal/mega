import math

def gamma(double point):
    if point <= 0:
        raise ValueError("nilai harus lebih besar dari nol")
    
    if point > 175.5:
        raise OverflowError("nilai rentang terlalu besar")

    if point == 0.5:
        return math.sqrt(math.pi)

    if point == 1:
        return 1.0

    return (point - 1) * gamma(point - 1)
