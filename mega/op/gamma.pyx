# MIT License

# Copyright (c) 2025 WargaSlowy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# cython: language_level=3
# mega/op/gamma.pyx

from mega.utils.constant cimport PI_NUMBER, SQRT_PI
from libc.math cimport sqrt, exp, log, sin, pow

cpdef double gamma(double point):
    """
    computing gamma function Γ(point) for real-valued `point > 0`

    implement using lanczos approximation which providing an efficient and
    accurate way to compute Γ(z) for non-integer values

    Parameter:
        point (double): real number greater than zero at which to evaluate  gamma function

    Return:
        (double): value of the gamma function at `gamma`

    Note:
        - Γ(0.5) = √π ≈ 1.77245385091
        - Γ(1.0) = 1
        - Γ(2.0) = 1

    Example:
    >>> gamma(0.5)
    1.77245385091
    
    >>> gamma(5.0)
    24.0

    Reference:
     - https://en.wikipedia.org/wiki/Gamma_function
     - https://en.wikipedia.org/wiki/Lanczos_approximation
    """
    cdef double z = point
    cdef double x, y, t, tmp, fact
    cdef int i, n
    cdef int neg = 0

    if point <= 0:
        raise ValueError("point must be bigger than zero")
    if point > 175.5:
        raise OverflowError("input value too large, will be OverflowError")

    # special case
    if point == 0.5:
        return SQRT_PI

    if point == 1.0 or point == 2.0:
        return 1.0

    if z == <int>z:
        n  = <int>z;
        fact = 1.0
        for i in range(2, n):
            fact *= i
        return fact

    if point == int(point):
        n = <int>point
        result = 1.0
        for i in range(2, n):
            result *= i
        return result

    z = point
    y = z
    
    neg = 0
    if z < 0.5:
        y = 1.0 - z
        neg = 1
    else:
        y = z

    x = 0.99999999999980993

    cdef double[8] LANCZOS_COEFF = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    for i in range(8):
        x += LANCZOS_COEFF[i] / (y + i)
    t = y + 7.5
    tmp = sqrt(2.0 * PI_NUMBER) * x * pow(t, y - 0.5) * exp(-t)
    if neg:
        return PI_NUMBER / (sin(PI_NUMBER * z) * tmp)
    return tmp
