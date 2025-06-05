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
# mega/op/function.pyx

from mega.utils.constant cimport PI_NUMBER, SQRT_PI
from libc.math cimport sqrt, exp, sin, pow, cos
from libc.stdlib cimport malloc, free


def prime_factors(int n, bint unique=False) -> list[int]:
    """"
    return prime factor of positive integer n

    Parameter:
        n (int): integer to be factorized, must greater than zero
        unique (bool): if True, return only distinct prime factors

    Return:
        list[int]: list of prime factor of `n`, sorted in increasing order

    Example:
    >>> prime_factors(60)
    [2, 2, 3, 5]
    >>> prime_factors(13, unique=True)
    [13]
    """
    if n <= 0:
        raise ValueError("only positive integer are acc")
    cdef int i = 2
    cdef list factors = []

    # trie division algorithm
    while i * i <= n:
        while n % i == 0:
            # append current factor
            factors.append(i)
            # divide n by the factor
            n //= i
        # move to next candidate divisor
        i += 1

    # remaining prime factors
    if n > 1:
        factors.append(n)
    # remove duplicate if unique flag is set
    if unique:
        seen = set()
        # using list comprehension to preserve order while removing duplicate
        return [x for x in factors if not (x in seen or seen.add(x))]
    else:
        return factors


cdef class Haversine:
    """
    compute the haversine function of an angle θ

    formula haversine function:
    haversine(θ) = sin^2(θ/2) = (1 - cos(θ)) / 2

    Attribute:
        theta (double): angle in radian passed during initialization

    Methods
        compute(self) (double): compute haversine(theta)
        get_theta(self) (double): get current theta value
        set_theta(self, double new_theta): set new angle in raadians

    Example:
    >>> h = Haversine(0)
    >>> h.compute()
    0.0
    >>> import math
    >>> h.set_theta(math.pi / 2)
    >>> h.compute()
    0.5
    """
    cdef double theta

    def __cinit__(self, double theta) -> None:
        """
        constructor for haversine class

        called once when object is created

        Parameter:
            theta (double): angle in radians - must be numeric float or double
        """
        self.theta = theta

    def __dealloc__(self) -> None:
        """
        currently empty because no dynamic memory allocation
        is used
        """
        pass

    cpdef double compute(self):
        """
        compute haversine function

        Return:
            (double): value of haversine(theta) = (1 - cos(theta)) / 2
        """
        # cosine based definition for avoiding compute square root or power
        return (1.0 - cos(self.theta)) / 2.0

    def get_theta(self):
        """
        get the current angle stored in the haversine instance
        usefull for debug or chaining operations

        Return:
            (double): current theta (in radians)
        """
        return self.theta

    def set_theta(self, double new_theta) -> None:
        """
        update internal angle theta

        allow reuse of same object for multiple computation

        Parameter:
            new_theta (double): new angle in radians
        """
        self.theta = new_theta

    def __repr__(self) -> str:
        return f"Haversine(self.theta) = {Haversine(self.theta).compute()}"

cdef class Gamma:
    """
    compute gamma function usign lanczos approximation

    Attribute:
        point (double): input value at witch evaluating gamma function
        LANCZOS_COEFF (double[8]): coefficient using in lanczos approximation
        cache (dict): optional dictionary for storing previously computed result for
                        performance

    Example:
    >>> g = Gamma(5.0)
    >>> print(g.compute())
    24.0
    """
    cdef double z
    cdef double[8] LANCZOS_COEFF
    cdef double point
    cdef dict cache

    def __cinit__(self, double point) -> None:
        """
        initialize gamma instance with a given point

        Parameter:
            point (double): value of z where gamma(z) will be evaluated
                            must be positive real number
        """
        self.LANCZOS_COEFF = [
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ]
        if point <= 0:
            raise ValueError("only positive real number acc")
        if point > 175.5:
            raise OverflowError("currently input to large for computation")

        self.point = point

    cpdef double compute(self):
        """
        compute value of the gamma function gamma(point) using lanczos approximation

        this method will be handling:
            - special cases: gamma(n) where n is integer (factorial)
            - half integers like gamma(0.5) = sqrt(pi)
            - reflection formula for z < 0.5
            - general case via lanczos approximation

        Return:
            (double): approximation of gamma(self.point)
        """
        cdef int i
        cdef double z = self.point
        cdef double y = z
        cdef double x = 0.99999999999980993
        cdef int neg = 0

        if self.point == 0.5:
            return SQRT_PI

        if self.point == 1.0 or self.point == 2.0:
            return 1.0

        cdef int n = <int>self.point
        cdef long result
        if self.point == n:
            result = 1
            for i in range(2, n):
                result *= i
            return <double>result

        if z < 0.5:
            y = 1.0 - z
            neg = 1

        for i in range(8):
            x += self.LANCZOS_COEFF[i] / (y + i)

        cdef double t = y + 7.5
        cdef double tmp = sqrt(2.0 * PI_NUMBER) * x * pow(t, y - 0.5) * exp(-t)
        if neg:
            return PI_NUMBER / (sin(PI_NUMBER * z) * tmp)
        else:
            return tmp

    def __repr__(self) -> str:
        computed_value = self.compute()
        return f"Gamma({self.point}) = {computed_value:.10f}"


cdef class JordanTotient:
    """
    compute jordan totient function

    Jordan totient generalizae euler totient function and formula
    defined as:

    Jₖ(n) = nᵏ × ∏(p|n) [1 - 1/pᵏ] = nᵏ × ∏(p|n) (pᵏ - 1)/pᵏ

    Attribute:
        n (int): number for which to compute the jordan totient
        k (int): exponent applied each div in the formula

    Methods:
        compute(): compute jordan totient based on stored value
        ___repr__(): string representation for debug
        __getitem__(): allow indexing like dictionary
        __setitem__(): manual cache value

    Example:
    >>> jt = JordanTotient(6, 1)
    >>> jt.compute()
    2
    """
    cdef int n, k
    cdef dict _cache

    def __cinit__(self, int n, int k) -> None:
        """
        initialize JordanTotient class

        Parameter:
            n (int): must be >= 1 - domain of jordan function
            k (int): exponent
        """
        if n <= 0:
            raise ValueError("only positive integers are accepted for n")
        if k < 0:
            raise ValueError("exponent k must be non-negative")

        self.n = n
        self.k = k
        self._cache = {}

    def __dealloc__(self) -> None:
        self._cache.clear()

    def __repr__(self) -> str:
        """
        return string representation of the object
        """
        return f"JordanTotient({self.n}, {self.k})"

    def __getitem__(self, tuple key) -> None:
        """
        custom indexing for precomputed value

        Parameter:
            key (tuple): should be (n, k)

        Return:
            (long): precomputed value if available
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("key must be tuple of (n, k)")

        n_key, k_key = key

        if (n_key, n_key) in self._cache:
            return self._cache[(n_key, k_key)]

        # create new instance to compute
        temp = JordanTotient(n_key, k_key)
        result = temp.compute()
        self._cache[(n_key, k_key)] = result
        return result

    def __setitem__(self, tuple key, long value) -> None:
        """
        manual cache value for faster future lookup

        Parameter:
            key (tuple): (n, k)
            value (long): precomputed value to store
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("key must be tuple of (n, k)")
        self._cache[key] = value

    cpdef long compute(self):
        """
        this method implement jordan totient function
        - for small n, using trial division and product over distinct primes
        - for k = 0 return 0
        - general case, compute n^k and applies the formula

        Return:
            (long): value of jordan totient
        """
        if self.k == 0:
            return 0
        if self.n == 1:
            return 1

        cdef long res = <long>(pow(self.n, self.k))
        cdef list primes = prime_factors(self.n)

        cdef int p
        cdef long pk, numerator, denominator

        for p in primes:
            pk = <long>pow(p, self.k)
            numerator = pk - 1
            denominator = pk
            res = res // denominator * numerator
        return res

cpdef int mobius(int n):
    """
    compute mobius function for given positive integer

    this implementing using sieve method to compute mu,
    based on the smallest prime factor sieve technique

    Parameter:
        n (int): positive integer >= 1

    Return:
        (int): value of mobius function

    Example:
    >>> mobius(6)
    1
    """
    if n < 1:
        raise ValueError("input must be at least 1")

    cdef int* mu = <int*>malloc((n + 1) * sizeof(int))
    cdef int* min_prime_factor = <int*>malloc((n + 1) * sizeof(int))

    if not mu or not min_prime_factor:
        free(mu)
        free(min_prime_factor)
        raise MemoryError()

    cdef int i, j, p

    for i in range(n + 1):
        min_prime_factor[i] = 0

    # base case
    mu[1] = 1
    # sieve of erathoneses-style looping for fill smallest
    # prime factors
    for i in range(2, n + 1):
        if min_prime_factor[i] == 0:
            # i is prime number
            min_prime_factor[i] = i  # marking i as own smallest prime factor
            j = i
            # mark all multiple of i that are divisible by i
            while j <= n:
                for k in range(j, n + 1, j):
                    if min_prime_factor[k] == 0:
                        min_prime_factor[k] = i
                j *= i
        # get smallest prime factor of current i
        p = min_prime_factor[i]
        # quotient after divide i by p
        j = i // p
        if min_prime_factor[j] == p:
            mu[i] = 0
        else:
            mu[i] = -mu[j]
    result = mu[n]
    free(mu)
    free(min_prime_factor)
    return result

cdef class Quartic:
    """
    compute quartic polynomial function

    formula quartic function:
    f(x) = ax⁴ + bx³ + cx² + dx + e

    use horner method for fast and numerical and stable eval

    Example:
    >>> quar = Quartic(1.0, 2.0, 3.0, 4.0, 5.0)
    >>> quar.compute(2)
    57.0
    """
    cdef double a, b, c, d, e

    def __init__(self, double a, double b, double c, double d, double e):
        """
        initialize quartic function with given coeffs

        Parameter:
            a (double): coefficient of x^4
            b (double): coefficient of x^3
            c (double): coefficient of x^2
            d (double): coefficient of x^
            e (double): constant term
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    cpdef double compute(self, double x):
        """
        compute quartic polynomial at given value of x using
        horner method

        Parameter:
            x (double): input value at which to compute polynomial

        Return:
            (double): result of compute f(x)

        Example:
        >>> quar = Quartic(1, 2, 3, 4, 5)
        >>> quar.compute(2)
        57.0
        """
        return (
            (
                (((self.a * x + self.b) * x + self.c) * x + self.d) * x
            ) + self.e
        )

    def __call__(self, double x):
        """
        make quartic instance callable like a function

        Parameter:
            x (double): input value

        Return:
            (double): result of f(x)
        """
        return self.compute(x)

    def coefficient(self):
        """
        compute current coefficient in order (a, b, c, d, e)

        using for debug or rinitializing another function with
        same parameters

        Return:
            tuple (double): (a, b, c, d, e)
        """
        return (self.a, self.b, self.c, self.d, self.e)

    def __repr__(self):
        return f"Quartic(a={self.a}, b={self.b}, c={self.c}, d={self.d}, e={self.e})"
