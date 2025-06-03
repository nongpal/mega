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
# mega/op/tensor.pyx

from libc.stdlib cimport malloc, free

cdef class Tensor:
    """
    class of implement tensor

    Attributes:
        shape (int*): pointer to an array storing the dimension of tensor
        ndim (int): number of dimension (rank) of the tensor
        size (int): total number of elements in tensor
        _dtype (str): data type of the tensor elements
        int_data (list[int]): python list used when dtype is int
        long_data (long*): pointer to memory block for `long` type data
        double_data (double*): pointer memory block for `double` type data
        float_data (float*): pointer to memory block for `float` type data

    Example:
    >>> tensor1 = Tensor((2, 3), [1, 2, 3, 4, 5, 6], dtype='long')
    >>> tensor2 = Tensor((2, 3), [10, 20, 30, 40, 50, 60], dtype='long')
    >>> tensor_add = tensor1.add(tensor2)
    >>> print(tensor_add)
    Tensor(shape=(2, 3), dtype=long, data=[11, 22, 33, ...])
    """
    cdef int *shape
    cdef int ndim
    cdef int size
    cdef str _dtype
    cdef object int_data
    cdef long *long_data
    cdef double *double_data
    cdef float *float_data

    def __cinit__(self, tuple shape, list data=None, str dtype="long"):
        """
        initialize tensor with given shape, optional data, and specified data type

        Parameter:
            shape (tuple or int): dimension of the tensor, each dimension must be > 0
            data (list, optional): initial data to populate tensor
            dtype (str, optional): data type of the tensor, supported type
                                    long, int, float, double, default was long
        """
        self._dtype = dtype.lower()
        if self._dtype not in ("int", "long", "double", "float"):
            raise ValueError(f"unsupported dtype: {dtype}")

        self.ndim = len(shape)
        self.shape = <int*>malloc(self.ndim * sizeof(int))
        if not self.shape:
            raise MemoryError("failed to allocated shape array")

        cdef int i
        self.size = 1
        for i in range(self.ndim):
            dim = shape[i]
            if dim <= 0:
                raise ValueError(f"dimension {i} must be > 0")
            self.shape[i] = dim
            self.size *= dim

        if self._dtype == "int":
            self.int_data = [0] * self.size
            if data is not None:
                for i in range(min(self.size, len(data))):
                    self.int_data[i] = data[i]

        elif self._dtype == "long":
            self.long_data = <long*>malloc(self.size * sizeof(long))
            if not self.long_data:
                raise MemoryError("failed to allocate long data")
            if data is None:
                for i in range(self.size):
                    self.long_data[i] = 0
            else:
                for i in range(self.size):
                    self.long_data[i] = data[i]

        elif self._dtype == "double":
            self.double_data = <double*>malloc(self.size * sizeof(double))
            if not self.double_data:
                raise MemoryError("failed to allocate double data")
            if data is None:
                for i in range(self.size):
                    self.double_data[i] = 0.0
            else:
                for i in range(self.size):
                    self.float_data[i] = <double>data[i]

        elif self._dtype == "float":
            self.float_data = <float*>malloc(self.size * sizeof(float))
            if not self.float_data:
                raise MemoryError("failed to allocate float data")
            if data is None:
                for i in range(self.size):
                    self.float_data[i] = 0.0
            else:
                for i in range(self.size):
                    self.float_data[i] = <float>data[i]

    def __dealloc__(self):
        """
        deallocate dynamically allocated memory when tensor was destroy

        make sure no memory leaks by freeing all alocated pointers.
        called automatically when tensor object was destroyed
        """
        if self.shape is not NULL:
            free(self.shape)
            self.shape = NULL
        if self._dtype == "long" and self.long_data is not NULL:
            free(<void*>self.long_data)
            self.long_data = NULL
        elif self._dtype == "double" and self.double_data is not NULL:
            free(<void*>self.double_data)
            self.double_data = NULL
        elif self._dtype == "float" and self.float_data is not NULL:
            free(<void*>self.float_data)
            self.float_data = NULL
        if hasattr(self, "int_data") and self.int_data is not None:
            self.int_data.clear()
        self.ndim = 0
        self.size = 0

    cdef _get_value(self, int index):
        if self._dtype == "int":
            return self.int_data[index]
        elif self._dtype == "long":
            return self.long_data[index]
        elif self._dtype == "double":
            return self.double_data[index]
        elif self._dtype == "float":
            return self.float_data[index]

    def tolist(self):
        """
        convert tensor into nested python with matching dimension

        Return:
            (list): nested list representing the tensor data
        """
        result = []
        if self.ndim == 1:
            for i in range(self.size):
                result.append(self._get_value(i))
        elif self.ndim == 2:
            rows, cols = self.shape[0], self.shape[1]
            for i in range(rows):
                row = []
                for j in range(cols):
                    row.append(self._get_value(i * cols + j))
                result.append(row)
        elif self.ndim == 3:
            d1, d2, d3 = self.shape[0], self.shape[1], self.shape[2]
            for i in range(d1):
                mat = []
                for j in range(d2):
                    vec = []
                    for k in range(d3):
                        vec.append(self._get_value(i * d2 * d3 + j * d3 + k))
                    mat.append(vec)
                result.append(mat)
        else:
            raise NotImplementedError("only support up to 3d tensor")
        return result

    @classmethod
    def fromlist(cls, list data, str dtype="double") -> object:
        """
        creating tensor from nested python list
        automatically infer shape from list nesting depth

        Parameter:
            data (list): nested list of value
            dtype (str): data type

        Return:
            (Tensor): constructed tensor from list
        """
        def infer_shape(lst):
            shape = []
            current_level = lst
            while isinstance(current_level, list):
                if not current_level:
                    return shape
                shape.append(len(current_level))
                current_level = current_level[0]
            return shape

        shape = infer_shape(data)
        if not shape:
            raise ValueError("cannot infer shape from scalar")

        cdef list flat_data = []
        cdef list stack = [data]

        while stack:
            item = stack.pop()
            if isinstance(item, list):
                for x in reversed(item):
                    stack.append(x)
            else:
                flat_data.append(item)

        for x in flat_data:
            if not isinstance(x, (int, float)):
                raise TypeError(f"non-numeric value found: {type(x)}")
        return cls(tuple(shape), flat_data, dtype=dtype)

    cpdef str dtype(self):
        """
        return data type of the tensor

        Return:
            str: data type string (int, long, float, or double)
        """
        return self._dtype

    def __getitem__(self, object idxs):
        """
        access element of tensor using multi-dimensional indices

        Parameter:
            idxs (tuple or int): indices along each axis

        Return:
            (int or float or double): value at the specified index
        """
        cdef tuple idx_tuple
        if isinstance(idxs, int):
            idx_tuple = (idxs,)
        elif isinstance(idxs, tuple):
            idx_tuple = idxs
        else:
            raise TypeError("Indices must be an int or a tuple of ints")

        if len(idx_tuple) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(idx_tuple)}")

        cdef long offset = 0
        cdef int i

        for i in range(self.ndim):
            pos = idx_tuple[i]
            if pos < 0 or pos >= self.shape[i]:
                raise IndexError(f"Index out of bounds at axis {i}: {pos}")
            offset += pos * self._compute_stride(i)

        if self._dtype == "int":
            return self.int_data[offset]
        elif self._dtype == "long":
            return self.long_data[offset]
        elif self._dtype == "double":
            return self.double_data[offset]
        elif self._dtype == "float":
            return self.float_data[offset]

    def __setitem__(self, object idxs, long value):
        """
        set value at specific multi-dimensional index

        Parameter:
            idxs (tuple of int): indices along each axis
            value (int or float): value to assign
        """
        cdef tuple idx_tuple
        if isinstance(idxs, int):
            idx_tuple = (idxs,)
        elif isinstance(idxs, tuple):
            idx_tuple = idxs
        else:
            raise TypeError("Indices must be an int or a tuple of ints")

        if len(idx_tuple) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(idx_tuple)}")

        cdef long offset = 0
        cdef int i

        for i in range(self.ndim):
            pos = idx_tuple[i]
            if pos < 0 or pos >= self.shape[i]:
                raise IndexError(f"Index out of bounds at axis {i}: {pos}")
            offset += pos * self._compute_stride(i)

        if self._dtype == "int":
            self.int_data[offset] = value
        elif self._dtype == "long":
            self.long_data[offset] = value
        elif self._dtype == "double":
            self.double_data[offset] = value
        elif self._dtype == "float":
            self.float_data[offset] = value

    cdef long _compute_stride(self, int axis):
        """
        compute the stride for given axis

        stride representing how many elements to skip in memory to move one step

        Parameter:
            axis (int): axis for which to compute the stride

        Return:
            (long): stride value for the given axis
        """
        cdef long stride = 1
        cdef int i
        for i in range(self.ndim - 1, axis, -1):
            stride *= self.shape[i]
        return stride

    cpdef Tensor add(self, Tensor other):
        """
        perform element-wise add between two tensor of same shape

        Return:
            (Tensor): new tensor after element-wise add
        """
        if self.size != other.size:
            raise ValueError("tensor must have the same number of elements")

        if self._dtype != other._dtype:
            raise TypeError("data type must match for add operation")

        cdef Tensor result = Tensor(
            tuple([self.shape[i] for i in range(self.ndim)]), dtype=self._dtype
        )
        cdef int i

        if self._dtype == "int":
            for i in range(self.size):
                result.int_data[i] = self.int_data[i] + other.int_data[i]
        elif self._dtype == "long":
            for i in range(self.size):
                result.long_data[i] = self.long_data[i] + other.long_data[i]
        elif self._dtype == "double":
            for i in range(self.size):
                result.double_data[i] = self.double_data[i] + other.double_data[i]
        elif self._dtype == "float":
            for i in range(self.size):
                result.float_data[i] = self.float_data[i] + other.float_data[i]
        return result

    cpdef Tensor multiply(self, Tensor other):
        """
        perform element-wise multiply two tensor of same shape

        Return:
            (Tensor): new Tensor after element-wise multiply
        """
        if self.size != other.size:
            raise ValueError("tensor must have same number oof elements")

        if self._dtype != other._dtype:
            raise TypeError("data type must match for multiply operation")

        cdef Tensor result = Tensor(
            tuple([self.shape[i] for i in range(self.ndim)]),
            dtype=self._dtype
        )
        cdef int i

        if self._dtype == "int":
            for i in range(self.size):
                result.int_data[i] = self.int_data[i] * other.int_data[i]
        elif self._dtype == "long":
            for i in range(self.size):
                result.long_data[i] = self.long_data[i] * other.long_data[i]
        elif self._dtype == "double":
            for i in range(self.size):
                result.double_data[i] = self.double_data[i] * other.double_data[i]
        elif self._dtype == "float":
            for i in range(self.size):
                result.float_data[i] = self.float_data[i] * other.float_data[i]

        return result

    def __repr__(self):
        """
        generate string representation of the tensor for debug

        Return:
            str: human-readable representation of tensor
        """
        shape_list = [str(self.shape[i]) for i in range(self.ndim)]
        shape_str = ", ".join(shape_list)

        data_preview = []
        if self._dtype == "int":
            for i in range(min(5, self.size)):
                data_preview.append(str(self.int_data[i]))
        elif self._dtype == "long":
            for i in range(min(5, self.size)):
                data_preview.append(str(self.long_data[i]))
        elif self._dtype == "double":
            for i in range(min(5, self.size)):
                data_preview.append(f"{self.double_data[i]:.4f}")
        elif self._dtype == "float":
            for i in range(min(5, self.size)):
                data_preview.append(f"{self.float_data[i]:.4f}")

        data_str = ", ".join(data_preview)
        if self.size > 50:
            data_str += ", ..."

        return f"Tensor(shape=({shape_str}), dtype={self._dtype}, data=[{data_str}])"

    def __str__(self):
        return self.__repr__()
