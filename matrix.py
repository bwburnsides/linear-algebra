from __future__ import annotations
from collections.abc import Sequence, Generator, Iterator
from collections import namedtuple
from typing import (
    Literal,
    Any,
    Union,
    overload,
    Protocol,
    TypeVar,
    Type,
    Optional,
)
import itertools as it
from fractions import Fraction

M = TypeVar("M", bound="Matrix")

init_args = namedtuple("init_args", ["strict", "fill", "fill_from"])
matrix_index = namedtuple("matrix_index", ["i", "j"])
matrix_size = namedtuple("matrix_size", ["m", "n"])


class SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


def _leading_idx(seq: list[float]) -> SupportsLessThan:
    for i, val in enumerate(seq):
        if val != 0:
            return i
    return len(seq) - 1


entry_types = (int, float, Fraction)
fill_from_vals = {"front", "back"}


class Matrix:
    """Represent Regular (Non-Ragged) Matrices of numeric types, and perform standard
    matrix operations on them. Supports most relevant Python protocols, making it easy
    to use in standardized applications, such as hash, len, subscripting, iteration and
    more. Implements support for standard matrix arithmetic operations, row operations,
    and other matrix quantities. Meant to be used as a loosely immutable type with
    1-based indexing.

    Args:
    - *rows (Sequence[float]) - any number of rows to build matrix from.
    - strict (bool) - optional, a strict matrix's *rows input MUST be non-ragged. For
    non-strict Matrix, :fill: and :fill_from: are used to produce the missing values.
    Default is strict = True.
    - fill (float) - optional, the value to enter into missing spaces for non-strict
    Matrix. Default is fill = 0.
    - fill_from (str) - optional, whether to fill from "front" (prepend) or "back"
    (append) for non-strict Matrix. Default is "back".

    Raises:
    ValueError - if *rows is len 0. (No rows present)
    ValueError - if row[j] max len is 0, j from 1 to n. (No cols present)
    ValueError - If any row contain non-numeric value.
    TypeError - If any row is non-Sequence.
    ValueError - If strict Matrix is constructed with ragged *rows.
    TypeError - If non strict Matrix is constructed with non-numeric :fill:.
    ValueError - If non strict Matrix is constructed with non
    {"front", "back"} :fill_from:."""

    def __init__(
        self: M,
        *rows: Sequence[float],
        strict: bool = True,
        fill: float = 0,
        fill_from: Union[Literal["front"], Literal["back"]] = "back",
    ):
        max_col = self._ensure_row_types(rows)
        self._rows = [list(row) for row in rows]
        self._size = matrix_size(len(rows), max_col)
        self._args = init_args(strict, fill, fill_from)
        self._normalize_dims()

    @staticmethod
    def _ensure_row_types(rows) -> int:
        """Helper method to check the row and element types and compute the max column
        count. Returns max col count (int).
        Raises:
        ValueError - if Matrix doesn't have at least 1 row and 1 column.
        TypeError - if rows are Sequence or elements aren't numeric."""
        # Ensure at least 1 row
        if not len(rows):
            raise ValueError("Matrix must have at least 1 row.")

        # Ensure all Sequences
        max_col = 0
        for i, row in enumerate(rows):
            if not isinstance(row, Sequence):
                raise TypeError(f"All rows must be Sequence: {i} = {row}.")

            # Keep track of max col count
            max_col = max(max_col, len(row))

            # Ensure all numeric elements
            for j, val in enumerate(row):
                if not isinstance(val, entry_types):
                    raise ValueError(
                        f"All vals must be of the "
                        f"following types: {(t.__name__ for t in entry_types)}. "
                        f"Got: {val} at {i}, {j} with type {type(val).__name__}."
                    )

        # Ensure at least 1 col
        if not max_col:
            raise ValueError("Matrix must have at least 1 column.")
        return max_col

    def _normalize_dims(self) -> None:
        """Helper method for normalizing Matrix's dimensions. For strict matrices,
        raises ValeError for ragged inputs. For non-strict, uses :fill_from: to apply
        :fill: to each row."""
        if self._args.strict:
            if not all(len(row) == self.size.n for row in self._rows):
                raise ValueError("Strict Matrix rows must have equal lengths.")
        else:
            if not isinstance(self._args.fill, entry_types):
                raise TypeError(
                    f"Matrix row fill value must be of the following types: "
                    f"{entry_types}. Got {self._args.fill} with type "
                    f"{type(self._args.fill).__name__}."
                )

            if self._args.fill_from not in {"back", "front"}:
                raise ValueError(
                    f"Matrix row fill from mode must be in {fill_from_vals}. "
                    f"Got '{self._args.fill_from}'"
                )

            self._pad_rows()

    def _pad_rows(self) -> None:
        """Helper method for prepending or appending self's fill value to each row,
        as needed."""
        for i, row in enumerate(self._rows):
            extra = [self._args.fill] * (self.size.n - len(row))
            if self._args.fill_from == "back":
                row.extend(extra)
            else:
                extra.extend(row)
                self._rows[i] = extra

    @classmethod
    def zeros(cls: Type[M], m: int, n: Optional[int] = None) -> M:
        """Create an m x n Matrix filled with zeros.
        m (int) - number of rows
        n (int) - optional, number of cols. If not given, the matrix is assumed
        square with m cols.
        """
        n = m if n is None else n
        return cls(*tuple(it.repeat(tuple(it.repeat(0, n)), m)))

    @classmethod
    def identity(cls: Type[M], n: int) -> M:
        """Create an n x n Identity Matrix
        (has 1s on the main diagonal, zero elsewhere).
        n (int) - number of rows and cols"""
        return cls(
            *tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
        )

    @classmethod
    def id(cls: Type[M], n: int) -> M:
        """Create an n x n Identity Matrix
        (has 1s on the main diagonal, zero elsewhere).
        n (int) - number of rows and cols"""
        return cls.identity(n)

    def __len__(self) -> int:
        """len(self)"""
        return len(self._rows)

    @property
    def size(self) -> matrix_size:
        """The size of the Matrix, in (m x n) form."""
        return self._size

    @property
    def shape(self) -> matrix_size:
        """The size of the Matrix in (m x n) form. (alias of self.size)"""
        return self.size

    @property
    def is_square(self) -> bool:
        """Whether the Matrix is square. Square matrices have the same
        number of rows and columns."""
        return self.size.m == self.size.n

    @property
    def is_row(self) -> bool:
        """Whether the Matrix is a row matrix. Row matrices have 1 row
        and n columns."""
        return self.size.m == 1

    @property
    def is_column(self) -> bool:
        """Whether the Matrix is a column matrix. Column matrices have m rows
        and 1 column."""
        return self.size.n == 1

    def __repr__(self) -> str:
        """repr(self)"""
        args_repr = ", ".join(repr(row) for row in self._rows)

        if not self._args.strict:
            args_repr = ", ".join((args_repr, "strict=False"))

            if (f := self._args.fill) != 0:
                args_repr = ", ".join((args_repr, f"fill={f}"))

            if (f := self._args.fill_from) != "back":
                args_repr = ", ".join((args_repr, f'fill_from="{f}"'))

        return f"Matrix({args_repr})"

    # TODO: make output have aligned columns for prettier printing
    def __str__(self) -> str:
        """str(self)"""
        mstr = ""
        for i, row in enumerate(self._rows):
            rstr = f"[{', '.join(str(val) for val in row)}]"
            if not i:
                mstr += f"Matrix({rstr}"
            else:
                mstr += f"\n       {rstr}"
        return mstr + ")"

    def __contains__(self, val: Any) -> bool:
        """val in self"""
        if not isinstance(val, entry_types):
            return False
        return any(val in row for row in self)

    def _key(self: M) -> tuple[Type[M], tuple[tuple[float, ...], ...]]:
        """A tuple-key for use during Matrix hashing.
        Returns:
        key tuple - (type(M), tuple[tuple[float, ...], ...]) where the tuples
        represent the underlying private self._rows member."""
        return (type(self), tuple(tuple(row) for row in self._rows))

    def __hash__(self) -> int:
        """hash(self)"""
        return hash(self._key())

    def __iter__(self) -> Generator[list[float], None, None]:
        """iter(self) - yield the rows, from i = 1 to i = m."""
        for row in self._rows:
            yield row

    def __reversed__(self) -> Generator[list[float], None, None]:
        """reversed(self) - yield the rows, from i = m to i = 1. The row elements
        remain in forward order, from j = 1 to j = n."""
        yield from reversed(self._rows)

    def cells(self) -> Generator[float, None, None]:
        """Return an iterator over the values in the Matrix. Starting with i = 1, yield
        the elements from j = 1 to j = n, then progreses to the next row until i = m."""
        for row in self:
            yield from row

    def ij_cell(self) -> Generator[tuple[matrix_index, float], None, None]:
        """Return an iterator over the values in the Matrix, including its index.
        Starting with i = i, yield (i, j) and the element for j = 1 to j = n. Continue
        towards i = m."""
        for i, row in enumerate(self, start=1):
            for j, val in enumerate(row, start=1):
                yield matrix_index(i, j), val

    def idx_cell(self) -> Generator[tuple[int, float], None, None]:
        """Return an iterator over the values in the Matrix, including its overall
        index within. Beginning with i = 0, yield idx and the element for j = 1
        to j = n. Continue towards i = m. idx is the total offset from the beginning
        of the Matrix. idx = ((i - 1) * n) + j"""
        for idx, val in enumerate(self.cells(), start=1):
            yield idx, val

    @overload
    def __getitem__(self, key: tuple[int, int]) -> float:
        ...

    @overload
    def __getitem__(self, key: int) -> tuple[float, ...]:
        ...

    def __getitem__(
        self, key: Union[tuple[int, int], int]
    ) -> Union[float, tuple[float, ...]]:
        """Get elements from the Matrix corresponding to the key. If k is an int, then
        return the corresponding row. If k is an int pair of the form [i, j], then
        return the jth element of the ith row.

        Args
        key (int | tuple[int, int]) - row or element to get.

        Raises
        ValueError - if the key is a tuple and is not length 2.
        TypeError - if the key is not a 2-tuple or an int.
        """
        if isinstance(key, tuple):
            if (l := len(key)) != 2:
                raise ValueError(f"Matrix tuple subscript must be length 2, got {l}.")
            return self._rows[key[0] - 1][key[1] - 1]

        if isinstance(key, int):
            return tuple(self._rows[key - 1])

        raise TypeError(
            f"Matrix subscript must be tuple[int, int] or int, "
            f"got {type(key).__name__}."
        )

    def index(
        self,
        val: Any,
        start: tuple[int, int] = (1, 1),
        end: Optional[tuple[int, int]] = None,
    ) -> matrix_index:
        """Return the (i, j) index of the first element in Matrix to match val.
        start and end represent the starting and ending rows and columns to search
        through.

        Args
        val (Any) - the val to look for in self
        start (tuple[int, int]) - the (starting row, starting column) to look from.
        Optional, default is first row and first col.
        end (tuple[int, int]) - the (ending row, ending column) to look to. Optional,
        the default is final row and final col.

        Raises
        ValueError - if the val is not found in self
        """
        if not isinstance(val, entry_types):
            raise ValueError(
                f"The val, {val}, not found in Matrix with {start = }, {end = }.\n"
                "Matrix only contains numeric entries."
            )

        end = (-1, -1) if end is None else end
        for i, row in enumerate(self._rows[slice(start[0] - 1, end[0])]):
            for j, cell in enumerate(row[slice(start[1] - 1, end[1])]):
                if cell == val:
                    return matrix_index(i + start[0], j + start[1])
        raise ValueError(
            f"The val, {val}, not found in Matrix with {start = }, {end = }."
        )

    def count(self, val: Any) -> int:
        """Return the number of occurces of val in self.

        Args
        val (Any) - the val to count occurrences of in self
        """
        if not isinstance(val, entry_types):
            return 0

        ct = 0
        for cell in self.cells():
            if cell == val:
                ct += 1
        return ct

    def copy(self: M) -> M:
        """Return a shallow copy of self. The rows are copied, but the entries themselves
        are not. This is fine as the Matrix should only contain immutable (numeric)
        datatypes.
        """
        return self.__class__(*tuple(row.copy() for row in self._rows))

    def __eq__(self, other: Any) -> bool:
        """self == other"""
        # A Matrix and another object can only be equal is other is also a Matrix.
        if not isinstance(other, type(self)):
            return False

        # If they're both Matrices, then they are equal if these hashes are the same.
        # Need to do the type check to ensure that a tuple matching self._key does not
        # give an equality false positive.
        return hash(other) == hash(self)

    def __mul__(self: M, other: Any) -> M:
        """self * other (scalar multiplication)"""
        if not isinstance(other, (float, int)):
            return NotImplemented

        # Multiply scalar with each Matrix element, return new Matrix
        return self.__class__(*tuple(tuple(val * other for val in row) for row in self))

    def __rmul__(self: M, other: Any) -> M:
        """other * self (scalar multiplication)"""
        return self * other

    def __add__(self: M, other: Any) -> M:
        """self + other (matrix addition)"""
        if not isinstance(other, Matrix):
            return NotImplemented

        if (s := self.size) != (o := other.size):
            raise ValueError(
                "Matrix addition is only defined for matrices of equal shape. "
                f"Got {s} and {o}."
            )

        # Perform elementwise addition, return new Matrix
        it_self: Iterator = iter(self)
        it_other: Iterator = iter(other)
        return self.__class__(
            *tuple(
                tuple(v1 + v2 for v1, v2 in zip(r1, r2))
                for r1, r2 in zip(it_self, it_other)
            )
        )

    def __radd__(self: M, other: Any) -> M:
        """other + self (matrix addition)"""
        return self + other

    def __neg__(self: M) -> M:
        return -1 * self

    def __sub__(self: M, other: Any) -> M:
        """self - other (matrix subtraction)"""
        return -other + self

    def __rsub__(self: M, other: Any) -> M:
        """other - self (matrix subtraction)"""
        return -self + other

    def __matmul__(self: M, other: Any) -> M:
        """self @ other (matrix multiplication)"""
        if not isinstance(other, Matrix):
            return NotImplemented

        # Matrix multiplication is only defined for matrices of shape (m, n) and (n, p),
        # where the result is (m, p) in shape.
        m, n1 = self.size
        n2, p = other.size

        if n1 != n2:
            raise ValueError(
                "Matrix Multiplication defined for matrices with m x n and n x p "
                f"dimensions. Got {n1} and {n2}"
            )
        n = n1

        # Each element in the matrix product AB is the dot product of the
        # corresponding row in A and the corresponding column in B.
        result: list[list[float]] = [[0 for _ in range(p)] for _ in range(m)]
        for i, j in it.product(range(m), range(p)):
            result[i][j] = sum(
                self[i + 1, k + 1] * other[k + 1, j + 1] for k in range(n)
            )

        return self.__class__(*result)

    def __rmatmul__(self: M, other: Any) -> M:
        """other @ self (matrix multiplication)"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return other @ self

    def transpose(self: M) -> M:
        """Return a transposed copy of self such that its columns become its rows,
        and its rows become its columns."""
        return self.__class__(*tuple(zip(*self)))

    def trans(self: M) -> M:
        """Return a transposed copy of self such that its columns become its rows,
        and its rows become its columns. (alias of self.transpose)"""
        return self.transpose()

    def row_swap(self: M, p: int, q: int) -> M:
        """Return a new Matrix with rows p and q of self swapped.

        Row opeation notation:
        p <-> q
        """
        return self.copy()._inplace_row_swap(p, q)

    def _inplace_row_swap(self: M, p: int, q: int) -> M:
        """Perform a row swap on self. Marked private as Matrix should be
        publically immutable."""
        self._rows[p - 1], self._rows[q - 1] = self._rows[q - 1], self._rows[p - 1]
        self._remove_fractions()
        return self

    def row_mult(self: M, p: int, k: Union[float, Fraction]) -> M:
        """Return a new Matrix with elements in row p multiplied by k.

        Row operation notation:
        p = k * p
        """
        return self.copy()._inplace_row_mult(p, k)

    def _inplace_row_mult(self: M, p: int, k: Union[float, Fraction]) -> M:
        """Perform a row multiplication on self. Marked private as Matrix should be
        publically immutable."""
        self._rows[p - 1] = [k * val for val in self._rows[p - 1]]
        self._remove_fractions()
        return self

    def row_add(self: M, p: int, q: int) -> M:
        """Return a new Matrix with elements in row q added to corresponding elements
        in row p.

        Row operation notation:
        p = p + q
        """
        return self.copy()._inplace_row_add(p, q)

    def _inplace_row_add(self: M, p: int, q: int) -> M:
        """Perform a row addition on self. Marked private as Matrix should be publically
        immutable."""
        self._rows[p - 1] = [
            x + y for x, y in zip(self._rows[p - 1], self._rows[q - 1])
        ]
        self._remove_fractions()
        return self

    def row_mult_add(self: M, p: int, k: Union[float, Fraction], q: int) -> M:
        """Return a new Matrix where a multiply-and-add operation has been performed.

        Row operation notation:
        q = kp + q
        """
        return self.copy()._inplace_row_mult_add(p, k, q)

    def _inplace_row_mult_add(self: M, p: int, k: Union[float, Fraction], q: int) -> M:
        """Perform a row addition and multiplication on self. Marked private as Matrix
        should be publically immutable."""
        old_row = self._rows[p - 1].copy()
        self._inplace_row_mult(p, k)
        self._inplace_row_add(q, p)
        self._rows[p - 1] = old_row
        self._remove_fractions()
        return self

    def _leading_coeff(self) -> tuple[float, ...]:
        """Return tuple of the leading coefficients for each row in Matrix."""
        return tuple(row[idx] for row, idx in zip(self, self._leading_coeff_idx()))

    def _leading_coeff_idx(self) -> tuple[int, ...]:
        """Return tuple of the indices of the leading coefficients for each row in
        Matrix. As this is a private method, the returned indices are zero-based,
        unlike most other public methods. Use with care."""
        idxs = []
        for row in self:
            for j, val in enumerate(row):
                if val != 0:
                    idxs.append(j)
                    break
            else:
                idxs.append(self.size.n - 1)
        return tuple(idxs)

    def is_ref(self) -> bool:
        """Whether self is in Row Echelon Form."""

        # A Matrix is said to be in Row Echelon Form if it meets the following
        # requirements:
        # 1. All rows consisting of only zeroes are at the bottom.
        # 2. The leading coefficient of a nonzero row is always strictly to the right
        # of the leading coefficient of the row above it.
        #
        # Some texts add the condition that the leading coefficient must be 1. This
        # function does not use that stipulation.

        non_zero_idx = [i for i, row in enumerate(self) if any(val != 0 for val in row)]
        all_zero_idx = [i for i, row in enumerate(self) if all(val == 0 for val in row)]

        # The lowest occuring row that is not all-zeros. In order to meet requirement
        # 1, all_zero_idx indices must all be greater than this value, implying that
        # they are further down in the matrix.
        last_non_zero_idx = max(non_zero_idx)

        if not all(idx > last_non_zero_idx for idx in all_zero_idx):
            return False

        coeff_idx = self._leading_coeff_idx()
        for i, idx in enumerate(coeff_idx):
            if i == 0:  # Skip the first row as there is no leading coeff above it
                continue

            # Get the prev leading coeff position. It's value MUST be less than the
            # current, otherwise condition 2 cannot be met.
            prev_idx = coeff_idx[i - 1]
            if not (prev_idx < idx):
                return False

        return True

    def _reverse_rows(self: M) -> M:
        """Reverse the rows of self in place, returning self. Marked private as it
        alters the internal state of the 'immutable' Matrix. Use with care."""
        self._rows = list(reversed(self._rows))
        return self

    def ref(self: M, display: bool = False) -> M:
        """Produce a Row Echelon-equivalent of self."""
        rv = self.copy()
        if rv.is_ref():
            return rv

        # First, sort the rows by leading coefficient position
        rv._rows.sort(key=_leading_idx)
        if rv != self:
            if display:
                print("Swapped rows. Before:\n", self)
                print("After:\n", rv)

        # Starting with row i, eliminate all non-zero values in column i below row i.
        # Repeat for row i to m. The result is that the lower triangle of the Matrix
        # becomes zeros.
        it_rv1: Iterator = iter(rv)
        for i, eqn in enumerate(it_rv1, start=1):

            it_rv2: Iterator = iter(rv)
            for j, row in enumerate(it_rv2, start=1):
                if j <= i or row[i - 1] == 0:
                    continue

                # If the values are integral, then we can use a Fraction() to help
                # avoid floating point errors.
                num_den = [-row[i - 1], eqn[i - 1]]
                frac = (float(val) if isinstance(val, int) else val for val in num_den)

                if all(isinstance(val, float) and val.is_integer() for val in frac):
                    k = Fraction(num_den[0], num_den[1])
                else:
                    k = num_den[0] / num_den[1]

                rv._inplace_row_mult_add(i, k, j)

                if display:
                    print(f"R{j} + {k}R{i} --> R{j}")
                    print(rv, end="\n\n")

        if not rv.is_ref():
            raise RuntimeError("Failed to compute Row Echelon Form")

        return rv

    def _remove_fractions(self: M) -> M:
        """Convert all values in the Matrix to an int or float as appropriate."""
        for i, row in enumerate(self._rows):
            for j, el in enumerate(row):
                int_ratio = el.as_integer_ratio()

                if abs(int_ratio[1]) == 1:
                    self._rows[i][j] = int(el)
                else:
                    self._rows[i][j] = float(el)

        return self

    def rref(self: M, display: bool = False) -> M:
        """ Return a Reduced Row Echelon-equivalent of self."""
        rv = self.ref(display)
        rv._reverse_rows()

        leading = list(rv._leading_coeff())
        leading_idx = list(rv._leading_coeff_idx())

        it_rv: Iterator = iter(rv)
        for i, row in enumerate(it_rv, start=1):
            if all(val == 0 for val in row):
                continue

            lead = leading[i - 1]
            k = 1 / lead
            rv._inplace_row_mult(i, k)

            if display:
                print(f"{k}R{i} --> R{i}")
                print(rv, end="\n\n")

        it_rv1: Iterator = iter(rv)
        for i, eqn in enumerate(it_rv1, start=1):
            lead = leading[i - 1]
            lead_idx = leading_idx[i - 1]

            it_rv2: Iterator = iter(rv)
            for j, row in enumerate(it_rv2, start=1):
                if j <= i:
                    continue

                val_to_cancel = row[lead_idx]
                val_to_add = -val_to_cancel

                dec = val_to_add
                rv._inplace_row_mult_add(i, dec, j)

                if display:
                    print(f"R{j} + {k}R{i} --> R{j}")
                    print(rv, end="\n\n")

        rv._reverse_rows()
        if display:
            print(rv, end="\n\n")
        return rv

    def submatrix(self: M, i: int, j: int) -> M:
        """Return the submatrix constructed by removing row i and col j from self."""
        it_self: Iterator = iter(self)
        return self.__class__(
            *tuple(
                tuple(val for mj, val in enumerate(row, start=1) if mj != i)
                for mi, row in enumerate(it_self, start=1)
                if mi != j
            )
        )

    def diagonal(self) -> tuple[float, ...]:
        """Return a tuple corresponding to the values on self's main diagonal."""
        return tuple(val for (i, j), val in self.ij_cell() if i == j)

    def diag(self) -> tuple[float, ...]:
        """Return a tuple corresponding to the values on self's main diagonal.
        (alias of self.diagonal)"""
        return self.diagonal()

    def trace(self) -> float:
        """Return the trace of self, which is the sum of the values on its main
        diagonal."""
        return sum(self.diag())

    def tr(self) -> float:
        """Return the trace of self, which is the sum of the values on its main
        diagonal. (alias of self.trace)"""
        return self.trace()

    def minor(self, i: int, j: int) -> float:
        """Return self's i, j minor, defined as the determinant of the submatrix
        constructed by removing the ith row and jth column from self."""
        if not self.is_square:
            raise ValueError("The minor is only defined for square matrices.")

        return self.submatrix(i, j).det()

    def determinant(self, i: int = 1) -> float:
        """Return the determinant of this Matrix."""
        if not self.is_square:
            raise ValueError("The determinant is only defined for square matrices.")

        if self.size.n == 1:
            return self[1, 1]
        else:
            return sum(
                self.cofactor(i, j) * aij for j, aij in enumerate(self[i], start=1)
            )

    def det(self, i: int = 1) -> float:
        """Return the determinant of this Matrix. (alias of self.determinant)"""
        return self.determinant(i)

    def is_invertible(self) -> bool:
        """Return whether self is invertible, defined as whether A has a matrix B such
        that AB = BA = I."""
        try:
            return bool(self.det())
        except ValueError:
            return False

    def is_orthogonal(self) -> bool:
        """Return whether self is invertible, defined as whether A satisfies:
        A^T = A^-1."""
        if not self.is_invertible():
            return False

        return self.transpose() == self.inv()

    def is_symmetric(self) -> bool:
        """Return whether self is symmetric, defined as whether A satisfies A = A^T."""
        return self == self.transpose()

    def is_diagonal(self) -> bool:
        """Return whether self is a diagonal Matrix, which is a propery defined as having
        having zero for all elements not on the main-diagonal."""
        return any(
            any(not val for j, val in enumerate(row) if i != j)
            for i, row in enumerate(self)
        )

    def cofactor(self, i: int, j: int) -> float:
        """Compute the cofactor of self[i, j]. Defined as:
        -1^(i + j) * Mij."""
        return ((-1) ** (i + j)) * self.minor(i, j)

    def comatrix(self: M) -> M:
        """Compute the comatrix of self, defined as the matrix consisting of self's
        cofactors."""
        it_self: Iterator = iter(self)
        return self.__class__(
            *tuple(
                tuple(self.cofactor(ci, cj) for cj, val in enumerate(row, start=1))
                for ci, row in enumerate(it_self, start=1)
            )
        )

    def adjugate(self: M) -> M:
        """Compute the adjugate matrix of self, defined as self's transposed comatrix.
        adj(A) = (C_A)^T"""
        return self.comatrix().transpose()

    def adj(self: M) -> M:
        """Compute the adjugate matrix of self, defined as self's transposed comatrix.
        adj(A) = (C_A)^T. (alias of self.adjugate)"""
        return self.adjugate()

    def inverse(self: M) -> M:
        """Compute the inverse of self."""
        if not self.is_invertible():
            raise ValueError("Matrix is Singular and has no Inverse.")

        return (1 / self.det()) * self.adjugate()

    def inv(self: M) -> M:
        """Compute the inverse of self. (alias of self.inverse)"""
        return self.inverse()

    def augment(self: M, vec: Sequence[float], j=None) -> M:
        """Augment a sequence of floats to a new Matrix.

        Args:
        j (int) - Optional, specify which column to insert vec into to. Default
        is the last column
        """
        raise NotImplementedError

    def aug(self: M, vec: Sequence[float], j=None) -> M:
        """Augment a sequence of floats to a new Matrix.

        Args:
        j (int) - Optional, specify which column to insert vec into to. Default
        is the last column
        """
        return self.augment(vec, j)
