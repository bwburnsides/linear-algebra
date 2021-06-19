from contextlib import contextmanager
import pytest
from matrix import Matrix as M
import math


@contextmanager
def doesnt_raise():
    yield


def test_init_empty():
    """Verify that matrices with 0 rows cannot be constructed."""
    with pytest.raises(ValueError):
        M()


def test_init_nocols():
    """Verify that matrices with 0 cols cannot be constructed"""
    with pytest.raises(ValueError):
        M([])

    with pytest.raises(ValueError):
        M([], [])


def test_init_nonsequence_row():
    with pytest.raises(TypeError):
        M(1.2)

    with pytest.raises(TypeError):
        M([0, 1], 1.2)


def test_init_nonnumber_val():
    with pytest.raises(ValueError):
        M([1, 2, 3], ["a", "b", "c"])


@pytest.mark.parametrize(
    "inp",
    [
        ([1, 2], [3, 4]),
    ],
)
def test_init(inp):
    M(*inp)


@pytest.mark.parametrize(
    "inp, expectation",
    [
        ([[0, 1], [0]], pytest.raises(ValueError)),
        ([[0, 1], [2, 3]], doesnt_raise()),
    ],
)
def test_init_strict(inp, expectation):
    with expectation:
        M(*inp)


@pytest.mark.parametrize(
    "inp, fill, fill_from, expected",
    [
        ([[0, 1], [0]], 0, "back", [[0, 1], [0, 0]]),
        ([[0, 1], [0]], 1, "back", [[0, 1], [0, 1]]),
        ([[0, 1], [0]], 0, "front", [[0, 1], [0, 0]]),
        ([[0, 1], [0]], 1, "front", [[0, 1], [1, 0]]),
        ([[0, 1], []], 2, "front", [[0, 1], [2, 2]]),
        ([[1, 2, 3], [4, 5], [6]], 0, "front", [[1, 2, 3], [0, 4, 5], [0, 0, 6]]),
    ],
)
def test_init_fill(inp, fill, fill_from, expected):
    assert M(*inp, strict=False, fill=fill, fill_from=fill_from)._rows == expected


def test_init_valid_fill_from():
    with pytest.raises(ValueError):
        M([0], fill_from="middle", strict=False)


@pytest.mark.parametrize(
    "inp, fill, expected",
    [
        ([[1], [1, 1]], 0, [[1, 0], [1, 1]]),
        ([[1], [1, 1]], 2, [[1, 2], [1, 1]]),
        ([[1, 1], [1, 1]], 0, [[1, 1], [1, 1]]),
    ],
)
def test_init_nonstrict(inp, fill, expected):
    assert M(*inp, strict=False, fill=fill)._rows == expected


def test_abc():
    from collections.abc import Hashable, Reversible, Collection, Sequence

    m = M([0])

    assert isinstance(m, Hashable)
    assert isinstance(m, Reversible)
    assert isinstance(m, Collection)

    # Although Matrix implements all methods of the Sequence interface,
    # its __getitem__ and index methods have altered signatures, and as such
    # it violates the Liskov substitution principle, and should not be considered
    # a Sequence, for safety.
    assert not isinstance(m, Sequence)


@pytest.mark.parametrize(
    "m, n, expected",
    [
        (2, 3, [[0, 0, 0], [0, 0, 0]]),
        (3, None, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    ],
)
def test_zeros(m, n, expected):
    assert M.zeros(m, n)._rows == expected


@pytest.mark.parametrize(
    "n, expected", [(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (2, [[1, 0], [0, 1]])]
)
def test_identity(n, expected):
    assert M.identity(n)._rows == expected
    assert M.id(n)._rows == expected


def test_len():
    assert len(M([0], [0], [0])) == 3


# TODO: Need more cases here
def test_repr():
    assert repr(M([0])) == "Matrix([0])"
    assert repr(M([0], strict=False)) == "Matrix([0], strict=False)"
    assert repr(M([0], [0, 0], strict=False)) == "Matrix([0, 0], [0, 0], strict=False)"
    assert (
        repr(M([0], [0, 0], strict=False, fill=1))
        == "Matrix([0, 1], [0, 0], strict=False, fill=1)"
    )
    assert (
        repr(M([0], [0, 0], strict=False, fill=1, fill_from="front"))
        == 'Matrix([1, 0], [0, 0], strict=False, fill=1, fill_from="front")'
    )


# TODO: Need more cases here
def test_str():
    assert str(M([0])) == "Matrix([0])"
    assert str(M([0], [1])) == "Matrix([0]\n       [1])"


def test_contains():
    m = M([0, 0, 0], [0, 1, 0], [0, 0, 0])
    assert 1 in m
    assert "foo" not in m


def test_hash():
    m = M([0, 1, 2], [3, 4, 5], [6, 7, 8])
    d = {m: "test"}  # noqa

    assert hash(m) == hash((type(m), ((0, 1, 2), (3, 4, 5), (6, 7, 8))))


def test_iter():
    rows = [[0, 1], [2, 3]]
    m = M(*rows)

    for mrow, row in zip(m, rows):
        assert mrow == row


def test_reversed():
    rows = [[0, 1], [2, 3]]
    m = M(*rows)

    for mrow, row in zip(reversed(m), reversed(rows)):
        assert mrow == row


def test_idx_cell():
    m = M([0, 1, 2], [3, 4, 5])

    # range(1, 6) instead of range(6) because typically Matrix index notation
    # is 1-based, rather than 0 based. This is a computer science sin, but
    # aligns better with the mathematical nature and purpose of the Matrix class.
    for (i, val), expected in zip(m.idx_cell(), range(1, 6)):
        assert i == expected


def test_ij_cell():
    m = M([0, 1, 2], [3, 4, 5])

    expected_idxs = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (idx, val), expected in zip(m.ij_cell(), expected_idxs):
        assert (idx.i, idx.j) == expected


def test_combined_indexed_cell_generators():
    m = M([0, 1, 2], [3, 4, 5])

    for idx_cell, ij_cell in zip(m.idx_cell(), m.ij_cell()):
        idx, c1 = idx_cell
        (i, j), c2 = ij_cell

        assert c1 == c2
        assert idx == ((i - 1) * m.size[1]) + j


def test_getitem():
    m = M([0, 1, 2], [3, 4, 5])

    assert m[1] == (0, 1, 2)
    assert m[1, 2] == 1

    with pytest.raises(TypeError):
        m[1:1]

    with pytest.raises(TypeError):
        m["1"]


@pytest.mark.parametrize(
    "val, idx, start, end, expectation",
    [
        (4, (2, 2), (1, 1), (None, None), doesnt_raise()),
        (6, None, (1, 1), (None, None), pytest.raises(ValueError)),
        ("test", None, (1, 1), (None, None), pytest.raises(ValueError)),
        (0, (1, 1), (1, 1), (None, None), doesnt_raise()),
        (0, None, (2, 2), (None, None), pytest.raises(ValueError)),
        (4, (2, 2), (2, 2), (None, None), doesnt_raise()),
    ],
)
def test_index(val, idx, start, end, expectation):
    m = M([0, 1, 2], [3, 4, 5])

    with expectation:
        assert m.index(val, start=start, end=end) == idx


def test_count():
    m = M([0, 1, 2, 6], [3, 4, 5, 6])

    assert m.count(0) == 1
    assert m.count(6) == 2


def test_copy():
    m = M([0, 1, 2], [3, 4, 5])
    n = m.copy()

    assert (m == n) and not (m is n)


def test_eq():
    m = M([0, 1, 2], [3, 4, 5])

    assert m == M([0, 1, 2], [3, 4, 5])
    assert m != (type(m), ((0, 1, 2), (3, 4, 5)))


def test_size():
    m = M([0, 1, 2], [3, 4, 5])
    assert m.size == (2, 3)
    assert m.shape == m.size


def test_transpose():
    m = M([0, 1, 2], [3, 4, 5])
    mT = M([0, 3], [1, 4], [2, 5])

    assert m.transpose() == mT
    assert m.trans() == mT


def test_row_swap():
    m = M([0, 0], [1, 1], [2, 2])
    mp = m.row_swap(1, 2)

    n = M([1, 1], [0, 0], [2, 2])

    assert mp == n


def test_row_mult():
    m = M([0, 0], [1, 1], [2, 2])
    mp = m.row_mult(2, 3)

    n = M([0, 0], [3, 3], [2, 2])

    assert mp == n


def test_row_add():
    m = M([0, 0], [1, 1], [2, 2])
    mp = m.row_add(1, 2)

    n = M([1, 1], [1, 1], [2, 2])

    assert mp == n


def test_row_mult_add():
    m = M([2, 2], [1, 1], [2, 2])
    mp = m.row_mult_add(2, 3, 1)

    n = M([5, 5], [1, 1], [2, 2])

    assert mp == n


def test_mul():
    m = M([0, 1, 2], [3, 4, 5])
    k = 3

    km = M([0, 3, 6], [9, 12, 15])
    assert m * k == km
    assert k * m == km


def test_add():
    A = M([0, 1, 2], [3, 4, 5])
    B = M([1, 4, 5], [4, 6, 3])

    summed = M([1, 5, 7], [7, 10, 8])

    assert A + B == summed
    assert B + A == summed


def test_neg():
    m = M([1, 5, 7], [7, 10, 8])
    nm = M([-1, -5, -7], [-7, -10, -8])

    assert -m == nm
    assert -(-m) == m


def test_sub():
    A = M([0, 1, 2], [3, 4, 5])
    B = M([1, 4, 5], [4, 6, 3])

    AsubB = M([-1, -3, -3], [-1, -2, 2])
    BsubA = M([1, 3, 3], [1, 2, -2])

    assert A - B == AsubB
    assert B - A == BsubA


@pytest.mark.parametrize(
    "A, B, expected",
    [
        (M([0, 1], [0, 0]), M([0, 0], [1, 0]), M([1, 0], [0, 0])),
        (M([0, 0], [1, 0]), M([0, 1], [0, 0]), M([0, 0], [0, 1])),
        (
            M([2, 1, 4], [0, 1, 1]),
            M([6, 3, -1, 0], [1, 1, 0, 4], [-2, 5, 0, 2]),
            M([5, 27, -2, 12], [-1, 6, 0, 6]),
        ),
        (M([-2, 1], [0, 4]), M([6, 5], [-7, 1]), M([-19, -9], [-28, 4])),
    ],
)
def test_matmul(A, B, expected):
    assert A @ B == expected


def test_leading_coeff():
    m = M([0, 0, 5, 6], [1, 0, 0, 0], [0, 0, 0, 2])

    assert m._leading_coeff() == (5, 1, 2)


def test_leading_coeff_idx():
    m = M([0, 0, 5, 6], [1, 0, 0, 0], [0, 0, 0, 2])

    assert m._leading_coeff_idx() == (2, 0, 3)


@pytest.mark.parametrize(
    "m, expect",
    [
        (M([4, 6], [0, 1]), True),
        (M([4, 6], [1, 1]), False),
        (M([4, 5, 6], [0, 0, 0]), True),
        (M([0, 0, 0], [4, 5, 6]), False),
        (M([1, 0, 0], [0, 1, 0], [0, 0, 1]), True),
    ],
)
def test_is_ref(m, expect):
    assert m.is_ref() == expect


def test_ref():
    m = M([2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3])
    expected_mref = M([2, 1, -1, 8], [0, 1 / 2, 1 / 2, 1], [0, 0, -1, 1])

    assert m.ref() == expected_mref


def test_rref():
    m = M([2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3])
    expected_mrref = M(
        [1, 0, 0, 2],
        [0, 1, 0, 3],
        [0, 0, 1, -1],
    )

    assert m.rref(display=True) == expected_mrref


def test_submatrix():
    m = M([2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3])
    expect = M([2, 1, -1], [-3, -1, 2])

    assert m.submatrix(4, 3) == expect


def test_diagonal():
    m = M([2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3])

    assert m.diagonal() == (2, -1, 2)
    assert m.diag() == (2, -1, 2)

    Ident = M([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert Ident.diagonal() == (1, 1, 1)
    assert Ident.diag() == (1, 1, 1)


def test_trace():
    m = M([2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3])

    assert m.trace() == 3
    assert m.tr() == 3


@pytest.mark.parametrize(
    "m, expected",
    [
        (M([2, 1], [7, 4]), True),
        (M([-5 / 3, 2 / 3, 11 / 3], [0, 4 / 3, 4], [2, -1, -5]), False),
        (M([3 / 2, 1 / 2, 5], [5 / 3, -2 / 3, 0], [1, 1, 1]), True),
        (M([-6, -5, -4], [4, 5, 2], [0, 1, -1]), True),
    ],
)
def test_is_invertible(m, expected):
    assert m.is_invertible() == expected


@pytest.mark.parametrize(
    "m, expected",
    [
        (M([2, 1], [7, 4]), M([4, 1], [-7, 2])),
        (
            M([-math.sin(4), -math.cos(4)], [math.cos(4), -math.sin(4)]),
            M([-math.sin(4), math.cos(4)], [-math.cos(4), -math.sin(4)]),
        ),
        (
            M([-6, -5, -4], [4, 5, 2], [0, 1, -1]),
            M([-7 / 6, -3 / 2, 5 / 3], [2 / 3, 1, -2 / 3], [2 / 3, 1, -5 / 3]),
        ),
    ],
)
def test_inverse(m, expected):
    assert m.inverse() == expected
    assert m.inv() == expected


@pytest.mark.parametrize(
    "m, expected",
    [
        (M([-2, -3, 3], [6, 1, -49]), M([-9], [5])),
        (M([4, -1, 1, -5], [2, 2, 3, 10], [5, -2, 6, 12]), M([-2, 1, 4])),
    ],
)
def test_solution(m, expected):
    assert m.solution() == expected


def test_linear_combination():
    A = M([2, 1, -1], [1, -3, 3])
    b = M([-5], [15])

    assert A.linear_combination(b) == M([0, 0, 5])


@pytest.mark.parametrize(
    "m, expected",
    [
        (M([-14]), -14),
        (M([2, 5], [3, 1]), -13),
        (M([5, 1, 3], [0, 2, 2], [0, 0, -6]), -60),
        (
            M(
                [5, 2, 0, 0, 5],
                [0, 1, 4, 2, 4],
                [0, 0, 6, 3, 6],
                [0, 0, 4, 3, 7],
                [0, 0, 0, 0, 7],
            ),
            210,
        ),
        (M([6, 9, -7, 2], [0, 0, 8, 0], [0, 0, 5, 4], [0, 0, 0, -1]), 0),
        (M([2, -4], [3, -6]), 0),
        (M([1, 0, 4], [-3, 1, 2], [4, 0, 4]), -12),
        (M([1, 6, -4], [1, 2, 1], [3, 9, 1]), -7),
        (M([-math.sin(4), -math.cos(4)], [math.cos(4), -math.sin(4)]), 1),
        (M([-3, 1], [9, -3]), 0),
        (M([1, 1], [0, -1]), -1),
        (M([6, 12], [3, -15]), -126),
        (M([2, 4], [1, -5]), -14),
        (M([-2, 1], [5, 0]), -5),
        (M([2, -1], [-5, 0]), -5),
    ],
)
def test_determinant(m, expected):
    assert m.determinant() == expected
    assert m.det() == expected


def test_hw9_num5():
    A = M([6, 1], [2, 7])

    assert (A.inv()).det() == (1 / 40)


def test_adjugate():
    A = M([-6, -5, -4], [4, 5, 2], [0, 1, -1])

    adjA = M([-7, -9, 10], [4, 6, -4], [4, 6, -10])

    assert A.adjugate() == adjA
    assert A.adjoint() == adjA
    assert A.adj() == adjA


def test_is_collinear():
    assert M.is_collinear((2, 7), (3, 10), (5, 16))


def test_is_coplanar():
    assert M.is_coplanar((0, -2, -4), (3, 0, -2), (0, 1, -1), (-2, 2, 0))
