import pytest
from main import sum_values


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, -1, -2),
        (100, 25, 125),
    ],
)
def test_sum_values(a, b, expected):
    assert sum_values(a, b) == expected
