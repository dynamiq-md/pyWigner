from nose.tools import (
    assert_equal, assert_not_equal, assert_almost_equal, assert_items_equal,
    raises
)
from nose.plugins.skip import Skip, SkipTest

from numpy.testing import assert_array_almost_equal

def check_function(function, dictionary):
    """Test a single-variable function based on key-value pairs.

    Takes `function` and calls it with the keys of `dictionary`. Asserts
    that the result should be "almost_equal" to the value of `dictionary`.
    """
    for test_input in dictionary.keys():
        assert_almost_equal(function(test_input), dictionary[test_input])


