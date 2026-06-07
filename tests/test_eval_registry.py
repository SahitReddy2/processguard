"""Registry-level tests: unknown types raise a clear error, and that error
includes the full list of registered types so a typo is obvious."""
from __future__ import annotations

import pytest

from processguard.evaluators import (
    AssertionTypeNotRegistered,
    get,
    registered_names,
)


def test_unknown_assertion_type_raises_with_registered_list():
    with pytest.raises(AssertionTypeNotRegistered) as excinfo:
        get("AssertNonExistent", {})

    msg = str(excinfo.value)
    # the bad name must appear, quoted
    assert '"AssertNonExistent"' in msg
    # so must the registered list, so the user can see the typo
    assert "AssertDetectorFired" in msg
    assert "AssertToolCalled" in msg


def test_registered_names_includes_all_v02_evaluators():
    names = set(registered_names())
    expected = {
        "AssertToolCalled",
        "AssertWithinStepBudget",
        "AssertDetectorFired",
        "AssertDetectorDidNotFire",
        "AssertEventCountByType",
        "AssertSingleTraceId",
    }
    missing = expected - names
    assert not missing, f"missing registered evaluators: {missing}"
