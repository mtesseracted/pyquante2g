import pytest

from tests.array_ops.conftest import INSTALLED_BACKENDS
from tests.array_ops.optest_helpers import (
    assert_same_output,
    available_testable_ops,
    call_op,
    make_ops,
)


REF_BACKEND = "torch"
COMPARE_BACKENDS = [name for name in INSTALLED_BACKENDS if name != REF_BACKEND]

pytestmark = [
    pytest.mark.skipif(
        REF_BACKEND not in INSTALLED_BACKENDS,
        reason="torch backend is required for torch reference comparisons",
    ),
    pytest.mark.skipif(
        not COMPARE_BACKENDS,
        reason="no non-torch backends installed",
    ),
]


def _comparison_params():
    if REF_BACKEND not in INSTALLED_BACKENDS:
        return []

    ref_methods = set(available_testable_ops(make_ops(REF_BACKEND)))
    params = []
    for backend_name in COMPARE_BACKENDS:
        other_methods = set(available_testable_ops(make_ops(backend_name)))
        for method_name in sorted(ref_methods & other_methods):
            params.append(
                pytest.param(
                    backend_name,
                    method_name,
                    id=f"{backend_name}:{method_name}",
                )
            )
    return params


@pytest.mark.parametrize("backend_name,method_name", _comparison_params())
def test_array_ops_vs_torch(backend_name, method_name):
    ref_ops = make_ops(REF_BACKEND)
    test_ops = make_ops(backend_name)

    ref_out = call_op(ref_ops, method_name)
    test_out = call_op(test_ops, method_name)

    assert_same_output(method_name, ref_out, test_out)
