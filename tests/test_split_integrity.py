"""Milestone 8: the geographic splits are frozen. Fingerprints recorded on 2026-10-31 (milestone-8 branch).

If any of these fail, a validation or test tile was added, removed or reassigned. That would
invalidate every comparison with earlier milestones, so it must be an explicit, documented decision.
"""

import hashlib

import pytest

from urbanmorph.data.m2_dataset import INDEX_PATH, load_index_with_expansion

pytestmark = pytest.mark.skipif(not INDEX_PATH.exists(), reason="dataset index not built")

FROZEN = {
    "val": (19, "c2110d3ca6bf0bae652e3367c1fef9db01c47c29101978564f70fc7c6e035413"),
    "test": (19, "c46aa7f76d969c5b809f9744299ba9ebb4484ce9b79bdc61283705950ec70f9e"),
    "train": (565, "495ac1b9651400016691d0a206ac83d6f4c7e5a7c96b48a01df582fb7bb45aa4"),
}


@pytest.mark.parametrize("split", sorted(FROZEN))
def test_split_membership_unchanged(split):
    d = load_index_with_expansion()
    ids = sorted(d[d["split"] == split]["sample_id"])
    n, digest = FROZEN[split]
    assert len(ids) == n
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == digest


def test_no_tile_in_two_splits():
    d = load_index_with_expansion()
    assert not d.duplicated("sample_id").any()
