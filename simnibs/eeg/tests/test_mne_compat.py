"""Regression tests for the bundled MNE-Python integration.

These tests exercise small, public-API operations whose behavior depends on
keeping the bundled MNE in sync with the numpy version declared in
``pyproject.toml``. They are fast (<1 s) and run without the FEM solver, so
they catch dependency-pin regressions long before any leadfield computation.
"""
import numpy as np
import mne


def test_mne_fiff_string_write_numpy2_compat(tmp_path):
    """MNE FIFF string writer must work under the declared numpy version.

    Background
    ----------
    SimNIBS pins ``numpy>=2`` in ``pyproject.toml``. MNE versions earlier
    than 1.6 emit the legacy ``">a"`` ASCII dtype alias when writing FIFF
    string tags (``mne/io/write.py:write_string``), which NumPy 2.0 no
    longer accepts (NEP 55). The failure mode in production is a
    ``TypeError: data type '>a' not understood`` raised at the very last
    step of ``prepare_eeg_forward`` — i.e. only after the multi-minute FEM
    leadfield solve has already completed.

    Upstream MNE fixed this in 1.6 by replacing ``">a"`` with ``">S"``.

    What this guards against
    ------------------------
    Any future regression of either bound — MNE downgraded below 1.6, or
    numpy pinned <2 to mask the symptom — fails this test in <1 s with
    the same traceback as the production crash, instead of after a
    multi-minute leadfield run.
    """
    info = mne.create_info(["E1"], sfreq=1000.0, ch_types="eeg")
    info["description"] = "simnibs-numpy2-regression"

    raw = mne.io.RawArray(
        np.zeros((1, 100), dtype=np.float32), info, verbose=False,
    )
    fname = tmp_path / "raw.fif"
    raw.save(str(fname), verbose=False)

    raw2 = mne.io.read_raw_fif(str(fname), verbose=False)
    assert raw2.info["description"] == "simnibs-numpy2-regression"
