"""
This contains a collection of functions to test the Machine API
"""

import glob
import os
import sys
import pytest
import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord, match_coordinates_3d

from .. import Machine, TPFMachine
from ..tpf import (
    _parse_TPFs,
    _wcs_from_tpfs,
    _preprocess,
    _get_coord_and_query_gaia,
)

tpfs = []
for idx in range(10):
    tpfs.append(lk.read(get_pkg_data_filename(f"data/tpf_test_{idx:02}.fits")))
tpfs = lk.collections.TargetPixelFileCollection(tpfs)


@pytest.mark.remote_data
def test_parse_TPFs():
    (
        times,
        flux,
        flux_err,
        column,
        row,
        unw,
        focus_mask,
        qual_mask,
        sat_mask,
    ) = _parse_TPFs(tpfs)

    assert times.shape == (10,)
    assert flux.shape == (10, 345)
    assert flux_err.shape == (10, 345)
    assert unw.shape == (345,)
    assert column.shape == (345,)
    assert row.shape == (345,)
    assert focus_mask.shape == (10,)
    assert qual_mask.shape == (10,)
    assert sat_mask.shape == (345,)

    locs, ra, dec = _wcs_from_tpfs(tpfs)
    assert locs.shape == (2, 345)
    assert ra.shape == (345,)
    assert dec.shape == (345,)

    flux, flux_err, unw, locs, ra, dec, column, row = _preprocess(
        flux,
        flux_err,
        unw,
        locs,
        ra,
        dec,
        column,
        row,
        tpfs,
        sat_mask,
    )

    assert np.isfinite(flux).all()
    assert np.isfinite(flux_err).all()
    assert np.isfinite(locs).all()
    assert np.isfinite(ra).all()
    assert np.isfinite(dec).all()

    assert locs.shape == (2, 285)
    assert ra.shape == (285,)
    assert dec.shape == (285,)
    assert flux.shape == (10, 285)
    assert flux_err.shape == (10, 285)

    sources = _get_coord_and_query_gaia(tpfs)

    assert isinstance(sources, pd.DataFrame)
    assert set(["ra", "dec", "phot_g_mean_mag"]).issubset(sources.columns)
    assert sources.shape == (16, 26)


@pytest.mark.remote_data
def test_from_TPFs():
    c = TPFMachine.from_TPFs(tpfs)
    assert isinstance(c, Machine)
