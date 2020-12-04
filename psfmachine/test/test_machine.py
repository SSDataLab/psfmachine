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
from astropy.coordinates import SkyCoord, match_coordinates_3d

sys.path.append("../../")
from psfmachine.machine import Machine

lib_path = os.getcwd()

tpfs = []
flist = glob.glob(f"{lib_path}/test/data/tpf_test_*.fits")
for f in flist:
    tpfs.append(lk.KeplerTargetPixelFile(f))
tpfs = lk.collections.TargetPixelFileCollection(tpfs)


def test_parse_TPFs():
    t, f, fe, unw = Machine._parse_TPFs(tpfs)
    assert t.shape == (10,)
    assert f.shape == (10, 319)
    assert fe.shape == (10, 319)
    assert unw.shape == (10, 319)


def test_convert_to_wcs():
    locs, ra, dec = Machine._convert_to_wcs(tpfs)
    assert locs.shape == (2, 319)
    assert ra.shape == (319,)
    assert dec.shape == (319,)


def test_preprocess():
    t, f, fe, unw = Machine._parse_TPFs(tpfs)
    locs, ra, dec = Machine._convert_to_wcs(tpfs)
    f, fe, unw, locs, ra, dec = Machine._preprocess(f, fe, unw, locs, ra, dec, tpfs)

    assert np.isfinite(f).all()
    assert np.isfinite(fe).all()
    assert np.isfinite(locs).all()
    assert np.isfinite(ra).all()
    assert np.isfinite(dec).all()

    assert locs.shape == (2, 287)
    assert ra.shape == (287,)
    assert dec.shape == (287,)
    assert f.shape == (10, 287)
    assert fe.shape == (10, 287)


def test_get_coord_and_query_gaia():
    t, f, fe, unw = Machine._parse_TPFs(tpfs)
    locs, ra, dec = Machine._convert_to_wcs(tpfs)
    f, fe, unw, locs, ra, dec = Machine._preprocess(f, fe, unw, locs, ra, dec, tpfs)
    sources = Machine._get_coord_and_query_gaia(ra, dec, unw, t[0])

    assert isinstance(sources, pd.DataFrame)
    assert set(["ra", "dec", "phot_g_mean_mag"]).issubset(sources.columns)
    assert sources.shape == (13, 98)


def test_clean_source_list():
    t, f, fe, unw = Machine._parse_TPFs(tpfs)
    locs, ra, dec = Machine._convert_to_wcs(tpfs)
    f, fe, unw, locs, ra, dec = Machine._preprocess(f, fe, unw, locs, ra, dec, tpfs)
    sources = Machine._get_coord_and_query_gaia(ra, dec, unw, t[0])
    sources, removed_sources = Machine._clean_source_list(sources, ra, dec)

    assert sources.shape == (11, 99)
    assert (sources.clean_flag.values == 0).all()

    s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
    midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]

    assert np.all(mdist.to("arcsec").value >= 6.0)


def test_from_TPFs():
    c = Machine.from_TPFs(tpfs)
    assert isinstance(c, Machine)


# def test_from_TPFs():
