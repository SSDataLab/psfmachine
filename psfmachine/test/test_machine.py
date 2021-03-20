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
from .. import Machine, TPFMachine

lib_path = os.getcwd()

tpfs = []
flist = glob.glob(f"{lib_path}/psfmachine/test/data/tpf_test_*.fits")
for f in flist:
    tpfs.append(lk.KeplerTargetPixelFile(f))
tpfs = lk.collections.TargetPixelFileCollection(tpfs)


def test_parse_TPFs():
    t, f, fe, unw = TPFMachine._parse_TPFs(tpfs)
    assert t.shape == (10,)
    assert f.shape == (10, 319)
    assert fe.shape == (10, 319)
    assert unw.shape == (10, 319)

    locs, ra, dec = TPFMachine._convert_to_wcs(tpfs)
    assert locs.shape == (2, 319)
    assert ra.shape == (319,)
    assert dec.shape == (319,)

    f, fe, unw, locs, ra, dec = TPFMachine._preprocess(f, fe, unw, locs, ra, dec, tpfs)

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

    sources = TPFMachine._get_coord_and_query_gaia(tpfs)

    assert isinstance(sources, pd.DataFrame)
    assert set(["ra", "dec", "phot_g_mean_mag"]).issubset(sources.columns)
    assert sources.shape == (13, 98)


def test_from_TPFs():
    c = TPFMachine.from_TPFs(tpfs)
    assert isinstance(c, Machine)
