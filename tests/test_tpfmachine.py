"""
This contains a collection of functions to test the Machine API
"""
import os
import numpy as np
import pandas as pd
import pytest
import lightkurve as lk
from astropy.utils.data import get_pkg_data_filename
from astropy.time import Time

from psfmachine import Machine, TPFMachine
from psfmachine.tpf import (
    _parse_TPFs,
    _wcs_from_tpfs,
    _preprocess,
    _get_coord_and_query_gaia,
    _clean_source_list,
)

from psfmachine.utils import do_tiled_query

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
    assert sources.shape == (21, 12)


@pytest.mark.remote_data
def test_from_TPFs():
    c = TPFMachine.from_TPFs(tpfs)
    assert isinstance(c, Machine)


def test_load_save_shape_model():
    # load sufficient TPFs to build a shape model
    # the 10 TPFs in ./data/tpf_test* are not enough to fit a model, singular matrix
    # error.
    tpfs_k16 = lk.search_targetpixelfile(
        "Kepler-16", mission="Kepler", quarter=12, radius=200, limit=10, cadence="long"
    ).download_all(quality_bitmask=None)
    # instantiate a machine object
    machine = TPFMachine.from_TPFs(tpfs_k16)
    # build a shape model from TPF data
    machine.build_shape_model(plot=False)
    # save object state
    org_state = machine.__dict__
    # save shape model to disk
    file_name = "%s/data/test_shape_model.fits" % os.path.abspath(
        os.path.dirname(__file__)
    )
    machine.save_shape_model(output=file_name)

    # instantiate a new machine object with same data but load shape model from disk
    machine = TPFMachine.from_TPFs(tpfs_k16)
    machine.load_shape_model(plot=False, input=file_name)
    new_state = machine.__dict__

    # check that critical attributes match
    assert org_state["n_r_knots"] == new_state["n_r_knots"]
    assert org_state["n_phi_knots"] == new_state["n_phi_knots"]
    assert org_state["rmin"] == new_state["rmin"]
    assert org_state["rmax"] == new_state["rmax"]
    assert org_state["cut_r"] == new_state["cut_r"]
    assert (org_state["psf_w"] == new_state["psf_w"]).all()
    assert ((org_state["mean_model"] == new_state["mean_model"]).data).all()
