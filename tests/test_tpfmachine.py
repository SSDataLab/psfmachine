"""
This contains a collection of functions to test the Machine API
"""
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


@pytest.mark.remote_data
def test_do_tiled_query():
    # unit test for TPF stack
    # test that the tiled query get the same results as the original psfmachine query
    epoch = Time(tpfs[0].time[len(tpfs[0]) // 2], format="jd").jyear
    sources_org = _get_coord_and_query_gaia(tpfs, magnitude_limit=18, dr=3)
    _, ra, dec = _wcs_from_tpfs(tpfs)
    sources_tiled = do_tiled_query(
        ra,
        dec,
        ngrid=(2, 2),
        magnitude_limit=18,
        dr=3,
        epoch=epoch,
    )
    assert isinstance(sources_tiled, pd.DataFrame)
    assert set(["ra", "dec", "phot_g_mean_mag"]).issubset(sources_tiled.columns)
    # check that the tiled query contain all sources from the non-tiled query.
    # tiled query is always bigger that the other for TPF stacks.
    assert set(sources_org.designation).issubset(sources_tiled.designation)

    # get ra,dec values for pixels then clean source list
    ras, decs = [], []
    for tpf in tpfs:
        r, d = np.hstack(tpf.get_coordinates(0)).T.reshape(
            [2, np.product(tpf.shape[1:])]
        )
        ras.append(r)
        decs.append(d)
    ras, decs = np.hstack(ras), np.hstack(decs)
    sources_tiled, _ = _clean_source_list(sources_tiled, ras, decs)
    # clean source lists must match between query versions
    assert sources_tiled.shape == sources_org.shape
    assert set(sources_org.designation) == set(sources_tiled.designation)

    # Unit test for 360->0 deg boundary. we use a smaller sky patch now.
    row = np.arange(100)
    column = np.arange(100)
    column_grid, row_grid = np.meshgrid(column, row)
    # I subtract pix position to move the grid into the 360->0 ra boundary
    column_grid -= 44200
    ra, dec = (
        tpfs[0]
        .wcs.wcs_pix2world(
            np.vstack([column_grid.ravel(), row_grid.ravel()]).T,
            0.0,
        )
        .T
    )
    # check that ra values are in the boundary
    assert not ((ra < 359) & (ra > 1)).all()
    boundary_sources = do_tiled_query(
        ra, dec, ngrid=(2, 2), magnitude_limit=16, epoch=epoch, dr=3
    )
    assert boundary_sources.shape == (85, 11)
    # check that no result objects are outside the boundary for ra
    assert not ((boundary_sources.ra < 359) & (boundary_sources.ra > 1)).all()
