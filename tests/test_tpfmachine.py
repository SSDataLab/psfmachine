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

from psfmachine import PACKAGEDIR
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
        pos_corr1,
        pos_corr2,
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
    assert pos_corr1.shape == (10, 10)
    assert pos_corr2.shape == (10, 10)

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


@pytest.mark.remote_data
def test_load_save_shape_model():
    # instantiate a machine object
    machine = TPFMachine.from_TPFs(tpfs, apply_focus_mask=False)
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
    machine = TPFMachine.from_TPFs(tpfs, apply_focus_mask=False)
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
    # remove shape model from disk
    os.remove(file_name)


@pytest.mark.remote_data
def test_load_shape_model_from_zenodo():
    # instantiate a machine object
    machine = TPFMachine.from_TPFs(tpfs, apply_focus_mask=False)

    # load a FFI-PRF model from zenodo repo
    machine.load_shape_model(plot=False, input=None)
    state = machine.__dict__

    # check that critical attributes match
    assert state["n_r_knots"] == 10
    assert state["n_phi_knots"] == 15
    assert state["rmin"] == 1
    assert state["rmax"] == 16
    assert state["cut_r"] == 6
    assert state["psf_w"].shape == (169,)

    # check file is in directory for future use
    assert os.path.isfile(
        f"{PACKAGEDIR}/data/"
        f"{machine.tpf_meta['mission'][0]}_FFI_PRFmodels_v1.0.tar.gz"
    )


@pytest.mark.remote_data
def test_get_source_centroids():
    # test centroid fx
    machine = TPFMachine.from_TPFs(tpfs, apply_focus_mask=False)
    machine._get_source_mask()
    machine.get_source_centroids(method="poscor")
    # check centroid arrays have the right shape
    assert machine.source_centroids_column_poscor.shape == (19, 10)
    assert machine.source_centroids_row_poscor.shape == (19, 10)
    machine.get_source_centroids(method="scene")
    assert machine.source_centroids_column_scene.shape == (19, 10)
    assert machine.source_centroids_row_scene.shape == (19, 10)

    tpf_idx = []
    for i in range(len(machine.sources)):
        tpf_idx.append(
            [k for k, ss in enumerate(machine.tpf_meta["sources"]) if i in ss]
        )
    # check centroids agree within a pixel for target sources when comparing vs
    # pipeline aperture moments aperture TPF.estimate_centroids()
    for i in range(machine.nsources):
        if machine.sources.tpf_id[i]:
            if len(tpf_idx[i]) > 1:
                continue
            col_cent, row_cent = tpfs[tpf_idx[i][0]].estimate_centroids()
            assert (
                np.abs(col_cent - machine.source_centroids_column_scene[i]).value < 1
            ).all()
            assert (
                np.abs(row_cent - machine.source_centroids_row_scene[i]).value < 1
            ).all()
            assert (
                np.abs(col_cent - machine.source_centroids_column_poscor[i]).value < 1
            ).all()
            assert (
                np.abs(row_cent - machine.source_centroids_row_poscor[i]).value < 1
            ).all()
