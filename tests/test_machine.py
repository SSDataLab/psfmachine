"""
This contains a collection of functions to test the Machine API
"""
import numpy as np
from scipy import sparse
import pytest
import lightkurve as lk
from astropy.utils.data import get_pkg_data_filename

from psfmachine import TPFMachine

tpfs = []
for idx in range(10):
    tpfs.append(lk.read(get_pkg_data_filename(f"data/tpf_test_{idx:02}.fits")))
tpfs = lk.collections.TargetPixelFileCollection(tpfs)


@pytest.mark.remote_data
def test_create_delta_sparse_arrays():
    machine = TPFMachine.from_TPFs(tpfs)
    # create numpy arrays
    machine._create_delta_arrays()
    non_sparse_arr = machine.__dict__.copy()

    # check for main attrs shape
    assert non_sparse_arr["time"].shape == (10,)
    assert non_sparse_arr["flux"].shape == (10, 285)
    assert non_sparse_arr["flux_err"].shape == (10, 285)
    assert non_sparse_arr["column"].shape == (285,)
    assert non_sparse_arr["row"].shape == (285,)
    assert non_sparse_arr["ra"].shape == (285,)
    assert non_sparse_arr["dec"].shape == (285,)
    assert non_sparse_arr["sources"].shape == (19, 13)

    assert non_sparse_arr["dra"].shape == (19, 285)
    assert non_sparse_arr["ddec"].shape == (19, 285)
    assert non_sparse_arr["r"].shape == (19, 285)
    assert non_sparse_arr["phi"].shape == (19, 285)
    # check dra ddec r and phi are numpy arrays
    assert isinstance(non_sparse_arr["dra"], np.ndarray)
    assert isinstance(non_sparse_arr["ddec"], np.ndarray)
    assert isinstance(non_sparse_arr["r"], np.ndarray)
    assert isinstance(non_sparse_arr["phi"], np.ndarray)

    # manually mask numpy arrays to compare them vs sparse array
    dist_lim = 40
    mask = (np.abs(non_sparse_arr["dra"].value) <= dist_lim / 3600) & (
        np.abs(non_sparse_arr["ddec"].value) <= dist_lim / 3600
    )

    # create sparse arrays
    machine._create_delta_sparse_arrays(dist_lim=dist_lim)
    sparse_arr = machine.__dict__.copy()

    assert sparse_arr["dra"].shape == non_sparse_arr["dra"].shape
    assert sparse_arr["ddec"].shape == non_sparse_arr["ddec"].shape
    assert sparse_arr["r"].shape == non_sparse_arr["r"].shape
    assert sparse_arr["phi"].shape == non_sparse_arr["phi"].shape
    # check dra ddec r and phi are sparse arrays
    assert isinstance(sparse_arr["dra"], sparse.csr_matrix)
    assert isinstance(sparse_arr["ddec"], sparse.csr_matrix)
    assert isinstance(sparse_arr["r"], sparse.csr_matrix)
    assert isinstance(sparse_arr["phi"], sparse.csr_matrix)
    # check for non-zero values shape
    assert sparse_arr["dra"].data.shape == (853,)
    assert sparse_arr["ddec"].data.shape == (853,)
    assert sparse_arr["r"].data.shape == (853,)
    assert sparse_arr["phi"].data.shape == (853,)

    # compare sparse array vs numpy array values
    assert (non_sparse_arr["dra"][mask].value == sparse_arr["dra"].data).all()
    assert (non_sparse_arr["ddec"][mask].value == sparse_arr["ddec"].data).all()
    assert (non_sparse_arr["r"][mask].value == sparse_arr["r"].data).all()
    assert (non_sparse_arr["phi"][mask].value == sparse_arr["phi"].data).all()


@pytest.mark.remote_data
def test_compute_aperture_photometry():
    # it tests aperture mask creation and flux metric computation
    machine = TPFMachine.from_TPFs(tpfs, apply_focus_mask=False)
    # load FFI shape model from file
    machine.load_shape_model(
        input=get_pkg_data_filename("data/shape_model_ffi.fits"),
        plot=False,
    )
    # compute max aperture
    machine.create_aperture_mask(percentile=0)
    assert machine.aperture_mask.shape == (19, 285)
    # some sources way outside the TPF can have 0-size aperture
    assert (machine.aperture_mask.sum(axis=1) >= 0).all()
    assert machine.FLFRCSAP.shape == (19,)
    assert machine.CROWDSAP.shape == (19,)
    # full apereture shoudl lead to FLFRCSAP = 1
    assert np.allclose(machine.FLFRCSAP[np.isfinite(machine.FLFRCSAP)], 1)
    assert (machine.CROWDSAP[np.isfinite(machine.CROWDSAP)] >= 0).all()
    assert (machine.CROWDSAP[np.isfinite(machine.CROWDSAP)] <= 1).all()

    # compute min aperture, here CROWDSAP not always will be 1, e.g. 2 sources in the
    # same pixel.
    machine.create_aperture_mask(percentile=99)
    assert (machine.CROWDSAP[np.isfinite(machine.CROWDSAP)] >= 0).all()
    assert (machine.CROWDSAP[np.isfinite(machine.CROWDSAP)] <= 1).all()
    assert (machine.FLFRCSAP[np.isfinite(machine.FLFRCSAP)] >= 0).all()
    assert (machine.FLFRCSAP[np.isfinite(machine.FLFRCSAP)] <= 1).all()

    machine.compute_aperture_photometry(aperture_size="optimal")
    assert machine.aperture_mask.shape == (19, 285)
    # some sources way outside the TPF can have 0-size aperture
    assert (machine.aperture_mask.sum(axis=1) >= 0).all()
    # check aperture size is within range
    assert (machine.optimal_percentile >= 0).all() and (
        machine.optimal_percentile <= 100
    ).all()
    assert machine.sap_flux.shape == (10, 19)
    assert machine.sap_flux_err.shape == (10, 19)
    assert (machine.sap_flux >= 0).all()
    assert (machine.sap_flux_err >= 0).all()
