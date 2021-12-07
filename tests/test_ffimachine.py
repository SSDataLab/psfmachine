"""
This contains a collection of functions to test the Machine API
"""
import os

# import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from psfmachine import Machine, FFIMachine


@pytest.mark.remote_data
def test_ffi_from_file():

    ffi_path = get_pkg_data_filename("./data/kplr-ffi_ch01_test.fits")
    ffi = FFIMachine.from_file(ffi_path, extension=1)
    # test `FFIMachine.from_file` is of Machine class
    assert isinstance(ffi, Machine)
    # test attributes have the right shapes
    assert ffi.time.shape == (1,)
    assert ffi.flux.shape == (1, 33812)
    assert ffi.flux_2d.shape == (1, 180, 188)
    assert ffi.flux_err.shape == (1, 33812)
    assert ffi.column.shape == (33812,)
    assert ffi.row.shape == (33812,)
    assert ffi.ra.shape == (33812,)
    assert ffi.dec.shape == (33812,)
    assert ffi.sources.shape == (269, 14)


@pytest.mark.remote_data
def test_save_shape_model():
    ffi_path = get_pkg_data_filename("./data/kplr-ffi_ch01_test.fits")
    ffi = FFIMachine.from_file(ffi_path, extension=1)
    # create a shape model
    ffi.build_shape_model()
    file_name = "%s/data/test_ffi_shape_model.fits" % os.path.abspath(
        os.path.dirname(__file__)
    )
    # save shape model to tmp file
    ffi.save_shape_model(file_name)

    # test that the saved shape model has the right metadata and shape
    shape_model = fits.open(file_name)
    assert shape_model[1].header["OBJECT"] == "PRF shape"
    assert shape_model[1].header["TELESCOP"] == ffi.meta["TELESCOP"]
    assert shape_model[1].header["MJD-OBS"] == ffi.time[0]

    assert shape_model[1].header["N_RKNOTS"] == ffi.n_r_knots
    assert shape_model[1].header["N_PKNOTS"] == ffi.n_phi_knots
    assert shape_model[1].header["RMIN"] == ffi.rmin
    assert shape_model[1].header["RMAX"] == ffi.rmax
    assert shape_model[1].header["CUT_R"] == ffi.cut_r
    assert shape_model[1].data.shape == ffi.psf_w.shape

    os.remove(file_name)


@pytest.mark.remote_data
def test_save_flux_values():
    ffi_path = get_pkg_data_filename("./data/kplr-ffi_ch01_test.fits")
    ffi = FFIMachine.from_file(ffi_path, extension=1)
    ffi.build_shape_model()
    file_name = "%s/data/ffi_test_phot.fits" % os.path.abspath(
        os.path.dirname(__file__)
    )
    # fit the shape model to sources, compute psf photometry, save catalog to disk
    ffi.save_flux_values(file_name)

    # check FITS file has the right metadata
    hdu = fits.open(file_name)
    assert hdu[0].header["TELESCOP"] == ffi.meta["TELESCOP"]
    assert hdu[0].header["DCT_TYPE"] == ffi.meta["DCT_TYPE"]
    assert hdu[1].header["MJD-OBS"] == ffi.time[0]

    # check FITS table has the right shapes, columns and units
    table = Table.read(file_name)
    assert len(table) == 269
    assert "psf_flux" in table.keys()
    assert "psf_flux_err" in table.keys()
    assert "gaia_id" in table.keys()
    assert "ra" in table.keys()
    assert "dec" in table.keys()
    assert table["psf_flux"].unit == "-e/s"
    assert table["psf_flux_err"].unit == "-e/s"
    assert table["ra"].unit == "deg"
    assert table["dec"].unit == "deg"

    os.remove(file_name)
