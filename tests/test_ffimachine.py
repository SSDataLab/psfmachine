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

from psfmachine import Machine, FFIMachine

from psfmachine.utils import do_tiled_query

ffi_path = get_pkg_data_filename("./ffi_ch01_test.fits")


@pytest.mark.remote_data
def test_ffi_from_file():
    ffi = pm.FFIMachine.from_file("./ffi_ch01_test.fits", channel=1)
    assert ffi.times.shape == (1,)
    assert ffi.flux.shape == (1, 33812)
    assert ffi.flux_2d.shape == (180, 188)
    assert ffi.flux_err.shape == (1, 33812)
    assert ffi.column.shape == (33812,)
    assert ffi.row.shape == (33812,)
    assert ffi.ra.shape == (33812,)
    assert ffi.dec.shape == (33812,)

    assert ffi.sources.shape == (259, 13)
