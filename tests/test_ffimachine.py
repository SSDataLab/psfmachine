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

ffi = None


@pytest.mark.remote_data
def test_ffi_from_file():
    pass
