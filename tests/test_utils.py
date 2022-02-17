import numpy as np
from psfmachine.utils import threshold_bin


def test_threshold_bin():

    # Pass a bunch of nans
    x, y, z = (
        np.random.normal(size=1000),
        np.random.normal(size=1000),
        np.ones(1000) * np.nan,
    )
    hist, xbin, ybin, zbin, zwbin = threshold_bin(x, y, z)
    assert xbin.shape[0] <= x.shape[0]
    assert np.isnan(zbin).all()
    assert hist.min() >= 1.0

    # Pass something that should actually pass
    x = np.random.normal(0, 1, 5000)
    y = np.random.normal(0, 1, 5000)
    z = np.power(10, -(((x ** 2 + y ** 2) / 0.5) ** 0.5) / 2)
    hist, xbin, ybin, zbin, zwbin = threshold_bin(x, y, z, abs_thresh=5, bins=50)
    # check that some points are binned, some points are not
    assert hist.shape == xbin.shape
    assert hist.shape == ybin.shape
    assert hist.shape == zbin.shape
    assert hist.shape == zwbin.shape
    assert xbin.shape[0] <= x.shape[0]
    assert np.isin(zbin, z).sum() > 0
    assert hist.min() >= 1.0

    # add random nan values to Z
    z[np.random.randint(0, 5000, 50)] = np.nan
    hist2, xbin2, ybin2, zbin2, zwbin = threshold_bin(x, y, z, abs_thresh=5, bins=50)
    # check that some points are binned, some points are not
    assert xbin2.shape[0] <= x.shape[0]
    assert np.isin(zbin2, z).sum() > 0
    assert hist2.min() >= 1.0
    # We should get the same outut for (hist, xbin, ybin) and same shape for zbin
    assert (hist == hist2).all()
    assert (xbin == xbin2).all()
    assert (ybin == ybin2).all()
    assert zbin.shape == hist2.shape
