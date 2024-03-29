"""test perturbation matrix"""
import numpy as np
from scipy import sparse
import pytest
from psfmachine.perturbation import PerturbationMatrix, PerturbationMatrix3D


def test_perturbation_matrix():
    time = np.arange(0, 10, 0.1)
    p = PerturbationMatrix(time=time, focus=False)
    assert p.vectors.shape == (100, 4)
    p = PerturbationMatrix(time=time, focus=True)
    assert p.vectors.shape == (100, 5)

    with pytest.raises(ValueError):
        p = PerturbationMatrix(
            time=time, other_vectors=np.random.normal(size=(2, 10)), focus=False
        )
        p = PerturbationMatrix(time=time, other_vectors=1, focus=False)
    p = PerturbationMatrix(
        time=time, other_vectors=np.random.normal(size=(2, 100)), focus=False
    )
    p = PerturbationMatrix(
        time=time, other_vectors=np.random.normal(size=(100, 2)), focus=False
    )
    assert p.vectors.shape == (100, 6)
    time = np.hstack([np.arange(0, 10, 0.1), np.arange(15, 25, 0.1)])
    p = PerturbationMatrix(time=time, focus=False)
    assert p.vectors.shape == (200, 8)
    p = PerturbationMatrix(time=time, focus=True)
    assert p.vectors.shape == (200, 10)

    assert p.matrix.shape == (200 / 10, 10)
    assert sparse.issparse(p.matrix)

    res = 10
    p = PerturbationMatrix(time=time, focus=False, resolution=res)
    y, ye = np.random.normal(1, 0.01, size=200), np.ones(200) * 0.01
    p.fit(y, ye)
    w = p.weights
    assert w.shape[0] == p.shape[1]
    assert np.isfinite(w).all()
    model = p.model()
    assert model.shape == y.shape
    chi = np.sum((y - model) ** 2 / (ye ** 2)) / (p.shape[0] - p.shape[1] - 1)
    assert chi < 3

    y, ye = np.random.normal(1, 0.01, size=200), np.ones(200) * 0.01
    for bin_method in ["downsample", "bin"]:
        s = 200 / res + 1 if bin_method == "downsample" else 200 / res
        p = PerturbationMatrix(
            time=time, focus=False, resolution=res, bin_method=bin_method
        )
        assert len(p.bin_func(y)) == s
        assert len(p.bin_func(ye, quad=True)) == s
        with pytest.raises(ValueError):
            p.bin_func(y[:-4])

        p.fit(y, ye)
        w = p.weights
        model = p.model()
        assert model.shape[0] == 200
        chi = np.sum((y - model) ** 2 / (ye ** 2)) / (p.shape[0] - p.shape[1] - 1)
        assert chi < 3

    # Test PCA
    flux = np.random.normal(1, 0.1, size=(200, 100))
    p = PerturbationMatrix(time=time, focus=False)
    assert p.matrix.shape == (20, 8)
    p.pca(flux, ncomponents=2)
    assert p.matrix.shape == (20, 12)
    # assert np.allclose((p.vectors.sum(axis=0) / (p.vectors != 0).sum(axis=0))[8:], 0)
    p.fit(y, ye)
    p = PerturbationMatrix(time=time, focus=False, segments=False)
    assert p.matrix.shape == (20, 4)
    p.pca(flux, ncomponents=2)
    assert p.matrix.shape == (20, 6)
    # assert np.allclose((p.vectors.sum(axis=0) / (p.vectors != 0).sum(axis=0))[8:], 0)
    p.fit(y, ye)


def test_perturbation_matrix3d():
    time = np.arange(0, 10, 1)
    # 13 x 13 pixels, evenly spaced in x and y
    dx, dy = np.mgrid[:13, :13] - 6 + 0.01
    dx, dy = dx.ravel(), dy.ravel()

    # ntime x npixels
    flux = np.random.normal(1, 0.01, size=(10, 169)) + dx[None, :] * dy[None, :]
    # the perturbation model assumes the perturbation is around 1
    flux_err = np.ones((10, 169)) * 0.01

    p3 = PerturbationMatrix3D(
        time=time, dx=dx, dy=dy, nknots=4, radius=5, resolution=5, poly_order=1
    )
    assert p3.cartesian_matrix.shape == (169, 81)
    assert p3.vectors.shape == (10, 2)
    assert p3.shape == (
        p3.cartesian_matrix.shape[0] * p3.ntime,
        p3.cartesian_matrix.shape[1] * p3.nvec,
    )
    assert p3.matrix.shape == (
        p3.cartesian_matrix.shape[0] * p3.nbins,
        p3.cartesian_matrix.shape[1] * p3.nvec,
    )
    p3.fit(flux, flux_err)
    w = p3.weights
    assert w.shape[0] == p3.cartesian_matrix.shape[1] * p3.nvec
    model = p3.model()
    assert model.shape == flux.shape

    chi = np.sum((flux - model) ** 2 / (flux_err ** 2)) / (
        p3.shape[0] - p3.shape[1] - 1
    )
    assert chi < 1.5

    time = np.arange(0, 100, 1)
    flux = np.random.normal(1, 0.01, size=(100, 169)) + dx[None, :] * dy[None, :]
    # the perturbation model assumes the perturbation is around 1
    flux_err = np.ones((100, 169)) * 0.01

    for bin_method in ["downsample", "bin"]:
        p3 = PerturbationMatrix3D(
            time=time,
            dx=dx,
            dy=dy,
            nknots=4,
            radius=5,
            poly_order=2,
            bin_method=bin_method,
        )
        p3.fit(flux, flux_err)
        w = p3.weights
        model = p3.model()
        assert model.shape == flux.shape
        chi = np.sum((flux - model) ** 2 / (flux_err ** 2)) / (
            p3.shape[0] - p3.shape[1] - 1
        )
        assert chi < 3

    p3 = PerturbationMatrix3D(
        time=time,
        dx=dx,
        dy=dy,
        nknots=4,
        radius=5,
        poly_order=2,
        bin_method=bin_method,
    )
    p3.pca(flux, ncomponents=5)
    p3.fit(flux, flux_err)

    # Add in one bad pixel
    flux[:, 100] = 1e5
    pixel_mask = np.ones(169, bool)
    pixel_mask[100] = False

    for bin_method in ["downsample", "bin"]:
        p3 = PerturbationMatrix3D(
            time=time,
            dx=dx,
            dy=dy,
            nknots=4,
            radius=5,
            poly_order=2,
            bin_method=bin_method,
        )
        p3.fit(flux, flux_err)
        w = p3.weights
        model = p3.model()
        chi = np.sum(
            (flux[:, pixel_mask] - model[:, pixel_mask]) ** 2
            / (flux_err[:, pixel_mask] ** 2)
        ) / (p3.shape[0] - p3.shape[1] - 1)
        # Without the pixel masking the model doesn't fit
        assert chi > 3
        p3.fit(flux, flux_err, pixel_mask=pixel_mask)
        w = p3.weights
        model = p3.model()
        chi = np.sum(
            (flux[:, pixel_mask] - model[:, pixel_mask]) ** 2
            / (flux_err[:, pixel_mask] ** 2)
        ) / (p3.shape[0] - p3.shape[1] - 1)
        # with pixel masking, it should fit
        assert chi < 3
