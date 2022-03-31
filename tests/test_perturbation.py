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

    assert p.matrix.shape == p.vectors.shape
    assert sparse.issparse(p.matrix)

    y, ye = np.random.normal(1, 0.01, size=200), np.ones(200) * 0.01
    w = p.fit(y, ye)
    assert w.shape[0] == p.shape[1]
    assert np.isfinite(w).all()
    model = p.dot(w)
    assert np.sum((y - model) ** 2 / (ye ** 2)) / (200) < 1.5

    # Fit multiple vectors
    y, ye = np.random.normal(1, 0.01, size=(200, 30)), np.ones(200) * 0.01
    w = p.fit(y, ye)
    assert w.shape == (p.shape[1], 30)
    assert np.isfinite(w).all()
    model = p.dot(w)
    assert np.all(
        np.sum((y - model) ** 2 / (ye[:, None] ** 2), axis=0) / (200 - 1) < 1.5
    )

    # Fit multiple vectors and multiple errors
    y, ye = np.random.normal(1, 0.01, size=(200, 30)), np.ones((200, 30)) * 0.01
    w = p.fit(y, ye)
    assert w.shape == (p.shape[1], 30)
    assert np.isfinite(w).all()
    model = p.dot(w)
    assert np.all(np.sum((y - model) ** 2 / (ye ** 2), axis=0) / (200 - 1) < 1.5)

    y, ye = np.random.normal(1, 0.01, size=200), np.ones(200) * 0.01
    assert len(p.bin_vectors(y)) == len(y)
    p_low = p.to_low_res(resolution=20)
    assert len(p_low.bin_vectors(y)) == 12
    assert p_low.bin_vectors(y[:, None]).shape == (12, 1)
    with pytest.raises(ValueError):
        p_low.bin_vectors(y[:-4])

    w = p_low.fit(p_low.bin_vectors(y))
    model = p.dot(w)
    assert model.shape[0] == 200


def test_perturbation_matrix3d():
    time = np.arange(0, 10, 1)
    # 13 x 13 pixels, evenly spaced in x and y
    dx, dy = np.mgrid[:13, :13] - 6 + 0.01
    dx, dy = dx.ravel(), dy.ravel()

    # ntime x npixels
    flux = (
        np.random.normal(0, 0.01, size=(10, 169))
        + 2 * dx[None, :] ** 2 * 3 * dy[None, :]
    )
    flux_err = np.ones((10, 169)) * 0.01

    p3 = PerturbationMatrix3D(time=time, dx=dx, dy=dy, nknots=4, radius=5)
    assert p3.cartesian_matrix.shape == (169, 81)
    assert p3.vectors.shape == (10, 4)
    assert p3.matrix.shape == (1690, 324)
    assert p3.shape == (1690, 324)
    w = p3.fit(flux.ravel(), flux_err.ravel())
    assert w.shape[0] == 324
    model = p3.dot(w).reshape(flux.shape)
    chi = np.sum((flux - model) ** 2 / (flux_err ** 2)) / (
        p3.shape[0] - p3.shape[1] - 1
    )
    assert chi < 1.5

    time = np.arange(0, 100, 1)
    flux = (
        np.random.normal(0, 0.01, size=(100, 169))
        + 2 * dx[None, :] ** 2 * 3 * dy[None, :]
    )
    flux_err = np.ones((100, 169)) * 0.01

    p3 = PerturbationMatrix3D(time=time, dx=dx, dy=dy, nknots=4, radius=5)
    assert p3.to_low_res(resolution=20).shape == (169 * 6, 81 * 4)
    assert p3.to_low_res(resolution=20).vectors.shape == (6, 4)

    p3l = p3.to_low_res(resolution=10)
    w = p3l.fit(p3l.bin_vectors(flux).ravel(), p3l.bin_vectors(flux_err).ravel())
    model = p3.dot(w).reshape(flux.shape)
    chi = np.sum((flux - model) ** 2 / (flux_err ** 2)) / (
        p3.shape[0] - p3.shape[1] - 1
    )
    assert chi < 1.5
