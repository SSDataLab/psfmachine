""" Collection of utility functions"""

import numpy as np
import pandas as pd
import functools

from scipy import sparse
from patsy import dmatrix
import pyia


@functools.lru_cache()
def get_gaia_sources(ras, decs, rads, magnitude_limit=18, epoch=2020, dr=2):
    """
    Will find gaia sources using a TAP query, accounting for proper motions.

    Inputs have be hashable, e.g. tuples

    Parameters
    ----------
    ras : tuple
        Tuple with right ascension coordinates to be queried
        shape nsources
    decs : tuple
        Tuple with declination coordinates to be queried
        shape nsources
    rads : tuple
        Tuple with radius query
        shape nsources
    magnitude_limit : int
        Limiting magnitued for query

    Returns
    -------
    Pandas DatFrame with number of result sources (rows) and Gaia columns

    """
    if not hasattr(ras, "__iter__"):
        ras = [ras]
    if not hasattr(decs, "__iter__"):
        decs = [decs]
    if not hasattr(rads, "__iter__"):
        rads = [rads]
    wheres = [
        f"""1=CONTAINS(
                  POINT('ICRS',ra,dec),
                  CIRCLE('ICRS',{ra},{dec},{rad}))"""
        for ra, dec, rad in zip(ras, decs, rads)
    ]

    where = """\n\tOR """.join(wheres)
    if dr == 2:
        # CH: We don't need a lot of these columns we could greatly reduce it
        gd = pyia.GaiaData.from_query(
            f"""SELECT designation,
            coord1(prop) AS ra, coord2(prop) AS dec, parallax,
            parallax_error, pmra, pmdec,
            phot_g_mean_flux,
            phot_g_mean_mag,
            phot_bp_mean_mag,
            phot_rp_mean_mag FROM (
     SELECT *,
     EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
     FROM gaiadr2.gaia_source
     WHERE {where}
    )  AS subquery
    WHERE phot_g_mean_mag<={magnitude_limit}

    """
        )
    elif dr == 3:
        gd = pyia.GaiaData.from_query(
            f"""SELECT designation,
            coord1(prop) AS ra, coord2(prop) AS dec, parallax,
            parallax_error, pmra, pmdec,
            phot_g_mean_flux,
            phot_g_mean_mag,
            phot_bp_mean_mag,
            phot_rp_mean_mag FROM (
             SELECT *,
             EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
             FROM gaiaedr3.gaia_source
             WHERE {where}
            )  AS subquery
            WHERE phot_g_mean_mag<={magnitude_limit}
            """
        )
    else:
        raise ValueError("Please pass a valid data release")
    return gd.data.to_pandas()


def do_tiled_query(ra, dec, ngrid=(5, 5), magnitude_limit=18, epoch=2020, dr=3):
    """
    Find the centers and radius of tiled queries when the sky area is large.
    This function divides the data into `ngrid` tiles and compute the ra, dec
    coordinates for each tile as well as its radius.
    This is meant to be used with dense data, e.g. FFI or cluster fields, and it is not
    optimized for sparse data, e.g. TPF stacks. For the latter use
    `psfmachine.tpf._get_coord_and_query_gaia()`.

    Parameters
    ----------
    ra : numpy.ndarray
        Data array with values of Right Ascension. Array can be 2D image or flatten.
    dec : numpy.ndarray
        Data array with values of Declination. Array can be 2D image or flatten.
    ngrid : tuple
        Tuple with number of bins in each axis. Default is (5, 5).
    magnitude_limit : int
        Limiting magnitude for query
    epoch : float
        Year of the observation (Julian year) used for proper motion correction.
    dr : int
        Gaia Data Release to be used, DR2 or EDR3. Default is EDR3.

    Returns
    -------
    sources : pandas.DatFrame
        Pandas DatFrame with number of result sources (rows) and Gaia columns
    """
    # find edges of the bins
    ra_edges = np.histogram_bin_edges(ra, ngrid[0])
    dec_edges = np.histogram_bin_edges(dec, ngrid[1])
    sources = []
    # iterate over 2d bins
    for idx in range(1, len(ra_edges)):
        for jdx in range(1, len(dec_edges)):
            # check if image data fall in the bin
            _in = (
                (ra_edges[idx - 1] <= ra)
                & (ra <= ra_edges[idx])
                & (dec_edges[jdx - 1] <= dec)
                & (dec <= dec_edges[jdx])
            )
            if not _in.any():
                continue
            # get the center coord of the query and radius to 7th decimal precision
            # (3 miliarcsec) to avoid not catching get_gaia_sources() due to
            # floating point error.
            ra_in = ra[_in]
            dec_in = dec[_in]
            # we use 50th percentile to get the centers and avoid 360-0 boundary
            ra_q = np.round(np.percentile(ra_in, 50), decimals=7)
            dec_q = np.round(np.percentile(dec_in, 50), decimals=7)
            # HARDCODED: +10/3600 to add a 10 arcsec to search radius, this is to get
            # sources off sensor up to 10" distance from sensor border.
            rad_q = np.round(
                np.hypot(ra_in - ra_q, dec_in - dec_q).max() + 10 / 3600, decimals=7
            )
            # query gaia with ra, dec, rad, epoch
            result = get_gaia_sources(
                tuple([ra_q]),
                tuple([dec_q]),
                tuple([rad_q]),
                magnitude_limit=magnitude_limit,
                epoch=epoch,
                dr=dr,
            )
            sources.append(result)
    #  concat results and remove duplicated sources
    sources = pd.concat(sources, axis=0).drop_duplicates(subset=["designation"])
    return sources.reset_index(drop=True)


def _make_A_polar(phi, r, cut_r=6, rmin=1, rmax=18, n_r_knots=12, n_phi_knots=15):
    # create the spline bases for radius and angle
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=n_phi_knots).T)
    r_knots = np.linspace(rmin ** 0.5, rmax ** 0.5, n_r_knots) ** 2
    cut_r_int = np.where(r_knots <= cut_r)[0].max()
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(np.hstack([r, rmin, rmax])), "knots": r_knots},
            )
        )
    )[:-2]

    # build full desing matrix
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    # find and remove the angle dependency for all basis for radius < 6
    cut = np.arange(0, phi_spline.shape[1] * cut_r_int)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r_int], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


def _make_A_cartesian(x, y, n_knots=10, radius=3.0):
    x_knots = np.linspace(-radius, radius, n_knots)
    x_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(x), "knots": x_knots},
            )
        )
    )
    y_knots = np.linspace(-radius, radius, n_knots)
    y_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(y), "knots": y_knots},
            )
        )
    )
    X = sparse.hstack(
        [x_spline.multiply(y_spline[:, idx]) for idx in range(y_spline.shape[1])],
        format="csr",
    )
    return X


def wrapped_spline(input_vector, order=2, nknots=10):
    """
    Creates a vector of folded-spline basis according to the input data. This is meant
    to be used to build the basis vectors for periodic data, like the angle in polar
    coordinates.

    Parameters
    ----------
    input_vector : numpy.ndarray
        Input data to create basis, angle values MUST BE BETWEEN -PI and PI.
    order : int
        Order of the spline basis
    nknots : int
         Number of knots for the splines

    Returns
    -------
    folded_basis : numpy.ndarray
        Array of folded-spline basis
    """

    if not ((input_vector >= -np.pi) & (input_vector <= np.pi)).all():
        raise ValueError("Must be between -pi and pi")
    x = np.copy(input_vector)
    x1 = np.hstack([x, x + np.pi * 2])
    nt = (nknots * 2) + 1

    t = np.linspace(-np.pi, 3 * np.pi, nt)
    dt = np.median(np.diff(t))
    # Zeroth order basis
    basis = np.asarray(
        [
            ((x1 >= t[idx]) & (x1 < t[idx + 1])).astype(float)
            for idx in range(len(t) - 1)
        ]
    )
    # Higher order basis
    for order in np.arange(1, 4):
        basis_1 = []
        for idx in range(len(t) - 1):
            a = ((x1 - t[idx]) / (dt * order)) * basis[idx]

            if ((idx + order + 1)) < (nt - 1):
                b = (-(x1 - t[(idx + order + 1)]) / (dt * order)) * basis[
                    (idx + 1) % (nt - 1)
                ]
            else:
                b = np.zeros(len(x1))
            basis_1.append(a + b)
        basis = np.vstack(basis_1)

    folded_basis = np.copy(basis)[: nt // 2, : len(x)]
    for idx in np.arange(-order, 0):
        folded_basis[idx, :] += np.copy(basis)[nt // 2 + idx, len(x) :]
    return folded_basis


def solve_linear_model(
    A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None, errors=False
):
    """
            Solves a linear model with design matrix A and observations y:
                Aw = y
            return the solutions w for the system assuming Gaussian priors.
            Alternatively the observation errors, priors, and a boolean mask for the
            observations (row axis) can be provided.

            Adapted from Luger, Foreman-Mackey & Hogg, 2017
            (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)

            Parameters
            ----------
            A: numpy ndarray or scipy sparce csr matrix
                Desging matrix with solution basis
                shape n_observations x n_basis
            y: numpy ndarray
                Observations
                shape n_observations
            y_err: numpy ndarray, optional
                Observation errors
                shape n_observations
            prior_mu: float, optional
                Mean of Gaussian prior values for the weights (w)
            prior_sigma: float, optional
                Standard deviation of Gaussian prior values for the weights (w)
            k: boolean, numpy ndarray, optional
                Mask that sets the observations to be used to solve the system
                shape n_observations
            errors: boolean
                Whether to return error estimates of the best fitting weights

            Returns
            -------
            w: numpy ndarray
                Array with the estimations for the weights
                shape n_basis
            werrs: numpy ndarray
                Array with the error estimations for the weights, returned if `error`
    is True
                shape n_basis
    """
    if k is None:
        k = np.ones(len(y), dtype=bool)

    if y_err is not None:
        sigma_w_inv = A[k].T.dot(A[k].multiply(1 / y_err[k, None] ** 2))
        B = A[k].T.dot((y[k] / y_err[k] ** 2))
    else:
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(y[k])

    if prior_mu is not None and prior_sigma is not None:
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        B += prior_mu / prior_sigma ** 2

    if isinstance(sigma_w_inv, (sparse.csr_matrix, sparse.csc_matrix)):
        sigma_w_inv = sigma_w_inv.toarray()
    if isinstance(sigma_w_inv, np.matrix):
        sigma_w_inv = np.asarray(sigma_w_inv)

    w = np.linalg.solve(sigma_w_inv, B)
    if errors is True:
        w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
        return w, w_err
    return w


def sparse_lessthan(arr, limit):
    """
    Compute less than operation on sparse array by evaluating only non-zero values
    and reconstructing the sparse array. This function return a sparse array, which is
    crutial to keep operating large matrices.

    Notes: when doing `x < a` for a sparse array `x` and `a > 0` it effectively compares
    all zero and non-zero values. Then we get a dense boolean array with `True` where
    the condition is met but also `True` where the sparse array was zero.
    To avoid this we evaluate the condition only for non-zero values in the sparse
    array and later reconstruct the sparse array with the right shape and content.
    When `x` is a [N * M] matrix and `a` is [N] array, and we want to evaluate the
    condition per row, we need to iterate over rows to perform the evaluation and then
    reconstruct the masked sparse array.

    Parameters
    ----------
    arr : scipy.sparse
        Sparse array to be masked, is a 2D matrix.
    limit : float, numpy.array
        Upper limit to evaluate less than. If float will do `arr < limit`. If array,
        shape has to match first dimension of `arr` to do `arr < limi[:, None]`` and
        evaluate the condition per row.

    Returns
    -------
    masked_arr : scipy.sparse.csr_matrix
        Sparse array after less than evaluation.
    """
    nonz_idx = arr.nonzero()
    # apply condition for each row
    if isinstance(limit, np.ndarray) and limit.shape[0] == arr.shape[0]:
        mask = [arr[s].data < limit[s] for s in set(nonz_idx[0])]
        # flatten mask
        mask = [x for sub in mask for x in sub]
    else:
        mask = arr.data < limit
    # reconstruct sparse array
    masked_arr = sparse.csr_matrix(
        (arr.data[mask], (nonz_idx[0][mask], nonz_idx[1][mask])),
        shape=arr.shape,
    ).astype(bool)
    return masked_arr
