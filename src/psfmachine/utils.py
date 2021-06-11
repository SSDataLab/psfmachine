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
