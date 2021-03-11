""" Collection of utility functions"""

import numpy as np
import functools

from scipy import sparse
from patsy import dmatrix
import pyia
from tqdm import tqdm

import astropy.units as u
from astropy.time import Time


@functools.lru_cache()
def get_sources(c, magnitude_limit=18):
    """ Will find gaia sources using a TAP query, accounting for proper motions."""
    epoch = int(Time(c.time.mean() + 2454833, format="jd").isot[:4])
    magnitude_limit

    def get_gaia(ras, decs, rads, magnitude_limit):
        wheres = [
            f"""1=CONTAINS(
                      POINT('ICRS',ra,dec),
                      CIRCLE('ICRS',{ra},{dec},{rad}))"""
            for ra, dec, rad in zip(ras, decs, rads)
        ]

        where = """\n\tOR """.join(wheres)
        gd = pyia.GaiaData.from_query(
            f"""SELECT solution_id, designation, source_id, random_index, ref_epoch,
            coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
            parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error,
            ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
            dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr,
            parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al,
            astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al,
            astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise,
            astrometric_excess_noise_sig, astrometric_params_solved,
            astrometric_primary_flag, astrometric_weight_al, astrometric_pseudo_colour,
            astrometric_pseudo_colour_error, mean_varpi_factor_al,
            astrometric_matched_observations, visibility_periods_used,
            astrometric_sigma5d_max, frame_rotator_object_type, matched_observations,
            duplicated_source, phot_g_n_obs, phot_g_mean_flux, phot_g_mean_flux_error,
            phot_g_mean_flux_over_error, phot_g_mean_mag, phot_bp_n_obs,
            phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_flux_over_error,
            phot_bp_mean_mag, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error,
            phot_rp_mean_flux_over_error, phot_rp_mean_mag, phot_bp_rp_excess_factor,
            phot_proc_mode, bp_rp, bp_g, g_rp, radial_velocity, radial_velocity_error,
            rv_nb_transits, rv_template_teff, rv_template_logg, rv_template_fe_h,
            phot_variable_flag, l, b, ecl_lon, ecl_lat, priam_flags, teff_val,
            teff_percentile_lower, teff_percentile_upper, a_g_val,
            a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val,
            e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags,
            radius_val, radius_percentile_lower, radius_percentile_upper, lum_val,
            lum_percentile_lower, lum_percentile_upper, datalink_url,
            epoch_photometry_url, ra as ra_gaia, dec as dec_gaia FROM (
     SELECT *,
     EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
     FROM gaiadr2.gaia_source
     WHERE {where}
    )  AS subquery
    WHERE phot_g_mean_mag<={magnitude_limit}
"""
        )
        return gd

    ras, decs, rads = [], [], []
    # for every TPF get the r,dec for that tpfs
    for l in np.unique(c.unw[0]):
        ra1 = c.ra[c.unw[0] == l]
        dec1 = c.dec[c.unw[0] == l]
        ras.append(ra1.mean())
        decs.append(dec1.mean())
        rads.append(
            np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max() / 2
            + (u.arcsecond * 6).to(u.deg).value
        )
    return get_gaia(ras, decs, rads, magnitude_limit)


# @functools.lru_cache()
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
            f"""SELECT solution_id, designation, source_id, random_index, ref_epoch,
            coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
            parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error,
            ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr,
            dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr,
            pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac,
            astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al,
            astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig,
            astrometric_params_solved, astrometric_primary_flag, astrometric_weight_al,
            astrometric_pseudo_colour, astrometric_pseudo_colour_error,
            mean_varpi_factor_al, astrometric_matched_observations,
            visibility_periods_used, astrometric_sigma5d_max, frame_rotator_object_type,
            matched_observations, duplicated_source, phot_g_n_obs, phot_g_mean_flux,
            phot_g_mean_flux_error, phot_g_mean_flux_over_error, phot_g_mean_mag,
            phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error,
            phot_bp_mean_flux_over_error, phot_bp_mean_mag, phot_rp_n_obs,
            phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_flux_over_error,
            phot_rp_mean_mag, phot_bp_rp_excess_factor, phot_proc_mode, bp_rp, bp_g, g_rp,
            radial_velocity, radial_velocity_error, rv_nb_transits, rv_template_teff,
            rv_template_logg, rv_template_fe_h, phot_variable_flag, l, b, ecl_lon, ecl_lat,
            priam_flags, teff_val, teff_percentile_lower, teff_percentile_upper, a_g_val,
            a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val,
            e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags,
            radius_val, radius_percentile_lower, radius_percentile_upper, lum_val,
            lum_percentile_lower, lum_percentile_upper, datalink_url, epoch_photometry_url,
            ra as ra_gaia, dec as dec_gaia FROM (
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
                    coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
                    parallax_error,
                    ruwe, phot_g_n_obs, phot_g_mean_flux,
                    phot_g_mean_flux_error, phot_g_mean_mag,
                    phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_mag, phot_rp_n_obs,
                    phot_rp_mean_flux, phot_rp_mean_flux_error,
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


def _make_A(phi, r, cut_r=5):
    """ Make spline design matrix in polar coordinates """
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=10).T)
    r_knots = np.linspace(0.25 ** 0.5, 5 ** 0.5, 8) ** 2
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(r), "knots": r_knots},
            )
        )
    )
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    cut = np.arange(phi_spline.shape[1] * 1, phi_spline.shape[1] * cut_r)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


def _make_A_wcs(phi, r, cut_r=6):
    """
    Makes a design matrix of b-spline basis in polar coordinates. For the azimutal
    angle axis, the splines are 2*pi periodic.

    The design matrix is build with all combinatios of radius and angle spline basis,
    with the exception of small angles (r < 6) where the angle dependency is removed
    to avoid artifacts at the center of the PSF.

    Parameters
    ----------
    phi : numpy.ndarray
        Azimutal angle used to build the design matrix
    r : numpy.ndarray
        Radial distances used to build the design matrix
    cut_r : int
        Radius at which the angle dependecy is removed from the basis

    Returns
    -------
    X1 : sparse.csr_matrix
        A sparse design matrix
    """
    # create the spline bases for radius and angle
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=10).T)
    r_knots = np.linspace(1 ** 0.5, 18 ** 0.5, 8) ** 2
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(r), "knots": r_knots},
            )
        )
    )
    # build full desing matrix
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    # find and remove the angle dependency for all basis for radius < 6
    cut = np.arange(phi_spline.shape[1] * 0, phi_spline.shape[1] * cut_r)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


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

    if not ((input_vector > -np.pi) & (input_vector < np.pi)).all():
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
