from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import functools

from scipy import sparse
from patsy import dmatrix
import lightkurve as lk
import pyia


from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.time import Time
import astropy.units as u
from matplotlib import animation


def get_sources_vizier(ra, dec, radius=None, height=None, width=None, magnitude_limit=14, epoch=2454833):

    """
    Deprecated

    Use `get_sources`

    This is slow
    """
    if radius is not None:
        if not hasattr(radius, 'unit'):
            radius = Angle(radius, "deg")
    if height is not None:
        if not hasattr(height, 'unit'):
            height = Angle(height, "deg")
    if width is not None:
        if not hasattr(width, 'unit'):
            width = Angle(width, "deg")

    c1 = SkyCoord(ra, dec, frame='icrs', unit='deg')
    Vizier.ROW_LIMIT = -1

    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=radius, height=height, width=width)["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    year = ((epoch - Time(2457206.375, format='jd'))).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec
    return result

@functools.lru_cache()
def get_sources(c, magnitude_limit=18):
    """ Will find gaia sources using a TAP query, accounting for proper motions."""
    epoch = int(Time(c.time.mean() + 2454833, format='jd').isot[:4])
    magnitude_limit
    def get_gaia(ras, decs, rads, magnitude_limit):
        wheres = [f"""1=CONTAINS(
                      POINT('ICRS',ra,dec),
                      CIRCLE('ICRS',{ra},{dec},{rad}))""" for ra, dec, rad in zip(ras, decs, rads)]

        where = """\n\tOR """.join(wheres)
        gd = pyia.GaiaData.from_query(
f"""SELECT solution_id, designation, source_id, random_index, ref_epoch, coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax, parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error, ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig, astrometric_params_solved, astrometric_primary_flag, astrometric_weight_al, astrometric_pseudo_colour, astrometric_pseudo_colour_error, mean_varpi_factor_al, astrometric_matched_observations, visibility_periods_used, astrometric_sigma5d_max, frame_rotator_object_type, matched_observations, duplicated_source, phot_g_n_obs, phot_g_mean_flux, phot_g_mean_flux_error, phot_g_mean_flux_over_error, phot_g_mean_mag, phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_flux_over_error, phot_bp_mean_mag, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_flux_over_error, phot_rp_mean_mag, phot_bp_rp_excess_factor, phot_proc_mode, bp_rp, bp_g, g_rp, radial_velocity, radial_velocity_error, rv_nb_transits, rv_template_teff, rv_template_logg, rv_template_fe_h, phot_variable_flag, l, b, ecl_lon, ecl_lat, priam_flags, teff_val, teff_percentile_lower, teff_percentile_upper, a_g_val, a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val, e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags, radius_val, radius_percentile_lower, radius_percentile_upper, lum_val, lum_percentile_lower, lum_percentile_upper, datalink_url, epoch_photometry_url, ra as ra_gaia, dec as dec_gaia FROM (
     SELECT *,
     EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, 2020) AS prop
     FROM gaiadr2.gaia_source
     WHERE {where}
    )  AS subquery
    WHERE phot_g_mean_mag<={magnitude_limit}

""")
        return gd

    ras, decs, rads = [], [], []
    for l in np.unique(c.unw[0]):
        ra1 = c.ra[c.unw[0] == l]
        dec1 = c.dec[c.unw[0] == l]
        ras.append(ra1.mean())
        decs.append(dec1.mean())
        rads.append(np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max()/2 + (u.arcsecond * 6).to(u.deg).value)
    return get_gaia(ras, decs, rads, magnitude_limit)



def _make_A(phi, r):
    phi_spline = wrapped_spline(phi, order=3, nknots=8).T


    r_spline = np.asarray(dmatrix('bs(x, knots=knots, degree=3, include_intercept=True)', {'x':r, 'knots':np.linspace(0.25, 3.25, 3)}))
#    r_spline = lk.designmatrix.create_spline_matrix(r, knots=list(np.linspace(0.25, 3.25, 3)), degree=3).append_constant().X
    r_spline = np.hstack([r_spline[:, 1:], r_spline[:, :1]])
    A = np.hstack([(np.atleast_2d(phi_spline[:, idx]).T * r_spline[:, 3:]) for idx in range(len(phi_spline.T))])
    A = np.hstack([A, r_spline[:, :3]])

#    r_spline = np.vstack([r**0, r**1]).T
#    A = np.hstack([(np.atleast_2d(phi_spline[:, idx]).T * r_spline) for idx in range(len(phi_spline.T))])

    return sparse.csr_matrix(A)

def wrapped_spline(input_vector, order=2, nknots=10):
    """ This took me forever. MUST BE BETWEEN -PI and PI"""

    if not ((input_vector > -np.pi) & (input_vector < np.pi)).all():
        raise ValueError('Must be between -pi and pi')
    x = np.copy(input_vector)
    x1 = np.hstack([x, x + np.pi*2])
    nt = (nknots * 2) + 1

    t = np.linspace(-np.pi, 3*np.pi, nt)
    dt = np.median(np.diff(t))
    # Zeroth order basis
    basis = np.asarray([((x1 >= t[idx]) & (x1 < t[idx + 1])).astype(float) for idx in range(len(t) - 1)])
    # Higher order basis
    for order in np.arange(1, 4):
        basis_1 = []
        for idx in range(len(t) - 1):
            a = ((x1 - t[idx])/(dt * order)) * basis[idx]

            if ((idx + order + 1)) < (nt - 1):
                b = (-(x1 - t[(idx + order + 1)])/(dt * order)) * basis[(idx + 1) % (nt - 1)]
            else:
                b = np.zeros(len(x1))
            basis_1.append(a + b)
        basis = np.vstack(basis_1)

    folded_basis = np.copy(basis)[:nt//2, :len(x)]
    for idx in np.arange(-order, 0):
        folded_basis[idx, :] += np.copy(basis)[nt//2 + idx, len(x):]
    return folded_basis


def movie(dat, title='', out='out.mp4', scale='linear', facecolor='red', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data = np.log10(np.copy(dat))
    else:
        data = dat
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.set_facecolor(facecolor)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanpercentile(data, 75)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(data, 5)
    im1 = ax.imshow(data[0], origin='bottom', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15)
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.ax.tick_params(labelsize=10)
    if scale == 'log':
        cbar1.set_label('log10(e$^-$s$^-1$)', fontsize=10)
    else:
        cbar1.set_label('e$^-$s$^-1$', fontsize=10)

    def animate(i):
        im1.set_array(data[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)
