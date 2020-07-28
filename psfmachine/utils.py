from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from scipy import sparse

from patsy import dmatrix

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.time import Time


def get_sources(ra, dec, pixel_radius, pix_scale=4.0, magnitude_limit=14, epoch=2454833):
    c1 = SkyCoord(ra, dec, frame='icrs', unit='deg')
    Vizier.ROW_LIMIT = -1

    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(pixel_radius * pix_scale, "arcsec"))["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    year = ((epoch - Time(2457206.375, format='jd'))).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.deg).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec
    return result

def _make_A(phi, r):
    phi_spline = wrapped_spline(phi, order=3, nknots=10).T
    r_spline = np.asarray(dmatrix('bs(x, knots=knots, degree=3, include_intercept=True)', {'x':r, 'knots':np.arange(0.25, 3, 0.5)}))
    r_spline = np.hstack([r_spline[:, 1:], r_spline[:, :1]])
    #r_spline = np.vstack([np.ones(len(r)), r, ])
    A = np.hstack([(np.atleast_2d(phi_spline[:, idx]).T * r_spline[:, 3:]) for idx in range(len(phi_spline.T))])
    A = np.hstack([A, r_spline[:, :3]])
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
