"""
Defines the main Machine object that fit a mean PSF model to sources
"""
import logging

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import sparse
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, sigma_clipped_stats

from .utils import get_gaia_sources, _make_A_wcs, _make_A_cartesian

log = logging.getLogger(__name__)

__all__ = ["Machine"]


# Machine Class
class Machine(object):
    """
    Class for calculating fast PSF photometry on a collection of images and
    a list of in image sources. The moethod fits a common mean PSF model for
    selected clean sources which is later fitted to al sources listed in `sources`.
    This method solves a linear model to assuming Gaussian priors on the weight of
    each linear components as explained by Luger, Foreman-Mackey & Hogg, 2017
    (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)

    This class is meant to be used with Kepler/K2 TPFs files, with a potential
    for general use if the right arguments are provided.

    The PSF model is build in polar coordinates.

    Attributes
    ----------
    sources : pandas.DataFrame
        Catalog of sources in the images
    source_flux_estiamtes : np.ndarray
        nsources

    time : numpy.ndarray
        Time values in JD
    flux : numpy.ndarray
        Flux values at each pixels and times in units of electrons / sec
    flux_err : numpy.ndarray
        Flux error values at each pixels and times in units of electrons / sec
    ra : numpy.ndarray
        Right Ascension coordinate of each pixel
    dec : numpy.ndarray
        Declination coordinate of each pixel
    sources : pandas.DataFram
        Data Frame with source present in the images
    limit_radius : numpy.ndarray
        Radius limit in arcsecs to select stars to be used for PSF modeling
    limit_flux : numpy.ndarray
        Flux limit in electrons / second to select stars to be used for PSF modeling
    nsources : int
        Number of sources to be extracted
    nt : int
        Number of onservations in the time series (aka number of cadences)
    npixels : int
        Total number of pixels with flux measurements
    source_flux_estimates : numpy.ndarray
        First estimation of pixel fluxes assuming values given by the sources catalog
        (e.g. Gaia phot_g_mean_flux)
    dra : numpy.ndarray
        Distance in right ascension between pixel and source coordinates, units of
        degrees
    ddec : numpy.ndarray
        Distance in declination between pixel and source coordinates, units of
        degrees
    r : numpy.ndarray
        Radial distance between pixel and source coordinates (polar coordinates),
        in units of arcseconds
    phi : numpy.ndarray
        Angle between pixel and source coordinates (polar coordinates),
        in units of radians
    source_mask : scipy.sparce.csr_matrix
        Sparce mask matrix with pixels that contains flux from sources
    uncontaminated_source_mask : scipy.sparce.csr_matrix
        Sparce mask matrix with selected uncontaminated pixels per source to be used to
        build the PSF model
    ra_centroid : numpy.ndarray
        Right ascension centroid values per source
    dec_centroid : numpy.ndarray
        Declination centroid values per source
    mean_model : scipy.sparce.csr_matrix
        Mean PSF model values per pixel used for PSF photometry
    psf_flux : numpy.ndarray
        PSF photometry for each source listed in `sources` as function of time in
        units of electrons / s
    psf_flux_err : numpy.ndarray
        PSF photometry uncertainties for each source listed in `sources` as function
        of time in units of electrons / s
    """

    def __init__(
        self,
        time,
        flux,
        flux_err,
        ra,
        dec,
        sources,
        column,
        row,
        limit_radius=24.0,
        pix2obs=None,
        plot=False,
    ):
        """
        Class constructor. This constructur will compute the basic atributes that will
        will be used by the object methods to perform PSF estimation and fitting.

        Parameters
        ----------
        time : numpy.ndarray
            Time in JD
            shape ntimes
        flux : numpy.ndarray
            Flux at each pixel at each time
            shape ntimes x npixel
        flux_err : numpy.ndarray
            Flux error at each pixel at each time
            shape ntimes x npixel
        ra : numpy.ndarray
            RA pixel position averaged in time
            shape npixel
        dec : numpy.ndarray
            Dec pixel position averaged in time
            shape npixel
        sources : pandas.DataFrame
            Catalog with soruces on the pixel data (e.g. Gaia catalog)
            shape nsources x nfeatures
        limit_radius : float
            Radius limit in arcsecs to select stars to be used for PSF modeling

        Exaples
        -------
        import lightkurve as lk
        from psfmachine.psfmachine import Machine
        # download a collection of 100 TPFs around the target star
        tpfs = lk.search_targetpixelfile(target='KIC 11904151', radius=1000,
                                         limit=100, cadence='long',
                                         mission='Kepler', quarter=4).download_all()
        # create a machine object from TPF collection
        mac = Machine().from_tpfs(tpfs)
        # build PSF model using selected stars
        mac.build_model()
        # fit PSF model to all sources listed in `sources`
        mac.fit_model()
        # access all sources light curves
        mac.time, mac.psf_flux
        """

        if not isinstance(sources, pd.DataFrame):
            raise TypeError("<sources> must be a of class Pandas Data Frame")

        # assigning initial attributes
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ra = ra
        self.dec = dec
        self.sources = sources
        self.column = column
        self.row = row
        self.limit_radius = limit_radius * u.arcsecond
        self.limit_flux = 1e4
        # Convert between pixel and the original observation
        self.pix2obs = pix2obs
        self.plot = plot

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # gaia estimate flux values per pixel to be used as flux priors
        self.source_flux_estimates = np.array(
            [
                np.zeros(self.npixels) + sources.phot_g_mean_flux[idx]
                for idx in range(len(sources))
            ]
        )

        # The distance in ra & dec from each source to each pixel
        self.dra, self.ddec = np.asarray(
            [
                [ra - self.sources["ra"][idx], dec - self.sources["dec"][idx]]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # convertion to polar coordinates
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        self.phi = np.arctan2(self.ddec, self.dra)

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        self._get_source_mask(plot=self.plot)
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        self._get_uncontaminated_pixel_mask()

        # Get the centroids of the images as a function of time
        self._get_centroids()

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    @staticmethod
    def _solve_linear_model(
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
        A : numpy ndarray or scipy sparce csr matrix
            Desging matrix with solution basis
            shape n_observations x n_basis
        y : numpy ndarray
            Observations
            shape n_observations
        y_err : numpy ndarray, optional
            Observation errors
            shape n_observations
        prior_mu : float, optional
            Mean of Gaussian prior values for the weights (w)
        prior_sigma : float, optional
            Standard deviation of Gaussian prior values for the weights (w)
        k : boolean, numpy ndarray, optional
            Mask that sets the observations to be used to solve the system
            shape n_observations

        Returns
        -------
        w : numpy ndarray
            Array with the estimations for the weights
            shape n_basis
        werrs : numpy ndarray
            Array with the error estimations for the weights, returned if y_err is
            provided
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

        if type(sigma_w_inv) == sparse.csr_matrix:
            sigma_w_inv = sigma_w_inv.toarray()

        if type(sigma_w_inv) == np.matrix:
            sigma_w_inv = np.asarray(sigma_w_inv)

        w = np.linalg.solve(sigma_w_inv, B)
        if errors is True:
            w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            return w, w_err
        return w

    def _get_source_mask(
        self,
        upper_radius_limit=20,
        lower_radius_limit=4.5,
        flux_cut_off=100,
        flux_limit=1e6,
        plot=False,
    ):

        mean_flux = np.nanmean(self.flux, axis=0)
        r = self.r

        temp_mask = (r.value < upper_radius_limit) & (
            self.source_flux_estimates < flux_limit
        )
        temp_mask &= temp_mask.sum(axis=0) == 1

        f = np.log10((temp_mask.astype(float) * mean_flux))

        weights = (
            (self.flux_err ** 0.5).sum(axis=0) ** 0.5 / self.flux.shape[0]
        ) * temp_mask

        mf = np.log10(self.source_flux_estimates[temp_mask])

        # Model is polynomial in r and log of the flux estimate.
        A = np.vstack(
            [
                r.value[temp_mask] ** 0,
                r.value[temp_mask],
                r.value[temp_mask] ** 2,
                r.value[temp_mask] ** 0 * mf,
                r.value[temp_mask] * mf,
                r.value[temp_mask] ** 2 * mf,
                r.value[temp_mask] ** 0 * mf ** 2,
                r.value[temp_mask] * mf ** 2,
                r.value[temp_mask] ** 2 * mf ** 2,
            ]
        ).T

        k = np.isfinite(f[temp_mask])
        for count in [0, 1, 2]:
            sigma_w_inv = A[k].T.dot(A[k] / weights[temp_mask][k, None] ** 2)
            B = A[k].T.dot(f[temp_mask][k] / weights[temp_mask][k] ** 2)
            w = np.linalg.solve(sigma_w_inv, B)
            res = np.ma.masked_array(f[temp_mask], ~k) - A.dot(w)
            k &= ~sigma_clip(res, sigma=3).mask

        test_f = np.linspace(
            np.log10(self.source_flux_estimates.min()),
            np.log10(self.source_flux_estimates.max()),
            100,
        )
        test_r = np.arange(lower_radius_limit, upper_radius_limit, 0.25)
        test_r2, test_f2 = np.meshgrid(test_r, test_f)

        test_val = (
            np.vstack(
                [
                    test_r2.ravel() ** 0,
                    test_r2.ravel(),
                    test_r2.ravel() ** 2,
                    test_r2.ravel() ** 0 * test_f2.ravel(),
                    test_r2.ravel() * test_f2.ravel(),
                    test_r2.ravel() ** 2 * test_f2.ravel(),
                    test_r2.ravel() ** 0 * test_f2.ravel() ** 2,
                    test_r2.ravel() * test_f2.ravel() ** 2,
                    test_r2.ravel() ** 2 * test_f2.ravel() ** 2,
                ]
            )
            .T.dot(w)
            .reshape(test_r2.shape)
        )

        l = np.zeros(len(test_f)) * np.nan
        for idx in range(len(test_f)):
            loc = np.where(10 ** test_val[idx] < flux_cut_off)[0]
            if len(loc) > 0:
                l[idx] = test_r[loc[0]]
        ok = np.isfinite(l)
        source_radius_limit = np.polyval(
            np.polyfit(test_f[ok], l[ok], 2), np.log10(self.source_flux_estimates[:, 0])
        )
        source_radius_limit[
            source_radius_limit > upper_radius_limit
        ] = upper_radius_limit
        source_radius_limit[
            source_radius_limit < lower_radius_limit
        ] = lower_radius_limit

        self.radius = source_radius_limit + 1
        self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(8, 3), facecolor="white")

            ax[0].scatter(
                r.value[temp_mask][k], f[temp_mask][k], s=0.4, c="k", label="Data"
            )
            ax[0].scatter(
                r.value[temp_mask][k], A[k].dot(w), c="r", s=0.4, label="Model"
            )
            ax[0].set(
                xlabel=("Radius from Source [arcsec]"),
                ylabel=("log$_{10}$ Kepler Flux"),
            )
            ax[0].legend(frameon=True)

            im = ax[1].pcolormesh(
                test_f2,
                test_r2,
                10 ** test_val,
                vmin=0,
                vmax=flux_cut_off * 2,
                cmap="viridis",
            )
            line = np.polyval(np.polyfit(test_f[ok], l[ok], 2), test_f)
            line[line > upper_radius_limit] = upper_radius_limit
            line[line < lower_radius_limit] = lower_radius_limit
            ax[1].plot(test_f, line, color="r", label="Best Fit PSF Edge")
            ax[1].set_ylim(lower_radius_limit, upper_radius_limit)
            ax[1].legend(frameon=True)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("PSF Flux [$e^-s^{-1}$]")

            ax[1].set(
                ylabel=("Radius from Source [arcsecond]"),
                xlabel=("log$_{10}$ Gaia Source Flux"),
            )
            return fig
        return

    #
    # def _get_source_mask(self):
    #     """
    #     Creates a mask of shape nsources x number of pixels, one where flux from a
    #     source exists, this mask is used to select which pixels will be used for
    #     PSF photometry.
    #
    #     First finds a linear relation between flux and radius from the source center
    #     to then calculate the radius at which the 97% of the flux is enclosed.
    #
    #     The created zero-ones-mask is a sparce csr_matrix, which helps to speed-up
    #     following computation, especially for large number of sources and pixels.
    #     """
    #     # mask pixels by Gaia flux and maximum distance
    #     self.mean_flux = np.nanmean(self.flux, axis=0)
    #     temp_mask = (self.r.to("arcsec") < self.limit_radius) & (
    #         self.source_flux_estimates > self.limit_flux
    #     )
    #     temp_mask &= temp_mask.sum(axis=0) == 1
    #
    #     # estimates the PSF edges as a function of flux
    #     f = np.log10((temp_mask.astype(float) * self.mean_flux))[temp_mask]
    #     A = np.vstack(
    #         [
    #             self.r[temp_mask].to("arcsec").value ** 0,
    #             self.r[temp_mask].to("arcsec").value,
    #             np.log10(self.source_flux_estimates[temp_mask]),
    #         ]
    #     ).T
    #     k = np.isfinite(f)
    #     w = self._solve_linear_model(A, f, k=k)
    #
    #     test_gaia = np.linspace(
    #         np.log10(np.nanmin(self.source_flux_estimates)),
    #         np.log10(np.nanmax(self.source_flux_estimates)),
    #         50,
    #     )
    #     # calculate radius of PSF edges from linear solution
    #     test_r = np.arange(0, 40, 1)
    #     radius_check = np.asarray(
    #         [
    #             np.vstack(
    #                 [[(np.ones(50) * v) ** idx for idx in range(2)], test_gaia]
    #             ).T.dot(w)
    #             for v in test_r
    #         ]
    #     )
    #
    #     # flux cap at which 97% of it is contained
    #     cut = np.percentile(np.abs(radius_check), 3) - np.min(np.abs(radius_check))
    #     x, y = np.asarray(np.meshgrid(test_gaia, test_r))[:, np.abs(radius_check) < cut]
    #     # calculate the radius at which the 97% is contained
    #     self.radius = np.polyval(
    #         np.polyfit(x, y, 5), np.log10(self.sources["phot_g_mean_flux"])
    #     )
    #     # cap the radius for faint and saturated sources
    #     # radius[np.log10(self.sources["phot_g_mean_flux"]) < 3] = 8.0
    #     # radius[np.log10(self.sources["phot_g_mean_flux"]) > 6.5] = 24.0
    #
    #     # mask all pixels with radial distance less than the source PSF edge.
    #     self.source_mask = sparse.csr_matrix(
    #         self.r.to("arcsec").value < self.radius[:, None]
    #     )
    #
    #     return

    def _get_uncontaminated_pixel_mask(self):
        """
        creates a mask of shape npixels (maybe by nt) where not saturated, not faint,
        and not contaminated. This mask is used to select pixels to build the PSF
        model.

        Pixel saturation defined by the limit where sensor response is linear,
        ~1.5e5 -e/s.
        Faint sources are filtered using the `sources` catalog, e.g.g
        Gaia phot_g_mean_flux < 10^3.
        Pixels with flux coming from multiple sources are flagged as contaminated.

        A pixel mask per sources is created combined all previus three masks, this is a
        sparce csr_matrix.

        """
        # find saturated pixels, the saturation cap per pixel is -e/s
        sat_pixel_mask = np.max(self.flux, axis=0) > 1.5e5

        # find pixels from faint Sources
        faint_sources = np.log10(self.sources["phot_g_mean_flux"]).values < 3

        # find pixels with flux from only one source
        one_source_pix = self.source_mask.sum(axis=0) == 1

        # combine non-saturated pixels and only-one-source pixels
        good_pixels = sparse.csr_matrix(~sat_pixel_mask).multiply(one_source_pix)

        # combine faint sources mask with good pixels
        self.uncontaminated_source_mask = (
            sparse.csr_matrix(~faint_sources[:, None])
            .dot(good_pixels)
            .multiply(self.source_mask)
        )

        # reduce to good pixels
        self.uncontaminated_pixel_mask = sparse.csr_matrix(
            self.uncontaminated_source_mask.sum(axis=0) > 0
        )

        return

    def _get_centroids(self):
        """
        Find the ra and dec centroid of the image, at each time.
        """
        # centroids are astropy quantities
        self.ra_centroid = np.zeros(self.nt)
        self.dec_centroid = np.zeros(self.nt)
        dra_m = self.source_mask.multiply(self.dra).data
        ddec_m = self.source_mask.multiply(self.ddec).data
        for t in range(self.nt):
            wgts = self.source_mask.multiply(self.flux[t]).data
            self.ra_centroid[t] = np.average(dra_m, weights=wgts)
            self.dec_centroid[t] = np.average(ddec_m, weights=wgts)
        del dra_m, ddec_m
        self.ra_centroid *= u.deg
        self.dec_centroid *= u.deg
        self.ra_centroid_avg = self.ra_centroid.mean()
        self.dec_centroid_avg = self.dec_centroid.mean()

        return

    def _time_bin(self, npoints=200):
        """Bin the data down in time

        Parameters
        ----------

        npoints: how many points in each time bin

        Returns
        -------

        time_original

        time_binned

        flux_binned_raw

        flux_binned

        flux_err_binned
        """
        splits = np.append(
            np.append(0, np.where(np.diff(self.time) > 0.1)[0]), len(self.time)
        )
        splits_a = splits[:-1] + 100
        splits_b = splits[1:]
        dsplits = (splits_b - splits_a) // npoints
        breaks = []
        for spdx in range(len(splits_a)):
            breaks.append(splits_a[spdx] + np.arange(0, dsplits[spdx] - 1) * npoints)
        breaks = np.hstack(breaks)

        # Time averaged
        tm = np.vstack(
            [t1.mean(axis=0) for t1 in np.array_split(self.time, breaks)]
        ).ravel()
        ta = (self.time - tm.mean()) / (tm.max() - tm.mean())

        ms = [
            np.in1d(np.arange(self.nt), i)
            for i in np.array_split(np.arange(self.nt), breaks)
        ]
        # Average Pixel values
        fm = np.asarray(
            [
                (
                    sparse.csr_matrix(self.uncontaminated_source_mask)
                    .multiply(self.flux[ms[tdx]].mean(axis=0))
                    .data
                )
                for tdx in range(len(ms))
            ]
        )
        fm_raw = np.asarray(
            [
                (
                    sparse.csr_matrix(self.uncontaminated_source_mask)
                    .multiply(self.flux[ms[tdx]].mean(axis=0))
                    .data
                )
                for tdx in range(len(ms))
            ]
        )
        fem = np.asarray(
            [
                (
                    sparse.csr_matrix(self.uncontaminated_source_mask)
                    .multiply((self.flux_err ** 2)[ms[tdx]].sum(axis=0) ** 0.5)
                    .data
                    / ms[tdx].sum()
                )
                for tdx in range(len(ms))
            ]
        )

        fem /= np.nanmean(fm, axis=0)
        fm /= np.nanmean(fm, axis=0)

        tm = ((tm - tm.mean()) / (tm.max() - tm.mean()))[:, None] * np.ones(fm.shape)

        return ta, tm, fm_raw, fm, fem

    def build_time_model(self, npoints=200, n_knots=10, plot=False):
        (
            time_original,
            time_binned,
            flux_binned_raw,
            flux_binned,
            flux_err_binned,
        ) = self._time_bin(npoints=npoints)

        self._time_model_knots = n_knots
        self._whitened_time = time_original
        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        A_c = _make_A_cartesian(dx, dy, n_knots=n_knots, radius=12)
        A2 = sparse.vstack([A_c] * time_binned.shape[0], format="csr")
        # Cartesian spline with time dependence
        A3 = sparse.hstack(
            [
                A2,
                A2.multiply(time_binned.ravel()[:, None]),
                A2.multiply(time_binned.ravel()[:, None] ** 2),
                A2.multiply(time_binned.ravel()[:, None] ** 3),
            ],
            format="csr",
        )

        # No saturated pixels
        k = (
            (flux_binned_raw < 1.4e5).any(axis=0)[None, :]
            * np.ones(flux_binned_raw.shape, bool)
        ).ravel()
        # No faint pixels
        k &= (
            (flux_binned_raw > 10).any(axis=0)[None, :]
            * np.ones(flux_binned_raw.shape, bool)
        ).ravel()
        # No huge variability
        k &= (
            (np.abs(flux_binned - 1) < 1).all(axis=0)[None, :]
            * np.ones(flux_binned.shape, bool)
        ).ravel()
        # No nans
        k &= np.isfinite(flux_binned.ravel()) & np.isfinite(flux_err_binned.ravel())
        prior_sigma = np.ones(A3.shape[1]) * 10
        prior_mu = np.zeros(A3.shape[1])

        for count in [0, 1, 2]:
            sigma_w_inv = A3[k].T.dot(A3[k])
            sigma_w_inv += np.diag(1 / prior_sigma ** 2)
            # Fit the flux - 1
            B = A3[k].T.dot((flux_binned.ravel() - 1)[k])
            B += prior_mu / (prior_sigma ** 2)
            velocity_aberration_w = np.linalg.solve(sigma_w_inv, B)
            res = flux_binned - A3.dot(velocity_aberration_w).reshape(flux_binned.shape)
            res = np.ma.masked_array(res, (~k).reshape(flux_binned.shape))
            bad_targets = sigma_clip(res, sigma=5).mask
            bad_targets = (
                np.ones(flux_binned.shape, bool) & bad_targets.any(axis=0)
            ).ravel()
            #    k &= ~sigma_clip(flux_binned.ravel() - A3.dot(velocity_aberration_w)).mask
            k &= ~bad_targets

        self.velocity_aberration_w = velocity_aberration_w
        if plot:
            model = A3.dot(velocity_aberration_w).reshape(flux_binned.shape) + 1
            fig, ax = plt.subplots(2, 2, figsize=(7, 6), facecolor="w")
            k1 = k.reshape(flux_binned.shape)[0]
            k2 = k.reshape(flux_binned.shape)[-1]
            im = ax[0, 0].scatter(
                dx[k1],
                dy[k1],
                c=flux_binned[0][k1],
                s=3,
                vmin=0.7,
                vmax=1.3,
                cmap="coolwarm",
            )
            ax[0, 1].scatter(
                dx[k2],
                dy[k2],
                c=flux_binned[-1][k2],
                s=3,
                vmin=0.7,
                vmax=1.3,
                cmap="coolwarm",
            )
            ax[1, 0].scatter(
                dx[k1], dy[k1], c=model[0][k1], s=3, vmin=0.7, vmax=1.3, cmap="coolwarm"
            )
            ax[1, 1].scatter(
                dx[k2],
                dy[k2],
                c=model[-1][k2],
                s=3,
                vmin=0.7,
                vmax=1.3,
                cmap="coolwarm",
            )
            ax[0, 0].set(title="Data First Cadence", ylabel="$\delta y$")
            ax[0, 1].set(title="Data Last Cadence")
            ax[1, 0].set(
                title="Model First Cadence", ylabel="$\delta y$", xlabel="$\delta x$"
            )
            ax[1, 1].set(title="Model Last Cadence", xlabel="$\delta x$")
            plt.subplots_adjust(hspace=0.3)

            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label("Normalized Flux")
            return fig
        return

    #
    # def _bin_data(self, mean_f, nphi=10, nr=10):
    #     """
    #     Bin the `mean_f` (mean fluxes) in polar coordinates space to then build the
    #     PSF model.
    #
    #     Note: For the sake of modeling a PSF, we assing a lower bound value to all
    #     bins with no counts and a significant large ditance (15 arcsec). This helps
    #     to set boundary conditions when doing the linear modeling of the binned data.
    #
    #     Radial distance bins are square spaced.
    #
    #     Parameters
    #     ----------
    #     mean_f : numpy.ndarray
    #         Values of mean flux at a given radial distance and azimutal angle
    #     nphi : int, optional
    #         Number of bins for the azimutal angle axis
    #     nphi : int, optional
    #         Number of bins for the radial distance axis
    #
    #     Returns
    #     -------
    #     r_b : numpy.ndarray
    #         Mesh grid of radial distance binned values
    #     phi_b : numpy.ndarray
    #         Mesh grid of azimutal angle binned values
    #     mean_f_b : numpy.ndarray
    #         Mean flux value per bin
    #     counts : numpy.ndarray
    #         Number of counts per bin
    #     """
    #     # defining bin edges
    #     phis = np.linspace(-np.pi, np.pi, nphi)
    #     rs = np.linspace(0 ** 0.5, (self.radius.max()) ** 0.5, nr) ** 2
    #
    #     phi_m = self.uncontaminated_source_mask.multiply(self.phi.value).data
    #     r_m = self.uncontaminated_source_mask.multiply(self.r.value).data
    #
    #     # binned data
    #     counts, _, _ = np.histogram2d(phi_m, r_m, bins=(phis, rs))
    #     mean_f_b, _, _ = np.histogram2d(phi_m, r_m, bins=(phis, rs), weights=mean_f)
    #     mean_f_b /= counts
    #     phi_b, r_b = np.asarray(
    #         np.meshgrid(phis[:-1] + np.median(np.diff(phis)) / 2, rs[:-1])
    #     )
    #     # mean_f_b[(r_b.T > 1) & (counts < 1)] = np.nan
    #     mean_f_b[(r_b.T > 15) & ~np.isfinite(mean_f_b)] = -6
    #
    #     return r_b, phi_b, mean_f_b.T, counts.T

    def build_shape_model(self, flux_estimates=None, sigma=5, niters=3, plot=False):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        sigma
        """
        if flux_estimates is None:
            flux_estimates = self.source_flux_estimates

        # mean flux values using uncontaminated mask and normalized by flux estimations
        # mean_f = np.log10(
        #     self.uncontaminated_source_mask.astype(float)
        #     .multiply(self.flux.mean(axis=0))
        #     .multiply(1 / flux_estimates)
        #     .data
        # )

        f, fe = (self.flux).mean(axis=0), ((self.flux_err ** 2).sum(axis=0) ** 0.5) / (
            self.nt
        )

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(f)
            .multiply(1 / flux_estimates)
            .data
        )
        mean_f_err = (
            self.uncontaminated_source_mask.astype(float)
            .multiply(fe / (f * np.log(10)))
            .multiply(1 / flux_estimates)
            .data
        )

        phi_b = self.uncontaminated_source_mask.multiply(self.phi.value).data
        r_b = self.uncontaminated_source_mask.multiply(self.r.value).data
        mean_f_b = mean_f

        # save them for later plotting
        self.mean_f = mean_f
        self.mean_f_b = mean_f_b
        self.phi_b = phi_b
        self.r_b = r_b

        # build a design matrix A with b-splines basis in radius and angle axis.
        A = _make_A_wcs(phi_b.ravel(), r_b.ravel())
        prior_sigma = np.ones(A.shape[1]) * 100
        prior_mu = np.zeros(A.shape[1])
        nan_mask = np.isfinite(mean_f_b.ravel())

        # we solve for A * psf_w = mean_f_b
        psf_w = self._solve_linear_model(
            A,
            y=mean_f_b.ravel(),
            #    y_err=mean_f_err.ravel(),
            k=nan_mask,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        bad = sigma_clip(mean_f_b.ravel() - A.dot(psf_w), sigma=5).mask

        psf_w = self._solve_linear_model(
            A,
            y=mean_f_b.ravel(),
            #    y_err=mean_f_err.ravel(),
            k=nan_mask & ~bad,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        self.psf_w = psf_w

        # We then build the same design matrix for all pixels with flux
        Ap = _make_A_wcs(
            self.source_mask.multiply(self.phi).data,
            self.source_mask.multiply(self.r).data,
        )

        # And create a `mean_model` that has the psf model for all pixels with fluxes
        mean_model = sparse.csr_matrix(self.r.shape)
        m = 10 ** Ap.dot(psf_w)
        mean_model[self.source_mask] = m
        mean_model.eliminate_zeros()
        self.mean_model = mean_model
        if plot:
            return self.plot_shape_model()
        return

    def plot_shape_model(self):
        """ Diagnostic plot of shape model..."""
        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.flux.mean(axis=0))
            .multiply(1 / self.source_flux_estimates)
            .data
        )

        fig, ax = plt.subplots(1, 4, figsize=(16, 3))
        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        im = ax[0].scatter(
            dx, dy, c=mean_f, cmap="viridis", vmin=-3, vmax=-1, s=4, rasterized=True
        )
        ax[0].set(xlabel='$\delta x$ ["]', ylabel='$\delta y$ ["]', title="Data")

        phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)
        im = ax[1].scatter(
            phi, r, c=mean_f, cmap="viridis", vmin=-3, vmax=-1, s=4, rasterized=True
        )
        ax[1].set(xlabel="$\phi$ [$^\circ$]", ylabel='$r$ ["]', title="Data")

        A = _make_A_wcs(phi, r)
        im = ax[2].scatter(
            phi, r, c=A.dot(self.psf_w), cmap="viridis", vmin=-3, vmax=-1, s=4
        )
        ax[2].set(xlabel="$\phi$ [$^\circ$]", ylabel='$r$ ["]', title="Model")

        im = ax[3].scatter(
            dx, dy, c=A.dot(self.psf_w), cmap="viridis", vmin=-3, vmax=-1, s=4
        )
        ax[3].set(xlabel='$\delta x$ ["]', ylabel='$\delta y$ ["]', title="Model")

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, location="right")
        cbar.set_label("log$_{10}$ Normalized Flux")

        return fig

    def fit_model(self, fit_va=False):

        if fit_va:
            if not hasattr(self, "velocity_aberration_w"):
                raise ValueError(
                    "Please use `build_time_model` before fitting with velocity aberration."
                )

            dx, dy = (
                self.source_mask.multiply(self.dra.value),
                self.source_mask.multiply(self.ddec.value),
            )
            dx = dx.data * u.deg.to(u.arcsecond)
            dy = dy.data * u.deg.to(u.arcsecond)

            A_cp = _make_A_cartesian(dx, dy, n_knots=self._time_model_knots, radius=12)
            A_cp3 = sparse.hstack([A_cp, A_cp, A_cp, A_cp], format="csr")
            m = sparse.csr_matrix(dx)

            prior_mu = self.source_flux_estimates[:, 0]  # np.zeros(A.shape[1])
            prior_sigma = (
                np.ones(self.mean_model.shape[0])
                * 10
                * self.source_flux_estimates[:, 0]
            )

            self.ws = np.zeros((self.nt, self.mean_model.shape[0]))
            self.werrs = np.zeros((self.nt, self.mean_model.shape[0]))

            self.ws_va = np.zeros((self.nt, self.mean_model.shape[0]))
            self.werrs_va = np.zeros((self.nt, self.mean_model.shape[0]))

            for tdx in tqdm(
                range(self.nt), desc=f"Fitting {self.nsources} Sources (w. VA)"
            ):
                X = self.mean_model.copy()
                X = X.T

                sigma_w_inv = X.T.dot(
                    X.multiply(1 / self.flux_err[tdx][:, None] ** 2)
                ).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((self.flux[tdx] / self.flux_err[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws[tdx] = np.linalg.solve(sigma_w_inv, B)
                self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

                # Divide through by expected velocity aberration
                X = self.mean_model.copy()
                t_mult = np.hstack(
                    (self._whitened_time[tdx] ** np.arange(4))[:, None]
                    * np.ones(A_cp3.shape[1] // 4)
                )
                X.data *= A_cp3.multiply(t_mult).dot(self.velocity_aberration_w) + 1
                X = X.T

                k = np.isfinite(self.flux[tdx])
                sigma_w_inv = X.T.dot(
                    X.multiply(1 / self.flux_err[tdx][:, None] ** 2)
                ).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((self.flux[tdx] / self.flux_err[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws_va[tdx] = np.linalg.solve(sigma_w_inv, B)
                self.werrs_va[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

            nodata = np.asarray(self.source_mask.sum(axis=1))[:, 0] == 0
            self.ws[:, nodata] *= np.nan
            self.werrs[:, nodata] *= np.nan
            self.ws_va[:, nodata] *= np.nan
            self.werrs_va[:, nodata] *= np.nan

        else:
            prior_mu = self.source_flux_estimates[:, 0]  # np.zeros(A.shape[1])
            prior_sigma = (
                np.ones(self.mean_model.shape[0])
                * 10
                * self.source_flux_estimates[:, 0]
            )

            self.ws = np.zeros((self.nt, self.mean_model.shape[0]))
            self.werrs = np.zeros((self.nt, self.mean_model.shape[0]))

            X = self.mean_model.copy()
            X = X.T

            for tdx in tqdm(
                range(self.nt), desc=f"Fitting {self.nsources} Sources (No VA)"
            ):
                sigma_w_inv = X.T.dot(
                    X.multiply(1 / self.flux_err[tdx][:, None] ** 2)
                ).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((self.flux[tdx] / self.flux_err[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws[tdx] = np.linalg.solve(sigma_w_inv, B)
                self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

            nodata = np.asarray(self.source_mask.sum(axis=1))[:, 0] == 0
            self.ws[:, nodata] *= np.nan
            self.werrs[:, nodata] *= np.nan

        return

    #
    # def fit_model(self):
    #     """
    #     Fits a sparse model matrix to all flux values iteratively in time to calculate
    #     PSF photometry. This is done in two iteretions, firs using priors and
    #     a mean model normalized with Gaia source expected fluxes, and a second
    #     iteration updating the priors and normalizations with the precomputed fluxes.
    #
    #     This method create two new class atributes that contain the PSF flux and
    #     uncertainties (`psf_flux` and `psf_flux_err`)
    #     """
    #     # self._fit_time_variable_model()
    #     A = (self.mean_model).T
    #     ws1 = np.zeros((self.nt, A.shape[1])) * np.nan
    #     werrs1 = np.zeros((self.nt, A.shape[1])) * np.nan
    #
    #     # first iteration uses Gaia flux as prior
    #     prior_mu = self.source_flux_estimates[:, 0]
    #     prior_sigma = np.ones(A.shape[1]) * 10 * self.source_flux_estimates[:, 0]
    #     k = np.isfinite(self.flux)
    #     for t in tqdm(np.arange(self.nt), desc="Fitting PSF model, Gaia prior"):
    #         ws1[t], werrs1[t] = self._solve_linear_model(
    #             A,
    #             self.flux[t],
    #             y_err=self.flux_err[t],
    #             prior_mu=prior_mu,
    #             prior_sigma=prior_sigma,
    #             k=k[t],
    #         )
    #     self.psf_flux1 = ws1.T
    #     self.psf_flux1_err = werrs1.T
    #
    #     ws2 = np.zeros_like(ws1) * np.nan
    #     werrs2 = np.zeros_like(werrs1) * np.nan
    #     # update priors using previous fitting, we keep wide priors for sigma
    #     prior_mu = ws1.mean(axis=0)
    #     # update mean model with the flux estimations
    #     self.build_model(update=True)
    #     self._plot_mean_model()
    #     A = (self.mean_model).T
    #     for t in tqdm(range(self.nt), desc="Fitting PSF model, 2nd iter"):
    #         ws2[t], werrs2[t] = self._solve_linear_model(
    #             A,
    #             self.flux[t],
    #             y_err=self.flux_err[t],
    #             prior_mu=prior_mu,
    #             prior_sigma=prior_sigma,
    #             k=k[t],
    #         )
    #     self.psf_flux2 = ws2.T
    #     self.psf_flux2_err = werrs2.T
    #
    # def _plot_mean_model(self):
    #     """
    #     Diagnostic plot of the mean PSF model. Run `self.build_model()` first.
    #
    #     Plot figures with the mean fluxes and mean model in polar coordinates, mean
    #     model in WC, radial and angle profile.
    #     Also includes the binned and counts of binned mean flux data if available
    #     """
    #     # Plotting r,phi,meanflux used to build PSF model
    #     fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    #     ax[0].set_title("Mean flux")
    #     cax = ax[0].scatter(
    #         self.uncontaminated_source_mask.multiply(self.phi).data,
    #         self.uncontaminated_source_mask.multiply(self.r).data,
    #         c=self.mean_f,
    #         marker=".",
    #     )
    #     ax[0].set_ylim(0, self.r_b.max())
    #     fig.colorbar(cax, ax=ax[0])
    #     ax[0].set_ylabel(r"$r^{\prime\prime}$ ")
    #     ax[0].set_xlabel(r"$\phi$ [rad]")
    #
    #     r_test, phi_test = np.meshgrid(
    #         np.linspace(0 ** 0.5, self.r_b.max() ** 0.5, 100) ** 2,
    #         np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 100),
    #     )
    #     A_test = _make_A_wcs(phi_test.ravel(), r_test.ravel())
    #     model_test = A_test.dot(self.psf_w)
    #     model_test = model_test.reshape(phi_test.shape)
    #
    #     ax[1].set_title("Average PSF Model")
    #     cax = ax[1].pcolormesh(phi_test, r_test, model_test)
    #     fig.colorbar(cax, ax=ax[1])
    #     ax[1].set_xlabel(r"$\phi$ [rad]")
    #     plt.show()
    #
    #     plt.figure(figsize=(9, 8))
    #     plt.pcolormesh(r_test * np.cos(phi_test), r_test * np.sin(phi_test), model_test)
    #     plt.show()
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    #     ax[0].set_title("Marginal Dist")
    #     ax[0].plot(r_test[0, :], model_test.sum(axis=0))
    #     ax[0].set_xlabel(r"$r$")
    #
    #     ax[1].set_title("Marginal Dist")
    #     ax[1].plot(phi_test[:, 0], model_test.sum(axis=1))
    #     ax[1].set_xlabel(r"$\phi$")
    #
    #     plt.show()
    #
    #     if self._binned_model:
    #         fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    #         ax[0].set_title("Counts per bin")
    #         cax = ax[0].pcolormesh(self.phi_b, self.r_b, self.counts)
    #         fig.colorbar(cax, ax=ax[0])
    #         ax[0].set_xlabel(r"$\phi$ [rad]")
    #         ax[0].set_ylabel(r"$r^{\prime\prime}$")
    #
    #         ax[1].set_title("Binned mean flux")
    #         cax = ax[1].pcolormesh(self.phi_b, self.r_b, self.mean_f_b)
    #         fig.colorbar(cax, ax=ax[1])
    #         ax[1].set_xlabel(r"$\phi$ [rad]")
    #         plt.show()
    #     else:
    #         fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    #         ax.set_title("Mean flux")
    #         cax = ax.scatter(self.phi_b, self.r_b, c=self.mean_f_b, marker=".")
    #         ax.set_ylim(0, self.r_b.max())
    #         fig.colorbar(cax, ax=ax)
    #         ax.set_ylabel(r"$r^{\prime\prime}$ ")
    #         ax.set_xlabel(r"$\phi$ [rad]")
    #     return

    def _plot_residual_scene():
        return

    def _plot_psf_fit():
        return

    def _parse_TPFs(tpfs):
        """
        Parse TPF collection to extract times, pixel fluxes, flux errors and tpf-index
        per pixel

        Parameters
        ----------
        tpfs : lightkurve TargetPixelFileCollection
            Collection of Target Pixel files

        Returns
        -------
        times : numpy.ndarray
            Array with time values
        flux : numpy.ndarray
            Array with flux values per pixel
        flux_err : numpy.ndarray
            Array with flux errors per pixel
        unw : numpy.ndarray
            Array with TPF index for each pixel
        """
        cadences = np.array([tpf.cadenceno for tpf in tpfs])

        # check if all TPFs has same cadences
        if not np.all(cadences[1:, :] - cadences[-1:, :] == 0):
            raise ValueError("All TPFs must have same time basis")
        # extract times
        times = np.asarray(tpfs[0].time.jd)

        locs = [
            np.mgrid[
                tpf.column : tpf.column + tpf.shape[2], tpf.row : tpf.row + tpf.shape[1]
            ].reshape(2, np.product(tpf.shape[1:]))
            for tpf in tpfs
        ]
        locs = np.hstack(locs)
        column, row = locs

        # put fluxes into ntimes x npix shape
        flux = np.hstack([np.hstack(tpf.flux.transpose([2, 0, 1])) for tpf in tpfs])
        flux_err = np.hstack(
            [np.hstack(tpf.flux_err.transpose([2, 0, 1])) for tpf in tpfs]
        )
        unw = np.hstack(
            [
                np.zeros((tpf.shape[0], tpf.shape[1] * tpf.shape[2]), dtype=int) + idx
                for idx, tpf in enumerate(tpfs)
            ]
        )
        return times, flux, flux_err, column, row, unw

    def _convert_to_wcs(tpfs):
        """
        Extract pairs of row, column coordinates per pixels and convert them into
        World Cordinate System ra, dec.

        Parameters
        ----------
        tpfs : lightkurve TargetPixelFileCollection
            Collection of Target Pixel files

        Returns
        -------
        locs : numpy.ndarray
            2D array with pixel locations (columns, rows) from the TPFs
        ra : numpy.ndarray
            Array with right ascension values per pixel
        dec : numpy.ndarray
            Array with declination values per pixel
        """
        # calculate x,y grid of each pixel
        locs = np.hstack(
            [
                np.mgrid[
                    tpf.column : tpf.column + tpf.shape[2],
                    tpf.row : tpf.row + tpf.shape[1],
                ].reshape(2, np.product(tpf.shape[1:]))
                for tpf in tpfs
            ]
        )

        # convert pixel coord to ra, dec using TPF's solution
        ra, dec = (
            tpfs[0]
            .wcs.wcs_pix2world(
                np.vstack([(locs[0] - tpfs[0].column), (locs[1] - tpfs[0].row)]).T, 0.0
            )
            .T
        )
        return locs, ra, dec

    def _preprocess(
        flux, flux_err, unw, locs, ra, dec, column, row, tpfs, saturation_limit=1.5e5
    ):
        """
        Clean pixels with nan values, bad cadences and removes duplicated pixels.
        """

        def _saturated_pixels_mask(flux, column, row, saturation_limit=1.5e5):
            """Finds and removes saturated pixels, including bleed columns."""
            # Which pixels are saturated
            saturated = np.nanpercentile(flux, 99, axis=0)
            saturated = np.where((saturated > saturation_limit).astype(float))[0]

            # Find bad pixels, including allowence for a bleed column.
            bad_pixels = np.vstack(
                [
                    np.hstack([column[saturated] + idx for idx in np.arange(-3, 3)]),
                    np.hstack([row[saturated] for idx in np.arange(-3, 3)]),
                ]
            ).T
            # Find unique row/column combinations
            bad_pixels = bad_pixels[
                np.unique(
                    ["".join(s) for s in bad_pixels.astype(str)], return_index=True
                )[1]
            ]
            # Build a mask of saturated pixels
            m = np.zeros(len(column), bool)
            for p in bad_pixels:
                m |= (column == p[0]) & (row == p[1])
            return m

        flux = np.asarray(flux)
        flux_err = np.asarray(flux_err)

        # Finite pixels
        not_nan = np.isfinite(flux).all(axis=0)
        # Unique Pixels
        _, unique_pix = np.unique(locs, axis=1, return_index=True)
        unique_pix = np.in1d(np.arange(len(ra)), unique_pix)
        # No saturation and bleed columns
        not_saturated = ~_saturated_pixels_mask(
            flux, column, row, saturation_limit=saturation_limit
        )

        mask = not_nan & unique_pix & not_saturated

        locs = locs[:, mask]
        column = column[mask]
        row = row[mask]
        ra = ra[mask]
        dec = dec[mask]
        flux = flux[:, mask]
        flux_err = flux_err[:, mask]
        unw = unw[:, mask]

        return flux, flux_err, unw, locs, ra, dec, column, row

    def _get_coord_and_query_gaia(ra, dec, unw, epoch, magnitude_limit, dr=2):
        """
        Calculate ra, dec coordinates and search radius to query Gaia catalog

        Parameters
        ----------
        ra : numpy.ndarray
            Right ascension coordinate of pixels to do Gaia search
        ra : numpy.ndarray
            Declination coordinate of pixels to do Gaia search
        unw : numpy.ndarray
            TPF index of each pixel
        epoch : float
            Epoch of obervation in Julian Days of ra, dec coordinates,
            will be used to propagate proper motions in Gaia.

        Returns
        -------
        sources : pandas.DataFrame
            Catalog with query result
        """
        # find the max circle per TPF that contain all pixel data to query Gaia
        ras, decs, rads = [], [], []
        for l in np.unique(unw[0]):
            ra1 = ra[unw[0] == l]
            dec1 = dec[unw[0] == l]
            ras.append(ra1.mean())
            decs.append(dec1.mean())
            rads.append(
                np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max()
                + (u.arcsecond * 6).to(u.deg).value
            )
        # query Gaia with epoch propagation
        sources = get_gaia_sources(
            tuple(ras),
            tuple(decs),
            tuple(rads),
            magnitude_limit=magnitude_limit,
            epoch=Time(epoch, format="jd").jyear,
            dr=dr,
        )
        return sources

    @staticmethod
    def from_TPFs(tpfs, magnitude_limit=18, dr=2):
        """
        Convert TPF input into Machine object:
            * Parse TPFs to extract time, flux, clux erros, and bookkeeping of
            the TPF-pixel correspondance
            * Convert pixel-based coordinates (row, column) into WCS (ra, dec) for
            all pixels
            * Clean pixels with no values, bad cadences, and remove duplicated pixels
            due to overlapping TPFs
            * Query Gaia DR2 data base to find all sources present in the TPF images
            * Clean unresolve sources (within 6`` = 1.5 pixels) and sources off silicon
            with a 1 pixel tolerance (distance from source gaia pocition and TPF edge)

        Parameters
        ----------
        tpfs : lightkurve TargetPixelFileCollection
            Collection of Target Pixel files

        Returns
        -------
        Machine : Machine object
            A Machine class object built from TPFs.
        """
        if not isinstance(tpfs, lk.collections.TargetPixelFileCollection):
            raise TypeError("<tpfs> must be a of class Target Pixel Collection")

        # parse tpfs
        times, flux, flux_err, column, row, unw = Machine._parse_TPFs(tpfs)

        # convert to RA Dec
        locs, ra, dec = Machine._convert_to_wcs(tpfs)

        # preprocess arrays
        flux, flux_err, unw, locs, ra, dec, column, row = Machine._preprocess(
            flux, flux_err, unw, locs, ra, dec, column, row, tpfs
        )

        sources = Machine._get_coord_and_query_gaia(
            ra, dec, unw, times[0], magnitude_limit, dr=dr
        )

        # soruce list cleaning
        #        sources, _ = Machine._clean_source_list(sources, ra, dec)

        # return a Machine object
        return Machine(
            time=times,
            flux=flux,
            flux_err=flux_err,
            ra=ra,
            dec=dec,
            sources=sources,
            column=column,
            row=row,
            pix2obs=unw,
        )

    def _clean_source_list(sources, ra, dec):
        """
        Removes sources that are too contaminated and/or off the edge of the image

        Parameters
        ----------
        sources : Pandas Dataframe
            Contains a list with cross-referenced Gaia results
            shape n sources x n Gaia features
        ra      : numpy ndarray
            RA pixel position averaged in time
            shape npixel
        dec     : numpy ndarray
            Dec pixel position averaged in time
            shape npixel

        Returns
        -------
        sources : Pandas.DataFrame
            Catalog with clean sources
        removed_sources : Pandas.DataFrame
            Catalog with removed sources
        """
        # find sources on the image
        inside = np.zeros(len(sources), dtype=bool)
        # max distance in arcsec from image edge to source ra, dec
        off = 3.0 / 3600
        for k in range(len(sources)):
            raok = (sources["ra"][k] > ra - off) & (sources["ra"][k] < ra + off)
            decok = (sources["dec"][k] > dec - off) & (sources["dec"][k] < dec + off)
            inside[k] = (raok & decok).any()
        del raok, decok

        # find well separated sources
        s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
        midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]
        # remove sources closer than 6" = 1.5 pix
        closest = mdist.arcsec < 2.0
        blocs = np.vstack([midx[closest], np.where(closest)[0]])
        bmags = np.vstack(
            [
                sources.phot_g_mean_mag[midx[closest]],
                sources.phot_g_mean_mag[np.where(closest)[0]],
            ]
        )
        faintest = [blocs[idx][s] for s, idx in enumerate(np.argmax(bmags, axis=0))]
        unresolved = np.in1d(np.arange(len(sources)), faintest)
        del s_coords, midx, mdist, closest, blocs, bmags

        # Keep track of sources that we removed
        sources.loc[:, "clean_flag"] = 0
        sources.loc[:, "clean_flag"][~inside] = 2 ** 0  # outside TPF
        # sources.loc[:, "clean_flag"][unresolved] += 2 ** 1  # close contaminant

        # combine 2 source masks
        clean = sources.clean_flag == 0
        removed_sources = sources[~clean].reset_index(drop=True)
        sources = sources[clean].reset_index(drop=True)

        return sources, removed_sources


"""
Notes on user functionality

machine.plot_tpf(4)
 machine.plot_source(13)
# show new sources
 machine.show_fresh()
"""
