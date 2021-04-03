"""
Defines the main Machine object that fit a mean PRF model to sources
"""
import numpy as np
import pandas as pd
from scipy import sparse
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from .utils import _make_A_polar, _make_A_cartesian

__all__ = ["Machine"]


class Machine(object):
    """
    Class for calculating fast PRF photometry on a collection of images and
    a list of in image sources.

    This method is discussed in detail in CITATION

    This method solves a linear model to assuming Gaussian priors on the weight of
    each linear components as explained by Luger, Foreman-Mackey & Hogg, 2017
    (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)
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
        time_mask=None,
        n_r_knots=10,
        n_phi_knots=15,
        n_time_knots=10,
        n_time_points=200,
        time_radius=8,
        rmin=1,
        rmax=16,
    ):
        """
        Class for calculating fast PRF photometry on a collection of images and
        a list of in image sources.

        This method is discussed in detail in CITATION

        This method solves a linear model to assuming Gaussian priors on the weight of
        each linear components as explained by Luger, Foreman-Mackey & Hogg, 2017
        (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)


        Parameters
        ----------
        time: numpy.ndarray
            Time values in JD
        flux: numpy.ndarray
            Flux values at each pixels and times in units of electrons / sec
        flux_err: numpy.ndarray
            Flux error values at each pixels and times in units of electrons / sec
        ra: numpy.ndarray
            Right Ascension coordinate of each pixel
        dec: numpy.ndarray
            Declination coordinate of each pixel
        sources: pandas.DataFrame
            DataFrame with source present in the images
        column: np.ndarray
            Data array containing the "columns" of the detector that each pixel is on.
        row: np.ndarray
            Data array containing the "columns" of the detector that each pixel is on.
        limit_radius: numpy.ndarray
            Radius limit in arcsecs to select stars to be used for PRF modeling
        time_mask:  np.ndarray of booleans
            A boolean array of shape time. Only values where this mask is `True`
            will be used to calculate the average image for fitting the PSF.
            Use this to e.g. select frames with low VA, or no focus change
        n_r_knots: int
            Number of radial knots in the spline model.
        n_phi_knots: int
            Number of azimuthal knots in the spline model.
        n_time_points: int
            Number of time points to bin by when fitting for velocity aberration.
        time_radius: float
            The radius around sources, out to which the velocity aberration model
            will be fit. (arcseconds)
        rmin: float
            The minimum radius for the PRF model to be fit. (arcseconds)
        rmax: float
            The maximum radius for the PRF model to be fit. (arcseconds)

        Attributes
        ----------
        nsources: int
            Number of sources to be extracted
        nt: int
            Number of onservations in the time series (aka number of cadences)
        npixels: int
            Total number of pixels with flux measurements
        source_flux_estimates: numpy.ndarray
            First estimation of pixel fluxes assuming values given by the sources catalog
            (e.g. Gaia phot_g_mean_flux)
        dra: numpy.ndarray
            Distance in right ascension between pixel and source coordinates, units of
            degrees
        ddec: numpy.ndarray
            Distance in declination between pixel and source coordinates, units of
            degrees
        r: numpy.ndarray
            Radial distance between pixel and source coordinates (polar coordinates),
            in units of arcseconds
        phi: numpy.ndarray
            Angle between pixel and source coordinates (polar coordinates),
            in units of radians
        source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with pixels that contains flux from sources
        uncontaminated_source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with selected uncontaminated pixels per source to be used to
            build the PSF model
        mean_model: scipy.sparce.csr_matrix
            Mean PSF model values per pixel used for PSF photometry
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
        self.n_r_knots = n_r_knots
        self.n_phi_knots = n_phi_knots
        self.n_time_knots = n_time_knots
        self.n_time_points = n_time_points
        self.time_radius = time_radius
        self.rmin = rmin
        self.rmax = rmax

        if time_mask is None:
            self.time_mask = np.ones(len(time), bool)
        else:
            self.time_mask = time_mask

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # The distance in ra & dec from each source to each pixel
        self.dra, self.ddec = np.asarray(
            [
                [self.ra - self.sources["ra"][idx], self.dec - self.sources["dec"][idx]]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # convertion to polar coordinates
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        self.phi = np.arctan2(self.ddec, self.dra)

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

        if isinstance(sigma_w_inv, (sparse.csr_matrix, sparse.csc_matrix, np.matrix)):
            sigma_w_inv = np.asarray(sigma_w_inv)

        w = np.linalg.solve(sigma_w_inv, B)
        if errors is True:
            w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            return w, w_err
        return w

    def _get_source_mask(
        self,
        upper_radius_limit=28.0,
        lower_radius_limit=4.5,
        upper_flux_limit=2e5,
        lower_flux_limit=100,
        plot=False,
    ):
        """Find the pixel mask that identifies pixels with contributions from ANY NUMBER of Sources

        Fits a simple polynomial model to the log of the pixel flux values, in radial dimension and source flux,
        to find the optimum circular apertures for every source.

        Parameters
        ----------
        upper_radius_limit: float
            The radius limit at which we assume there is no flux from a source of any brightness (arcsec)
        lower_radius_limit: float
            The radius limit at which we assume there is flux from a source of any brightness (arcsec)
        upper_flux_limit: float
            The flux at which we assume as source is saturated
        lower_flux_limit: float
            The flux at which we assume a source is too faint to model
        plot: bool
            Whether to show diagnostic plot. Default is False
        """

        if not hasattr(self, "source_flux_estimates"):
            # gaia estimate flux values per pixel to be used as flux priors
            self.source_flux_estimates = np.copy(
                np.asarray(self.sources.phot_g_mean_flux)
            )
        # We will use the radius a lot, this is for readibility
        r = self.r

        # The average flux, which we assume is a good estimate of the whole stack of images
        mean_flux = np.nanmean(self.flux[self.time_mask], axis=0)
        # mean_flux_err = (self.flux_err[self.time_mask] ** 0.5).sum(
        #     axis=0
        # ) ** 0.5 / self.time_mask.sum()

        # First we make a guess that each source has exactly the gaia flux
        source_flux_estimates = np.asarray(self.sources.phot_g_mean_flux)[
            :, None
        ] * np.ones((self.nsources, self.npixels))

        # Mask out sources that are above the flux limit, and pixels above the radius limit
        source_rad = 0.5 * np.log10(self.source_flux_estimates) ** 1.5 + 3
        temp_mask = (r.value < source_rad[:, None]) & (
            source_flux_estimates < upper_flux_limit
        )
        temp_mask &= temp_mask.sum(axis=0) == 1
        #        temp_mask &= temp_mask.sum(axis=1)[:, None] == 1

        # log of flux values
        f = np.log10((temp_mask.astype(float) * mean_flux))
        #        weights = (
        #            (self.flux_err ** 0.5).sum(axis=0) ** 0.5 / self.flux.shape[0]
        #        ) * temp_mask

        # flux estimates
        mf = np.log10(source_flux_estimates[temp_mask])

        # Model is polynomial in r and log of the flux estimate.
        # Here I'm using a 1st order polynomial, to ensure it's monatonic in each dimension
        A = np.vstack(
            [
                r.value[temp_mask] ** 0,
                r.value[temp_mask],
                #                r.value[temp_mask] ** 2,
                r.value[temp_mask] ** 0 * mf,
                r.value[temp_mask] * mf,
                #                r.value[temp_mask] ** 2 * mf,
                #                r.value[temp_mask] ** 0 * mf ** 2,
                #                r.value[temp_mask] * mf ** 2,
                #                r.value[temp_mask] ** 2 * mf ** 2,
            ]
        ).T

        # Iteratively fit
        k = np.isfinite(f[temp_mask])
        for count in [0, 1, 2]:
            sigma_w_inv = A[k].T.dot(A[k])
            B = A[k].T.dot(f[temp_mask][k])
            w = np.linalg.solve(sigma_w_inv, B)
            res = np.ma.masked_array(f[temp_mask], ~k) - A.dot(w)
            k &= ~sigma_clip(res, sigma=3).mask

        # Now find the radius and source flux at which the model reaches the flux limit
        test_f = np.linspace(
            np.log10(source_flux_estimates.min()),
            np.log10(source_flux_estimates.max()),
            100,
        )
        test_r = np.arange(lower_radius_limit, upper_radius_limit, 0.25)
        test_r2, test_f2 = np.meshgrid(test_r, test_f)

        test_val = (
            np.vstack(
                [
                    test_r2.ravel() ** 0,
                    test_r2.ravel(),
                    #                    test_r2.ravel() ** 2,
                    test_r2.ravel() ** 0 * test_f2.ravel(),
                    test_r2.ravel() * test_f2.ravel(),
                    #                    test_r2.ravel() ** 2 * test_f2.ravel(),
                    #                    test_r2.ravel() ** 0 * test_f2.ravel() ** 2,
                    #                    test_r2.ravel() * test_f2.ravel() ** 2,
                    #                    test_r2.ravel() ** 2 * test_f2.ravel() ** 2,
                ]
            )
            .T.dot(w)
            .reshape(test_r2.shape)
        )
        l = np.zeros(len(test_f)) * np.nan
        for idx in range(len(test_f)):
            loc = np.where(10 ** test_val[idx] < lower_flux_limit)[0]
            if len(loc) > 0:
                l[idx] = test_r[loc[0]]
        ok = np.isfinite(l)
        source_radius_limit = np.polyval(
            np.polyfit(test_f[ok], l[ok], 1), np.log10(source_flux_estimates[:, 0])
        )
        source_radius_limit[
            source_radius_limit > upper_radius_limit
        ] = upper_radius_limit
        source_radius_limit[
            source_radius_limit < lower_radius_limit
        ] = lower_radius_limit

        # Here we set the radius for each source. We add two pixels, to be generous
        self.radius = source_radius_limit + 2

        # This sparse mask is one where there is ANY number of sources in a pixel
        self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])

        self._get_uncontaminated_pixel_mask()

        # Now we can update the r and phi estimates, allowing for a slight centroid offset

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data
        dy = dy.data

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.flux[self.time_mask].mean(axis=0))
            .multiply(1 / self.source_flux_estimates[:, None])
            .data
        )
        k = np.isfinite(mean_f)
        ra_cent = np.average(dx[k], weights=mean_f[k])
        dec_cent = np.average(dy[k], weights=mean_f[k])

        self.dra, self.ddec = np.asarray(
            [
                [
                    self.ra - self.sources["ra"][idx] - ra_cent,
                    self.dec - self.sources["dec"][idx] - dec_cent,
                ]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        self.phi = np.arctan2(self.ddec, self.dra)

        if plot:
            k = np.isfinite(f[temp_mask])
            # Make a nice diagnostic plot
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
                vmax=lower_flux_limit * 2,
                cmap="viridis",
            )
            line = np.polyval(np.polyfit(test_f[ok], l[ok], 1), test_f)
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

    def _get_uncontaminated_pixel_mask(self):
        """
        creates a mask of shape nsources x npixels where targets are not contaminated.
        This mask is used to select pixels to build the PSF model.
        """

        # This could be a property, but it is a pain to calculate on the fly, perhaps with lru_cache
        self.uncontaminated_source_mask = self.source_mask.multiply(
            np.asarray(self.source_mask.sum(axis=0) == 1)[0]
        ).tocsr()

        # # reduce to good pixels
        # self.uncontaminated_pixel_mask = sparse.csr_matrix(
        #     self.uncontaminated_source_mask.sum(axis=0) > 0
        # )

        return

    # CH: We're not currently using this, but it might prove useful later so I will leave for now
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
        """Bin the flux data down in time

        Parameters
        ----------
        npoints: int
            How many points should be in each time bin

        Returns
        -------
        time_original: np.ndarray
            The time array of the data, whitened
        time_binned: np.ndarray
            The binned time array
        flux_binned_raw: np.ndarray
            The binned flux, raw
        flux_binned: np.ndarray
            The binned flux, whitened by the mean of the flux in time
        flux_err_binned:
            The binned flux error, whitened by the mean of the flux
        """

        # Where there are break points in the data
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

    def build_time_model(self, plot=False):
        (
            time_original,
            time_binned,
            flux_binned_raw,
            flux_binned,
            flux_err_binned,
        ) = self._time_bin(npoints=self.n_time_points)

        self._whitened_time = time_original
        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        A_c = _make_A_cartesian(
            dx, dy, n_knots=self.n_time_knots, radius=self.time_radius
        )
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
            sigma_w_inv = A3[k].T.dot(
                (A3.multiply(1 / flux_err_binned.ravel()[:, None] ** 2)).tocsr()[k]
            )
            sigma_w_inv += np.diag(1 / prior_sigma ** 2)
            # Fit the flux - 1
            B = A3[k].T.dot(
                ((flux_binned.ravel() - 1) / flux_err_binned.ravel() ** 2)[k]
            )
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
        self._time_masked = k
        if plot:
            return self.plot_time_model()
        return

    def plot_time_model(self):
        (
            time_original,
            time_binned,
            flux_binned_raw,
            flux_binned,
            flux_err_binned,
        ) = self._time_bin(npoints=self.n_time_points)

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        A_c = _make_A_cartesian(dx, dy, n_knots=self.n_time_knots, radius=8)
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

        model = A3.dot(self.velocity_aberration_w).reshape(flux_binned.shape) + 1
        fig, ax = plt.subplots(2, 2, figsize=(7, 6), facecolor="w")
        k1 = self._time_masked.reshape(flux_binned.shape)[0]
        k2 = self._time_masked.reshape(flux_binned.shape)[-1]
        im = ax[0, 0].scatter(
            dx[k1],
            dy[k1],
            c=flux_binned[0][k1],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[0, 1].scatter(
            dx[k2],
            dy[k2],
            c=flux_binned[-1][k2],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[1, 0].scatter(
            dx[k1],
            dy[k1],
            c=model[0][k1],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[1, 1].scatter(
            dx[k2],
            dy[k2],
            c=model[-1][k2],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[0, 0].set(title="Data First Cadence", ylabel=r"$\delta y$")
        ax[0, 1].set(title="Data Last Cadence")
        ax[1, 0].set(
            title="Model First Cadence", ylabel=r"$\delta y$", xlabel=r"$\delta x$"
        )
        ax[1, 1].set(title="Model Last Cadence", xlabel=r"$\delta x$")
        plt.subplots_adjust(hspace=0.3)

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("Normalized Flux")
        return fig

    def build_shape_model(self, plot=False, flux_cut_off=1):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        flux_cut_off: float
            the flux in COUNTS at which to stop evaluating the model!
        """

        # gaia estimate flux values per pixel to be used as flux priors
        self.source_flux_estimates = np.copy(np.asarray(self.sources.phot_g_mean_flux))

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        self._get_source_mask()
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        self._get_uncontaminated_pixel_mask()

        # for iter in range(niters):
        flux_estimates = self.source_flux_estimates[:, None]

        f, fe = (self.flux[self.time_mask]).mean(axis=0), (
            (self.flux_err[self.time_mask] ** 2).sum(axis=0) ** 0.5
        ) / (self.nt)

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(f)
            .multiply(1 / flux_estimates)
            .data
        )
        # Actual Kepler errors cause all sorts of instability
        # mean_f_err = (
        #     self.uncontaminated_source_mask.astype(float)
        #     .multiply(fe / (f * np.log(10)))
        #     .multiply(1 / flux_estimates)
        #     .data
        # )
        # We only need these weights for the wings, so we'll use poisson noise
        mean_f_err = (
            self.uncontaminated_source_mask.astype(float)
            .multiply((f ** 0.5) / (f * np.log(10)))
            .multiply(1 / flux_estimates)
            .data
        )
        mean_f_err.data = np.abs(mean_f_err.data)

        phi_b = self.uncontaminated_source_mask.multiply(self.phi.value).data
        r_b = self.uncontaminated_source_mask.multiply(self.r.value).data
        mean_f_b = mean_f

        # save them for later plotting
        self.mean_f = mean_f
        self.mean_f_b = mean_f_b
        self.phi_b = phi_b
        self.r_b = r_b

        # build a design matrix A with b-splines basis in radius and angle axis.
        A = _make_A_polar(
            phi_b.ravel(),
            r_b.ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        prior_sigma = np.ones(A.shape[1]) * 10
        prior_mu = np.zeros(A.shape[1]) - 10

        nan_mask = np.isfinite(mean_f_b.ravel())

        # we solve for A * psf_w = mean_f_b
        psf_w, psf_w_err = self._solve_linear_model(
            A,
            y=mean_f_b.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        bad = sigma_clip(mean_f_b.ravel() - A.dot(psf_w), sigma=5).mask

        psf_w, psf_w_err = self._solve_linear_model(
            A,
            y=mean_f_b.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask & ~bad,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        self.psf_w = psf_w
        self.psf_w_err = psf_w_err

        # We then build the same design matrix for all pixels with flux
        self._get_mean_model()

        # Re-estimate source flux
        # -----
        prior_mu = self.source_flux_estimates
        prior_sigma = (
            np.ones(self.mean_model.shape[0]) * 10 * self.source_flux_estimates
        )

        f, fe = (self.flux).mean(axis=0), ((self.flux_err ** 2).sum(axis=0) ** 0.5) / (
            self.nt
        )

        X = self.mean_model.copy()
        X = X.T

        sigma_w_inv = X.T.dot(X.multiply(1 / fe[:, None] ** 2)).toarray()
        sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
        B = X.T.dot((f / fe ** 2))
        B += prior_mu / (prior_sigma ** 2)
        ws = np.linalg.solve(sigma_w_inv, B)
        werrs = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

        # -----

        # Rebuild source mask
        ok = np.abs(ws - self.source_flux_estimates) / werrs > 3
        ok &= ((ws / self.source_flux_estimates) < 10) & (
            (self.source_flux_estimates / ws) < 10
        )
        ok &= ws > 10
        ok &= werrs > 0

        self.source_flux_estimates[ok] = ws[ok]

        self.source_mask = (
            self.mean_model.multiply(
                self.mean_model.T.dot(self.source_flux_estimates)
            ).tocsr()
            > flux_cut_off
        )

        # Recreate uncontaminated mask
        self._get_uncontaminated_pixel_mask()
        # self.uncontaminated_source_mask = self.uncontaminated_source_mask.multiply(
        #    (self.mean_model.max(axis=1) < 1)
        # )

        # Recreate mean model!
        self._get_mean_model()
        if plot:
            return self.plot_shape_model()
        return

    def _get_mean_model(self):
        """Convenience function to make the scene model"""
        Ap = _make_A_polar(
            self.source_mask.multiply(self.phi).data,
            self.source_mask.multiply(self.r).data,
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )

        # And create a `mean_model` that has the psf model for all pixels with fluxes
        mean_model = sparse.csr_matrix(self.r.shape)
        m = 10 ** Ap.dot(self.psf_w)
        m[~np.isfinite(m)] = 0
        mean_model[self.source_mask] = m
        mean_model.eliminate_zeros()
        self.mean_model = mean_model

    def plot_shape_model(self, radius=20):
        """ Diagnostic plot of shape model..."""

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.flux[self.time_mask].mean(axis=0))
            .multiply(1 / self.source_flux_estimates[:, None])
            .data
        )

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra.value),
            self.uncontaminated_source_mask.multiply(self.ddec.value),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        fig, ax = plt.subplots(2, 2, figsize=(8, 6.5))
        im = ax[0, 0].scatter(
            dx, dy, c=mean_f, cmap="viridis", vmin=-3, vmax=-1, s=3, rasterized=True
        )
        ax[0, 0].set(
            ylabel=r'$\delta y$ ["]',
            title="Data",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )

        phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)
        im = ax[0, 1].scatter(
            phi, r, c=mean_f, cmap="viridis", vmin=-3, vmax=-1, s=3, rasterized=True
        )
        ax[0, 1].set(
            ylabel='$r$ ["]',
            title="Data",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )

        A = _make_A_polar(
            phi,
            r,
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        im = ax[1, 1].scatter(
            phi,
            r,
            c=A.dot(self.psf_w),
            cmap="viridis",
            vmin=-3,
            vmax=-1,
            s=3,
            rasterized=True,
        )
        ax[1, 1].set(
            xlabel=r"$\phi$ [$^\circ$]",
            ylabel=r'$r$ ["]',
            title="Model",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )

        im = ax[1, 0].scatter(
            dx,
            dy,
            c=A.dot(self.psf_w),
            cmap="viridis",
            vmin=-3,
            vmax=-1,
            s=3,
            rasterized=True,
        )
        ax[1, 0].set(
            xlabel=r'$\delta x$ ["]',
            ylabel=r'$\delta y$ ["]',
            title="Model",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, location="right")
        cbar.set_label("log$_{10}$ Normalized Flux")

        return fig

    def fit_model(self, fit_va=False):
        """Finds the best fitting weights for every source, simultaneously"""
        prior_mu = self.source_flux_estimates  # np.zeros(A.shape[1])
        prior_sigma = (
            np.ones(self.mean_model.shape[0])
            * 5
            * np.abs(self.source_flux_estimates) ** 0.5
        )

        self.model_flux = np.zeros(self.flux.shape) * np.nan

        self.ws = np.zeros((self.nt, self.mean_model.shape[0]))
        self.werrs = np.zeros((self.nt, self.mean_model.shape[0]))

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

            A_cp = _make_A_cartesian(dx, dy, n_knots=self.n_time_knots, radius=8)
            A_cp3 = sparse.hstack([A_cp, A_cp, A_cp, A_cp], format="csr")

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
                self.ws[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
                self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

                # Divide through by expected velocity aberration
                X = self.mean_model.copy()
                t_mult = np.hstack(
                    (self._whitened_time[tdx] ** np.arange(4))[:, None]
                    * np.ones(A_cp3.shape[1] // 4)
                )
                X.data *= A_cp3.multiply(t_mult).dot(self.velocity_aberration_w) + 1
                X = X.T

                sigma_w_inv = X.T.dot(
                    X.multiply(1 / self.flux_err[tdx][:, None] ** 2)
                ).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((self.flux[tdx] / self.flux_err[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws_va[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
                self.werrs_va[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
                self.model_flux[tdx] = X.dot(self.ws_va[tdx])

            nodata = np.asarray(self.mean_model.sum(axis=1))[:, 0] == 0
            self.ws[:, nodata] *= np.nan
            self.werrs[:, nodata] *= np.nan
            self.ws_va[:, nodata] *= np.nan
            self.werrs_va[:, nodata] *= np.nan

        else:

            X = self.mean_model.copy()
            X = X.T
            f = self.flux
            fe = self.flux_err

            for tdx in tqdm(
                range(self.nt), desc=f"Fitting {self.nsources} Sources (No VA)"
            ):
                sigma_w_inv = X.T.dot(X.multiply(1 / fe[tdx][:, None] ** 2)).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((f[tdx] / fe[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
                self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
                self.model_flux[tdx] = X.dot(self.ws[tdx])

            nodata = np.asarray(self.source_mask.sum(axis=1))[:, 0] == 0
            # These sources are poorly estimated
            nodata |= (self.mean_model.max(axis=1) > 1).toarray()[:, 0]
            self.ws[:, nodata] *= np.nan
            self.werrs[:, nodata] *= np.nan

        return
