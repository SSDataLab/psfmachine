"""Defines the main XXX object that fit PSF model to sources"""

import numpy as np
import lightkurve as lk
from scipy import sparse
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import get_gaia_sources, _make_A_wcs

# help functions


class Machine(object):
    def __init__(self, time, flux, flux_err, ra, dec, sources, limit_radius=24.0):
        """

        Parameters
        ----------
        time : np.ndarray
            Time in JD
            shape nt
        flux : np.ndarray
            Flux at each pixel at each time
            shape nt x npixel
        flux_err : np.ndarray
            Flux error at each pixel at each time
            shape nt x npixel
        x : np.ndarray
            X pixel position averaged in time
            shape npixel
        y :  np.ndarray
            Y pixel position averaged in time
            shape npixel
        ra : np.ndarray
            RA pixel position averaged in time
            shape npixel
        dec : np.ndarray
            Dec pixel position averaged in time
            shape npixel
        sources : pandas DataFrame
            Catalog with soruces on the pixel data (e.g. Gaia catalog)
            shape nsources x nfeatures
        limit_radius : float astropy units
            Radius limit in arcsecs to select stars to be used for PSF modeling


        Attributes
        ----------

        sources : pd.DataFrame

        source_flux_estiamtes : np.ndarray
            nsources
        """

        # assigning initial attributes
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ra = ra
        self.dec = dec
        self.sources = sources
        self.limit_radius = limit_radius * u.arcsecond
        self.limit_flux = 1e4

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # gaia estimate flux value per pixel
        self.source_flux_estimates = np.array(
            [
                np.zeros(self.npixels) + sources.phot_g_mean_flux[idx]
                for idx in range(len(sources))
            ]
        )

        # The distance in ra & dec from each source at each pixel
        # shape nsources x npixels, arcsec units
        self.dra, self.ddec = np.asarray(
            [
                [ra - self.sources["ra"][idx], dec - self.sources["dec"][idx]]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # polar coordinates, remember these two are astropy quantity objects
        # radial distance from source in arcseconds
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        # azimuthal angle is in radians
        self.phi = np.arctan2(self.ddec, self.dra)

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        # self._get_source_mask()
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        # self._get_uncontaminated_pixel_mask()

        # Get the centroids of the images as a function of time
        # self._get_centroids()

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    @staticmethod
    def _solve_linear_model(A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None):
        """
        Solves a linear model with design matrix A and observations y
            Aw = y
        return the solutions w for the system.
        Alternative, the observation errors, priors, and a boolean mask can be provided.

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

        if type(sigma_w_inv) == sparse.csr_matrix:
            sigma_w_inv = sigma_w_inv.toarray()

        if prior_mu is not None and prior_sigma is not None:
            sigma_w_inv += np.diag(1 / prior_sigma ** 2)
            B += prior_mu / prior_sigma ** 2
        w = np.linalg.solve(sigma_w_inv, B)
        if y_err is not None:
            w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            return w, w_err
        return w

    def _get_source_mask(self):
        """
        Mask of shape nsources x number of pixels, one where flux from a source
        exists

        Returns
        -------
        Mask with all pixels containing flux from sources.
        """
        # mask by Gaia flux and maximum distance
        self.mean_flux = np.nanmean(self.flux, axis=0)
        temp_mask = (self.r.to("arcsec") < self.limit_radius) & (
            self.source_flux_estimates > self.limit_flux
        )
        temp_mask &= temp_mask.sum(axis=0) == 1

        # estimates the PSF edges as a function of flux
        f = np.log10((temp_mask.astype(float) * self.mean_flux))[temp_mask]
        A = np.vstack(
            [
                self.r[temp_mask].to("arcsec").value ** 0,
                self.r[temp_mask].to("arcsec").value,
                np.log10(self.source_flux_estimates[temp_mask]),
            ]
        ).T
        k = np.isfinite(f)
        w = self._solve_linear_model(A, f, k=k)

        test_gaia = np.linspace(
            np.log10(self.source_flux_estimates.min()),
            np.log10(self.source_flux_estimates.max()),
            50,
        )
        # r is in arcsecs
        test_r = np.arange(0, 40, 1)
        radius_check = np.asarray(
            [
                np.vstack(
                    [[(np.ones(50) * v) ** idx for idx in range(2)], test_gaia]
                ).T.dot(w)
                for v in test_r
            ]
        )

        # flux cap at which 95% of it is contained
        cut = np.percentile(np.abs(radius_check), 5) - np.min(np.abs(radius_check))
        x, y = np.asarray(np.meshgrid(test_gaia, test_r))[:, np.abs(radius_check) < cut]
        # calculate the radius at which the 95% is contained
        self.radius = np.polyval(
            np.polyfit(x, y, 5), np.log10(self.sources["phot_g_mean_flux"])
        )
        # cap the radius for faint and saturated sources
        # radius[np.log10(self.sources["phot_g_mean_flux"]) < 3] = 8.0
        # radius[np.log10(self.sources["phot_g_mean_flux"]) > 6.5] = 24.0

        self.source_mask = sparse.csr_matrix(
            self.r.to("arcsec").value < self.radius[:, None]
        )

        return

    def _get_uncontaminated_pixel_mask(self):
        """
        Mask of shape npixels (maybe by nt) where not saturated, not faint,
        not contaminated etc.

        """
        # find saturated pixels, the saturation cap per pixel in -e/s
        sat_pixel_mask = np.max(self.flux, axis=0) > 1.5e5

        # find pixels from faint Sources
        faint_sources = np.log10(self.sources["phot_g_mean_flux"]).values < 3

        # find pixels with flux from only one source
        one_source_pix = self.source_mask.sum(axis=0) == 1

        # combine isolated and not faint sources
        good_pixels = sparse.csr_matrix(~sat_pixel_mask).multiply(one_source_pix)

        # combine source mask with good sources and not saturated pixels
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

    def _build_time_variable_model():
        return

    def _fit_time_variable_model():
        return

    def _bin_data(self, mean_f, nphi=40, nr=30):
        """
        Bin data (radius and angle) to speed-up the linear modeling
        """
        # bin data
        phis = np.linspace(-np.pi, np.pi, nphi)
        # try r in squared space
        # rs = np.linspace(0, self.limit_radius.value, nr)
        rs = np.linspace(0 ** 0.5, self.limit_radius.value ** 0.5, nr) ** 2

        # phi_m = self.phi[self.uncontaminated_source_mask].value
        # r_m = self.r[self.uncontaminated_source_mask].to("arcsec").value

        phi_m = self.uncontaminated_source_mask.multiply(self.phi.value).data
        r_m = self.uncontaminated_source_mask.multiply(self.r.value).data

        counts, _, _ = np.histogram2d(phi_m, r_m, bins=(phis, rs))
        mean_f_b, _, _ = np.histogram2d(phi_m, r_m, bins=(phis, rs), weights=mean_f)
        mean_f_b /= counts
        phi_b, r_b = np.asarray(
            np.meshgrid(phis[:-1] + np.median(np.diff(phis)) / 2, rs[:-1])
        )

        return r_b, phi_b, mean_f_b.T, counts.T

    def build_model(self, nphi=40, nr=30, bin=True, update=False):
        """
        Builds a sparse model matrix of shape nsources x npixels

        Builds the model using self.uncontaminated_pixel_mask
        """
        # self._build_time_variable_model()
        # only use pixels near source
        # mean frame
        if update:
            if "psf_flux1" not in dir(self):
                raise AttributeError(
                    "Update model is set to True but no fluxes were "
                    + "precomputed. Run self.fit_model() before."
                )
            flux_estimates = np.repeat(
                self.psf_flux1.mean(axis=1)[:, None], self.npixels, axis=1
            )
        else:
            flux_estimates = self.source_flux_estimates
        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.flux.mean(axis=0))
            .multiply(1 / flux_estimates)
            .data
        )
        if bin:
            r_b, phi_b, mean_f_b, counts = self._bin_data(mean_f)
            self.counts = counts
        else:
            phi_b = self.uncontaminated_source_mask.multiply(self.phi.value).data
            r_b = self.uncontaminated_source_mask.multiply(self.r.value).data
            mean_f_b = mean_f
            counts = mean_f

        self.mean_f = mean_f
        self.mean_f_b = mean_f_b
        self.phi_b = phi_b
        self.r_b = r_b
        # mean_f_b[(r_b.T > 1) & (counts < 3)] = np.nan
        # mean_f_b[(r_b.T > 4) & ~np.isfinite(mean_f_b)] = -5

        A = _make_A_wcs(phi_b.ravel(), r_b.ravel())
        prior_sigma = np.ones(A.shape[1]) * 100
        prior_mu = np.zeros(A.shape[1])
        nan_mask = np.isfinite(mean_f_b.ravel())

        psf_w = self._solve_linear_model(
            A,
            mean_f_b.ravel(),
            k=nan_mask,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )
        self.psf_w = psf_w

        # We then build the same design matrix for all pixels with flux
        Ap = _make_A_wcs(
            self.source_mask.multiply(self.phi).data,
            self.source_mask.multiply(self.r).data,
        )

        # And create a `mean_model` that has the psf model multiplied
        # by the expected gaia flux.
        mean_model = sparse.csr_matrix(self.r.shape)
        m = 10 ** Ap.dot(psf_w)
        mean_model[self.source_mask] = m
        mean_model.eliminate_zeros()

        self.mean_model = mean_model

    def fit_model(self):
        """
        Fits sparse model matrix to all flux values iteratively in time

        Fits PSF model first using gaia flux, then updates `self.flux_estimates`
        Second iteration, uses best fit flux estimates to find PSF shape

        Fits model using self.source_mask
        """
        # self._fit_time_variable_model()
        A = (self.mean_model).T
        ws1 = np.zeros((self.nt, A.shape[1])) * np.nan
        werrs1 = np.zeros((self.nt, A.shape[1])) * np.nan

        # firs iteration uses Gaia flux as prior
        prior_mu = self.source_flux_estimates[:, 0]
        prior_sigma = np.ones(A.shape[1]) * 10 * self.source_flux_estimates[:, 0]
        k = np.isfinite(self.flux)
        for t in tqdm(np.arange(self.nt), desc="Fitting PSF model, Gaia prior"):
            ws1[t], werrs1[t] = self._solve_linear_model(
                A,
                self.flux[t],
                y_err=self.flux_err[t],
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                k=k[t],
            )
        self.psf_flux1 = ws1.T
        self.psf_flux1_err = werrs1.T

        ws2 = np.zeros_like(ws1) * np.nan
        werrs2 = np.zeros_like(werrs1) * np.nan
        # update priors using previous fitting, do we average in time?
        prior_mu = ws1.mean(axis=0)
        # prior_sigma = werrs1.mean(axis=1) * 100
        # think about how to avoidsouble werrs invertion calculation, catching?
        # update mean model
        self.build_model(update=True)
        A = (self.mean_model).T
        for t in tqdm(np.arange(self.nt), desc="Fitting PSF model, 2nd iter"):
            ws2[t], werrs2[t] = self._solve_linear_model(
                A,
                self.flux[t],
                y_err=self.flux_err[t],
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                k=k[t],
            )
        self.psf_flux2 = ws2.T
        self.psf_flux2_err = werrs2.T

    def _plot_mean_model(self):
        """
        Diagnostic plot of the mean PSF model.
        run self.build_model(first)
        """
        # Plotting r,phi,meanflux used to build PSF model
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].set_title("Mean flux")
        cax = ax[0].scatter(
            self.uncontaminated_source_mask.multiply(self.phi).data,
            self.uncontaminated_source_mask.multiply(self.r).data,
            c=self.mean_f,
            marker=".",
        )
        fig.colorbar(cax, ax=ax[0])
        ax[0].set_ylabel(r"$r^{\prime\prime}$ ")
        ax[0].set_xlabel(r"$\phi$ [rad]")

        r_test, phi_test = np.meshgrid(
            np.linspace(0 ** 0.5, self.r_b.max() ** 0.5, 50) ** 2,
            np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 50),
        )
        A_test = _make_A_wcs(phi_test.ravel(), r_test.ravel())
        model_test = A_test.dot(self.psf_w)
        model_test = model_test.reshape(phi_test.shape)

        ax[1].set_title("Average PSF Model")
        cax = ax[1].pcolormesh(phi_test, r_test, model_test)
        fig.colorbar(cax, ax=ax[1])
        ax[1].set_xlabel(r"$\phi$ [rad]")
        plt.show()

        if "counts" in dir(self):
            if self.counts.shape != self.mean_f_b.shape:
                return
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            ax[0].set_title("Counts per bin")
            cax = ax[0].pcolormesh(self.phi_b, self.r_b, self.counts)
            fig.colorbar(cax, ax=ax[0])
            ax[0].set_xlabel(r"$\phi$ [rad]")
            ax[0].set_ylabel(r"$r^{\prime\prime}$")

            ax[1].set_title("Binned mean flux")
            cax = ax[1].pcolormesh(self.phi_b, self.r_b, self.mean_f_b)
            fig.colorbar(cax, ax=ax[1])
            ax[1].set_xlabel(r"$\phi$ [rad]")
            plt.show()
        return

    def _plot_residual_scene():
        return

    def _plot_psf_fit():
        return

    @staticmethod
    def from_TPFs(tpfs):
        """
        Convert TPF input into machine object
        """
        # Checks that all TPFs have identical time sampling
        times = np.array([tpf.astropy_time.jd for tpf in tpfs])
        # 2e-5 d = 1.7 sec
        if not np.all(times[1:, :] - times[-1:, :] < 1e-4):
            raise ValueError("All TPFs must have same time basis")
        times = times[0]

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

        # Remove nan pixels
        nan_mask = np.isnan(flux)
        flux = np.array([fx[~ma] for fx, ma in zip(flux, nan_mask)])
        flux_err = np.array([fx[~ma] for fx, ma in zip(flux_err, nan_mask)])
        unw = np.array([ii[~ma] for ii, ma in zip(unw, nan_mask)])

        # Remove bad cadences where the pointing is rubbish
        bad_cadences = np.hypot(tpfs[0].pos_corr1, tpfs[0].pos_corr2) > 10
        flux_err[bad_cadences] *= 1e2
        del bad_cadences

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
        locs = locs[:, ~np.all(nan_mask, axis=0)]

        #  check for overlapped pixels between nearby tpfs, i.e. unique pixels
        _, unique_pix = np.unique(locs, axis=1, return_index=True)
        # important to srot indexes, np.unique return idx sorted by array values
        unique_pix.sort()
        locs = locs[:, unique_pix]
        flux = flux[:, unique_pix]
        flux_err = flux_err[:, unique_pix]
        unw = unw[:, unique_pix]

        # convert pixel coord to ra, dec using TPF's solution
        # what happens if tpfs don't have WCS solutions?
        ra, dec = (
            tpfs[0]
            .wcs.wcs_pix2world(
                np.vstack([(locs[0] - tpfs[0].column), (locs[1] - tpfs[0].row)]).T, 0.0
            )
            .T
        )
        # find the max circle per TPF that contain all pixel data to query Gaia
        ras, decs, rads = [], [], []
        for l in np.unique(unw[0]):
            ra1 = ra[unw[0] == l]
            dec1 = dec[unw[0] == l]
            ras.append(ra1.mean())
            decs.append(dec1.mean())
            rads.append(
                np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max() / 2
                + (u.arcsecond * 6).to(u.deg).value
            )
        # query Gaia with epoch propagation
        sources = get_gaia_sources(
            tuple(ras),
            tuple(decs),
            tuple(rads),
            magnitude_limit=18,
            epoch=Time(times[0], format="jd").jyear,
        )
        del locs, nan_mask, ra1, dec1, ras, decs, rads

        # soruce list cleaning
        sources = Machine._clean_source_list(sources, ra, dec)

        # return a Machine object
        return Machine(
            time=times, flux=flux, flux_err=flux_err, ra=ra, dec=dec, sources=sources
        )

    @staticmethod
    def _clean_source_list(sources, ra, dec):
        """
        Removes sources that are too contaminated or off the edge of the image

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
        """
        # find souerces on the image
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
        # all sources closer than 6" = 1.5 pix, this is a model parameters
        # that need to be tested
        closest = mdist.arcsec < 6.0
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
        sources.loc[:, "clean_flag"][unresolved] += 2 ** 1  # close contaminant

        # combine 2 source masks
        clean = sources.clean_flag == 0

        removed_sources = sources[~clean].reset_index(drop=True)
        sources = sources[clean].reset_index(drop=True)

        return sources


"""
Notes on user functionality

mac = Machine().from_tpfs()
mac.build_model()
mac.fit_model()

mac.model_flux  # Values
mac.model_flux_err  # values


machine.plot_tpf(4)
 machine.plot_source(13)
# show new sources
 machine.show_fresh()
"""
