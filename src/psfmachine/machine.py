"""
Defines the main Machine object that fit a mean PRF model to sources
"""
import logging
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from .utils import (
    _make_A_polar,
    solve_linear_model,
    sparse_lessthan,
    threshold_bin,
    bspline_smooth,
)
from .aperture import optimize_aperture, compute_FLFRCSAP, compute_CROWDSAP
from .perturbation import PerturbationMatrix3D

log = logging.getLogger(__name__)
__all__ = ["Machine"]


class Machine(object):
    """
    Class for calculating fast PRF photometry on a collection of images and
    a list of in image sources.

    This method is discussed in detail in
    [Hedges et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210608411H/abstract).

    This method solves a linear model to assuming Gaussian priors on the weight of
    each linear components as explained by
    [Luger, Foreman-Mackey & Hogg, 2017](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)
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
        time_nknots=10,
        time_resolution=200,
        time_radius=8,
        rmin=1,
        rmax=16,
        cut_r=6,
        sparse_dist_lim=40,
        sources_flux_column="phot_g_mean_flux",
    ):
        """
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
            Data array containing the "rows" of the detector that each pixel is on.
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
        time_nknots: int
            Number og knots for cartesian DM in time model.
        time_resolution: int
            Number of time points to bin by when fitting for velocity aberration.
        time_radius: float
            The radius around sources, out to which the velocity aberration model
            will be fit. (arcseconds)
        rmin: float
            The minimum radius for the PRF model to be fit. (arcseconds)
        rmax: float
            The maximum radius for the PRF model to be fit. (arcseconds)
        cut_r : float
            Radius distance whithin the shape model only depends on radius and not
            angle.
        sparse_dist_lim : float
            Radial distance used to include pixels around sources when creating delta
            arrays (dra, ddec, r, and phi) as sparse matrices for efficiency.
            Default is 40" (recommended for kepler). (arcseconds)
        sources_flux_column : str
            Column name in `sources` table to be used as flux estimate. For Kepler data
            gaia.phot_g_mean_flux is recommended, for TESS use gaia.phot_rp_mean_flux.

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
        cartesian_knot_spacing: string
            Defines the type of spacing between knots in cartessian space to generate
            the design matrix, options are "linear" or "sqrt".
        quiet: booleans
            Quiets TQDM progress bars.
        contaminant_mag_limit: float
          The limiting magnitude at which a sources is considered as contaminant
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
        self.time_nknots = time_nknots
        self.time_resolution = time_resolution
        self.time_radius = time_radius
        self.rmin = rmin
        self.rmax = rmax
        self.cut_r = cut_r
        self.sparse_dist_lim = sparse_dist_lim * u.arcsecond
        self.cartesian_knot_spacing = "sqrt"
        # disble tqdm prgress bar when running in HPC
        self.quiet = False
        self.contaminant_mag_limit = None

        self.source_flux_estimates = np.copy(self.sources[sources_flux_column].values)

        if time_mask is None:
            self.time_mask = np.ones(len(time), bool)
        else:
            self.time_mask = time_mask

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # Hardcoded: sparse implementation is efficient when nsourxes * npixels < 1e7
        # (JMP profile this)
        # https://github.com/SSDataLab/psfmachine/pull/17#issuecomment-866382898
        if self.nsources * self.npixels < 1e7:
            self._create_delta_arrays()
        else:
            self._create_delta_sparse_arrays()

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    def _create_delta_arrays(self, centroid_offset=[0, 0]):
        """
        Creates dra, ddec, r and phi numpy ndarrays .

        Parameters
        ----------
        centroid_offset : list
            Centroid offset for [ra, dec] to be included in dra and ddec computation.
            Default is [0, 0].
        """
        # The distance in ra & dec from each source to each pixel
        # when centroid offset is 0 (i.e. first time creating arrays) create delta
        # arrays from scratch
        if centroid_offset[0] == centroid_offset[1] == 0:
            self.dra, self.ddec = np.asarray(
                [
                    [
                        self.ra - self.sources["ra"][idx] - centroid_offset[0],
                        self.dec - self.sources["dec"][idx] - centroid_offset[1],
                    ]
                    for idx in range(len(self.sources))
                ]
            ).transpose(1, 0, 2)
            self.dra = self.dra * (u.deg)
            self.ddec = self.ddec * (u.deg)
        # when offsets are != 0 (i.e. updating dra and ddec arrays) we just substract
        # the ofsets avoiding the for loop
        else:
            self.dra -= centroid_offset[0] * u.deg
            self.ddec -= centroid_offset[1] * u.deg

        # convertion to polar coordinates
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        self.phi = np.arctan2(self.ddec, self.dra)

    def _create_delta_sparse_arrays(self, centroid_offset=[0, 0]):
        """
        Creates dra, ddec, r and phi arrays as sparse arrays to be used for dense data,
        e.g. Kepler FFIs or cluster fields. Assuming that there is no flux information
        further than `dist_lim` for a given source, we only keep pixels within the
        `dist_lim`.
        dra, ddec, ra, and phi are unitless because they are `sparse.csr_matrix`. But
        keep same scale as '_create_delta_arrays()'.
        dra and ddec in deg. r in arcseconds and phi in rads

        Parameters
        ----------
        centroid_offset : list
            Centroid offset for [ra, dec] to be included in dra and ddec computation.
            Default is [0, 0].
        """
        # If not centroid offsets or  centroid correction are larget than a pixel,
        # then we need to compute the sparse delta arrays from scratch
        if (centroid_offset[0] == centroid_offset[1] == 0) or (
            np.maximum(*np.abs(centroid_offset)) > 4 / 3600
        ):
            # iterate over sources to only keep pixels within self.sparse_dist_lim
            # this is inefficient, could be done in a tiled manner? only for squared data
            dra, ddec, sparse_mask = [], [], []
            for i in tqdm(
                range(len(self.sources)),
                desc="Creating delta arrays",
                disable=self.quiet,
            ):
                dra_aux = self.ra - self.sources["ra"].iloc[i] - centroid_offset[0]
                ddec_aux = self.dec - self.sources["dec"].iloc[i] - centroid_offset[1]
                box_mask = sparse.csr_matrix(
                    (np.abs(dra_aux) <= self.sparse_dist_lim.to("deg").value)
                    & (np.abs(ddec_aux) <= self.sparse_dist_lim.to("deg").value)
                )
                dra.append(box_mask.multiply(dra_aux))
                ddec.append(box_mask.multiply(ddec_aux))
                sparse_mask.append(box_mask)

            del dra_aux, ddec_aux, box_mask
            # we stack dra, ddec of each object to create a [nsources, npixels] matrices
            self.dra = sparse.vstack(dra, "csr")
            self.ddec = sparse.vstack(ddec, "csr")
            sparse_mask = sparse.vstack(sparse_mask, "csr")
            sparse_mask.eliminate_zeros()
        # if centroid correction is less than 1 pixel, then we just update dra and ddec
        # sparse arrays arrays and r and phi.
        else:
            self.dra = self.dra - sparse.csr_matrix(
                (
                    np.repeat(centroid_offset[0], self.dra.data.shape),
                    (self.dra.nonzero()),
                ),
                shape=self.dra.shape,
                dtype=float,
            )
            self.ddec = self.ddec - sparse.csr_matrix(
                (
                    np.repeat(centroid_offset[1], self.ddec.data.shape),
                    (self.ddec.nonzero()),
                ),
                shape=self.ddec.shape,
                dtype=float,
            )
            sparse_mask = self.dra.astype(bool)

        # convertion to polar coordinates. We can't apply np.hypot or np.arctan2 to
        # sparse arrays. We keep track of non-zero index, do math in numpy space,
        # then rebuild r, phi as sparse.
        nnz_inds = sparse_mask.nonzero()
        # convert radial dist to arcseconds
        r_vals = np.hypot(self.dra.data, self.ddec.data) * 3600
        phi_vals = np.arctan2(self.ddec.data, self.dra.data)
        self.r = sparse.csr_matrix(
            (r_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask.shape,
            dtype=float,
        )
        self.phi = sparse.csr_matrix(
            (phi_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask.shape,
            dtype=float,
        )
        del r_vals, phi_vals, nnz_inds, sparse_mask
        return

    def _get_source_mask(
        self,
        upper_radius_limit=28.0,
        lower_radius_limit=4.5,
        upper_flux_limit=2e5,
        lower_flux_limit=100,
        correct_centroid_offset=True,
        plot=False,
    ):
        """Find the pixel mask that identifies pixels with contributions from ANY
        NUMBER of Sources

        Fits a simple polynomial model to the log of the pixel flux values, in radial
        dimension and source flux, to find the optimum circular apertures for every
        source.

        Parameters
        ----------
        upper_radius_limit: float
            The radius limit at which we assume there is no flux from a source of
            any brightness (arcsec)
        lower_radius_limit: float
            The radius limit at which we assume there is flux from a source of any
            brightness (arcsec)
        upper_flux_limit: float
            The flux at which we assume as source is saturated
        lower_flux_limit: float
            The flux at which we assume a source is too faint to model
        correct_centroid_offset: bool
            Correct the dra, ddec arrays from centroid offsets. If centroid offsets are
            larger than 1 arcsec, `source_mask` will be also updated.
        plot: bool
            Whether to show diagnostic plot. Default is False
        """
        # We will use the radius a lot, this is for readibility
        # don't do if sparse array
        if isinstance(self.r, u.quantity.Quantity):
            r = self.r.value
        else:
            r = self.r

        # The average flux, which we assume is a good estimate of the whole stack of images
        max_flux = np.nanmax(self.flux[self.time_mask], axis=0)

        # Mask out sources that are above the flux limit, and pixels above the
        # radius limit. This is a good estimation only when using phot_g_mean_flux
        source_rad = 0.5 * np.log10(self.source_flux_estimates) ** 1.5 + 3
        # temp_mask for the sparse array case should also be a sparse matrix. Then it is
        # applied to r, max_flux, and, source_flux_estimates to be used later.
        # Numpy array case:
        if not isinstance(r, sparse.csr_matrix):
            # First we make a guess that each source has exactly the gaia flux
            source_flux_estimates = np.asarray(self.source_flux_estimates)[
                :, None
            ] * np.ones((self.nsources, self.npixels))
            temp_mask = (r < source_rad[:, None]) & (
                source_flux_estimates < upper_flux_limit
            )
            temp_mask &= temp_mask.sum(axis=0) == 1

            # apply temp_mask to r
            r_temp_mask = r[temp_mask]

            # log of flux values
            f = np.log10((temp_mask.astype(float) * max_flux))
            f_temp_mask = f[temp_mask]
            # weights = (
            #     (self.flux_err ** 0.5).sum(axis=0) ** 0.5 / self.flux.shape[0]
            # ) * temp_mask

            # flux estimates
            mf = np.log10(source_flux_estimates[temp_mask])
        # sparse array case:
        else:
            source_flux_estimates = self.r.astype(bool).multiply(
                self.source_flux_estimates[:, None]
            )
            temp_mask = sparse_lessthan(r, source_rad)
            temp_mask = temp_mask.multiply(
                sparse_lessthan(source_flux_estimates, upper_flux_limit)
            ).tocsr()
            temp_mask = temp_mask.multiply(temp_mask.sum(axis=0) == 1).tocsr()
            temp_mask.eliminate_zeros()

            f = np.log10(temp_mask.astype(float).multiply(max_flux).data)
            k = np.isfinite(f)
            f_temp_mask = f[k]
            r_temp_mask = temp_mask.astype(float).multiply(r).data[k]
            mf = np.log10(
                temp_mask.astype(float).multiply(source_flux_estimates).data[k]
            )

        # Model is polynomial in r and log of the flux estimate.
        # Here I'm using a 1st order polynomial, to ensure it's monatonic in each dimension
        A = np.vstack(
            [
                r_temp_mask ** 0,
                r_temp_mask,
                # r_temp_mask ** 2,
                r_temp_mask ** 0 * mf,
                r_temp_mask * mf,
                # r_temp_mask ** 2 * mf,
                # r_temp_mask ** 0 * mf ** 2,
                # r_temp_mask * mf ** 2,
                # r_temp_mask ** 2 * mf ** 2,
            ]
        ).T

        # Iteratively fit
        k = np.isfinite(f_temp_mask)
        for count in [0, 1, 2]:
            sigma_w_inv = A[k].T.dot(A[k])
            B = A[k].T.dot(f_temp_mask[k])
            w = np.linalg.solve(sigma_w_inv, B)
            res = np.ma.masked_array(f_temp_mask, ~k) - A.dot(w)
            k &= ~sigma_clip(res, sigma=3).mask

        # Now find the radius and source flux at which the model reaches the flux limit
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
                    # test_r2.ravel() ** 2,
                    test_r2.ravel() ** 0 * test_f2.ravel(),
                    test_r2.ravel() * test_f2.ravel(),
                    # test_r2.ravel() ** 2 * test_f2.ravel(),
                    # test_r2.ravel() ** 0 * test_f2.ravel() ** 2,
                    # test_r2.ravel() * test_f2.ravel() ** 2,
                    # test_r2.ravel() ** 2 * test_f2.ravel() ** 2,
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
            np.polyfit(test_f[ok], l[ok], 1), np.log10(self.source_flux_estimates)
        )
        source_radius_limit[
            source_radius_limit > upper_radius_limit
        ] = upper_radius_limit
        source_radius_limit[
            source_radius_limit < lower_radius_limit
        ] = lower_radius_limit

        # Here we set the radius for each source. We add two pixels, to be generous
        self.radius = source_radius_limit + 2

        # This sparse mask is one where where is ANY number of sources in a pixel
        if not isinstance(self.r, sparse.csr_matrix):
            self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])
        else:
            # for a sparse matrix doing < self.radius is not efficient, it
            # considers all zero values in the sparse matrix and set them to True.
            # this is a workaround to this problem.
            self.source_mask = sparse_lessthan(r, self.radius)

        self._get_uncontaminated_pixel_mask()

        # Now we can update the r and phi estimates, allowing for a slight centroid
        # calculate image centroids and correct dra,ddec for offset.
        if correct_centroid_offset:
            self._get_centroids()
            # print(self.ra_centroid_avg.to("arcsec"), self.dec_centroid_avg.to("arcsec"))
            # re-estimate dra, ddec with centroid shifts, check if sparse case applies.
            # Hardcoded: sparse implementation is efficient when nsourxes * npixels < 1e7
            # (JMP profile this)
            # https://github.com/SSDataLab/psfmachine/pull/17#issuecomment-866382898
            if self.nsources * self.npixels < 1e7:
                self._create_delta_arrays(
                    centroid_offset=[
                        self.ra_centroid_avg.value,
                        self.dec_centroid_avg.value,
                    ]
                )
            else:
                self._create_delta_sparse_arrays(
                    centroid_offset=[
                        self.ra_centroid_avg.value,
                        self.dec_centroid_avg.value,
                    ]
                )
            # if centroid offset id larger than 1" then we need to recalculate the
            # source mask to include/reject correct pixels.
            # this 1 arcsec limit only works for Kepler/K2
            if (
                np.abs(self.ra_centroid_avg.to("arcsec").value) > 1
                or np.abs(self.dec_centroid_avg.to("arcsec").value) > 1
            ):
                self._get_source_mask(correct_centroid_offset=False)

        if plot:
            k = np.isfinite(f_temp_mask)
            # Make a nice diagnostic plot
            fig, ax = plt.subplots(1, 2, figsize=(11, 3), facecolor="white")

            ax[0].scatter(r_temp_mask[k], f_temp_mask[k], s=0.4, c="k", label="Data")
            ax[0].scatter(r_temp_mask[k], A[k].dot(w), c="r", s=0.4, label="Model")
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
                shading="auto",
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

        # we flag sources fainter than mag_limit as non-contaminant
        if isinstance(self.contaminant_mag_limit, (float, int)):
            aux = self.source_mask.multiply(
                self.source_flux_estimates[:, None] < self.contaminant_mag_limit
            )
            aux.eliminate_zeros()
            self.uncontaminated_source_mask = aux.multiply(
                np.asarray(aux.sum(axis=0) == 1)[0]
            ).tocsr()
        # all sources are accounted for contamination
        else:
            self.uncontaminated_source_mask = self.source_mask.multiply(
                np.asarray(self.source_mask.sum(axis=0) == 1)[0]
            ).tocsr()

        # have to remove leaked zeros
        self.uncontaminated_source_mask.eliminate_zeros()
        return

    # CH: We're not currently using this, but it might prove useful later so I will leave for now
    def _get_centroids(self):
        """
        Find the ra and dec centroid of the image, at each time.
        """
        # centroids are astropy quantities
        self.ra_centroid = np.zeros(self.nt)
        self.dec_centroid = np.zeros(self.nt)
        dra_m = self.uncontaminated_source_mask.multiply(self.dra).data
        ddec_m = self.uncontaminated_source_mask.multiply(self.ddec).data
        for t in range(self.nt):
            wgts = self.uncontaminated_source_mask.multiply(
                np.sqrt(np.abs(self.flux[t]))
            ).data
            # mask out non finite values and background pixels
            k = (np.isfinite(wgts)) & (
                self.uncontaminated_source_mask.multiply(self.flux[t]).data > 100
            )
            self.ra_centroid[t] = np.average(dra_m[k], weights=wgts[k])
            self.dec_centroid[t] = np.average(ddec_m[k], weights=wgts[k])
        del dra_m, ddec_m
        self.ra_centroid *= u.deg
        self.dec_centroid *= u.deg
        self.ra_centroid_avg = self.ra_centroid.mean()
        self.dec_centroid_avg = self.dec_centroid.mean()

        return

    def build_time_model(
        self,
        plot=False,
        bin_method="bin",
        poly_order=3,
        segments=False,
        focus=False,
        focus_exptime=50,
        pca_ncomponents=0,
        pca_smooth_time_scale=0,
        positions=False,
        other_vectors=None,
    ):
        """
        Builds a time model that moves the PRF model to account for the scene movement
        due to velocity aberration. It has two methods to choose from using the
        attribute `self.time_corrector`, if `"polynomial"` (default) will use a
        polynomial in time, if `"poscorr"` will use the pos_corr vectos that can be found
        in the TPFs. The time polynomial gives a more flexible model vs the pos_corr
        option, but can lead to light curves with "weird" long-term trends. Using
        pos_corr is recomended for Kepler data.

        Parameters
        ----------
        plot: boolean
            Plot a diagnostic figure.
        bin_method: string
            Type of bin method, options are "bin" and "downsample".
        poly_order: int
            Degree of the time polynomial used for the time model. Default is 3.
        segments : boolean
            If `True` will split the light curve into segments to fit different time
            models with a common pixel normalization. If `False` will fit the full
            time series as one segment. Segments breaks are infered from time
            discontinuities.
        focus: boolean
            Add a component that models th focus change at the begining of a segment.
        focus_exptime : int
            Characteristic decay for focus component modeled as exponential decay when
            `focus` is True. In the same units as `PerturbationMatrix3D.time`.
        pca_ncomponents : int
            Number of PCA components used in `PerturbationMatrix3D`. The components are
            derived from pixel light lighrcurves.
        pca_smooth_time_scale : float
            Smooth time sacel for PCA components.
        positions : boolean or string
            If one of strings `"poscorr", "centroid"` then the perturbation matrix will
            add other vectors accounting for position shift.
        other_vectors : list or numpy.ndarray
            Set of other components used to include  in the perturbed model.
            See `psfmachine.perturbation.PerturbationMatrix` object for details.
            Posible use case are using Kepler CBVs or engeneering data. Shape has to be
            (ncomponents, ntimes). Default is `None`.
        """
        # create the time and space basis
        _whitened_time = (self.time - self.time.mean()) / (
            self.time.max() - self.time.mean()
        )
        dx, dy, uncontaminated_pixels = (
            self.source_mask.multiply(self.dra),
            self.source_mask.multiply(self.ddec),
            self.source_mask.multiply(self.uncontaminated_source_mask.todense()),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)
        # uncontaminated pixels mask
        uncontaminated_pixels = uncontaminated_pixels.data

        # add other vectors if asked, centroids or poscorrs
        if positions:
            if other_vectors is not None:
                raise ValueError("When using `positions` do not provide `other_vector`")
            if positions == "poscorr" and hasattr(self, "pos_corr1"):
                mpc1 = np.nanmedian(self.pos_corr1, axis=0)
                mpc2 = np.nanmedian(self.pos_corr2, axis=0)
            elif positions == "centroid" and hasattr(self, "ra_centroid"):
                mpc1 = self.ra_centroid.to("arcsec").value / 4
                mpc2 = self.dec_centroid.to("arcsec").value / 4
            else:
                raise ValueError(
                    "`position` not valid, use one of {None, 'poscorr', 'centroid'}"
                )

            # k2 centroids do not need smoothing
            if self.tpf_meta["mission"][0] == "K2":
                mpc1_smooth, mpc2_smooth = mpc1, mpc2
            else:
                # smooth the vectors
                if not segments:
                    log.warning(
                        "Segments will still be used to smooth the position vectors."
                        "See https://github.com/SSDataLab/psfmachine/pull/63 for details."
                    )
                # knot spacing every 1 day
                mpc1_smooth, mpc2_smooth = bspline_smooth(
                    [mpc1, mpc2],
                    x=self.time,
                    do_segments=True,
                    n_knots=int((self.time[-1] - self.time[0]) / 1.0),
                )
            # normalize components
            mpc1_smooth = (mpc1_smooth - mpc1_smooth.mean()) / (
                mpc1_smooth.max() - mpc1_smooth.mean()
            )
            mpc2_smooth = (mpc2_smooth - mpc2_smooth.mean()) / (
                mpc2_smooth.max() - mpc2_smooth.mean()
            )
            # combine them as the first order
            other_vectors = [mpc1_smooth, mpc2_smooth, mpc1_smooth * mpc2_smooth]
        if other_vectors is not None:
            if not isinstance(other_vectors, (list, np.ndarray)):
                raise ValueError("`other vector` is not a list of arrays or a ndarray")

        # create a 3D perturbation matrix
        P = PerturbationMatrix3D(
            time=_whitened_time,
            dx=dx,
            dy=dy,
            poly_order=poly_order,
            segments=segments,
            focus=focus,
            other_vectors=other_vectors,
            bin_method=bin_method,
            focus_exptime=focus_exptime,
            resolution=self.time_resolution,
            radius=self.time_radius,
            nknots=self.time_nknots,
            knot_spacing_type=self.cartesian_knot_spacing,
        )

        # get uncontaminated pixel norm-flux
        flux, flux_err = np.array(
            [
                np.array(
                    [
                        self.source_mask.multiply(self.flux[idx]).data,
                        self.source_mask.multiply(self.flux_err[idx]).data,
                    ]
                )
                for idx in range(self.nt)
            ]
        ).transpose((1, 0, 2))
        flux_norm = flux / np.nanmean(flux, axis=0)
        flux_err_norm = flux_err / np.nanmean(flux, axis=0)

        # create pixel mask for model fitting
        # No saturated pixels, 1e5 is a hardcoded value for Kepler.
        k = (flux < 1e5).all(axis=0)[None, :] * np.ones(flux.shape, bool)
        # No faint pixels, 100 is a hardcoded value for Kepler.
        k &= (flux > 100).all(axis=0)[None, :] * np.ones(flux.shape, bool)
        # No huge variability
        _per_pix = np.percentile(flux_norm, [3, 97], axis=1)
        k &= (
            (
                (flux_norm < _per_pix[0][:, None]) | (flux_norm > _per_pix[1][:, None])
            ).sum(axis=0)
            < flux_norm.shape[0] * 0.95
        )[None, :] * np.ones(flux_norm.shape, bool)
        # No nans
        k &= np.isfinite(flux_norm) & np.isfinite(flux_err_norm)
        k = np.all(k, axis=0)
        # combine good-behaved pixels with uncontaminated pixels
        k &= uncontaminated_pixels

        # adding PCA components to pertrubation matrix
        if pca_ncomponents > 0:
            # select only bright pixels
            k &= (flux > 300).all(axis=0)
            P.pca(
                flux_norm[:, k],
                ncomponents=pca_ncomponents,
                smooth_time_scale=pca_smooth_time_scale,
            )

        # bindown flux arrays
        flux_binned = P.bin_func(flux_norm)
        flux_err_binned = P.bin_func(flux_err_norm, quad=True)

        # iterate to remvoe outliers
        for count in [0, 1, 2]:
            P.fit(flux_norm, flux_err=flux_err_norm, pixel_mask=k)
            res = flux_binned - P.matrix.dot(P.weights).reshape(flux_binned.shape)
            chi2 = np.sum((res) ** 2 / (flux_err_binned ** 2), axis=0) / P.nbins
            bad_targets = sigma_clip(chi2, sigma=5).mask
            bad_targets = bad_targets.all(axis=0)
            k &= ~bad_targets

        # book keeping
        self.flux_binned = flux_binned
        self._time_masked_pix = k
        self.P = P
        if plot:
            return self.plot_time_model()
        return

    def perturbed_model(self, time_index):
        """
        Computes the perturbed model at a given time
        Parameters
        ----------
        time_index : int or np.ndarray
            Time index where to evaluate the perturbed model.
        """
        X = self.mean_model.copy()
        X.data *= self.P.model(time_indices=time_index).ravel()
        return X

    def plot_time_model(self):
        """
        Diagnostic plot of time model.

        Returns
        -------
        fig : matplotlib.Figure
            Figure.
        """
        model_binned = self.P.matrix.dot(self.P.weights).reshape(self.flux_binned.shape)
        fig1, ax = plt.subplots(2, 2, figsize=(9, 7), facecolor="w")
        # k1 = self._time_masked_pix.reshape(self.flux_binned.shape)[0].astype(bool)
        im = ax[0, 0].scatter(
            self.P.dx[self._time_masked_pix],
            self.P.dy[self._time_masked_pix],
            c=self.flux_binned[0, self._time_masked_pix],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[0, 1].scatter(
            self.P.dx[self._time_masked_pix],
            self.P.dy[self._time_masked_pix],
            c=self.flux_binned[-1, self._time_masked_pix],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[1, 0].scatter(
            self.P.dx[self._time_masked_pix],
            self.P.dy[self._time_masked_pix],
            c=model_binned[0, self._time_masked_pix],
            s=3,
            vmin=0.5,
            vmax=1.5,
            cmap="coolwarm",
            rasterized=True,
        )
        ax[1, 1].scatter(
            self.P.dx[self._time_masked_pix],
            self.P.dy[self._time_masked_pix],
            c=model_binned[-1, self._time_masked_pix],
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

        cbar = fig1.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("Normalized Flux")

        flux = np.array(
            [self.source_mask.multiply(self.flux[idx]).data for idx in range(self.nt)]
        )
        flux_sort = np.argsort(np.nanmean(flux[:, self._time_masked_pix], axis=0))
        data_binned_clean = self.flux_binned[:, self._time_masked_pix]
        model_binned_clean = model_binned[:, self._time_masked_pix]

        fig2, ax = plt.subplots(1, 3, figsize=(18, 5))
        im = ax[0].imshow(
            data_binned_clean.T[flux_sort],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.9,
            vmax=1.1,
        )
        ax[0].set(
            xlabel="Binned Time Index", ylabel="Flux-Sorted Clean Pixels", title="Data"
        )

        im = ax[1].imshow(
            model_binned_clean.T[flux_sort],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.9,
            vmax=1.1,
        )
        ax[1].set(xlabel="Binned Time Index", title="Perturbed Model")

        cbar = fig2.colorbar(
            im, ax=ax[:2], shrink=0.7, orientation="horizontal", location="bottom"
        )
        cbar.set_label("Normalized Flux")

        im = ax[2].imshow(
            (model_binned_clean / data_binned_clean).T[flux_sort],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.97,
            vmax=1.03,
        )
        ax[2].set(xlabel="Binned Time Index", title="Perturbed Model / Data")

        cbar = fig2.colorbar(
            im, ax=ax[2], shrink=0.7, orientation="horizontal", location="bottom"
        )
        cbar.set_label("")

        return fig1, fig2

    def build_shape_model(
        self, plot=False, flux_cut_off=1, frame_index="mean", bin_data=False, **kwargs
    ):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        plot : boolean
            Make a diagnostic plot
        flux_cut_off: float
            the flux in COUNTS at which to stop evaluating the model!
        frame_index : string or int
            The frame index used to build the shape model, if "mean" then use the
            mean value across time
        bin_data : boolean
            Bin flux data spatially to increase SNR before fitting the shape model
        **kwargs
            Keyword arguments to be passed to `_get_source_mask()`
        """

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        # if not hasattr(self, "source_mask"):
        self._get_source_mask(**kwargs)

        # for iter in range(niters):
        flux_estimates = self.source_flux_estimates[:, None]

        if frame_index == "mean":
            f = (self.flux[self.time_mask]).mean(axis=0)
        elif isinstance(frame_index, int):
            f = self.flux[frame_index]
        # f, fe = (self.flux[self.time_mask]).mean(axis=0), (
        #     (self.flux_err[self.time_mask] ** 2).sum(axis=0) ** 0.5
        # ) / (self.nt)

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

        # take value from Quantity is not necessary
        phi_b = self.uncontaminated_source_mask.multiply(self.phi).data
        r_b = self.uncontaminated_source_mask.multiply(self.r).data

        if bin_data:
            # number of bins is hardcoded to work with FFI or TPFs accordingly
            # I found 30 wirks good with TPF stacks (<10000 pixels),
            # 90 with FFIs (tipically >50k pixels), and 60 in between.
            # this could be improved later if necessary
            nbins = (
                30 if mean_f.shape[0] <= 1e4 else (60 if mean_f.shape[0] <= 5e4 else 90)
            )
            _, phi_b, r_b, mean_f, mean_f_err = threshold_bin(
                phi_b,
                r_b,
                mean_f,
                z_err=mean_f_err,
                bins=nbins,
                abs_thresh=5,
            )

        # build a design matrix A with b-splines basis in radius and angle axis.
        A = _make_A_polar(
            phi_b.ravel(),
            r_b.ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        prior_sigma = np.ones(A.shape[1]) * 10
        prior_mu = np.zeros(A.shape[1]) - 10

        nan_mask = np.isfinite(mean_f.ravel())

        # we solve for A * psf_w = mean_f
        psf_w, psf_w_err = solve_linear_model(
            A,
            y=mean_f.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        bad = sigma_clip(mean_f.ravel() - A.dot(psf_w), sigma=5).mask

        psf_w, psf_w_err = solve_linear_model(
            A,
            y=mean_f.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask & ~bad,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        self.psf_w = psf_w
        self.psf_w_err = psf_w_err
        self.normalized_shape_model = False

        # We then build the same design matrix for all pixels with flux
        # this non-normalized mean model is temporary and used to re-create a better
        # `source_mask`
        self._get_mean_model()
        # remove background pixels and recreate mean model
        self._update_source_mask_remove_bkg_pixels(
            flux_cut_off=flux_cut_off, frame_index=frame_index
        )

        if plot:
            return self.plot_shape_model(frame_index=frame_index, bin_data=bin_data)
        return

    def _update_source_mask_remove_bkg_pixels(self, flux_cut_off=1, frame_index="mean"):
        """
        Update the `source_mask` to remove pixels that do not contribuite to the PRF
        shape.
        First, re-estimate the source flux usign the precomputed `mean_model`.
        This re-estimation is used to remove sources with bad prediction and update
        the `source_mask` by removing background pixels that do not contribuite to
        the PRF shape.
        Pixels with normalized flux > `flux_cut_off` are kept.

        Parameters
        ----------
        flux_cut_off : float
            Lower limit for the normalized flux predicted from the mean model.
        frame_index : string or int
            The frame index to be used, if "mean" then use the
            mean value across time
        """

        # Re-estimate source flux
        # -----
        prior_mu = self.source_flux_estimates
        prior_sigma = (
            np.ones(self.mean_model.shape[0]) * 10 * self.source_flux_estimates
        )

        if frame_index == "mean":
            f, fe = (self.flux).mean(axis=0), (
                (self.flux_err ** 2).sum(axis=0) ** 0.5
            ) / (self.nt)
        elif isinstance(frame_index, int):
            f, fe = self.flux[frame_index], self.flux_err[frame_index]

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

        # create the final normalized mean model!
        self._get_normalized_mean_model()
        # self._get_mean_model()

    def _get_mean_model(self):
        """Convenience function to make the scene model"""
        Ap = _make_A_polar(
            self.source_mask.multiply(self.phi).data,
            self.source_mask.multiply(self.r).data,
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
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

    def _get_normalized_mean_model(self, npoints=300, plot=False):
        """
        Renomarlize shape model to sum 1

        Parameters
        ----------
        npoints : int
            Number of points used to build a high resolution grid in polar coordinates
        plot : boolean
            Create a diagnostic plot
        """

        # create a high resolution polar grid
        r = self.source_mask.multiply(self.r).data
        phi_hd = np.linspace(-np.pi, np.pi, npoints)
        r_hd = np.linspace(0, r.max(), npoints)
        phi_hd, r_hd = np.meshgrid(phi_hd, r_hd)

        # high res DM
        Ap = _make_A_polar(
            phi_hd.ravel(),
            r_hd.ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        # evaluate the high res model
        mean_model_hd = Ap.dot(self.psf_w)
        mean_model_hd[~np.isfinite(mean_model_hd)] = np.nan
        mean_model_hd = mean_model_hd.reshape(phi_hd.shape)

        # mask out datapoint that don't contribuite to the psf
        mean_model_hd_ma = mean_model_hd.copy()
        mask = mean_model_hd > -3
        mean_model_hd_ma[~mask] = -np.inf
        mask &= ~((r_hd > 14) & (np.gradient(mean_model_hd_ma, axis=0) > 0))
        mean_model_hd_ma[~mask] = -np.inf

        # double integral using trapezoidal rule
        self.mean_model_integral = np.trapz(
            np.trapz(10 ** mean_model_hd_ma, r_hd[:, 0], axis=0),
            phi_hd[0, :],
            axis=0,
        )
        # renormalize weights and build new shape model
        if not self.normalized_shape_model:
            self.psf_w *= np.log10(self.mean_model_integral)
            self.normalized_shape_model = True
        self._get_mean_model()

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            im = ax[0].scatter(
                phi_hd.ravel(),
                r_hd.ravel(),
                c=mean_model_hd_ma.ravel(),
                vmin=-3,
                vmax=-1,
                s=1,
                label=r"$\int = $" + f"{self.mean_model_integral:.4f}",
            )
            im = ax[1].scatter(
                r_hd.ravel() * np.cos(phi_hd.ravel()),
                r_hd.ravel() * np.sin(phi_hd.ravel()),
                c=mean_model_hd_ma.ravel(),
                vmin=-3,
                vmax=-1,
                s=1,
            )
            ax[0].legend()
            fig.colorbar(im, ax=ax, location="bottom")
            plt.show()

    def get_psf_metrics(self, npoints_per_pixel=10):
        """
        Computes three metrics for the PSF model:
            source_psf_fraction: the amount of PSF in the data. Tells how much of a
                sources is used to estimate the PSF, values are in between [0, 1].
            perturbed_ratio_mean: the ratio between the mean model and perturbed model
                for each source. Usefull to find when the time model affects the
                mean value of the light curve.
            perturbed_std: the standard deviation of the perturbed model for each
                source. USeful to find when the time model introduces variability in the
                light curve.

        If npoints_per_pixel > 0, it creates high npoints_per_pixel shape models for
        each source by dividing each pixels into a grid of
        [npoints_per_pixel x npoints_per_pixel]. This provides a better estimate of
        `source_psf_fraction`.

        Parameters
        ----------
        npoints_per_pixel : int
            Value in which each pixel axis is split to increase npoints_per_pixel.
            Default is 0 for no subpixel npoints_per_pixel.

        """
        if npoints_per_pixel > 0:
            # find from which observation (TPF) a sources comes
            obs_per_pixel = self.source_mask.multiply(self.pix2obs).tocsr()
            tpf_idx = []
            for k in range(self.source_mask.shape[0]):
                pix = obs_per_pixel[k].data
                mode = stats.mode(pix)[0]
                if len(mode) > 0:
                    tpf_idx.append(mode[0])
                else:
                    tpf_idx.append(
                        [x for x, ss in enumerate(self.tpf_meta["sources"]) if k in ss][
                            0
                        ]
                    )
            tpf_idx = np.array(tpf_idx)

            # get the pix coord for each source, we know how to increase resolution in
            # the pixel space but not in WCS
            row = self.source_mask.multiply(self.row).tocsr()
            col = self.source_mask.multiply(self.column).tocsr()
            mean_model_hd_sum = []
            # iterating per sources avoids creating a new super large `source_mask`
            # with high resolution, which a priori is hard
            for k in range(self.nsources):
                # find row, col combo for each source
                row_ = row[k].data
                col_ = col[k].data
                colhd, rowhd = [], []
                # pixels are divided into `resolution` - 1 subpixels
                for c, r in zip(col_, row_):
                    x = np.linspace(c - 0.5, c + 0.5, npoints_per_pixel + 1)
                    y = np.linspace(r - 0.5, r + 0.5, npoints_per_pixel + 1)
                    x, y = np.meshgrid(x, y)
                    colhd.extend(x[:, :-1].ravel())
                    rowhd.extend(y[:-1].ravel())
                colhd = np.array(colhd)
                rowhd = np.array(rowhd)
                # convert to ra, dec beacuse machine shape model works in sky coord
                rahd, dechd = self.tpfs[tpf_idx[k]].wcs.wcs_pix2world(
                    colhd - self.tpfs[tpf_idx[k]].column,
                    rowhd - self.tpfs[tpf_idx[k]].row,
                    0,
                )
                drahd = rahd - self.sources["ra"][k]
                ddechd = dechd - self.sources["dec"][k]
                drahd = drahd * (u.deg)
                ddechd = ddechd * (u.deg)
                rhd = np.hypot(drahd, ddechd).to("arcsec").value
                phihd = np.arctan2(ddechd, drahd).value
                # create a high resolution DM
                Ap = _make_A_polar(
                    phihd.ravel(),
                    rhd.ravel(),
                    rmin=self.rmin,
                    rmax=self.rmax,
                    cut_r=self.cut_r,
                    n_r_knots=self.n_r_knots,
                    n_phi_knots=self.n_phi_knots,
                )
                # evaluate the HD model
                modelhd = 10 ** Ap.dot(self.psf_w)
                # compute the model sum for source, how much of the source is in data
                mean_model_hd_sum.append(
                    np.trapz(modelhd, dx=1 / npoints_per_pixel ** 2)
                )

            # get normalized psf fraction metric
            self.source_psf_fraction = np.array(
                mean_model_hd_sum
            )  # / np.nanmax(mean_model_hd_sum)
        else:
            self.source_psf_fraction = np.array(self.mean_model.sum(axis=1)).ravel()

        # time model metrics
        if hasattr(self, "P"):
            perturbed_lcs = np.vstack(
                [
                    np.array(self.perturbed_model(time_index=k).sum(axis=1)).ravel()
                    for k in range(self.time.shape[0])
                ]
            )
            self.perturbed_ratio_mean = (
                np.nanmean(perturbed_lcs, axis=0)
                / np.array(self.mean_model.sum(axis=1)).ravel()
            )
            self.perturbed_std = np.nanstd(perturbed_lcs, axis=0)

    def plot_shape_model(self, frame_index="mean", bin_data=False):
        """
        Diagnostic plot of shape model.

        Parameters
        ----------
        frame_index : string or int
            The frame index used to plot the shape model, if "mean" then use the
            mean value across time
        bin_data : bool
            Bin or not the pixel data in a 2D historgram, default is False.

        Returns
        -------
        fig : matplotlib.Figure
            Figure.
        """
        if frame_index == "mean":
            mean_f = np.log10(
                self.uncontaminated_source_mask.astype(float)
                .multiply(self.flux[self.time_mask].mean(axis=0))
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
        elif isinstance(frame_index, int):
            mean_f = np.log10(
                self.uncontaminated_source_mask.astype(float)
                .multiply(self.flux[frame_index])
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra),
            self.uncontaminated_source_mask.multiply(self.ddec),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        radius = np.maximum(np.abs(dx).max(), np.abs(dy).max()) * 1.1
        vmin, vmax = np.nanpercentile(mean_f, [5, 90])

        if bin_data:
            nbins = 30 if mean_f.shape[0] <= 5e3 else 90
            _, dx, dy, mean_f, _ = threshold_bin(
                dx, dy, mean_f, bins=nbins, abs_thresh=5
            )

        fig, ax = plt.subplots(3, 2, figsize=(9, 10.5), constrained_layout=True)
        im = ax[0, 0].scatter(
            dx,
            dy,
            c=mean_f,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[0, 0].set(
            ylabel=r'$\delta y$ ["]',
            title="Data (cadence %s)" % str(frame_index),
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        # arrow to show centroid offset correction
        if hasattr(self, "ra_centroid_avg"):
            ax[0, 0].arrow(
                0,
                0,
                self.ra_centroid_avg.to("arcsec").value,
                self.dec_centroid_avg.to("arcsec").value,
                width=1e-6,
                shape="full",
                head_width=0.05,
                head_length=0.1,
                color="tab:red",
            )

        phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)
        im = ax[0, 1].scatter(
            phi, r, c=mean_f, cmap="viridis", vmin=vmin, vmax=vmax, s=3, rasterized=True
        )
        ax[0, 1].set(
            ylabel='$r$ ["]',
            title="Data (cadence %s)" % str(frame_index),
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )

        A = _make_A_polar(
            phi,
            r,
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        # if the mean model is normalized, we revert it only for plotting to make
        # easier the comparisson between the data and model.
        # the normalization is a multiplicative factor
        if self.normalized_shape_model:
            model = A.dot(self.psf_w / np.log10(self.mean_model_integral))
        else:
            model = A.dot(self.psf_w)
        im = ax[1, 1].scatter(
            phi,
            r,
            c=model,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[1, 1].set(
            ylabel=r'$r$ ["]',
            title="Model",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )
        im = ax[1, 0].scatter(
            dx,
            dy,
            c=model,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[1, 0].set(
            ylabel=r'$\delta y$ ["]',
            title="Model",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        ax[0, 0].set_aspect("equal", adjustable="box")
        ax[1, 0].set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(im, ax=ax[:2, 1], shrink=0.7, location="right")
        cbar.set_label("log$_{10}$ Normalized Flux")
        mean_f = 10 ** mean_f
        model = 10 ** model

        im = ax[2, 0].scatter(
            dx,
            dy,
            c=(model - mean_f) / mean_f,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            s=3,
            rasterized=True,
        )
        ax[2, 1].scatter(
            phi,
            r,
            c=(model - mean_f) / mean_f,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            s=3,
            rasterized=True,
        )
        ax[2, 0].set_aspect("equal", adjustable="box")
        ax[2, 0].set(
            xlabel=r'$\delta x$ ["]',
            ylabel=r'$\delta y$ ["]',
            title="Residuals",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        ax[2, 1].set(
            xlabel=r"$\phi$ [$^\circ$]",
            ylabel='$r$ ["]',
            title="Residuals",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )
        cbar = fig.colorbar(im, ax=ax[2, 1], shrink=0.9, location="right")
        cbar.set_label("(F$_M$ - F$_D$)/F$_D$")

        return fig

    def fit_model(self, fit_va=False):
        """
        Finds the best fitting weights for every source, simultaneously

        Parameters
        ----------
        fit_va : boolean
            Fitting model accounting for velocity aberration. If `True`, then a time
            model has to be built previously with `build_time_model`.
        """

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
            self.ws_va = np.zeros((self.nt, self.mean_model.shape[0]))
            self.werrs_va = np.zeros((self.nt, self.mean_model.shape[0]))

        for tdx in tqdm(
            range(self.nt),
            desc=f"Fitting {self.nsources} Sources (w. VA)",
            disable=self.quiet,
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
            self.model_flux[tdx] = X.dot(self.ws[tdx])

            if fit_va:
                if not hasattr(self, "P"):
                    raise ValueError(
                        "Please use `build_time_model` before fitting with velocity "
                        "aberration."
                    )

                X = self.perturbed_model(tdx)
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

        # check bad estimates
        nodata = np.asarray(self.mean_model.sum(axis=1))[:, 0] == 0
        # These sources are poorly estimated
        # nodata |= (self.mean_model.max(axis=1) > 1).toarray()[:, 0]
        self.ws[:, nodata] *= np.nan
        self.werrs[:, nodata] *= np.nan
        if fit_va:
            self.ws_va[:, nodata] *= np.nan
            self.werrs_va[:, nodata] *= np.nan

        return

    # aperture photometry functions
    def create_aperture_mask(self, percentile=50):
        """
        Function to create the aperture mask of a given source for a given aperture
        size. This function can compute aperutre mask for all sources in the scene.

        It creates three new attributes:
            * `self.aperture_mask` has the aperture mask, shape is [n_surces, n_pixels]
            * `self.FLFRCSAP` has the completeness metric, shape is [n_sources]
            * `self.CROWDSAP` has the crowdeness metric, shape is [n_sources]

        Parameters
        ----------
        percentile : float or list of floats
            Percentile value that defines the isophote from the distribution
            of values in the PRF model of the source. If float, then
            all sources will use the same percentile value. If list, then it has to
            have lenght that matches `self.nsources`, then each source has its own
            percentile value.

        """
        if type(percentile) == int:
            percentile = [percentile] * self.nsources
        if len(percentile) != self.nsources:
            raise ValueError("Lenght of percentile doesn't match number of sources.")
        # compute isophot limit allowing for different source percentile
        cut = np.array(
            [
                np.nanpercentile(obj.data, per)
                for obj, per in zip(self.mean_model, percentile)
            ]
        )
        # create aperture mask
        self.aperture_mask = np.array(self.mean_model >= cut[::, None])
        # make sure there are no all-pixel apertures due to `mean_model[i] = 0`
        self.aperture_mask[self.aperture_mask.sum(axis=1) == self.npixels] = False
        # compute flux metrics. Have to round to 10th decimal due to floating point
        self.FLFRCSAP = np.round(
            compute_FLFRCSAP(self.mean_model, self.aperture_mask), 10
        )
        self.CROWDSAP = np.round(
            compute_CROWDSAP(self.mean_model, self.aperture_mask), 10
        )

    def compute_aperture_photometry(
        self, aperture_size="optimal", target_complete=0.9, target_crowd=0.9
    ):
        """
        Computes aperture photometry for all sources in the scene. The aperture shape
        follow the PRF profile.

        Parameters
        ----------
        aperture_size : string or int
            Size of the aperture to be used. If "optimal" the aperture will be optimized
            using the flux metric targets. If int between [0, 100], then the boundaries
            of the aperture are calculated from the normalized flux value of the given
            ith percentile.
        target_complete : float
            Target flux completeness metric (FLFRCSAP) used if aperture_size is
             "optimal".
        target_crowd : float
            Target flux crowding metric (CROWDSAP) used if aperture_size is "optimal".
        """
        if aperture_size == "optimal":
            # raise NotImplementedError
            optimal_percentile = optimize_aperture(
                self.mean_model,
                target_complete=target_complete,
                target_crowd=target_crowd,
                quiet=self.quiet,
            )
            self.create_aperture_mask(percentile=optimal_percentile)
            self.optimal_percentile = optimal_percentile
        else:
            self.create_aperture_mask(percentile=aperture_size)

        self.sap_flux = np.zeros((self.flux.shape[0], self.nsources))
        self.sap_flux_err = np.zeros((self.flux.shape[0], self.nsources))

        for sdx in tqdm(
            range(len(self.aperture_mask)),
            desc="SAP",
            leave=True,
            disable=self.quiet,
        ):
            self.sap_flux[:, sdx] = self.flux[:, self.aperture_mask[sdx]].sum(axis=1)
            self.sap_flux_err[:, sdx] = (
                np.power(self.flux_err[:, self.aperture_mask[sdx]], 2).sum(axis=1)
                ** 0.5
            )

        return
