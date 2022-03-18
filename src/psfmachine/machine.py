"""
Defines the main Machine object that fit a mean PRF model to sources
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from .utils import (
    _make_A_polar,
    _make_A_cartesian,
    solve_linear_model,
    sparse_lessthan,
    _combine_A,
    threshold_bin,
    _find_uncontaminated_pixels,
)
from .aperture import optimize_aperture, compute_FLFRCSAP, compute_CROWDSAP

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
        n_time_knots=10,
        n_time_points=200,
        time_radius=8,
        rmin=1,
        rmax=16,
        cut_r=6,
        sparse_dist_lim=40,
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
        n_time_points: int
            Number of time points to bin by when fitting for velocity aberration.
        time_radius: float
            The radius around sources, out to which the velocity aberration model
            will be fit. (arcseconds)
        rmin: float
            The minimum radius for the PRF model to be fit. (arcseconds)
        rmax: float
            The maximum radius for the PRF model to be fit. (arcseconds)
        sparse_dist_lim : float
            Radial distance used to include pixels around sources when creating delta
            arrays (dra, ddec, r, and phi) as sparse matrices for efficiency.
            Default is 40" (recommended for kepler). (arcseconds)

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
        rough_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with pixels that are close to sources, simple round mask
        source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with pixels that contains flux from sources
        uncontaminated_source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with selected uncontaminated pixels per source to be used to
            build the PSF model
        mean_model: scipy.sparce.csr_matrix
            Mean PSF model values per pixel used for PSF photometry
        time_corrector: string
            The type of time corrector that will be used to build the time model,
            default is a "polynomial" for a polynomial in time, it can also be "pos_corr"
        poscorr_filter_size: int
            Standard deviation for Gaussian kernel to be used to smooth the pos_corrs
        cartesian_knot_spacing: string
            Defines the type of spacing between knots in cartessian space to generate
            the design matrix, options are "linear" or "sqrt".
        quiet: booleans
            Quiets TQDM progress bars.
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
        self.cut_r = cut_r
        self.sparse_dist_lim = sparse_dist_lim * u.arcsecond
        self.time_corrector = "centroid"
        # if pos_corr, then we can smooth the vector with a Gaussian kernel of size
        # poscorr_filter_size, if this is < 0.5 -> no smoothing, default is 12
        # beacause of 6hr-CDPP
        self.poscorr_filter_size = 12
        self.cartesian_knot_spacing = "linear"
        self.pixel_scale = (
            np.hypot(np.min(np.diff(self.ra)), np.min(np.diff(self.dec))) * u.deg
        ).to(u.arcsecond)

        # disble tqdm prgress bar when running in HPC
        self.quiet = False

        if time_mask is None:
            self.time_mask = np.ones(len(time), bool)
        else:
            self.time_mask = time_mask

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        self.source_flux_estimates = np.copy(np.asarray(self.sources.phot_g_mean_flux))
        # Hardcoded: sparse implementation is efficient when nsourxes * npixels < 1e7
        # (JMP profile this)
        # https://github.com/SSDataLab/psfmachine/pull/17#issuecomment-866382898
        self.ra_centroid, self.dec_centroid = np.zeros((2)) * u.deg
        self._update_delta_arrays()
        self._get_centroid()
        self._update_delta_arrays()
        self._get_source_mask()
        # self._update_delta_arrays()
        return

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    @property
    def dx(self):
        """Delta RA, corrected for centroid shift"""
        if not sparse.issparse(self.dra):
            return self.dra - self.ra_centroid.value
        else:
            ra_offset = sparse.csr_matrix(
                (
                    np.repeat(
                        self.ra_centroid.value,
                        self.dra.data.shape,
                    ),
                    (self.dra.nonzero()),
                ),
                shape=self.dra.shape,
                dtype=float,
            )
            return self.dra - ra_offset

    @property
    def dy(self):
        """Delta Dec, corrected for centroid shift"""
        if not sparse.issparse(self.ddec):
            return self.ddec - self.dec_centroid.value
        else:
            dec_offset = sparse.csr_matrix(
                (
                    np.repeat(
                        self.dec_centroid.value,
                        self.ddec.data.shape,
                    ),
                    (self.ddec.nonzero()),
                ),
                shape=self.ddec.shape,
                dtype=float,
            )
            return self.ddec - dec_offset

    def _update_delta_arrays(self, frame_indices="mean"):
        if self.nsources * self.npixels < 1e7:
            self._update_delta_numpy_arrays()
        else:
            self._update_delta_sparse_arrays()

    def _update_delta_numpy_arrays(self, frame_indices="mean"):
        """
        Creates dra, ddec, r and phi numpy ndarrays .

        Parameters
        ----------
        frame_indices : list or str
            "mean" takes the mean of all the centroids in "time_mask"

        """
        # The distance in ra & dec from each source to each pixel
        # when centroid offset is 0 (i.e. first time creating arrays) create delta
        # arrays from scratch

        if not hasattr(self, "dra"):
            self.dra, self.ddec = np.asarray(
                [
                    [
                        self.ra - self.sources["ra"][idx],
                        self.dec - self.sources["dec"][idx],
                    ]
                    for idx in range(len(self.sources))
                ]
            ).transpose(1, 0, 2)
            self.dra = self.dra * (u.deg)
            self.ddec = self.ddec * (u.deg)

        # convertion to polar coordinates
        self.r = (
            np.hypot(
                self.dx,
                self.dy,
            )
            * 3600
        )
        self.phi = np.arctan2(
            self.dy,
            self.dx,
        )

    def _update_delta_sparse_arrays(self, frame_indices="mean", dist_lim=50):
        """
        Creates dra, ddec, r and phi arrays as sparse arrays to be used for dense data,
        e.g. Kepler FFIs or cluster fields. Assuming that there is no flux information
        further than `dist_lim` for a given source, we only keep pixels within the
        `dist_lim`.
        dra, ddec, ra, and phi are unitless because they are `sparse.csr_matrix`. But
        keep same scale as '_update_delta_arrays()'.
        dra and ddec in deg. r in arcseconds and phi in rads

        Parameters
        ----------
        dist_lim : float
            Distance limit (in arcsecds) at which pixels are keep.
        centroid_offset : list
            Centroid offset for [ra, dec] to be included in dra and ddec computation.
            Default is [0, 0].
        """
        if frame_indices == "mean":
            frame_indices = np.where(self.time_mask)[0]
        # If not centroid offsets or  centroid correction are larget than a pixel,
        # then we need to compute the sparse delta arrays from scratch
        if not hasattr(self, "dra"):
            # iterate over sources to only keep pixels within dist_lim
            # this is inefficient, could be done in a tiled manner? only for squared data
            dra, ddec, sparse_mask = [], [], []
            for i in tqdm(
                range(len(self.sources)),
                desc="Creating delta arrays",
                disable=self.quiet,
            ):
                dra_aux = self.ra - self.sources["ra"].iloc[i]
                ddec_aux = self.dec - self.sources["dec"].iloc[i]
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

        sparse_mask = self.dra.astype(bool)

        # convertion to polar coordinates. We can't apply np.hypot or np.arctan2 to
        # sparse arrays. We keep track of non-zero index, do math in numpy space,
        # then rebuild r, phi as sparse.
        nnz_inds = sparse_mask.nonzero()
        # convert radial dist to arcseconds
        r_vals = np.hypot(self.dx.data, self.dy.data) * 3600
        phi_vals = np.arctan2(self.dy.data, self.dx.data)
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

    def _get_source_mask(self, source_flux_limit=1):
        """Find the round pixel mask that identifies pixels with contributions from ANY of source

        Firstly, makes a `rough_mask` that is ~2 pixels in radius. Then fits a simple
        linear trend in radius and flux. Uses this linear trend to identify pixels
        that are likely to be over the flux limit, the `source_mask`.

        We then iterate, masking out contaminated pixels in the `source_mask`, to get a better fit
        to the simple linear trend.

        Parameters
        ----------
        source_radius_limit: float
            Upper limit on radius of circular apertures in arcsecond.
        """
        self.radius = 2 * self.pixel_scale.to(u.arcsecond).value
        if not isinstance(self.r, sparse.csr_matrix):
            self.rough_mask = sparse.csr_matrix(self.r < self.radius[:, None])
        else:
            self.rough_mask = sparse_lessthan(self.r, self.radius)
        self.source_mask = self.rough_mask.copy()
        self.source_mask.eliminate_zeros()
        self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)
        for count in [0, 1]:
            # self._get_centroid()
            # self._update_delta_arrays()

            mask = self.uncontaminated_source_mask
            r = mask.multiply(self.r).data
            max_f = np.log10(
                mask.astype(float)
                .multiply(np.max(self.flux[self.time_mask], axis=0))
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
            sourcef = np.log10(
                mask.astype(float).multiply(self.source_flux_estimates[:, None]).data
            )
            rbins = np.linspace(0, self.radius * 5, 50)
            masks = np.asarray(
                [
                    (r > rbins[idx]) & (r <= rbins[idx + 1])
                    for idx in range(len(rbins) - 1)
                ]
            )
            fbins = np.asarray([np.nanpercentile(max_f[m], 20) for m in masks])
            rbins = rbins[1:] - np.median(np.diff(rbins))
            k = np.isfinite(fbins)
            l = np.polyfit(rbins[k], fbins[k], 1)

            mean_model = self.r.copy()
            mean_model.data = 10 ** np.polyval(l, mean_model.data)
            self.source_mask = (
                mean_model.multiply(self.source_flux_estimates[:, None])
            ) > source_flux_limit
            self.uncontaminated_source_mask = _find_uncontaminated_pixels(
                self.source_mask
            )
        return
        # self.radius = (
        #     rbins[k][
        #         np.where(
        #             np.cumsum(10 ** fbins[k]) / np.nansum(10 ** fbins[k]) > 0.95
        #         )[0][0]
        #     ]
        #     + 5
        # )
        #     if not isinstance(self.r, sparse.csr_matrix):
        #         self.source_mask = sparse.csr_matrix(self.r < self.radius[:, None] * 5)
        #     else:
        #         self.source_mask = sparse_lessthan(self.r, self.radius * 5)
        #
        #     m = self.source_mask.multiply(self.r).copy()
        #     # scipy sparse does -not- like that we are setting the `data` attribute
        #     m.data = 10 ** np.polyval(l, self.source_mask.multiply(self.r).data)
        #     m = m.multiply(self.source_flux_estimates[:, None])
        #     self.source_mask = m > source_flux_limit
        # # if not isinstance(self.r, sparse.csr_matrix):
        # #     self.rough_mask = sparse.csr_matrix(self.r < self.radius[:, None])
        # # else:
        # #     self.rough_mask = sparse_lessthan(self.r, self.radius)
        # self.source_mask.eliminate_zeros()
        # self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)

    # def dra(self):
    #     if not hasattr(self, 'ra_centroid'):
    #         return self.uncontaminated_source_mask.multiply(self.dra).data
    #     else:

    # def _get_centroids(self, plot=False):
    #     """
    #     Find the ra and dec centroid of the image, at each time.
    #     """
    #     # centroids are astropy quantities
    #     self.ra_centroid = np.zeros(self.nt)
    #     self.dec_centroid = np.zeros(self.nt)
    #     dra_m = self.uncontaminated_source_mask.multiply(self.dra).data
    #     ddec_m = self.uncontaminated_source_mask.multiply(self.ddec).data
    #     for t in range(self.nt):
    #         wgts = (
    #             self.uncontaminated_source_mask.multiply(self.flux[t])
    #             .multiply(1 / self.source_flux_estimates[:, None])
    #             .data
    #             ** 2
    #         )
    #         # mask out non finite values and background pixels
    #         k = (np.isfinite(wgts)) & (wgts > 0.01)
    #         self.ra_centroid[t] = np.average(dra_m[k], weights=wgts[k])
    #         self.dec_centroid[t] = np.average(ddec_m[k], weights=wgts[k])
    #     if plot:
    #         plt.figure()
    #         plt.scatter(
    #             dra_m, ddec_m, c=wgts ** 0.5, s=1, vmin=0, vmax=0.2, cmap="Greys"
    #         )
    #         plt.scatter(dra_m[k], ddec_m[k], c=wgts[k] ** 0.5, s=1, vmin=0, vmax=0.2)
    #         plt.scatter(self.ra_centroid[t], self.dec_centroid[t], c="r")
    #         plt.gca().set_aspect("equal")
    #
    #     self.ra_centroid *= u.deg
    #     self.dec_centroid *= u.deg
    #     del dra_m, ddec_m
    #     return

    # def _get_centroid(self, frame_indices="mean", binsize=0.0005):
    #     if isinstance(frame_indices, str):
    #         if frame_indices == "mean":
    #             frame_indices = np.where(self.time_mask)[0]
    #     else:
    #         frame_indices = np.atleast_1d(frame_indices)
    #
    #     x, y = (
    #         self.rough_mask.multiply(self.dra).data,
    #         self.rough_mask.multiply(self.ddec).data,
    #     )
    #     ar, X, Y = np.histogram2d(
    #         x,
    #         y,
    #         (np.arange(-0.01, 0.01, binsize), np.arange(-0.01, 0.011, binsize)),
    #     )
    #     X, Y = np.meshgrid(X, Y)
    #     j = np.zeros(np.asarray(ar.T.shape) + 1, bool)
    #     j[:-1, :-1] = ar.T != 0
    #
    #     c = (
    #         self.rough_mask.multiply(self.flux[frame_indices].mean(axis=0))
    #         .multiply(1 / self.source_flux_estimates[:, None])
    #         .data
    #     )
    #     k = c < 0.8
    #     x_k, y_k, c_k = x[k], y[k], c[k]
    #     ar2 = np.asarray(
    #         [
    #             np.nanmedian(
    #                 c_k[
    #                     (x_k >= X1)
    #                     & (x_k < X1 + binsize)
    #                     & (y_k >= Y1)
    #                     & (y_k < Y1 + binsize)
    #                 ]
    #             )
    #             for X1, Y1 in zip(X[j], Y[j])
    #         ]
    #     )
    #     self.ra_centroid = (
    #         np.average(X[j] + binsize / 2, weights=np.nan_to_num(ar2 ** 3)) * u.deg
    #     )
    #     self.dec_centroid = (
    #         np.average(Y[j] + binsize / 2, weights=np.nan_to_num(ar2 ** 3)) * u.deg
    #     )
    #     return
    def _get_centroid(self, plot=False):
        radius = 1.5 * self.pixel_scale.to(u.arcsecond).value
        if not isinstance(self.r, sparse.csr_matrix):
            mask = sparse.csr_matrix(self.r < radius[:, None])
        else:
            mask = sparse_lessthan(self.r, radius)
        mask = _find_uncontaminated_pixels(mask)
        x, y = (
            mask.multiply(self.dra).data,
            mask.multiply(self.ddec).data,
        )
        c = (
            mask.multiply(self.flux[self.time_mask].mean(axis=0))
            .multiply(1 / self.source_flux_estimates[:, None])
            .data
        )
        bm, nx, ny, nz, nze = threshold_bin(x, y, c, bins=20)
        k = bm > 10
        self.ra_centroid, self.dec_centroid = (
            np.average(nx[k], weights=(nz / nze)[k]) * u.deg,
            np.average(ny[k], weights=(nz / nze)[k]) * u.deg,
        )
        if plot:
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(nx, ny, c=nz / nze, s=200)
            plt.scatter(
                self.ra_centroid.value,
                self.dec_centroid.value,
                c="r",
                marker="*",
                s=100,
            )

    def _time_bin(self, npoints=200, downsample=False):
        """Bin the flux data down in time. If using `pos_corr`s as corrector, it will
        return also the binned and smooth versions of the pos_coors vectors.

        Parameters
        ----------
        npoints: int
            How many points should be in each time bin
        downsample: bool
            If True, the arrays will be downsampled instead of bin-averaged

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
        flux_err_binned:: np.ndarray
            The binned flux error, whitened by the mean of the flux
        pc1_smooth and pc2_smooth: np.ndarray
            Smooth version of the median poscorr vectors 1 and 2 used for time correction.
        pc1_bin and pc2_bin: np.ndarray
            Binned version of poscorrs vectors 1 and 2 used for setting knots.
        """

        # Where there are break points in the time array
        splits = np.append(
            np.append(0, np.where(np.diff(self.time) > 0.1)[0] + 1), len(self.time)
        )
        # if using poscorr, find and add discontinuity in poscorr data
        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            if self.time_corrector == "pos_corr":
                # take the scene-median poscorr
                mpc1 = np.nanmedian(self.pos_corr1, axis=0)
                mpc2 = np.nanmedian(self.pos_corr2, axis=0)
            else:
                # if usig centroids need to convert to pixels
                mpc1 = self.ra_centroids.to("arcsec").value / 4
                mpc2 = self.dec_centroids.to("arcsec").value / 4

            # find poscorr discontinuity in each axis
            grads1 = np.gradient(mpc1, self.time)
            grads2 = np.gradient(mpc2, self.time)
            # the 7-sigma here is hardcoded and found to work ok
            splits1 = np.where(grads1 > 7 * grads1.std())[0]
            splits2 = np.where(grads2 > 7 * grads2.std())[0]
            # merging breaks
            splits = np.unique(np.concatenate([splits, splits1[1::2], splits2[1::2]]))
            del grads1, grads2, splits1, splits2

        # the adition is to set the first knot after breaks @ 1% of the sequence lenght
        splits_a = splits[:-1] + int(self.nt * 0.01)
        splits_b = splits[1:]
        dsplits = (splits_b - splits_a) // npoints
        # this isteration is to avoid knots right at poscorr/time discontinuity by
        # iterating over segments between splits and creating evenly spaced knots
        # within them
        breaks = []
        for spdx in range(len(splits_a)):
            breaks.append(splits_a[spdx] + np.arange(0, dsplits[spdx]) * npoints)
        # we include the last cadance as 99% of the sequence lenght
        breaks.append(int(self.nt * 0.99))
        breaks = np.hstack(breaks)

        if not downsample:
            # time averaged between breaks
            tm = np.vstack(
                [t1.mean(axis=0) for t1 in np.array_split(self.time, breaks)]
            ).ravel()
            # whiten the time array
            ta = (self.time - tm.mean()) / (tm.max() - tm.mean())
            # find the time index for each segments between breaks to then average flux
            ms = [
                np.in1d(np.arange(self.nt), i)
                for i in np.array_split(np.arange(self.nt), breaks)
            ]
            # Average Pixel flux values
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
            # average flux err values at knots
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
        else:
            dwns_idx = (
                np.vstack(
                    [np.median(t1) for t1 in np.array_split(np.arange(self.nt), breaks)]
                )
                .ravel()
                .astype(int)
            )
            tm = self.time[dwns_idx]
            ta = (self.time - tm.mean()) / (tm.max() - tm.mean())
            fm = np.asarray(
                [
                    self.uncontaminated_source_mask.multiply(self.flux[idx]).data
                    for idx in dwns_idx
                ]
            )
            fem = np.asarray(
                [
                    self.uncontaminated_source_mask.multiply(self.flux_err[idx]).data
                    for idx in dwns_idx
                ]
            )

        fm_raw = fm.copy()
        fem /= np.nanmean(fm, axis=0)
        fm /= np.nanmean(fm, axis=0)

        tm = ((tm - tm.mean()) / (tm.max() - tm.mean()))[:, None] * np.ones(fm.shape)

        # poscor
        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            # we smooth the poscorr with a Gaussian kernel and 12 cadence window
            # (6hr-CDPP) to not introduce too much noise, the smoothing is aware
            # of focus-change breaks
            pc1_smooth = []
            pc2_smooth = []
            # if poscorr_filter_size == 0 then the smoothing will fail. we take care of
            # it by setting every value < .5 to 0.1 which leads to no smoothing
            # (residuals < 1e-4)
            self.poscorr_filter_size = (
                0.1 if self.poscorr_filter_size < 0.5 else self.poscorr_filter_size
            )
            for i in range(1, len(splits)):
                pc1_smooth.extend(
                    gaussian_filter1d(
                        mpc1[splits[i - 1] : splits[i]],
                        self.poscorr_filter_size,
                        mode="mirror",
                    )
                )
                pc2_smooth.extend(
                    gaussian_filter1d(
                        mpc2[splits[i - 1] : splits[i]],
                        self.poscorr_filter_size,
                        mode="mirror",
                    )
                )
            pc1_smooth = np.array(pc1_smooth)
            pc2_smooth = np.array(pc2_smooth)

            # do poscorr binning
            if not downsample:
                pc1_bin = np.vstack(
                    [np.median(t1, axis=0) for t1 in np.array_split(pc1_smooth, breaks)]
                ).ravel()[:, None] * np.ones(fm.shape)
                pc2_bin = np.vstack(
                    [np.median(t1, axis=0) for t1 in np.array_split(pc2_smooth, breaks)]
                ).ravel()[:, None] * np.ones(fm.shape)
            else:
                pc1_bin = pc1_smooth[dwns_idx][:, None] * np.ones(fm.shape)
                pc2_bin = pc2_smooth[dwns_idx][:, None] * np.ones(fm.shape)

            return (
                ta,
                tm,
                fm_raw,
                fm,
                fem,
                pc1_smooth,
                pc2_smooth,
                pc1_bin,
                pc2_bin,
            )

        return ta, tm, fm_raw, fm, fem

    def _time_matrix(self, mask):
        dx, dy = (
            mask.multiply(self.dx),
            mask.multiply(self.dy),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        A_c = _make_A_cartesian(
            dx,
            dy,
            n_knots=self.n_time_knots,
            radius=self.time_radius,
            spacing=self.cartesian_knot_spacing,
        )
        return A_c

    def build_time_model(self, plot=False, downsample=False):
        """
        Builds a time model that moves the PRF model to account for the scene movement
        due to velocity aberration. It has two methods to choose from using the
        attribute `self.time_corrector`, if `"polynomial"` (default) will use a
        polynomial in time, if `"pos_corr"` will use the pos_corr vectos that can be found
        in the TPFs. The time polynomial gives a more flexible model vs the pos_corr
        option, but can lead to light curves with "weird" long-term trends. Using
        pos_corr is recomended for Kepler data.

        Parameters
        ----------
        plot: boolean
            Plot a diagnostic figure.
        downsample: boolean
            If True the `time` and `pos_corr` arrays will be downsampled instead of
            binned.
        """
        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            (
                time_original,
                time_binned,
                flux_binned_raw,
                flux_binned,
                flux_err_binned,
                poscorr1_smooth,
                poscorr2_smooth,
                poscorr1_binned,
                poscorr2_binned,
            ) = self._time_bin(npoints=self.n_time_points, downsample=downsample)
            self.pos_corr1_smooth = poscorr1_smooth
            self.pos_corr2_smooth = poscorr2_smooth
        else:
            (
                time_original,
                time_binned,
                flux_binned_raw,
                flux_binned,
                flux_err_binned,
            ) = self._time_bin(npoints=self.n_time_points, downsample=downsample)

        self._whitened_time = time_original
        A_c = self._time_matrix(self.uncontaminated_source_mask)
        A2 = sparse.vstack(
            [A_c] * time_binned.shape[0],
            format="csr",
        )

        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            # Cartesian spline with poscor dependence
            A3 = _combine_A(
                A2, poscorr=[poscorr1_binned, poscorr2_binned], time=time_binned
            )
        else:
            # Cartesian spline with time dependence
            A3 = _combine_A(A2, time=time_binned)

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
            time_model_w = np.linalg.solve(sigma_w_inv, B)
            res = flux_binned - A3.dot(time_model_w).reshape(flux_binned.shape)
            res = np.ma.masked_array(res, (~k).reshape(flux_binned.shape))
            bad_targets = sigma_clip(res, sigma=5).mask
            bad_targets = (
                np.ones(flux_binned.shape, bool) & bad_targets.any(axis=0)
            ).ravel()
            #    k &= ~sigma_clip(flux_binned.ravel() - A3.dot(time_model_w)).mask
            k &= ~bad_targets

        self.time_model_w = time_model_w
        self._time_masked = k
        if plot:
            return self.plot_time_model()
        return

    def plot_time_model(self):
        """
        Diagnostic plot of time model.

        Returns
        -------
        fig : matplotlib.Figure
            Figure.
        """
        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            (
                time_original,
                time_binned,
                flux_binned_raw,
                flux_binned,
                flux_err_binned,
                poscorr1_smooth,
                poscorr2_smooth,
                poscorr1_binned,
                poscorr2_binned,
            ) = self._time_bin(npoints=self.n_time_points)
        else:
            (
                time_original,
                time_binned,
                flux_binned_raw,
                flux_binned,
                flux_err_binned,
            ) = self._time_bin(npoints=self.n_time_points)

        A_c = self._time_matrix(self.uncontaminated_source_mask)
        A2 = sparse.vstack(
            [A_c] * time_binned.shape[0],
            format="csr",
        )
        # Cartesian spline with time dependence
        # Cartesian spline with time dependence
        if hasattr(self, "pos_corr1") and self.time_corrector in [
            "pos_corr",
            "centroid",
        ]:
            # Cartesian spline with poscor dependence
            A3 = _combine_A(
                A2, poscorr=[poscorr1_binned, poscorr2_binned], time=time_binned
            )
        else:
            # Cartesian spline with time dependence
            A3 = _combine_A(A2, time=time_binned)

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dx).data * 3600,
            self.uncontaminated_source_mask.multiply(self.dy).data * 3600,
        )
        model = A3.dot(self.time_model_w).reshape(flux_binned.shape) + 1
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

    def build_shape_model(
        self, plot=False, flux_cut_off=1, frame_index="mean", bin_data=False, **kwargs
    ):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        flux_cut_off: float
            the flux in COUNTS at which to stop evaluating the model!
        frame_index : string or int
            The frame index used to build the shape model, if "mean" then use the
            mean value across time
        **kwargs
            Keyword arguments to be passed to `_get_source_mask()`
        """
        # gaia estimate flux values per pixel to be used as flux priors
        self.source_flux_estimates = np.copy(np.asarray(self.sources.phot_g_mean_flux))

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        if not hasattr(self, "uncontaminated_source_mask"):
            mask = self.rough_mask
        else:
            mask = self.uncontaminated_source_mask
        #            self._get_source_mask(**kwargs)
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        #        self._get_uncontaminated_pixel_mask()

        # for iter in range(niters):
        flux_estimates = self.source_flux_estimates[:, None]

        if frame_index == "mean":
            f = (self.flux[self.time_mask]).mean(axis=0)
        elif frame_index == "max":
            f = (self.flux[self.time_mask]).max(axis=0)
        elif isinstance(frame_index, (int, np.int64, np.int32, np.int16)):
            f = self.flux[frame_index]
        else:
            raise ValueError(f"frame_index {frame_index} not valid")
        # f, fe = (self.flux[self.time_mask]).mean(axis=0), (
        #     (self.flux_err[self.time_mask] ** 2).sum(axis=0) ** 0.5
        # ) / (self.nt)

        mean_f = np.log10(
            mask.astype(float).multiply(f).multiply(1 / flux_estimates).data
        )
        # Actual Kepler errors cause all sorts of instability
        # mean_f_err = (
        #     mask.astype(float)
        #     .multiply(fe / (f * np.log(10)))
        #     .multiply(1 / flux_estimates)
        #     .data
        # )
        # We only need these weights for the wings, so we'll use poisson noise
        mean_f_err = (
            mask.astype(float)
            .multiply((f ** 0.5) / (f * np.log(10)))
            .multiply(1 / flux_estimates)
            .data
        )
        mean_f_err.data = np.abs(mean_f_err.data)

        # take value from Quantity is not necessary
        phi_b = mask.multiply(self.phi).data
        r_b = mask.multiply(self.r).data

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
        prior_mu = np.zeros(A.shape[1]) - 100

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

        self.psf_w = psf_w
        self.psf_w_err = psf_w_err

        # # We build the mean model because this builds our source mask,
        # # and cuts out pixels that have really low flux values or are in the wings
        # self._get_mean_model()
        # self.source_mask = mean_model != 0
        # self.source_mask.eliminate_zeros()
        # self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)

        bad = sigma_clip((mean_f.ravel() - A.dot(psf_w)), sigma=5).mask

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

        # We then build the same design matrix for all pixels with flux
        self._get_mean_model()
        # self.source_mask = mean_model != 0
        # self.source_mask.eliminate_zeros()
        # self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)

        #        self._update_source_mask_remove_bkg_pixels()
        # remove background pixels and recreate mean model
        # self._update_source_mask_remove_bkg_pixels(
        #     flux_cut_off=flux_cut_off, frame_index=frame_index
        # )

        if plot:
            return self.plot_shape_model(self.source_mask, frame_index=frame_index)
        return

    #
    # def _update_source_mask_remove_bkg_pixels(self, flux_cut_off=1, frame_index="max"):
    #     """
    #     Update the `source_mask` to remove pixels that do not contribuite to the PRF
    #     shape.
    #     First, re-estimate the source flux usign the precomputed `mean_model`.
    #     This re-estimation is used to remove sources with bad prediction and update
    #     the `source_mask` by removing background pixels that do not contribuite to
    #     the PRF shape.
    #     Pixels with normalized flux > `flux_cut_off` are kept.
    #
    #     Parameters
    #     ----------
    #     flux_cut_off : float
    #         Lower limit for the normalized flux predicted from the mean model.
    #     frame_index : string or int
    #         The frame index to be used, if "mean" then use the
    #         mean value across time
    #     """
    #
    #     # Re-estimate source flux
    #     # -----
    #     prior_mu = self.source_flux_estimates
    #     prior_sigma = (
    #         np.ones(self.mean_model.shape[0]) * 10 * self.source_flux_estimates
    #     )
    #
    #     if frame_index == "mean":
    #         f, fe = (self.flux).mean(axis=0), (
    #             (self.flux_err ** 2).sum(axis=0) ** 0.5
    #         ) / (self.nt)
    #     if frame_index == "max":
    #         f, fe = (self.flux).max(axis=0), (
    #             (self.flux_err ** 2).sum(axis=0) ** 0.5
    #         ) / (self.nt)
    #     elif isinstance(frame_index, int):
    #         f, fe = self.flux[frame_index], self.flux_err[frame_index]
    #
    #     X = self.mean_model.copy()
    #     X = X.T
    #
    #     sigma_w_inv = X.T.dot(X.multiply(1 / fe[:, None] ** 2)).toarray()
    #     sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
    #     B = X.T.dot((f / fe ** 2))
    #     B += prior_mu / (prior_sigma ** 2)
    #     ws = np.linalg.solve(sigma_w_inv, B)
    #     werrs = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
    #
    #     # -----
    #
    #     # Rebuild source mask
    #     ok = np.abs(ws - self.source_flux_estimates) / werrs > 3
    #     ok &= ((ws / self.source_flux_estimates) < 10) & (
    #         (self.source_flux_estimates / ws) < 10
    #     )
    #     ok &= ws > 10
    #     ok &= werrs > 0
    #
    #     self.source_flux_estimates[ok] = ws[ok]
    #
    #     self.source_mask = (
    #         self.mean_model.multiply(
    #             self.mean_model.T.dot(self.source_flux_estimates)
    #         ).tocsr()
    #         > flux_cut_off
    #     )
    #
    #     # Recreate uncontaminated mask
    #     self._get_uncontaminated_pixel_mask()
    #     # self.uncontaminated_source_mask = self.uncontaminated_source_mask.multiply(
    #     #    (self.mean_model.max(axis=1) < 1)
    #     # )
    #
    #     # Recreate mean model!
    #     self._get_mean_model()

    def _get_mean_model(
        self, mask=None  # , relative_flux_limit=0.001, absolute_flux_limit=1
    ):

        if mask is None:
            if not hasattr(self, "source_mask"):
                mask = self.rough_mask
            else:
                mask = self.source_mask

        """Convenience function to make the scene model"""
        Ap = _make_A_polar(
            mask.multiply(self.phi).data,
            mask.multiply(self.r).data,
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
        mean_model[mask] = m
        # mean_model = mean_model.multiply(mean_model > relative_flux_limit)
        # mean_model = mean_model.multiply(
        #     mean_model.multiply(self.source_flux_estimates[:, None])
        #     > absolute_flux_limit
        # )
        mean_model.eliminate_zeros()
        self.mean_model = mean_model

    def plot_shape_model(self, mask, bin_data=False, radius=None, frame_index="max"):
        """
        Diagnostic plot of shape model.

        Parameters
        ----------
        radius : float
            Radius (in arcseconds) limit to be shown in the figure.
        frame_index : string or int
            The frame index used to plot the shape model, if "mean" then use the
            mean value across time

        Returns
        -------
        fig : matplotlib.Figure
            Figure.
        """
        #        mask = plot_mask.multiply(self.mean_model != 0)
        #        mask.eliminate_zeros()
        if frame_index == "mean":
            mean_f = np.log10(
                mask.astype(float)
                .multiply(self.flux[self.time_mask].mean(axis=0))
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
        if frame_index == "max":
            mean_f = np.log10(
                mask.astype(float)
                .multiply(self.flux[self.time_mask].max(axis=0))
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
        elif isinstance(frame_index, int):
            mean_f = np.log10(
                mask.astype(float)
                .multiply(self.flux[frame_index])
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
        vmin, vmax = np.nanpercentile(mean_f, [10, 90])
        dx, dy = (
            mask.multiply(self.dx),
            mask.multiply(self.dy),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)
        phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)
        if radius is None:
            radius = np.nanmax(r)
        if bin_data:
            nbins = 30 if mean_f.shape[0] <= 5e3 else 90
            _, dx, dy, mean_f, _ = threshold_bin(
                dx, dy, mean_f, bins=nbins, abs_thresh=5
            )

        fig, ax = plt.subplots(3, 2, figsize=(9, 10.5), constrained_layout=True)
        im = ax[0, 0].scatter(
            dx, dy, c=mean_f, cmap="viridis", vmin=vmin, vmax=vmax, s=3, rasterized=True
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

        self._n_time_components = [
            6
            if (
                (
                    self.time_corrector
                    in [
                        "pos_corr",
                        "centroid",
                    ]
                )
                & (hasattr(self, "pos_corr1"))
            )
            else 4
        ][0]

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
            if not hasattr(self, "time_model_w"):
                raise ValueError(
                    "Please use `build_time_model` before fitting with velocity aberration."
                )

            # not necessary to take value from Quantity to do .multiply()
            A_cp = self._time_matrix(self.mean_model != 0)
            A_cp3 = sparse.hstack([A_cp] * self._n_time_components, format="csr")

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

                # Divide through by expected velocity aberration
                X = self.mean_model.copy()
                if hasattr(self, "pos_corr1") and self.time_corrector in [
                    "pos_corr",
                    "centroid",
                ]:
                    # use median pos_corr
                    t_mult = np.hstack(
                        np.array(
                            [
                                1,
                                self._whitened_time[tdx],
                                self._whitened_time[tdx] ** 2,
                                self.pos_corr1_smooth[tdx],
                                self.pos_corr2_smooth[tdx],
                                self.pos_corr1_smooth[tdx] * self.pos_corr2_smooth[tdx],
                            ]
                        )[:, None]
                        * np.ones(A_cp3.shape[1] // self._n_time_components)
                    )
                else:
                    # use time
                    t_mult = np.hstack(
                        (
                            self._whitened_time[tdx]
                            ** np.arange(self._n_time_components)
                        )[:, None]
                        * np.ones(A_cp3.shape[1] // self._n_time_components)
                    )
                X.data *= A_cp3.multiply(t_mult).dot(self.time_model_w) + 1
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
                range(self.nt),
                desc=f"Fitting {self.nsources} Sources (No VA)",
                disable=self.quiet,
            ):
                sigma_w_inv = X.T.dot(X.multiply(1 / fe[tdx][:, None] ** 2)).toarray()
                sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
                B = X.T.dot((f[tdx] / fe[tdx] ** 2))
                B += prior_mu / (prior_sigma ** 2)
                self.ws[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
                self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
                self.model_flux[tdx] = X.dot(self.ws[tdx])

            # These sources are poorly estimated
            nodata = (self.mean_model.max(axis=1) > 1).toarray()[:, 0]
            self.ws[:, nodata] *= np.nan
            self.werrs[:, nodata] *= np.nan

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
