"""Classes to deal with perturbation matrices"""

import numpy as np
import numpy.typing as npt
from typing import Optional
from scipy import sparse
from psfmachine.utils import _make_A_cartesian
import matplotlib.pyplot as plt
from fbpca import pca
from .utils import spline1d


class PerturbationMatrix(object):
    """
    Class to handle perturbation matrices in PSFMachine

    Parameters
    ----------
    time : np.ndarray
        Array of time values
    other_vectors: list or np.ndarray
        Other detrending vectors (e.g. centroids)
    poly_order: int
        Polynomial order to use for detrending, default 3
    focus : bool
        Whether to correct focus using a simple exponent model
    segments: bool
        Whether to fit portions of data where there is a significant time break as separate segments
    resolution: int
        How many cadences to bin down via `bin_method`
    bin_method: str
        How to bin the data under the hood. Default is by mean binning. Options are 'downsample' and 'bin'
    focus_exptime: float
        Time for the exponent for focus change, if used
    """

    def __init__(
        self,
        time: npt.ArrayLike,
        other_vectors: Optional[list] = None,
        poly_order: int = 3,
        focus: bool = False,
        segments: bool = True,
        resolution: int = 10,
        bin_method: str = "bin",
        focus_exptime=2,
    ):

        self.time = time
        self.other_vectors = np.nan_to_num(other_vectors)
        self.poly_order = poly_order
        self.focus = focus
        self.segments = segments
        self.resolution = resolution
        self.bin_method = bin_method
        self.focus_exptime = focus_exptime
        self._vectors = np.vstack(
            [
                (self.time - self.time.mean()) ** idx
                for idx in range(self.poly_order + 1)
            ]
        ).T
        if self.focus:
            self._get_focus_change()
        if self.other_vectors is not None:
            if isinstance(self.other_vectors, (list, np.ndarray)):
                self.other_vectors = np.atleast_2d(self.other_vectors)
                if self.other_vectors.shape[0] != len(self.time):
                    if self.other_vectors.shape[1] == len(self.time):
                        self.other_vectors = self.other_vectors.T
                    else:
                        raise ValueError("Must pass other vectors in the right shape")
            else:
                raise ValueError("Must pass a list as other vectors")
            self.vectors = np.hstack([self._vectors.copy(), self.other_vectors])
        else:
            self.vectors = self._vectors.copy()
        if self.segments:
            self.vectors = self._cut_segments(self.vectors)
        # self._clean_vectors()
        self.matrix = sparse.csr_matrix(self.bin_func(self.vectors))

    def __repr__(self):
        return "PerturbationMatrix"

    @property
    def prior_mu(self):
        return np.ones(self.shape[1])

    @property
    def prior_sigma(self):
        return np.ones(self.shape[1]) * 0.5

    @property
    def breaks(self):
        return np.where(np.diff(self.time) / np.median(np.diff(self.time)) > 5)[0] + 1

    @property
    def segment_masks(self):
        x = np.array_split(np.arange(len(self.time)), self.breaks)
        return np.asarray(
            [np.in1d(np.arange(len(self.time)), x1).astype(float) for x1 in x]
        ).T

    def _cut_segments(self, vectors):
        """
        Cuts the data into "segments" wherever there is a break. Breaks are defined
        as anywhere where there is a gap in data of more than 5 times the median
        time between observations.

        Parameters
        ----------
        vectors : np.ndarray
            Vector arrays to be break into segments.
        """
        return np.hstack(
            [
                vectors[:, idx][:, None] * self.segment_masks
                for idx in range(vectors.shape[1])
            ]
        )

    def _get_focus_change(self):
        """Finds a simple model for the focus change"""
        focus = np.asarray(
            [
                np.exp(-self.focus_exptime * (self.time - self.time[b]))
                for b in np.hstack([0, self.breaks])
            ]
        )
        focus *= np.asarray(
            [
                ((self.time - self.time[b]) >= 0).astype(float)
                for b in np.hstack([0, self.breaks])
            ]
        )
        focus[focus < 1e-10] = 0
        self._vectors = np.hstack([self._vectors, np.nansum(focus, axis=0)[:, None]])
        return

    # def _clean_vectors(self):
    #     """Remove time polynomial from other vectors"""
    #     nvec = self.poly_order + 1
    #     if self.focus:
    #         nvec += 1
    #     if self.segments:
    #         s = nvec * (len(self.breaks) + 1)
    #     else:
    #         s = nvec
    #
    #     if s != self.vectors.shape[1]:
    #         X = self.vectors[:, :s]
    #         w = np.linalg.solve(X.T.dot(X), X.T.dot(self.vectors[:, s:]))
    #         self.vectors[:, s:] -= X.dot(w)
    #         # Each segment has mean zero
    #         self.vectors[:, s:] -= np.asarray(
    #             [v[v != 0].mean() * (v != 0) for v in self.vectors[:, s:].T]
    #         ).T
    #     return

    def plot(self):
        """Plot basis vectors"""
        fig, ax = plt.subplots()
        ax.plot(self.time, self.vectors + np.arange(self.vectors.shape[1]) * 0.1)
        ax.set(xlabel="Time", ylabel="Vector", yticks=[], title="Vectors")
        return fig

    def _fit_linalg(self, y, ye, k=None):
        """Hidden method to fit data with linalg"""
        if k is None:
            k = np.ones(y.shape[0], bool)
        X = self.matrix[k]
        sigma_w_inv = X.T.dot(X.multiply(1 / ye[k, None] ** 2)) + np.diag(
            1 / self.prior_sigma ** 2
        )
        B = X.T.dot(y[k] / ye[k] ** 2) + self.prior_mu / self.prior_sigma ** 2
        return np.linalg.solve(sigma_w_inv, B)

    def fit(self, flux: npt.ArrayLike, flux_err: Optional[npt.ArrayLike] = None):
        """
        Fits flux to find the best fit model weights. Optionally will include flux errors.
        Sets the `self.weights` attribute with best fit weights.

        Parameters
        ----------
        flux: npt.ArrayLike
            Array of flux values. Should have shape ntimes.
        flux: npt.ArrayLike
            Optional flux errors. Should have shape ntimes.

        Returns
        -------
        weights: npt.ArrayLike
            Array with computed weights
        """
        if flux_err is None:
            flux_err = np.ones_like(flux)

        y, ye = self.bin_func(flux).ravel(), self.bin_func(flux_err, quad=True).ravel()
        self.weights = self._fit_linalg(y, ye)
        return self.weights

    def model(self, time_indices: Optional[list] = None):
        """Returns the best fit model at given `time_indices`.

        Parameters
        ----------
        time_indices: list
            Optionally pass a list of integers. Model will be evaluated at those indices.

        Returns
        -------
        model: npt.ArrayLike
            Array of values with the same shape as the `flux` used in `self.fit`
        """
        if not hasattr(self, "weights"):
            raise ValueError("Run `fit` first.")
        if time_indices is None:
            time_indices = np.ones(len(self.time), bool)
        return self.vectors[time_indices].dot(self.weights)

    @property
    def shape(self):
        return self.vectors.shape

    @property
    def nvec(self):
        return self.vectors.shape[1]

    @property
    def ntime(self):
        return self.time.shape[0]

    def bin_func(self, var, **kwargs):
        """
        Bins down an input variable to the same time resolution as `self`

        Parameters
        ----------
        var: npt.ArrayLike
            Array of values with at least 1 dimension. The first dimension must be
            the same shape as `self.time`

        Returns
        -------
        func: object
            An object function according to `self.bin_method`
        """
        if self.bin_method.lower() == "downsample":
            func = self._get_downsample_func()
        elif self.bin_method.lower() == "bin":
            func = self._get_bindown_func()
        else:
            raise NotImplementedError
        return func(var, **kwargs)

    def _get_downsample_func(self):
        """Builds a function to lower the resolution of the data through downsampling"""
        points = []
        b = np.hstack([0, self.breaks, len(self.time) - 1])
        for b1, b2 in zip(b[:-1], b[1:]):
            p = np.arange(b1, b2, self.resolution)
            if p[-1] != b2:
                p = np.hstack([p, b2])
            points.append(p)
        points = np.unique(np.hstack(points))
        self.nbins = len(points)

        def func(x, quad=False):
            """
            Bins down an input variable to the same time resolution as `self`
            """
            if x.shape[0] == len(self.time):
                return x[points]
            else:
                raise ValueError("Wrong size to bin")

        return func

    def _get_bindown_func(self):
        """Builds a function to lower the resolution of the data through binning"""
        b = np.hstack([0, self.breaks, len(self.time) - 1])
        points = np.hstack(
            [np.arange(b1, b2, self.resolution) for b1, b2 in zip(b[:-1], b[1:])]
        )
        points = points[~np.in1d(points, np.hstack([0, len(self.time) - 1]))]
        points = np.unique(np.hstack([points, self.breaks]))
        self.nbins = len(points) + 1

        def func(x, quad=False):
            """
            Bins down an input variable to the same time resolution as `self`
            """
            if x.shape[0] == len(self.time):
                if not quad:
                    return np.asarray(
                        [i.mean(axis=0) for i in np.array_split(x, points)]
                    )
                else:
                    return (
                        np.asarray(
                            [
                                np.sum(i ** 2, axis=0) / (len(i) ** 2)
                                for i in np.array_split(x, points)
                            ]
                        )
                        ** 0.5
                    )
            else:
                raise ValueError("Wrong size to bin")

        return func

    def pca(self, y, ncomponents=5, smooth_time_scale=0):
        """Adds the first `ncomponents` principal components of `y` to the design
        matrix. `y` is smoothen with a spline function and scale `smooth_time_scale`.

        Parameters
        ----------
        y: np.ndarray
            Input flux array to take PCA of.
        ncomponents: int
            Number of principal components to use
        smooth_time_scale: float
            Amount to smooth the components, using a spline in time.
             If 0, the components will not be smoothed.
        """
        return self._pca(
            y, ncomponents=ncomponents, smooth_time_scale=smooth_time_scale
        )

    def _pca(self, y, ncomponents=3, smooth_time_scale=0):
        """This hidden method allows us to update the pca method for other classes"""
        if not y.ndim == 2:
            raise ValueError("Must pass a 2D `y`")
        if not y.shape[0] == len(self.time):
            raise ValueError(f"Must pass a `y` with shape ({len(self.time)}, X)")

        # Clean out any time series have significant contribution from one component
        k = np.nansum(y, axis=0) != 0

        if smooth_time_scale != 0:
            X = sparse.hstack(
                [
                    spline1d(
                        self.time,
                        np.linspace(
                            self.time[m].min(),
                            self.time[m].max(),
                            int(
                                np.ceil(
                                    (self.time[m].max() - self.time[m].min())
                                    / smooth_time_scale
                                )
                            ),
                        ),
                        degree=3,
                    )[:, 1:]
                    for m in self.segment_masks.astype(bool).T
                ]
            )
            X = sparse.hstack([X, sparse.csr_matrix(np.ones(X.shape[0])).T]).tocsr()
            X = X[:, np.asarray(X.sum(axis=0) != 0)[0]]
            smoothed_y = X.dot(
                np.linalg.solve(
                    X.T.dot(X).toarray() + np.diag(1 / (np.ones(X.shape[1]) * 1e10)),
                    X.T.dot(y),
                )
            )
        else:
            smoothed_y = np.copy(y)

        for count in range(3):
            self._pca_components, s, V = pca(
                np.nan_to_num(smoothed_y)[:, k], ncomponents, n_iter=30
            )
            k[k] &= (np.abs(V) < 0.5).all(axis=0)

        if self.other_vectors is not None:
            self.vectors = np.hstack(
                [self._vectors, self.other_vectors, self._pca_components]
            )
        else:
            self.vectors = np.hstack([self._vectors, self._pca_components])
        if self.segments:
            self.vectors = self._cut_segments(self.vectors)
        # self._clean_vectors()
        self.matrix = sparse.csr_matrix(self.bin_func(self.vectors))


class PerturbationMatrix3D(PerturbationMatrix):
    """Class to handle 3D perturbation matrices in PSFMachine

    Parameters
    ----------
    time : np.ndarray
        Array of time values
    dx: np.ndarray
        Pixel positions in x separation from source center
    dy : np.ndaray
        Pixel positions in y separation from source center
    other_vectors: list or np.ndarray
        Other detrending vectors (e.g. centroids)
    poly_order: int
        Polynomial order to use for detrending, default 3
    nknots: int
        Number of knots for the cartesian spline
    radius: float
        Radius out to which to calculate the cartesian spline
    focus : bool
        Whether to correct focus using a simple exponent model
    segments: bool
        Whether to fit portions of data where there is a significant time break as
        separate segments
    resolution: int
        How many cadences to bin down via `bin_method`
    bin_method: str
        How to bin the data under the hood. Default is by mean binning.
    focus_exptime: float
        Time for the exponent for focus change, if used
    degree: int
        Polynomial degree used to build the row/column cartesian design matrix
    knot_spacing_type: str
        Type of spacing bewtwen knots used for cartesian design matrix, options are
        {"linear", "sqrt"}
    """

    def __init__(
        self,
        time: npt.ArrayLike,
        dx: npt.ArrayLike,
        dy: npt.ArrayLike,
        other_vectors: Optional[list] = None,
        poly_order: int = 3,
        nknots: int = 7,
        radius: float = 8,
        focus: bool = False,
        segments: bool = True,
        resolution: int = 30,
        bin_method: str = "downsample",
        focus_exptime: float = 2,
        degree: int = 2,
        knot_spacing_type: str = "linear",
    ):
        self.dx = dx
        self.dy = dy
        self.nknots = nknots
        self.radius = radius
        self.degree = degree
        self.knot_spacing_type = knot_spacing_type
        self.cartesian_matrix = _make_A_cartesian(
            self.dx,
            self.dy,
            n_knots=self.nknots,
            radius=self.radius,
            knot_spacing_type=self.knot_spacing_type,
            degree=self.degree,
        )
        super().__init__(
            time=time,
            other_vectors=other_vectors,
            poly_order=poly_order,
            focus=focus,
            segments=segments,
            resolution=resolution,
            bin_method=bin_method,
            focus_exptime=focus_exptime,
        )
        self._get_cartesian_stacked()

    def _get_cartesian_stacked(self):
        """
        Stacks cartesian design matrix in preparation to be combined with
        time basis vectors.
        """
        self._cartesian_stacked = sparse.hstack(
            [self.cartesian_matrix for idx in range(self.vectors.shape[1])],
            format="csr",
        )
        repeat1d = np.repeat(
            self.bin_func(self.vectors), self.cartesian_matrix.shape[1], axis=1
        )
        repeat2d = np.repeat(repeat1d, self.cartesian_matrix.shape[0], axis=0)
        self.matrix = (
            sparse.vstack([self._cartesian_stacked] * self.nbins)
            .multiply(repeat2d)
            .tocsr()
        )
        self.matrix.eliminate_zeros()

    def __repr__(self):
        return "PerturbationMatrix3D"

    @property
    def shape(self):
        return (
            self.cartesian_matrix.shape[0] * self.time.shape[0],
            self.cartesian_matrix.shape[1] * self.vectors.shape[1],
        )

    def fit(
        self,
        flux: npt.ArrayLike,
        flux_err: Optional[npt.ArrayLike] = None,
        pixel_mask: Optional[npt.ArrayLike] = None,
    ):
        """
        Fits flux to find the best fit model weights. Optionally will include flux
        errors. Sets the `self.weights` attribute with best fit weights.

        Parameters
        ----------
        flux: npt.ArrayLike
            Array of flux values. Should have shape ntimes x npixels.
        flux_err: npt.ArrayLike
            Optional flux errors. Should have shape ntimes x npixels.
        pixel_mask: npt.ArrayLike
            Pixel mask to apply. Values that are `True` will be used in the fit.
            Values that are `False` will be masked. Should have shape npixels.
        """
        if pixel_mask is not None:
            if not isinstance(pixel_mask, np.ndarray):
                raise ValueError("`pixel_mask` must be an `np.ndarray`")
            if not pixel_mask.shape[0] == flux.shape[-1]:
                raise ValueError(
                    f"`pixel_mask` must be shape {flux.shape[-1]} (npixels)"
                )
        else:
            pixel_mask = np.ones(flux.shape[-1], bool)
        if flux_err is None:
            flux_err = np.ones_like(flux)

        y, ye = self.bin_func(flux).ravel(), self.bin_func(flux_err, quad=True).ravel()
        k = (np.ones(self.nbins, bool)[:, None] * pixel_mask).ravel()
        self.weights = self._fit_linalg(y, ye, k=k)
        return

    def model(self, time_indices: Optional[list] = None):
        """Returns the best fit model at given `time_indices`.

        Parameters
        ----------
        time_indices: list
            Optionally pass a list of integers. Model will be evaluated at those indices.

        Returns
        -------
        model: npt.ArrayLike
            Array of values with the same shape as the `flux` used in `self.fit`
        """
        if not hasattr(self, "weights"):
            raise ValueError("Run `fit` first")
        if time_indices is None:
            time_indices = np.arange(len(self.time))
        time_indices = np.atleast_1d(time_indices)
        if isinstance(time_indices[0], bool):
            time_indices = np.where(time_indices[0])[0]

        return np.asarray(
            [
                self._cartesian_stacked.multiply(
                    np.repeat(self.vectors[time_index], self.cartesian_matrix.shape[1])
                ).dot(self.weights)
                for time_index in time_indices
            ]
        )

    def pca(self, y, ncomponents=3, smooth_time_scale=0):
        """Adds the first `ncomponents` principal components of `y` to the design
        matrix. `y` is smoothen with a spline function and scale `smooth_time_scale`.

        Parameters
        ----------
        y: np.ndarray
            Input flux array to take PCA of.
        n_components: int
            Number of components to take
        smooth_time_scale: float
            Amount to smooth the components, using a spline in time.
             If 0, the components will not be smoothed.
        """
        self._pca(
            y,
            ncomponents=ncomponents,
            smooth_time_scale=smooth_time_scale,
        )
        self._get_cartesian_stacked()

    def plot_model(self, time_index=0):
        """
        Plot perturbation model

        Parameters
        ----------
        time_index : int
            Time index to plot the perturbed model

        """
        if not hasattr(self, "weights"):
            raise ValueError("Run `fit` first.")
        fig, ax = plt.subplots()
        ax.scatter(self.dx, self.dy, c=self.model(time_index)[0])
        ax.set(
            xlabel=r"$\delta$x",
            ylabel=r"$\delta$y",
            title=f"Perturbation Model [Cadence {time_index}]",
        )
        return fig
