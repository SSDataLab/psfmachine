"""Classes to deal with perturbation matrices"""

import numpy as np
from scipy import sparse
from psfmachine.utils import _make_A_cartesian
import matplotlib.pyplot as plt


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
        How to bin the data under the hood. Default is by mean binning.
    """

    def __init__(
        self,
        time,
        other_vectors=None,
        poly_order=3,
        focus=False,
        cbvs=False,
        segments=True,
        resolution=10,
        bin_method="bin",
    ):

        self.time = time
        self.other_vectors = other_vectors
        self.poly_order = poly_order
        self.focus = focus
        self.cbvs = cbvs
        self.segments = segments
        self.resolution = resolution
        self.bin_method = bin_method
        self.vectors = np.vstack(
            [self.time ** idx for idx in range(self.poly_order + 1)]
        ).T
        if self.cbvs:
            self._get_cbvs()
        if self.focus:
            self._get_focus_change()
        self._validate()
        if self.segments:
            self._cut_segments()
        self._clean_vectors()
        self.matrix = sparse.csr_matrix(self.bin_func(self.vectors))

    def __repr__(self):
        return "PerturbationMatrix"

    @property
    def prior_mu(self):
        return np.zeros(self.shape[1])

    @property
    def prior_sigma(self):
        return np.ones(self.shape[1]) * 1e4

    @property
    def breaks(self):
        return np.where(np.diff(self.time) / np.median(np.diff(self.time)) > 5)[0] + 1

    def _validate(self):
        # Check that the shape is correct etc
        # Check the priors are right
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
            self.vectors = np.hstack([self.vectors, self.other_vectors])
        return

    def _cut_segments(self):
        x = np.array_split(np.arange(len(self.time)), self.breaks)
        masks = np.asarray(
            [np.in1d(np.arange(len(self.time)), x1).astype(float) for x1 in x]
        ).T
        self.vectors = np.hstack(
            [
                self.vectors[:, idx][:, None] * masks
                for idx in range(self.vectors.shape[1])
            ]
        )

    def _get_focus_change(self, exptime=100):
        focus = np.asarray(
            [
                np.exp(-exptime * (self.time - self.time[b]))
                for b in np.hstack([0, self.breaks])
            ]
        )
        focus *= np.asarray(
            [
                ((self.time - self.time[b]) >= 0).astype(float)
                for b in np.hstack([0, self.breaks])
            ]
        )
        self.vectors = np.hstack([self.vectors, np.nansum(focus, axis=0)[:, None]])
        return

    def _clean_vectors(self):
        """Remove time polynomial from other vectors"""
        nvec = self.poly_order + 1
        if self.focus:
            nvec += 1
        if self.cbvs:
            nvec += self.ncbvs
        if self.segments:
            s = nvec * (len(self.breaks) + 1)
        else:
            s = nvec
        X = self.vectors[:, :s]
        w = np.linalg.solve(X.T.dot(X), X.T.dot(self.vectors[:, s:]))
        self.vectors[:, s:] -= X.dot(w)
        return

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.vectors + np.arange(self.vectors.shape[1]) * 0.1)
        ax.set(xlabel="Time", ylabel="Vector", yticks=[], title="Vectors")
        return fig

    def _get_cbvs(self):
        self.ncbvs = 0
        # use lightkurve to get CBVs for a given time?????????
        # Might need channel information maybe
        # self.ncbvs = ....
        #        self.vectors = np.hstack([self.vectors, cbvs])
        raise NotImplementedError

    def fit(self, flux, flux_err=None):
        if flux_err is None:
            flux_err = np.ones(len(flux_err))

        y, ye = self.bin_func(flux).ravel(), self.bin_func(flux_err, quad=True).ravel()
        X = self.matrix
        sigma_w_inv = X.T.dot(X.multiply(1 / ye[:, None] ** 2)) + np.diag(
            1 / self.prior_sigma ** 2
        )
        B = X.T.dot(y / ye ** 2) + self.prior_mu / self.prior_sigma ** 2
        self.weights = np.linalg.solve(sigma_w_inv, B)
        return self.weights

    def model(self, time_indices=None):
        if time_indices is None:
            time_indices = np.ones(len(self.time), bool)
        return self.vectors[time_indices].dot(self.weights)

    @property
    def shape(self):
        return self.vectors.shape

    def bin_func(self, var, **kwargs):
        """
        Bins down an input variable to the same time resolution as `self`
        """
        if self.bin_method.lower() == "downsample":
            func = self._get_downsample_func()
        elif self.bin_method.lower() == "bin":
            func = self._get_bindown_func()
        else:
            raise NotImplementedError
        return func(var, **kwargs)

    def _get_downsample_func(self):
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
        Whether to fit portions of data where there is a significant time break as separate segments
    resolution: int
        How many cadences to bin down via `bin_method`
    bin_method: str
        How to bin the data under the hood. Default is by mean binning.
    """

    def __init__(
        self,
        time,
        dx,
        dy,
        other_vectors=None,
        poly_order=3,
        nknots=10,
        radius=8,
        focus=False,
        cbvs=False,
        segments=True,
        resolution=10,
        bin_method="downsample",
    ):
        self.dx = dx
        self.dy = dy
        self.nknots = nknots
        self.radius = radius
        self.cartesian_matrix = _make_A_cartesian(
            self.dx,
            self.dy,
            n_knots=self.nknots,
            radius=self.radius,
            knot_spacing_type="linear",
        )
        super().__init__(
            time=time,
            other_vectors=other_vectors,
            poly_order=poly_order,
            focus=focus,
            cbvs=cbvs,
            segments=segments,
            resolution=resolution,
            bin_method=bin_method,
        )

        self._cartesian_stacked = sparse.hstack(
            [self.cartesian_matrix for idx in range(self.vectors.shape[1])],
            format="csr",
        )
        repeat1d = np.repeat(
            self.bin_func(self.vectors), self.cartesian_matrix.shape[1], axis=1
        )
        repeat2d = np.repeat(repeat1d, self.cartesian_matrix.shape[0], axis=0)
        self.matrix = sparse.vstack([self._cartesian_stacked] * self.nbins).multiply(
            repeat2d
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

    def model(self, time_indices=None):
        """We build the matrix for every frame"""
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

    def plot_model(self, time_index=0):
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
