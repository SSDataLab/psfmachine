"""Classes to deal with perturbation matrices"""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from typing import Optional
from psfmachine.utils import _make_A_cartesian
import matplotlib.pyplot as plt


class PerturbationMatrix(object):
    """
    Class to handle perturbation matrices in PSFMachine
    """

    def __init__(
        self,
        time,
        other_vectors=None,
        poly_order=3,
        focus=False,
        cbvs=False,
        segments=True,
        clean=True,
    ):

        self.time = time
        self.other_vectors = other_vectors
        self.poly_order = poly_order
        self.focus = focus
        self.cbvs = cbvs
        self.segments = segments
        self.clean = clean
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
        if self.clean:
            self._clean_vectors()

    def __repr__(self):
        return f"PerturbationMatrix"

    @property
    def prior_mu(self):
        return np.ones(self.shape[1])

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
        s = nvec * (len(self.breaks) + 1)
        X = self.vectors[:, :s]
        w = np.linalg.solve(X.T.dot(X), X.T.dot(self.vectors[:, s:]))
        self.vectors[:, s:] -= X.dot(w)
        return

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.vectors + np.arange(self.shape[1]) * 0.1, c="k")
        ax.set(xlabel="Time", ylabel="Vector", yticks=[], title="Vectors")
        return fig

    def _get_cbvs(self):
        self.ncbvs = 0
        # use lightkurve to get CBVs for a given time?????????
        # Might need channel information maybe
        # self.ncbvs = ....
        #        self.vectors = np.hstack([self.vectors, cbvs])
        raise NotImplementedError

    def fit(self, y, ye=None, mask=None):
        if mask is None:
            mask = np.ones(len(y), bool)
        elif not isinstance(mask, np.ndarray):
            raise ValueError("Pass a mask of booleans")
        if ye is None:
            ye = np.ones(len(y))
        if mask.sum() == 0:
            raise ValueError("All points in perturbation matrix are masked")
        X = self.matrix[mask]
        if (y.ndim == 1) and (ye.ndim == 1):
            sigma_w_inv = X.T.dot(X.multiply(1 / ye[mask, None] ** 2)) + np.diag(
                1 / self.prior_sigma ** 2
            )
            B = X.T.dot(y[mask] / ye[mask] ** 2) + self.prior_mu / self.prior_sigma ** 2
            w = np.linalg.solve(sigma_w_inv, B)
        elif (y.ndim == 2) and (ye.ndim == 1):
            sigma_w_inv = X.T.dot(X.multiply(1 / ye[mask, None] ** 2)) + np.diag(
                1 / self.prior_sigma ** 2
            )
            B = (
                X.T.dot(y[mask] / ye[mask, None] ** 2)
                + self.prior_mu[:, None] / self.prior_sigma[:, None] ** 2
            )
            w = np.linalg.solve(sigma_w_inv, B)
        elif (y.ndim == 2) and (ye.ndim == 2):
            w = []
            for y1, ye1 in zip(y.T, ye.T):
                sigma_w_inv = X.T.dot(X.multiply(1 / ye1[mask, None] ** 2)) + np.diag(
                    1 / self.prior_sigma ** 2
                )
                B = (
                    X.T.dot(y1[mask] / ye1[mask] ** 2)
                    + self.prior_mu / self.prior_sigma ** 2
                )
                w.append(np.linalg.solve(sigma_w_inv, B))
            w = np.asarray(w).T
        else:
            raise ValueError("Can not parse input dimensions")
        return w

    def dot(self, weights):
        return np.asarray(self.matrix.dot(weights))

    @property
    def matrix(self):
        return sparse.csr_matrix(self.vectors)

    @property
    def shape(self):
        return self.vectors.shape

    def bin_vectors(self, var):
        """
        Bins down an input variable to the same time resolution as `self`
        """
        return var

    def _get_downsample_func(self, resolution):
        points = []
        b = np.hstack([0, self.breaks, len(self.time) - 1])
        for b1, b2 in zip(b[:-1], b[1:]):
            p = np.arange(b1, b2, resolution)
            if p[-1] != b2:
                p = np.hstack([p, b2])
            points.append(p)
        points = np.hstack(points)

        def bin_func(x):
            """
            Bins down an input variable to the same time resolution as `self`
            """
            if x.shape[0] == len(self.time):
                return x[points]
            else:
                raise ValueError("Wrong size to bin")

        return bin_func

    def to_low_res(self, resolution=20, method="downsample"):
        """
        Convert to a lower resolution matrix
        Parameters
        ----------
        resolution: int
            Number of points to either bin down to or downsample to
        """

        if method == "downsample":
            bin_func = self._get_downsample_func(resolution)

        # Make new object, turn off all additional vectors and cleaning
        # It will inherrit those from the "other_vectors" variable
        low_res_pm = PerturbationMatrix(
            time=bin_func(self.time),
            other_vectors=bin_func(self.vectors[:, self.poly_order + 1 :]),
            segments=False,
            clean=False,
            focus=False,
            cbvs=False,
        )
        low_res_pm.bin_vectors = bin_func
        return low_res_pm


class PerturbationMatrix3D(PerturbationMatrix):
    """3D perturbation matrix in time, row, column

    NOTE: Radius seems like a bad way of parameterizing something in -cartesian- space?!
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
        clean=True,
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
            clean=clean,
        )

    def __repr__(self):
        return f"PerturbationMatrix3D"

    @property
    def shape(self):
        return (
            self.cartesian_matrix.shape[0] * self.time.shape[0],
            self.cartesian_matrix.shape[1] * self.vectors.shape[1],
        )

    @property
    def matrix(self):
        C = sparse.hstack(
            [
                sparse.vstack(
                    [self.cartesian_matrix * vector[idx] for idx in range(len(vector))],
                    format="csr",
                )
                for vector in self.vectors.T
            ],
            format="csr",
        )
        return C

    def to_low_res(self, resolution=20, method="downsample"):
        """
        Convert to a lower resolution matrix
        Parameters
        ----------
        resolution: int
            Number of points to either bin down to or downsample to
        """

        if method == "downsample":
            bin_func = self._get_downsample_func(resolution)

        # Make new object, turn off all additional vectors and cleaning
        # It will inherrit those from the "other_vectors" variable
        low_res_pm = PerturbationMatrix3D(
            time=bin_func(self.time),
            dx=self.dx,
            dy=self.dy,
            nknots=self.nknots,
            radius=self.radius,
            other_vectors=bin_func(self.vectors[:, self.poly_order + 1 :]),
            segments=False,
            clean=False,
            focus=False,
            cbvs=False,
        )
        low_res_pm.bin_vectors = bin_func
        return low_res_pm


#     """"
