"""Classes to deal with perturbation matrices"""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from typing import Optional
import matplotlib.pyplot as plt


@dataclass
class PerturbationMatrix:
    """
    Class to handle perturbation matrices in PSFMachine
    """

    time: np.ndarray
    other_vectors: Optional = None
    poly_order: int = 3
    bin_or_downsample: str = "bin"
    focus: bool = False
    cbvs: bool = False
    segments: bool = True
    clean: bool = True

    def __post_init__(self):
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
        self.prior_mu = np.ones(self.shape[1])
        self.prior_sigma = np.ones(self.shape[1]) * 1e4

    def __repr__(self):
        return f"PerturbationMatrix {self.shape}"

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

    def to_low_res(self, resolution=20, method="downsample"):
        """
        Convert to a lower resolution matrix
        Parameters
        ----------
        resolution: int
            Number of points to either bin down to or downsample to
        """

        if method is "downsample":
            points = []
            b = np.hstack([0, self.breaks, self.shape[0] - 1])
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
                if x.shape[0] == self.shape[0]:
                    return x[points]
                else:
                    raise ValueError("Wrong size to bin")

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


# @dataclass
# class PerturbationMatrix3d(PerturbationMatrix):
#     """"
#     Makes a perturbation matrix that accounts for spatial extent
#     """"
#     time
#     other_vectors
#     dx
#     dy
#     ntime_knots
#     knot_radius
#     knot_spacing
#
#     def __post_init__(self):
#         self.time_matrix = PerturbationMatrix(self.time, ....)
#         self._get_cartesian_matrix(dx, dy)
#
#     def __repr__(self):
#         return f'PerturbationMatrix {self.shape}'
#
#     @property
#     def shape(self):
#
#     def _get_cartesian_matrix(self, dx, dy):
#         ...
#         self.cartesian_matrix =
#
#     @property
#     def matrix(self):
#         # Merge cartesian and time
#
#     def to_low_res(self):
#         return PerturbationMatrix3d(self.time_matrix.to_low_res....)
#
#     def from_time_matrix(self, perturbationmatrix):
#         return PerturbationMatrix3d(....)
#
#
# A = PerturbationMatrix3d(time=time, dx=dx, dy=dy, other_vectors=[], focus=True, bin_style='bin')
# w = A.to_low_res.fit(y, ye, mask)
# model = A.dot(w)
#
#
# mac.mean_model
# mac.perturbed_model
