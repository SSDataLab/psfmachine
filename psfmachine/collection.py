"""tools to deal with collections of tpfs"""
import numpy as np
from scipy import sparse
from tqdm.notebook import tqdm

from astropy.coordinates import SkyCoord, match_coordinates_3d

from .utils import get_sources, _make_A

from scipy.integrate import simps


class Collection(object):
    """ Class to work with TPF collections. """

    def __init__(self, tpfs, radius_limit=6, flux_limit=1e4):
        """Class to work with TPF collections

        Parameters:
        -----------
        tpfs : lightkurve.TargetPixelFileCollection
            Collection of target pixel files from Kepler or TESS. These TPFs must be
            from the same quarter, campaign or sector (they must share the same
            time basis)
        radius_limit : int
            The radius in pixels out to which we will consider flux to be part of
            the PSF
        flux_limit : float
            The limit in flux where we will calculate the PSF model. The PSF model
            will still be applied to fainter targets than this limit.

        Attributes:
        -----------
        time: np.ndarray
            Time array of measurements
        flux: np.ndarray
            Flux values from TPFs with shape (ntimes x npixels)
        flux_err: np.ndarray
            Flux error values from TPFs with shape (ntimes x npixels)
        unw: np.ndarray
            Array which specifies which TPF a pixel came from. Has dimenions of
            ntimes x npixels. (Name is "unw[rap]")
        ra : np.ndarray
            The RA of every pixel
        dec : np.ndarray
            The declination of every pixel
        GaiaData : pyia.GaiaData
            The gaia data for all sources nearby to pixels
        sources : pd.DataFrame
            Dataframe containing all the sources nearby to pixels. This is separate
            from GaiaData attribute for convenience
        dx : np.ndarray
            The x distance from every source at every pixel.
            Has dimensions nsources x npixels
        dy : np.ndarray
            The y distance from every source at every pixel.
            Has dimensions nsources x npixels
        gv : np.ndarray
            The gaia flux of each target, has dimensions nsources x npixels
        close_mask : np.ndarray
            Boolean mask, True where the distance to a source is under `radius_limit`.
            Has shape nsources x npixels.
        mask : np.ndarray
            Boolean mask, True where there is expected to be significant flux from
            the PSF of a source.
            Has shape nsources x npixels.
        xcent : np.ndarray
            The x centroid position of the sources, as a function of time.
            Has dimensions (ntime)
        ycent : np.ndarray
            The y centroid position of the sources, as a function of time.
            Has dimensions (ntime)
        """
        self.tpfs = tpfs
        self.time = tpfs[0].time.value
        bad_cadences = np.hypot(tpfs[0].pos_corr1, tpfs[0].pos_corr2) > 10

        self.flux = np.hstack(
            [np.hstack(tpf.flux.value.transpose([2, 0, 1])) for tpf in tpfs]
        )
        self.flux_err = np.hstack(
            [np.hstack(tpf.flux_err.value.transpose([2, 0, 1])) for tpf in tpfs]
        )
        self.flux_err[bad_cadences] *= 1e2

        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]
        self.ntpfs = len(self.tpfs)

        # We need this to know which pixels belong to which TPF later.
        self.unw = np.hstack(
            [
                np.zeros((tpf.shape[0], tpf.shape[1] * tpf.shape[2]), dtype=int) + idx
                for idx, tpf in enumerate(tpfs)
            ]
        )

        # Find the locations of all the pixels
        locs = [
            np.mgrid[
                tpf.column : tpf.column + tpf.shape[2], tpf.row : tpf.row + tpf.shape[1]
            ].reshape(2, np.product(tpf.shape[1:]))
            for tpf in tpfs
        ]
        locs = np.hstack(locs)
        self.locs = locs
        self.ra, self.dec = (
            tpfs[0]
            .wcs.wcs_pix2world(
                np.vstack([(locs[0] - tpfs[0].column), (locs[1] - tpfs[0].row)]).T, 0.0
            )
            .T
        )

        # Create a set of sources from Gaia
        c = SkyCoord(self.ra, self.dec, unit="deg")
        self.GaiaData = get_sources(self, magnitude_limit=18)
        sources = self.GaiaData.data.to_pandas()
        coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))

        slocs = np.asarray(
            [
                tpfs[0].wcs.wcs_world2pix(
                    np.vstack([sources.ra[idx], sources.dec[idx]]).T, 0.0
                )[0]
                for idx in range(len(sources))
            ]
        )
        self.slocs = slocs

        sources["y"] = slocs[:, 1] + tpfs[0].row
        sources["x"] = slocs[:, 0] + tpfs[0].column

        self.dx, self.dy, self.gv = np.asarray(
            [
                np.vstack(
                    [
                        locs[0] - sources["x"][idx],
                        locs[1] - sources["y"][idx],
                        np.zeros(len(locs[0])) + sources.phot_g_mean_flux[idx],
                    ]
                )
                for idx in range(len(sources))
            ]
        ).transpose([1, 0, 2])

        l, d = match_coordinates_3d(coords, coords, nthneighbor=2)[:2]
        dmag = np.abs(
            np.asarray(sources.phot_g_mean_mag) - np.asarray(sources.phot_g_mean_mag[l])
        )
        pixel_dist = np.abs(d.to("arcsec")).value / 4

        # Find the faint contaminated stars
        bad = pixel_dist < 1.5
        blocs = np.vstack([l[bad], np.where(bad)[0]]).T
        mlocs = np.vstack(
            [sources.phot_g_mean_mag[l[bad]], sources.phot_g_mean_mag[np.where(bad)[0]]]
        ).T
        faintest = [blocs[idx][i] for idx, i in enumerate(np.argmax(mlocs, axis=1))]
        bad = np.in1d(np.arange(len(sources)), faintest)
        self.too_close = bad

        r = np.hypot(self.dx, self.dy)
        self.close_mask = r < radius_limit
        #        close_to_pixels = (np.hypot(self.dx, self.dy) < 1).sum(axis=1)

        # Identifing targets that are ON silicon
        surrounded = np.zeros(len(sources), bool)
        for t in np.arange(self.ntpfs):
            xok = (
                np.asarray(sources["x"]) > self.locs[0, self.unw[0] == t][:, None]
            ) & (
                np.asarray(sources["x"]) < (self.locs[0, self.unw[0] == t][:, None] + 1)
            )
            yok = (
                np.asarray(sources["y"]) > self.locs[1, self.unw[0] == t][:, None]
            ) & (
                np.asarray(sources["y"]) < (self.locs[1, self.unw[0] == t][:, None] + 1)
            )
            surrounded |= (xok & yok).any(axis=0)

        self.surrounded = surrounded
        bad |= ~surrounded
        self.bad_sources = sources[bad].reset_index(drop=True)

        source_mask = np.asarray(
            [(c.separation(d1).min().arcsec) < 12 for d1 in coords]
        )
        self.source_dist_mask = source_mask
        source_mask &= ~bad

        sources = sources[source_mask].reset_index(drop=True)
        self.dx, self.dy, self.gv = (
            self.dx[source_mask],
            self.dy[source_mask],
            self.gv[source_mask],
        )
        self.GaiaData = self.GaiaData[source_mask]
        self.sources = sources
        self.nsources = len(self.sources)
        r = np.hypot(self.dx, self.dy)
        self.close_mask = r < radius_limit
        #

        self._find_PSF_edge(radius_limit=radius_limit, flux_limit=flux_limit)

        m = sparse.csr_matrix(self.mask.astype(float))
        dx1, dy1 = self.dx[self.mask], self.dy[self.mask]
        self.xcents, self.ycents = np.ones(len(self.flux)), np.ones(len(self.flux))
        for tdx in range(len(self.flux)):
            self.xcents[tdx] = np.average(
                dx1, weights=np.nan_to_num(m.multiply(self.flux[tdx]).data)
            )
            self.ycents[tdx] = np.average(
                dy1, weights=np.nan_to_num(m.multiply(self.flux[tdx]).data)
            )

        d = SkyCoord(self.sources.ra, self.sources.dec, unit="deg")
        close = match_coordinates_3d(d, d, nthneighbor=2)[1].to("arcsecond").value < 8
        sources = SkyCoord(self.sources.ra, self.sources.dec, unit=("deg"))
        tpfloc = SkyCoord([SkyCoord(tpf.ra, tpf.dec, unit="deg") for tpf in tpfs])
        idx = np.asarray([match_coordinates_3d(loc, sources)[0] for loc in tpfloc])
        jdx = np.asarray(
            [match_coordinates_3d(source, tpfloc)[0] for source in sources]
        )
        self.fresh = ~np.in1d(np.arange(0, self.nsources), idx)

        self.mean_model = self._build_model()
        self._fit_model()

    def _find_PSF_edge(self, radius_limit=6, flux_limit=1e4):
        """ Find the edges of the PSF as a function of flux"""

        # This is the average measured flux by Kepler
        mean_flux = np.nanmean(self.flux, axis=0)
        r = np.hypot(self.dx, self.dy)

        temp_mask = (r < radius_limit) & (self.gv > flux_limit)
        temp_mask &= temp_mask.sum(axis=0) == 1

        f = np.log10((temp_mask.astype(float) * mean_flux))
        A = np.vstack([r[temp_mask] ** 0, r[temp_mask], np.log10(self.gv[temp_mask])]).T
        k = np.isfinite(f[temp_mask])
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(f[temp_mask][k])
        w = np.linalg.solve(sigma_w_inv, B)

        test_gaia = np.linspace(np.log10(self.gv.min()), np.log10(self.gv.max()), 100)
        test_r = np.arange(1, 10, 0.25)
        radius_check = np.asarray(
            [
                np.vstack(
                    [[(np.ones(100) * v) ** idx for idx in range(2)], test_gaia]
                ).T.dot(w)
                for v in test_r
            ]
        )
        self.radius_check = radius_check

        cut = np.percentile(np.abs(radius_check - 1), 3) - np.min(
            np.abs(radius_check - 1)
        )
        x, y = np.asarray(np.meshgrid(test_gaia, test_r))[
            :, np.abs(radius_check - 1) < cut
        ]

        radius = np.polyval(
            np.polyfit(x, y, 5), np.log10(self.sources["phot_g_mean_flux"])
        )
        radius[np.log10(self.sources["phot_g_mean_flux"]) < 3] = 2
        radius[np.log10(self.sources["phot_g_mean_flux"]) > 6.5] = 6
        radius = np.ceil(radius)
        # radius has estimation of the PSF edge per Gaia Source
        self.radius = radius

        self.mask = r < radius[:, None]

    #        self.mask &= (self.mask.sum(axis=0) == 1)

    def __repr__(self):
        return f"Collection [{self.ntpfs} TPFs, {len(self.sources)} Sources]"

    def _build_model(self, nphi=40, nr=30, bin=True):
        """Use the TPF data to build a model of the PSF

        We use the sources above `flux_limit` to build a model of the PSF,
        in the "mean" frame. We do this by fitting a spline in polar coordinates
        to the mean frame of each source, normalized by the Gaia flux.
        (We assume on average the Gaia flux is correlated with the measured flux).
        We fit a spline to the binned flux data, and then apply this spline model
        to all sources.

        We make an assumption that all the PSFs are the same shape, regardless of flux.

        This is a poor assumption.

        Parameters
        ----------
        nphi : int
            Number of bins in phi (angle in polar coordinates)
        nr : int
            Number of bins in r (distance in polar coordinates)
        """

        # Find the mean frame
        f = np.log10(
            sparse.csr_matrix(self.mask.astype(float))
            .multiply(self.flux.mean(axis=0))
            .multiply(1 / self.gv)
            .data
        )
        self.mean_f = f
        xcent_avg = np.average(self.dx[self.mask], weights=np.nan_to_num(f))
        ycent_avg = np.average(self.dy[self.mask], weights=np.nan_to_num(f))

        # Find the distance from the source in polar coordinates, for every source.
        # We use only pixels near sources for this.
        r = np.hypot(self.dx[self.mask] - xcent_avg, self.dy[self.mask] - ycent_avg)
        phi = np.arctan2(self.dy[self.mask] - ycent_avg, self.dx[self.mask] - xcent_avg)
        self.r = r
        self.phi = phi

        # Bin the data in radius/phi space.
        k = np.isfinite(f)
        if bin:
            phis = np.linspace(-np.pi, np.pi, nphi)
            # rs = np.linspace(0, 8, nr)
            rs = np.linspace(0 ** 0.5, 8 ** 0.5, nr) ** 2
            ar = np.zeros((len(phis) - 1, len(rs) - 1))
            count = np.zeros((len(phis) - 1, len(rs) - 1))
            for idx, phi1 in enumerate(phis[1:]):
                for jdx, r1 in enumerate(rs[1:]):
                    m = (phi > phis[idx]) & (phi <= phi1) & (r > rs[jdx]) & (r <= r1)
                    count[idx, jdx] = m.sum()
                    ar[idx, jdx] = np.nanmean(f[m])

            phi_b, r_b = np.asarray(
                np.meshgrid(phis[:-1] + np.median(np.diff(phis)) / 2, rs[:-1])
            )
            ar[(r_b.T > 1) & (count < 1)] = np.nan
            ar[(r_b.T > 4) & ~np.isfinite(ar)] = -5
        else:
            phi_b = phi
            r_b = r
            ar = f
            count = f
            # phi_b = np.append(phi, np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 100))
            # r_b = np.append(r, np.ones(100) * r.max() * 1.1)
            # ar = np.append(f, np.ones(100) * -6)
            # count = np.append(f, np.ones(100) * -6)

        self.count = count
        self.ar = ar
        self.phi_b = phi_b
        self.r_b = r_b

        #        norm = simps(simps(10**np.nan_to_num(ar), r_b[:, 0]), phi_b[0])
        #        ar = np.log10(10**(ar) / norm)

        # Fit the binned data
        A = _make_A(phi_b.ravel(), r_b.ravel())
        self.A = A

        prior_sigma = np.ones(A.shape[1]) * 100
        prior_mu = np.zeros(A.shape[1])
        k = np.isfinite(ar.T.ravel())

        sigma_w_inv = A[k].T.dot(A[k]).toarray()
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        B = A[k].T.dot(ar.T.ravel()[k])
        B += prior_mu / prior_sigma ** 2

        # These weights now give the best fit spline model to the data.
        psf_w = np.linalg.solve(sigma_w_inv, B)
        self._psf_w = psf_w

        # Build the r and phi for -all- pixels
        r = np.hypot(self.dx - xcent_avg, self.dy - ycent_avg)
        phi = np.arctan2(self.dy - ycent_avg, self.dx - xcent_avg)

        # We then build the same design matrix
        Ap = _make_A(phi[self.close_mask], r[self.close_mask])

        # And create a `mean_model` that has the psf model multiplied
        # by the expected gaia flux.
        mean_model = sparse.csr_matrix(r.shape)

        bad = Ap.dot(psf_w) <= -4
        m = 10 ** Ap.dot(psf_w)  # * self.gv[self.close_mask])

        mean_model[self.close_mask] = m
        mean_model.eliminate_zeros()
        return mean_model

    def _fit_model(self):
        """Fit the PSF model to the data

        Using the model built in `_build_model`, we'll fit each
        soure, allowing the flux to vary. We assume that for Kepler,
        the motion of the source is negligible. This is not a great assumption.

        We also assume the PSF shape isn't changing from the mean frame, not a great
        assumption.

        BUT these assumptions are also in SAP flux, so...
        """
        # Use PSF model from the average image to fit flux.
        A = (self.mean_model).T
        ws = np.zeros((self.nt, A.shape[1])) * np.nan
        werrs = np.zeros((self.nt, A.shape[1])) * np.nan

        prior_mu = self.gv[:, 0]  # np.zeros(A.shape[1])
        prior_sigma = np.ones(A.shape[1]) * 10 * self.gv[:, 0]
        for tdx in tqdm(np.arange(len(self.time)), desc="Fitting PSF model"):
            k = np.isfinite(self.flux[tdx])
            sigma_w_inv = (
                A[k]
                .T.dot(A[k].multiply(1 / self.flux_err[tdx][k, None] ** 2))
                .toarray()
            )
            sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
            B = A[k].T.dot((self.flux[tdx][k] / self.flux_err[tdx][k] ** 2))
            B += prior_mu / (prior_sigma ** 2)
            ws[tdx] = np.linalg.solve(sigma_w_inv, B)
            werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

        self.f = ws.T
        self.f_err = werrs.T

        sap = np.zeros((self.mask.shape[0], self.time.shape[0]))
        sap_e = np.zeros((self.mask.shape[0], self.time.shape[0]))
        m = sparse.csr_matrix(self.mask.astype(float))
        for tdx in tqdm(range(self.time.shape[0]), desc="Simple SAP flux"):
            l = m.multiply(self.flux[tdx])
            le = m.multiply(self.flux_err[tdx])
            l.data[~np.isfinite(l.data)] = 0
            l.data[~np.isfinite(l.data)] = 0
            l.eliminate_zeros()
            le.eliminate_zeros()
            sap[:, tdx] = np.asarray(l.sum(axis=1))[:, 0]
            sap_e[:, tdx] = np.asarray(le.power(2).sum(axis=1))[:, 0] ** 0.5

        self.sap = sap
        self.sap_e = sap_e
