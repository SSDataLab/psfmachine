"""Defines the main XXX object that fit PSF model to sources"""

import numpy as np
import lightkurve as lk
from scipy import sparse
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.time import Time
import astropy.units as u

from .utils import get_gaia_sources, wrapped_spline


# help functions


class Machine(object):
    def __init__(
        self, time, flux, flux_err, ra, dec, sources, radius_limit=24.0 * u.arcsecond
    ):
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
        radius_limit : float astropy units
            Radius limit in arcsecs to select stars to be used for PSF modeling


        Attributes
        ----------

        sources : pd.DataFrame

        source_flux_estiamtes : np.ndarray
            nsources
        """

        # assigning initial attribute
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ra = ra
        self.dec = dec
        self.sources = sources
        self.radius_limit = radius_limit
        self.flux_limit = 1e4

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # gaia estimate flux value for each pixel
        self.source_flux_estimates = np.array(
            [
                np.zeros(len(locs[0])) + sources.phot_g_mean_flux[idx]
                for idx in range(len(sources))
            ]
        )

        # The separation in ra & dec from each source at each pixel shape
        # nsources x npixels, arcsec units
        self.dra, self.ddec = np.asarray(
            [
                [ra - self.sources["ra"][idx], dec - self.sources["dec"][idx]]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # Get the centroids of the images as a function of time
        self._get_centroids()

        # polar coordinates, remember these two are astropy quantity objects
        # radial distance from source in arcseconds
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        # azimuthal angle from source in radians
        self.phi = np.arctan2(self.ddec, self.dra)

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        self.source_mask = self._get_source_mask()
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        self.uncontaminated_pixel_mask = self._get_uncontaminated_pixel_mask()

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    def _solve_linear_model(A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None):
        """
        Solves a linear model with dsign matrix A and observations y
            Aw = y
        return the solutions w for the system.
        Alternative, the observations errors, priors, and a boolean mask can be provided.
        """
        if not k:
            k = np.ones(len(y), dtype=bool)

        if y_err:
            sigma_w_inv = A[k].T.dot(A[k]).multiply(1 / y_err[k, None] ** 2)
            B = A[k].T.dot((y[k] / y_err[k] ** 2))
        else:
            sigma_w_inv = A[k].T.dot(A[k])
            B = A[k].T.dot(y[k])

        if type(sigma_w_inv) == sparse.csr_matrix:
            sigma_w_inv = sigma_w_inv.toarray()

        if prior_mu and prior_sigma:
            sigma_w_inv += np.diag(1 / prior_sigma ** 2)
            B += prior_mu / prior_sigma ** 2
        w = np.linalg.solve(sigma_w_inv, B)
        if y_err:
            w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            return w, w_err
        return w

    def _get_source_mask(self):
        """
        Mask of shape nsources x number of pixels, one where flux from a source
        exists
        """
        # mask by Gaia flux and maximum distance
        mean_flux = np.nanmean(self.flux, axis=0)
        temp_mask = (sell.r.to("arcsec") < self.limit_radius) & (
            self.source_flux_estimates > self.limit_flux
        )
        temp_mask &= temp_mask.sum(axis=0) == 1

        f = np.log10((temp_mask.astype(float) * mean_flux))[temp_mask]
        A = np.vstack(
            [
                self.r[temp_mask].to("arcsec").value ** 0,
                self.r[temp_mask].to("arcsec").value,
                np.log10(self.source_flux_estimates[temp_mask]),
            ]
        ).T
        w = _solve_linear_model(A, f)

        test_gaia = np.linspace(
            np.log10(source_flux_estimates.min()),
            np.log10(source_flux_estimates.max()),
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

        cut = np.percentile(np.abs(radius_check), 3) - np.min(np.abs(radius_check))
        x, y = np.asarray(np.meshgrid(test_gaia, test_r))[:, np.abs(radius_check) < cut]

        radius = np.polyval(np.polyfit(x, y, 5), np.log10(sources["phot_g_mean_flux"]))
        # cap the radius for faint and saturated sources
        radius[np.log10(sources["phot_g_mean_flux"]) < 3] = 8.0
        radius[np.log10(sources["phot_g_mean_flux"]) > 6.5] = 24.0

        return self.r.to("arcsec") < radius[:, None]

    def _get_uncontaminated_pixel_mask(self):
        """
        Mask of shape npixels (maybe by nt) where not saturated, not faint,
        not contaminated etc.
        """
        return uncontaminated_pixel_mask

    def _get_centroids(self):
        """
        Find the ra and dec centroid of the image, at each time.
        """
        # centroids are astropy quantities
        pixmask = np.ones(self.dra.shape, dtype=bool)
        # pixmask = source.source_mask.astype(float)
        # centroids are astropy quantities
        self.ra_centroid = np.zeros(self.nt)
        self.dec_centroid = np.zeros(self.nt)
        dra_, ddec_ = dra[pixmask].value, ddec[pixmask].value
        for t in range(self.nt):
            self.ra_centroid[t] = np.average(
                dra_, weights=np.multiply(pixmask, self.flux[t])
            )
            self.dec_centroid[t] = np.average(
                ddec_, weights=np.multiply(pixmask, self.flux[t])
            )
        del pixmask, dra_, ddec_
        self.ra_centroid *= u.deg
        self.dec_centroid *= u.deg
        self.ra_centroid_avg = self.ra_centroid.mean()
        self.dec_centroid_avg = self.dec_centroid.mean()

    def _build_time_variable_model():
        return

    def _fit_time_variable_model():
        return

    def build_model():
        """
        Builds a sparse model matrix of shape nsources x npixels

        Builds the model using self.uncontaminated_pixel_mask
        """
        self._build_time_variable_model()
        self.model =

    def fit_model():
        """
        Fits sparse model matrix to all flux values iteratively in time

        Fits PSF model first using gaia flux, then updates `self.flux_estimates`
        Second iteration, uses best fit flux estimates to find PSF shape

        Fits model using self.source_mask
        """
        self._fit_time_variable_model()
        self.model_flux = None

    def diagnostic_plotting(self):
        """
        Some helpful diagnostic plots
        """

    return

    @static_method
    def from_TPFs(tpfs):
        """
        Convert TPF input into machine object
        """
        # Checks that all TPFs have identical time sampling
        times = np.array([tpf.astropy_time.jd for tpf in tpfs])
        # 1.2e-5 d = 1 sec
        if not np.all(times[1:, :] - times[-1:, :] < 1.2e-5):
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

        #  check for overlapped pixels betweent nearby tpfs, i.e. unique pixels
        # this keeps the first time a repeated pixels show up, need to
        # prioritize TPF size?
        _, unique_idx = np.unique(locs, axis=1, return_index=True)
        # important to srot indexes, np.unique return idx sorted by array values
        unique_idx.sort()
        # remove is a mask with overlapped pixels from the small TPF
        # this could split a primary target
        # would this remove significant pixel correlation information?
        # need to test if this changes the flux results
        # remove = np.zeros(locs.shape[1], dtype=bool)
        # locst = locs.T
        # tpfs_size = np.unique(unw[0], return_counts=True)[1]
        # for k, xy in enumerate(locst):
        #    rep = np.all(xy == locst, axis=1)
        #    if rep.sum() == 1:
        #        continue
        #    ind = np.where(rep)[0]
        #    remove[ind[np.argmin(tpfs_size[unw[0, ind]])]] = True

        locs = locs[:, unique_idx]
        flux = flux[:, unique_idx]
        flux_err = flux_err[:, unique_idx]
        unw = unw[:, unique_idx]

        # what happens if tpfs don't have WCS solutions?
        ra, dec = (
            tpfs[0]
            .wcs.wcs_pix2world(
                np.vstack([(locs[0] - tpfs[0].column), (locs[1] - tpfs[0].row)]).T, 0.0
            )
            .T
        )
        # Find sources using Gaia to pandas DF
        ras, decs, rads = [], [], []
        # find the max circle per TPF that contain pixel data to query Gaia
        for l in np.unique(unw[0]):
            ra1 = ra[unw[0] == l]
            dec1 = dec[unw[0] == l]
            ras.append(ra1.mean())
            decs.append(dec1.mean())
            rads.append(
                np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max() / 2
                + (u.arcsecond * 6).to(u.deg).value
            )
        sources = get_gaia_sources(
            tuple(ras),
            tuple(decs),
            tuple(rads),
            magnitude_limit=18,
            epoch=Time(times[0], format="jd").jyear,
        )
        del locs, nan_mask, ra1, dec1, ras, decs, rads

        # soruce cleaning happens here
        # We need a way to pair down the source list to only sources that are
        # 1) separated 2) on the image
        sources = self._clean_source_list(sources, ra, dec)

        return Machine(
            time=times, flux=flux, flux_err=flux_err, ra=ra, dec=dec, sources=sources
        )

    def _clean_source_list(sources, ra, dec):
        """
        Removes sources that are too contaminated or off the edge of the image
        """
        # find souerces on the image
        inside = np.zeros(len(sources), dtype=bool)
        off = 3.0 / 3600  # max distance in arcsec from image edge to source ra, dec
        for k in range(len(sources)):
            # masks for ra and dec
            raok = (sources["ra"][k] > ra - off) & (sources["ra"][k] < ra + off)
            decok = (sources["dec"][k] > dec - off) & (sources["dec"][k] < dec + off)
            inside[k] = (raok & decok).any()
        del raok, decok

        # find well separated sources
        s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
        midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)
        # all sources closer than 6" = 1.5 pix, this is a model Parameters
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
        bad = np.in1d(np.arange(len(sources)), faintest)
        del s_coords, midx, mdist, closest, blocs, bmags

        # combine 2 source masks
        clean = inside | ~bad

        # Keep track of sources that we removed
        # Flag each source for why they were removed?
        sources[:, "clean_flag"] = 0
        sources[:, "clean_flag"][~inside] = 2 ** 0  # outside TPF
        sources[:, "clean_flag"][bad] += 2 ** 1  # close contaminant
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
"""
