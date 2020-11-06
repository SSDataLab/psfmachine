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

    def __init__(self, time, flux, flux_err, ra, dec, sources,
                 radius_limit=20*u.arcsecond):
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
        radius_limit:
            helpful docstring


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

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]


        self.source_flux_estimates =  # Get from gaia values

        # Get the centroids of the images as a function of time
        self._get_centroids()

        # The separation in ra & dec from each source at each pixel shape
        # nsources x npixels, arcsec units, use astropy units!!!
        self.dra, self.ddec = np.asarray([[
            ra - self.sources['ra'][idx],
            dec - self.sources['dec'][idx]] for idx in range(len(self.sources))]).transpose(1,0,2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # polar coordinates
        self.r = np.hypot(self.dra, self.ddec)  # radial distance from source in arcseconds
        self.phi = np.arctan2(self.ddec, self.dra)  # azimuthal angle from source in radians

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        self.source_mask = self._get_source_mask()
        # Mask of shape npixels (maybe by nt) where not saturated, not faint,
        # not contaminated etc
        self.uncontaminated_pixel_mask = self._get_uncontaminated_pixel_mask()



    @property
    def shape(self):
        return (self.nsources, self.nt,  self.npixels)

    def __repr__(self):
        return f'Machine (N sources, N times, N pixels): {self.shape}'

    def _clean_source_list(self):
        """
        Removes sources that are too contaminated or off the edge of the image
        """
        # find souerces on the image
        for tpf in range(self.ntpfs):
            # masks for ra and dec
            raok = (self.sources['ra'] > self.ra[self.unw[0] == tpf])
            decok =
        clean =  # Some source mask

        # Keep track of sources that we removed
        # Flag each source for why they were removed?
        self._removed_sources = self.sources[~clean]

        self.sources = self.sources[clean]

    def _get_source_mask(self):
        """
        Mask of shape nsources x number of pixels, one where flux from a source
        exists
        """
        return source_mask

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
        #
        self.ra_centroid =
        self.dec_centroid =
        self.ra_centroid_avg =
        self.dec_centroid_avg =

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
        self.model_flux =

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
        # 1e-5 d = .8 sec
        if not np.all(times[1:, :] - times[-1:, :] < 1e-5):
            raise ValueError('All TPFs must have same time basis')
        times = times[0]

        # put fluxes into ntimes x npix shape
        flux = np.hstack([np.hstack(tpf.flux.transpose([2, 0, 1])) for tpf in tpfs])
        flux_err = np.hstack([np.hstack(tpf.flux_err.transpose([2, 0, 1])) for tpf in tpfs])

        unw = np.hstack([np.zeros((
            tpf.shape[0], tpf.shape[1] * tpf.shape[2]),
                                  dtype=int) + idx for idx,tpf in enumerate(tpfs)])

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
        locs = np.hstack([np.mgrid[
            tpf.column:tpf.column + tpf.shape[2],
            tpf.row: tpf.row + tpf.shape[1]].reshape(
                2, np.product(tpf.shape[1:])) for tpf in tpfs])
        locs = locs[:, ~np.all(nan_mask, axis=0)]

        # check for overlaped pixels betweent nearby tpfs, i.e. unique pixels
        # this keeps the first time a repeated pixels show up, need to
        # prioritize TPF size?
        _, unique_idx = np.unique(locs, axis=1, return_index=True)
        # important to srot indexes, np.unique return idx sorted by array values
        unique_idx.sort()
        locs = locs[:, unique_idx]
        flux = flux[:, unique_idx]
        flux_err = flux_err[:, unique_idx]
        unw = unw[:, unique_idx]

        # what happens if tpfs don't have WCS solutions?
        ra, dec = tpfs[0].wcs.wcs_pix2world(np.vstack([(locs[0] - tpfs[0].column),
                                                       (locs[1] - tpfs[0].row)]).T, 1).T
        # Find sources using Gaia to pandas DF
        coord = SkyCoord(ra, dec, unit='deg')
        ras, decs, rads = [], [], []
        # find the max circle per TPF that contain pixel data to query Gaia
        for l in np.unique(unw[0]):
            ra1 = ra[unw[0] == l]
            dec1 = dec[unw[0] == l]
            ras.append(ra1.mean())
            decs.append(dec1.mean())
            rads.append(np.hypot(ra1 - ra1.mean(), dec1 - dec1.mean()).max()/2 +
                        (u.arcsecond * 6).to(u.deg).value)
        sources = get_gaia_sources(tuple(ras), tuple(decs), tuple(rads),
                                   magnitude_limit=18)
        del locs, nan_mask, ra1, dec1, ras, decs, rads

        # soruce cleaning happens here
        # We need a way to pair down the source list to only sources that are
        # 1) separated 2) on the image
        self._clean_source_list()


        return Machine(time=times, flux=flux, flux_err=flux_err,
                       ra=ra, dec=dec, sources=sources)


"""
Notes on user functionality

mac = Machine().from_tpfs()
mac.build_model()
mac.fit_model()

mac.model_flux  # Values
mac.model_flux_err  # values
"""
