"""Subclass of `Machine` that Specifically work with FFIs"""
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.time import Time
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground, BkgZoomInterpolator

# from . import PACKAGEDIR
from .utils import do_tiled_query
from .machine import Machine
from .version import __version__

__all__ = ["FFIMachine"]

r_min, r_max = 20, 1044
c_min, c_max = 12, 1112


class FFIMachine(Machine):
    """Subclass of Machine for working with FFI data"""

    # Probably don't need a very new init function over Machine.
    def __init__(self, channel=1, quarter=5, wcs=None, **kwargs):

        self.column = kwargs["column"].ravel()
        self.row = kwargs["row"].ravel()
        self.ra = kwargs["ra"].ravel()
        self.dec = kwargs["dec"].ravel()
        self.flux_2d = kwargs["flux"]
        self.flux = np.atleast_2d(kwargs["flux"].ravel())
        self.flux_err = np.atleast_2d(kwargs["flux_err"].ravel())
        self.sources = kwargs["sources"]

        # remove background and mask bright/saturated pixels
        # these steps need to be done before `machine` init, so sparse delta
        # and flux arrays have the same shape
        self._remove_background()
        self._mask_pixels()

        # init `machine` object
        super().__init__(
            kwargs["time"],
            self.flux,
            self.flux_err,
            self.ra,
            self.dec,
            self.sources,
            self.column,
            self.row,
            n_r_knots=5,
            n_phi_knots=15,
            cut_r=6,
            do_sparse=True,
        )
        self.meta = kwargs["metadata"]
        self.channel = channel
        self.quarter = quarter
        self.wcs = wcs
        self.flux_2d = kwargs["flux"]

    def __repr__(self):
        return f"FFIMachine (N sources, N times, N pixels): {self.shape}"

    @staticmethod
    def from_file(fname, channel=1):
        """Reads data from files and initiates a new class...
        Parameters
        ----------
        fname : str
            Filename"""
        (
            wcs,
            time,
            flux,
            flux_err,
            ra,
            dec,
            column,
            row,
            metadata,
        ) = _load_file(fname, channel=channel)

        sources = _get_sources(
            ra, dec, wcs, magnitude_limit=18, epoch=time.jyear, ngrid=(5, 5), dr=3
        )
        # return wcs, time, flux, flux_err, ra, dec, column, row, sources
        return FFIMachine(
            time=np.array([time.jd]),
            flux=flux,
            flux_err=flux_err,
            ra=ra,
            dec=dec,
            sources=sources,
            column=column,
            row=row,
            channel=channel,
            quarter=metadata["QUARTER"],
            wcs=wcs,
            metadata=metadata,
        )

    def save_shape_model(self, output=None):
        """Saves the weights of a PRF fit to a file
        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        """
        raise NotImplementedError

    def load_shape_model(self):
        """Loads a PRF"""
        raise NotImplementedError

    def save_flux_values(self, output=None, format="fits"):
        """Saves the flux values of all sources to a file
        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        format : str
            Something like a format, maybe feather, csv, fits?
        """
        # check if model was fitted
        if not hasattr(self, "ws"):
            self.fit_model(fit_va=False)

        # asign default output file name
        if output is None:
            output = "./source_catalog_ch%02i_q%02i_mjd%s.fits" % (
                self.channel,
                self.quarter,
                str(self.time[0]),
            )
        # create bin table with photometry
        id_col = fits.Column(
            name="gaia_id", array=self.sources.designation, format="29A"
        )
        ra_col = fits.Column(name="ra", array=self.sources.ra, format="D", unit="deg")
        dec_col = fits.Column(
            name="dec", array=self.sources.dec, format="D", unit="deg"
        )
        flux_col = fits.Column(
            name="psf_flux", array=self.ws[0, :], format="D", unit="-e/s"
        )
        flux_err_col = fits.Column(
            name="psf_flux_err", array=self.werrs[0, :], format="D", unit="-e/s"
        )
        table_hdu = fits.BinTableHDU.from_columns(
            [id_col, ra_col, dec_col, flux_col, flux_err_col]
        )
        table_hdu.header["EXTNAME"] = "CATALOG"

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["object"] = ("Photometric Catalog", "Photometry")
        primary_hdu.header["origin"] = ("PSFmachine.FFIMachine", "Software of origin")
        primary_hdu.header["version"] = (__version__, "Software version")
        primary_hdu.header["TELESCOP"] = (self.meta["TELESCOP"], "Telescope")
        primary_hdu.header["mission"] = ("kepler", "Mission name")
        primary_hdu.header["OBSMODE"] = (self.meta["OBSMODE"], "Observing mode")
        primary_hdu.header["DCT_TYPE"] = (self.meta["DCT_TYPE"], "Data type")
        primary_hdu.header["quarter"] = (self.quarter, "Quarter of observations")
        primary_hdu.header["SEASON"] = (self.meta["SEASON"], "Observation season")
        primary_hdu.header["channel"] = (self.channel, "CCD channel")
        primary_hdu.header["MODULE"] = (self.meta["MODULE"], "CCD module")
        primary_hdu.header["OUTPUT"] = (self.meta["OUTPUT"], "CCD module")
        primary_hdu.header["aperture"] = ("PSF", "Type of photometry")
        primary_hdu.header["MJD-OBS"] = (self.time[0], "MJD of observation")
        primary_hdu.header["DATSETNM"] = (self.meta["DATSETNM"], "data set name")
        primary_hdu.header["RADESYS"] = (
            self.meta["RADESYS"],
            "reference frame of celestial coordinates",
        )
        primary_hdu.header["EQUINOX"] = (
            self.meta["EQUINOX"],
            "equinox of celestial coordinate system",
        )

        hdul = fits.HDUList([primary_hdu, table_hdu])

        hdul.writeto(output, checksum=True, overwrite=True)

        return

    def _remove_background(self, mask=None):
        """kepler-apertures probably used some background removal functions,
        e.g. mean filter, fine to subclass astropy here"""
        model = Background2D(
            self.flux_2d,
            mask=mask,
            box_size=(64, 50),
            filter_size=15,
            exclude_percentile=20,
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
            bkg_estimator=MedianBackground(),
            interpolator=BkgZoomInterpolator(order=3),
        )
        self.flux_2d -= model.background
        self.flux = self.flux_2d.ravel()[None, :]
        return

    def _saturated_pixels_mask(self, saturation_limit=1.5e5, tolerance=3):
        """
        Finds and removes saturated pixels, including bleed columns.

        Parameters
        ----------
        saturation_limit : foat
            Saturation limit at which pixels are removed.
        tolerance : float
            Number of pixels masked around the saturated pixel, remove bleeding.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
        # Which pixels are saturated
        # this nanpercentile takes forever to compute for a single cadance ffi
        # saturated = np.nanpercentile(self.flux, 99, axis=0)
        # assume we'll use ffi for 1 single cadence
        saturated = np.where(self.flux > saturation_limit)[1]
        # Find bad pixels, including allowence for a bleed column.
        bad_pixels = np.vstack(
            [
                np.hstack(
                    [
                        self.column[saturated] + idx
                        for idx in np.arange(-tolerance, tolerance)
                    ]
                ),
                np.hstack(
                    [self.row[saturated] for idx in np.arange(-tolerance, tolerance)]
                ),
            ]
        ).T
        # Find unique row/column combinations
        bad_pixels = bad_pixels[
            np.unique(["".join(s) for s in bad_pixels.astype(str)], return_index=True)[
                1
            ]
        ]
        # Build a mask of saturated pixels
        m = np.zeros(len(self.column), bool)
        # this works for FFIs but is slow
        for p in bad_pixels:
            m |= (self.column == p[0]) & (self.row == p[1])

        saturated = (self.flux > saturation_limit)[0]
        return m

    def _bright_sources_mask(self, magnitude_limit=8, tolerance=30):
        """
        Finds and mask pixels with halos produced by bright stars (<8 mag).

        Parameters
        ----------
        magnitude_limit : foat
            Magnitude limit at which bright sources are identified.
        tolerance : float
            Radius limit (in pixels) at which pixels around bright sources are masked.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
        bright_mask = self.sources["phot_g_mean_mag"] <= magnitude_limit

        mask = [
            np.hypot(self.column - s.column, self.row - s.row) < tolerance
            for _, s in self.sources[bright_mask].iterrows()
        ]
        mask = np.array(mask).sum(axis=0) > 0

        return mask

    def _mask_pixels(self, pixel_saturation_limit=1.2e5, magnitude_bright_limit=8):
        """
        Mask saturated pixels and halo/difraction from bright sources.

        Parameters
        ----------
        pixel_saturation_limit: float
            Flux value at which pixels saturate.
        magnitude_bright_limit: float
            Magnitude limit for sources at which pixels are masked.
        """

        # mask saturated pixels.
        self.non_sat_pixel_mask = ~self._saturated_pixels_mask(
            saturation_limit=pixel_saturation_limit
        )
        self.non_bright_source_mask = ~self._bright_sources_mask(
            magnitude_limit=magnitude_bright_limit
        )
        good_pixels = self.non_sat_pixel_mask & self.non_bright_source_mask

        self.column = self.column[good_pixels]
        self.row = self.row[good_pixels]
        self.ra = self.ra[good_pixels]
        self.dec = self.dec[good_pixels]
        self.flux = self.flux[:, good_pixels]
        self.flux_err = self.flux_err[:, good_pixels]
        return

    def plot_image(self, ax=None, sources=False):
        """
        Function to plot the Full Frame Image and the Gaia Sources

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned
        sources : boolean
            Whether to overplot or not the source catalog

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 10))
        ax = plt.subplot(projection=self.wcs)
        im = ax.imshow(
            self.flux_2d,
            cmap=plt.cm.viridis,
            origin="lower",
            norm=colors.SymLogNorm(linthresh=200, vmin=0, vmax=2000, base=10),
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042)

        ax.set_title("FFI Ch %i MJD %f" % (self.channel, self.time[0]))
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.grid(color="white", ls="solid")
        ax.set_aspect("equal", adjustable="box")

        if sources:
            ax.scatter(
                self.sources.column,
                self.sources.row,
                facecolors="none",
                edgecolors="r",
                linewidths=0.5,
                alpha=0.9,
            )
        return ax

    def plot_pixel_masks(self, ax=None):
        """
        Function to plot the mask used to reject saturated and bright pixels

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
        row_2d, col_2d = np.mgrid[: self.flux_2d.shape[0], : self.flux_2d.shape[1]]

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 10))
        if hasattr(self, "non_bright_source_mask"):
            ax.scatter(
                col_2d.ravel()[~self.non_bright_source_mask],
                row_2d.ravel()[~self.non_bright_source_mask],
                c="y",
                marker="s",
                s=1,
                label="bright mask",
            )
        if hasattr(self, "non_sat_pixel_mask"):
            ax.scatter(
                col_2d.ravel()[~self.non_sat_pixel_mask],
                row_2d.ravel()[~self.non_sat_pixel_mask],
                c="r",
                marker="s",
                s=1,
                label="saturated pixels",
            )
        ax.legend(loc="best")

        ax.set_xlabel("Column Pixel Number")
        ax.set_ylabel("Row Pixel Number")
        ax.set_title("Pixel Mask")

        return ax


def _load_file(fname, channel=1):
    """Helper function to load file?"""
    img_path = "./data/ffi/%s-cal.fits" % (fname)
    err_path = "./data/ffi/%s-uncert.fits" % (fname)
    if not os.path.isfile(img_path):
        raise FileNotFoundError("FFI calibrated fits file does not exist.")
    if not os.path.isfile(err_path):
        raise FileNotFoundError("FFI uncertainty fits file does not exist.")

    hdul = fits.open(img_path)
    header = hdul[0].header

    # Have to do some checks here that it's the right kind of data.
    #  We could loosen these checks in future.
    if header["TELESCOP"] == "Kepler":
        if header["DCT_TYPE"] == "FFI":
            pass
        else:
            raise TypeError("File is not Kepler FFI type.")
    elif header["TELESCOP"] == "TESS":
        raise NotImplementedError
    else:
        raise TypeError("File is not from Kepler or TESS mission")

    if hdul[channel].header["CHANNEL"] != channel:
        raise ValueError("Woring channel number")

    hdr = hdul[channel].header
    img = hdul[channel].data
    err = fits.open(err_path)[channel].data
    # err = np.sqrt(np.abs(img))
    wcs = WCS(hdr)
    # mid observ time
    time = Time((hdr["MJDEND"] + hdr["MJDSTART"]) / 2, format="mjd")
    row_2d, col_2d = np.mgrid[: img.shape[0], : img.shape[1]]
    col_2d = col_2d[r_min:r_max, c_min:c_max]
    row_2d = row_2d[r_min:r_max, c_min:c_max]
    flux_2d = img[r_min:r_max, c_min:c_max]
    flux_err_2d = err[r_min:r_max, c_min:c_max]
    ra, dec = wcs.all_pix2world(np.vstack([col_2d.ravel(), row_2d.ravel()]).T, 0.0).T
    col_2d -= c_min
    row_2d -= r_min
    ra_2d = ra.reshape(flux_2d.shape)
    dec_2d = dec.reshape(flux_2d.shape)

    attrs = [
        "TELESCOP",
        "INSTRUME",
        "OBSMODE",
        "DCT_TYPE",
        "DATSETNM",
        "QUARTER",
        "SEASON",
    ]
    meta = {k: header[k] for k in attrs}
    attrs = [
        "CHANNEL",
        "MODULE",
        "OUTPUT",
        "MJDSTART",
        "MJDEND",
        "TIMEDEL",
        "RADESYS",
        "EQUINOX",
    ]
    meta.update({k: hdr[k] for k in attrs})

    del hdul, header, hdr, img, err, ra, dec

    return (
        wcs,
        time,
        flux_2d,
        flux_err_2d,
        ra_2d,
        dec_2d,
        col_2d,
        row_2d,
        meta,
    )


def _get_sources(ra, dec, wcs, **kwargs):
    """"""
    sources = do_tiled_query(ra, dec, **kwargs)
    sources["column"], sources["row"] = wcs.all_world2pix(
        sources.loc[:, ["ra", "dec"]].values, 0.0
    ).T

    # correct col,row columns for gaia sources
    sources.row -= r_min
    sources.column -= c_min
    # remove sources outiside the ccd
    tolerance = 0
    inside = (
        (sources.row > 0 - tolerance)
        & (sources.row < 1023 + tolerance)
        & (sources.column > 0 - tolerance)
        & (sources.column < 1099 + tolerance)
    )
    sources = sources[inside].reset_index(drop=True)
    return sources


def buildKeplerPRFDatabase(fnames):
    """Procedure to build the database of Kepler PRF shape models.
    Parameters
    ---------
    fnames: list of str
        List of filenames for Kepler FFIs.
    """

    # This proceedure should be stored as part of the module, because it will
    # be vital for reproducability.

    # 1. Do some basic checks on FFI files that they are Kepler FFIs, and that
    # all 53 are present, all same channel etc.

    # 2. Iterate through files
    # for fname in fnames:
    #     f = FFIMachine.from_file(fname, HARD_CODED_PARAMETERS)
    #     f.build_shape_model()
    #     f.fit_model()
    #
    #     output = (
    #         PACKAGEDIR
    #         + f"src/psfmachine/data/q{quarter}_ch{channel}_{params}.csv"
    #     )
    #     f.save_shape_model(output=output)
    raise NotImplementedError
