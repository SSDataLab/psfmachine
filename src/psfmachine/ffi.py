"""Subclass of `Machine` that Specifically work with FFIs"""
import os

# import warnings
import wget

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import sparse
from tqdm.auto import tqdm
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip
from astropy.time import Time
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground, BkgZoomInterpolator

from . import PACKAGEDIR
from .utils import do_tiled_query
from .machine import Machine

__all__ = ["FFIMachine"]

r_min, r_max = 20, 1044
c_min, c_max = 12, 1112


class FFIMachine(Machine):
    """Subclass of Machine for working with FFI data"""

    # Probably don't need a very new init function over Machine.
    def __init__(self, channel=1, quarter=5, wcs=None, **kwargs):
        super().__init__(
            kwargs["time"],
            np.atleast_2d(kwargs["flux"].ravel()),
            np.atleast_2d(kwargs["flux_err"].ravel()),
            kwargs["ra"].ravel(),
            kwargs["dec"].ravel(),
            kwargs["sources"],
            kwargs["column"].ravel(),
            kwargs["row"].ravel(),
            n_r_knots=5,
            n_phi_knots=15,
        )
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
        wcs, time, quarter, flux, flux_err, ra, dec, column, row = _load_file(
            fname, channel=channel
        )
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
            quarter=quarter,
            wcs=wcs,
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

    def save_flux_values(self, output=None, format="feather"):
        """Saves the flux values of all sources to a file
        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        format : str
            Something like a format, maybe feather, csv, fits?
        """
        raise NotImplementedError

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
        self.flux_2d -= model
        raise NotImplementedError

    def _mask_pixels(self):
        """kepler-apertures probably needed some bespoke masking functions?"""
        raise NotImplementedError

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
                self.sources.col,
                self.sources.row,
                facecolors="none",
                edgecolors="r",
                linewidths=0.5,
                alpha=0.9,
            )
        return ax


def _load_file(fname, channel=1):
    """Helper function to load file?"""
    img_path = "./data/ffi/%s-cal.fits" % (fname)
    err_path = "./data/ffi/%s-uncert.fits" % (fname)
    if not os.path.isfile(img_path):
        print("Downloading FFI calibrated fits file")
        download_ffi(img_path.split("/")[-1])
    if not os.path.isfile(err_path):
        print("Downloading FFI uncertainty fits file")
        download_ffi(err_path.split("/")[-1])

    header = fits.open(img_path)[0].header
    quarter = header["QUARTER"]

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

    hdr = fits.open(img_path)[channel].header
    img = fits.open(img_path)[channel].data
    err = fits.open(err_path)[channel].data
    wcs = WCS(hdr)
    time = Time(hdr["MJDSTART"], format="mjd")
    row_2d, col_2d = np.mgrid[: img.shape[0], : img.shape[1]]
    col_2d = col_2d[r_min:r_max, c_min:c_max]
    row_2d = row_2d[r_min:r_max, c_min:c_max]
    flux_2d = img[r_min:r_max, c_min:c_max]
    flux_err_2d = err[r_min:r_max, c_min:c_max]
    ra, dec = wcs.all_pix2world(np.vstack([row_2d.ravel(), col_2d.ravel()]).T, 0.0).T
    col_2d -= c_min
    row_2d -= r_min
    ra_2d = ra.reshape(flux_2d.shape)
    dec_2d = dec.reshape(flux_2d.shape)

    return (wcs, time, quarter, flux_2d, flux_err_2d, ra_2d, dec_2d, col_2d, row_2d)


def _get_sources(ra, dec, wcs, **kwargs):
    """"""
    sources = do_tiled_query(ra, dec, **kwargs)
    sources["col"], sources["row"] = wcs.all_world2pix(
        sources.loc[:, ["ra", "dec"]].values, 0.0
    ).T

    # correct col,row columns for gaia sources
    sources.row -= r_min
    sources.col -= c_min
    # remove sources outiside the ccd
    tolerance = 0
    inside = (
        (sources.row > 0 - tolerance)
        & (sources.row < 1023 + tolerance)
        & (sources.col > 0 - tolerance)
        & (sources.col < 1099 + tolerance)
    )
    sources = sources[inside].reset_index(drop=True)
    return sources


def download_ffi(fits_name):
    """
    Download FFI fits file to a dedicated quarter directory

    Parameters
    ----------
    fits_name : string
        Name of FFI fits file
    """
    url = "https://archive.stsci.edu/missions/kepler/ffi"
    if fits_name == "":
        raise ValueError("Invalid fits file name")

    if not os.path.isdir("./data/ffi"):
        os.makedirs("./data/ffi")

    out = "./data/ffi/%s" % (fits_name)
    wget.download("%s/%s" % (url, fits_name), out=out)

    return


def _buildKeplerPRFDatabase(fnames):
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
    for fname in fnames:
        f = FFIMachine.from_file(fname, HARD_CODED_PARAMETERS)
        f.build_shape_model()
        f.fit_model()

        output = (
            PACKAGEDIR
            + f"src/psfmachine/data/q{self.quarter}_ch{self.channel}_{params}.csv"
        )
        f.save_shape_model(output=output)
