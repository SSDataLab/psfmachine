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


class FFIMachine(Machine):
    """
    Subclass of Machine for working with FFI data. It is a subclass of Machine
    """

    def __init__(
        self,
        channel=1,
        quarter=5,
        wcs=None,
        limit_radius=32.0,
        n_r_knots=10,
        n_phi_knots=15,
        cut_r=6,
        rmin=1,
        rmax=16,
        **kwargs,
    ):
        """
        Class to work with FFI data.

        Parameters
        ----------
        channel : int
            Channel number to be used
        quarter : int
            Quarter/Campagn nunmber to be used (for Kepler data).
        wcs : astropy.wcs
            World coordinates system solution for the FFI. Used for plotting.
        **kwargs
            Keyword attributes that contain information parsed from `from_file()` and
            is used to initialize a `Machine` class object.

        Attributes
        ----------
        All attributes inherited from Machine.

        meta : dictionary
            Meta data information related to the FFI
        wcs : astropy.wcs
            World coordinates system solution for the FFI. Used for plotting.
        flux_2d : numpy.ndarray
            2D image representation of the FFI, used for plotting.
        channel : int
            Channel number to be used
        quarter : int
            Quarter/Campagn nunmber to be used (for Kepler data).
        """
        self.column = kwargs["column"].ravel()
        self.row = kwargs["row"].ravel()
        self.ra = kwargs["ra"].ravel()
        self.dec = kwargs["dec"].ravel()
        # keep 2d image for easy plotting
        self.flux_2d = kwargs["flux"]
        # reshape flux and flux_err as [ntimes, npix]
        self.flux = kwargs["flux"].reshape(kwargs["flux"].shape[0], -1)
        self.flux_err = kwargs["flux_err"].reshape(kwargs["flux_err"].shape[0], -1)
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
            n_r_knots=n_r_knots,
            n_phi_knots=n_phi_knots,
            cut_r=cut_r,
            rmin=rmin,
            rmax=rmax,
        )
        self.meta = kwargs["metadata"]
        self.channel = channel
        self.quarter = quarter
        self.wcs = wcs
        self.flux_2d = kwargs["flux"]

    def __repr__(self):
        return f"FFIMachine (N sources, N times, N pixels): {self.shape}"

    @staticmethod
    def from_file(fname, channel=1, cutout_size=None, cutout_origin=[0, 0], **kwargs):
        """
        Reads data from files and initiates a new FFIMachine class.

        Parameters
        ----------
        fname : str
            Filename of the FFI file
        channel : int
            Channel number to be used
        cutout_size : int
            Size of the cutout in pixels, assumed to be square
        cutout_origin : tuple
            Origin pixel coordinates where to start the cut out. Follows matrix indexing
        **kwargs : dictionary
            Keyword arguments that defines shape model in a `machine` class object.

        Returns
        -------
        FFIMachine : Machine object
            A Machine class object built from the FFI.
        """
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
        if cutout_size is not None:
            flux, flux_err, ra, dec, column, row = do_image_cutout(
                flux,
                flux_err,
                ra,
                dec,
                column,
                row,
                cutout_size=cutout_size,
                cutout_origin=cutout_origin,
            )

        sources = _get_sources(
            ra,
            dec,
            wcs,
            magnitude_limit=18,
            epoch=time.jyear.mean(),
            ngrid=(2, 2) if flux.shape[1] <= 500 else (4, 4),
            dr=3,
            img_limits=[[row.min(), row.max()], [column.min(), column.max()]],
        )
        # return wcs, time, flux, flux_err, ra, dec, column, row, sources
        return FFIMachine(
            time=time.jd,
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
            **kwargs,
        )

    def save_shape_model(self, output=None):
        """
        Saves the weights of a PRF fit to a disk.

        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        """
        # asign a file name
        if output is None:
            output = "./%s-ffi_shape_model_ch%02i_q%02i.fits" % (
                self.meta["TELESCOP"],
                self.channel,
                self.quarter,
            )

        # create data structure (DataFrame) to save the model params
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="psf_w", array=self.psf_w, format="D")]
        )
        # include metadata and descriptions
        table.header["object"] = ("PRF shape", "PRF shape parameters")
        table.header["datatype"] = ("FFI", "Type of data used to fit shape model")
        table.header["origin"] = ("PSFmachine.FFIMachine", "Software of origin")
        table.header["version"] = (__version__, "Software version")
        table.header["TELESCOP"] = (self.meta["TELESCOP"], "Telescope name")
        table.header["mission"] = ("Kepler", "Mission name")
        table.header["quarter"] = (self.quarter, "Quarter of observations")
        table.header["channel"] = (self.channel, "Channel output")
        table.header["MJD-OBS"] = (self.time[0], "MJD of observation")
        table.header["n_rknots"] = (
            self.n_r_knots,
            "Number of knots for spline basis in radial axis",
        )
        table.header["n_pknots"] = (
            self.n_phi_knots,
            "Number of knots for spline basis in angle axis",
        )
        table.header["rmin"] = (self.rmin, "Minimum value for knot spacing")
        table.header["rmax"] = (self.rmax, "Maximum value for knot spacing")
        table.header["cut_r"] = (
            self.cut_r,
            "Radial distance to remove angle dependency",
        )
        # spline degree is hardcoded in `_make_A_polar` implementation.
        table.header["spln_deg"] = (3, "Degree of the spline basis")

        table.writeto(output, checksum=True, overwrite=True)

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
        """
        Background removal. It models the background using a median estimator, rejects
        flux values with sigma clipping. It modiffies the attributes `flux` and
        `flux_2d`.

        Parameters
        ----------
        mask : numpy.ndarray of booleans
            Mask to reject pixels containing source flux. Default None.
        """
        # model background for all cadences
        self.background_model = np.array(
            [
                Background2D(
                    flux_2d,
                    mask=mask,
                    box_size=(64, 50),
                    filter_size=15,
                    exclude_percentile=20,
                    sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
                    bkg_estimator=MedianBackground(),
                    interpolator=BkgZoomInterpolator(order=3),
                ).background
                for flux_2d in self.flux_2d
            ]
        )
        # substract background
        self.flux_2d -= self.background_model
        # flatten flix image
        self.flux = self.flux_2d.reshape(self.flux_2d.shape[0], -1)
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

    def residuals(self, plot=False, zoom=False, metric="residuals"):
        """
        Get the residuals (model - image) and compute statistics

        Parameters
        ----------
        plot : bool
            Do plotting
        zoom : bool
            Zoom into a section of the image for better visualization

        Return
        ------
        fig : matplotlib figure
            Figure
        """
        if not hasattr(self, "ws"):
            self.fit_model(fit_va=False)

        # evaluate mean model
        ffi_model = self.mean_model.T.dot(self.ws[0])
        ffi_model_err = self.mean_model.T.dot(self.werrs[0])
        # compute residuals
        residuals = ffi_model - self.flux[0]
        weighted_chi = (ffi_model - self.flux[0]) ** 2 / ffi_model_err
        # mask background
        source_mask = ffi_model != 0.0
        # rms
        self.rms = np.sqrt((residuals[source_mask] ** 2).mean())
        self.frac_esidual_median = np.median(
            residuals[source_mask] / self.flux[0][source_mask]
        )
        self.frac_esidual_std = np.std(
            residuals[source_mask] / self.flux[0][source_mask]
        )

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(15, 15))

            ax[0, 0].scatter(
                self.column,
                self.row,
                c=self.flux[0],
                marker="s",
                s=7.5 if zoom else 1,
                norm=colors.SymLogNorm(linthresh=500, vmin=0, vmax=5000, base=10),
            )
            ax[0, 0].set_aspect("equal", adjustable="box")

            ax[0, 1].scatter(
                self.column,
                self.row,
                c=ffi_model,
                marker="s",
                s=7.5 if zoom else 1,
                norm=colors.SymLogNorm(linthresh=500, vmin=0, vmax=5000, base=10),
            )
            ax[0, 1].set_aspect("equal", adjustable="box")

            if metric == "residuals":
                to_plot = residuals
                norm = colors.SymLogNorm(linthresh=500, vmin=-5000, vmax=5000, base=10)
                cmap = "RdBu"
            elif metric == "chi2":
                to_plot = weighted_chi
                norm = colors.LogNorm(vmin=1, vmax=5000)
                cmap = "viridis"
            else:
                raise ValueError("wrong type of metric")

            cbar = ax[1, 0].scatter(
                self.column[source_mask],
                self.row[source_mask],
                c=to_plot[source_mask],
                marker="s",
                s=7.5 if zoom else 1,
                cmap=cmap,
                norm=norm,
            )
            ax[1, 0].set_aspect("equal", adjustable="box")
            plt.colorbar(
                cbar, ax=ax[1, 0], label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042
            )

            ax[1, 1].hist(
                residuals[source_mask] / self.flux[0][source_mask],
                bins=50,
                log=True,
                label=(
                    "RMS (model - data) = %.3f" % self.rms
                    + "\nMedian = %.3f" % self.frac_esidual_median
                    + "\nSTD = %3f" % self.frac_esidual_std
                ),
            )
            ax[1, 1].legend(loc="best")

            ax[0, 0].set_ylabel("Pixel Row Number")
            ax[0, 0].set_xlabel("Pixel Column Number")
            ax[0, 1].set_xlabel("Pixel Column Number")
            ax[1, 0].set_ylabel("Pixel Row Number")
            ax[1, 0].set_xlabel("Pixel Column Number")
            ax[1, 1].set_xlabel("(model - data) / data")

            if zoom:
                ax[0, 0].set_xlim(self.column.min(), self.column.min() + 100)
                ax[0, 0].set_ylim(self.row.min(), self.row.min() + 100)
                ax[0, 1].set_xlim(self.column.min(), self.column.min() + 100)
                ax[0, 1].set_ylim(self.row.min(), self.row.min() + 100)
                ax[1, 0].set_xlim(self.column.min(), self.column.min() + 100)
                ax[1, 0].set_ylim(self.row.min(), self.row.min() + 100)

            return fig

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
        row_2d, col_2d = np.mgrid[
            self.row.min() : self.row.max() + 1,
            self.column.min() : self.column.max() + 1,
        ]
        im = ax.pcolormesh(
            col_2d,
            row_2d,
            self.flux_2d[0],
            cmap=plt.cm.viridis,
            shading="nearest",
            # origin="lower",
            norm=colors.SymLogNorm(linthresh=200, vmin=0, vmax=2000, base=10),
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042)

        ax.set_title("FFI Ch %i MJD %f" % (self.channel, self.time[0]))
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
        ax.set_xlim(self.column.min() - 2, self.column.max() + 2)
        ax.set_ylim(self.row.min() - 2, self.row.max() + 2)

        ax.set_aspect("equal", adjustable="box")

        if sources:
            ax.scatter(
                self.sources.column,
                self.sources.row,
                facecolors="none",
                edgecolors="r",
                linewidths=0.5 if self.sources.shape[0] > 1000 else 1,
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
        row_2d, col_2d = np.mgrid[: self.flux_2d.shape[1], : self.flux_2d.shape[2]]

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
    """
    Helper function to load FFI files and parse data.

    Parameters
    ----------
    fname : string or list of strings
        Name of the FFI files
    channel : int
        Number of channel to be used.

    Returns
    -------
    wcs : astropy.wcs
        World coordinates system solution for the FFI. Used to convert RA, Dec to pixels
    time : numpy.array
        Array with time values in MJD
    flux_2d : numpy.ndarray
        Array with 2D (image) representation of flux values
    flux_err_2d : numpy.ndarray
        Array with 2D (image) representation of flux errors
    ra_2d : numpy.ndarray
        Array with 2D (image) representation of flux RA
    dec_2d : numpy.ndarray
        Array with 2D (image) representation of flux Dec
    col_2d : numpy.ndarray
        Array with 2D (image) representation of pixel column
    row_2d : numpy.ndarray
        Array with 2D (image) representation of pixel row
    meta : dict
        Dictionary with metadata
    """
    if not isinstance(fname, list):
        fname = np.sort([fname])
    imgs = []
    times = []
    telescopes = []
    dct_types = []
    quarters = []
    channels = []

    for i, f in enumerate(fname):
        if not os.path.isfile(f):
            raise FileNotFoundError("FFI calibrated fits file does not exist: ", f)

        hdul = fits.open(f)
        header = hdul[0].header
        telescopes.append(header["TELESCOP"])
        dct_types.append(header["DCT_TYPE"])
        if header["DATSETNM"].startswith("kplr"):
            quarters.append(header["QUARTER"])
        elif header["DATSETNM"].startswith("ktwo"):
            quarters.append(header["CAMPAIGN"])
        elif header["TELESCOP"] == "TESS":
            raise NotImplementedError

        hdr = hdul[channel].header
        times.append((hdr["MJDEND"] + hdr["MJDSTART"]) / 2)
        imgs.append(hdul[channel].data)
        channels.append(hdul[channel].header["CHANNEL"])

        if i == 0:
            wcs = WCS(hdr)

    # check for integrity of files, same telescope, all FFIs and same quarter
    if len(set(telescopes)) != 1:
        raise ValueError("All FFIs must be from same telescope")
    if len(set(dct_types)) != 1 or set(dct_types).pop() != "FFI":
        raise ValueError("All images must be FFIs")
    if len(set(quarters)) != 1:
        raise ValueError("All FFIs must be of same quarter/campaign/sector.")
    if len(set(channels)) != 1 or set(channels).pop() != channel:
        raise ValueError("Woring channel number")
    # Have to do some checks here that it's the right kind of data.
    #  We could loosen these checks in future.
    if telescopes[0] == "Kepler":
        if dct_types[0] == "FFI":
            pass
        else:
            raise TypeError("File is not Kepler FFI type.")
        # CCD overscan for Kepler
        r_min = 20
        r_max = 1044
        c_min = 12
        c_max = 1112
    elif telescopes[0] == "TESS":
        # CCD overscan for TESS
        r_min = 0
        r_max = 2048
        c_min = 45
        c_max = 2093
        raise NotImplementedError
    else:
        raise TypeError("File is not from Kepler or TESS mission")

    # collect meta data, I get everthing from one header.
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
        "RADESYS",
        "EQUINOX",
    ]
    meta.update({k: hdr[k] for k in attrs})
    # sort by times
    times = Time(times, format="mjd")
    tdx = np.argsort(times)
    times = times[tdx]

    # remove overscan
    row_2d, col_2d = np.mgrid[: imgs[0].shape[0], : imgs[0].shape[1]]
    col_2d = col_2d[r_min:r_max, c_min:c_max]
    row_2d = row_2d[r_min:r_max, c_min:c_max]
    flux_2d = np.array(imgs)[tdx, r_min:r_max, c_min:c_max]
    flux_err_2d = np.sqrt(np.abs(flux_2d))

    ra, dec = wcs.all_pix2world(np.vstack([col_2d.ravel(), row_2d.ravel()]).T, 0.0).T
    ra_2d = ra.reshape(flux_2d.shape[1:])
    dec_2d = dec.reshape(flux_2d.shape[1:])

    del hdul, header, hdr, imgs, ra, dec

    return (
        wcs,
        times,
        flux_2d,
        flux_err_2d,
        ra_2d,
        dec_2d,
        col_2d,
        row_2d,
        meta,
    )


def _get_sources(ra, dec, wcs, img_limits=[[0, 0], [0, 0]], **kwargs):
    """
    Query Gaia catalog in a tiled manner and clean sources off sensor.

    Parameters
    ----------
    ra : numpy.ndarray
        Data array with pixel RA values used to create the grid for tiled query and
        compute centers and radius of cone search
    dec : numpy.ndarray
        Data array with pixel Dec values used to create the grid for tiled query and
        compute centers and radius of cone search
    wcs : astropy.wcs
        World coordinates system solution for the FFI. Used to convert RA, Dec to pixels
    img_limits :
        Image limits in pixel numbers to remove sources outside the CCD
    **kwargs
        Keyword arguments to be passed to `do_tiled_query`.

    Returns
    -------
    sources : pandas.DataFrame
        Data Frame with query result
    """
    sources = do_tiled_query(ra, dec, **kwargs)
    sources["column"], sources["row"] = wcs.all_world2pix(
        sources.loc[:, ["ra", "dec"]].values, 0.0
    ).T

    # remove sources outiside the ccd with a tolerance
    tolerance = 0
    inside = (
        (sources.row > img_limits[0][0] - tolerance)
        & (sources.row < img_limits[0][1] + tolerance)
        & (sources.column > img_limits[1][0] - tolerance)
        & (sources.column < img_limits[1][1] + tolerance)
    )
    sources = sources[inside].reset_index(drop=True)
    return sources


def do_image_cutout(
    flux, flux_err, ra, dec, column, row, cutout_size=100, cutout_origin=[0, 0]
):
    """
    Creates a cutout of the full image

    Parameters
    ----------
    cutout_size : int
        Size in pixels of the cutout
    cutout_origin : list
        Origin of the cutout following matrix indexing
    """

    if cutout_size + cutout_origin[0] < np.minimum(*flux.shape):
        column = column[
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
        row = row[
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
        flux = flux[
            :,
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
        flux_err = flux_err[
            :,
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
        ra = ra[
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
        dec = dec[
            cutout_origin[0] : cutout_origin[0] + cutout_size,
            cutout_origin[1] : cutout_origin[1] + cutout_size,
        ]
    else:
        raise ValueError("Cutout size is larger than image shape ", flux.shape)

    return flux, flux_err, ra, dec, column, row


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
