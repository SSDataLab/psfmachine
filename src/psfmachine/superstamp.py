"""Subclass of `Machine` that Specifically work with FFIs"""
import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import imageio
from ipywidgets import interact

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from .ffi import FFIMachine, _get_sources
from .utils import _do_image_cutout

__all__ = ["SSMachine"]


class SSMachine(FFIMachine):
    """
    Subclass of Machine for working with FFI data. It is a subclass of Machine
    """

    def __init__(
        self,
        time,
        flux,
        flux_err,
        ra,
        dec,
        sources,
        column,
        row,
        wcs=None,
        limit_radius=32.0,
        n_r_knots=10,
        n_phi_knots=15,
        time_nknots=10,
        time_resolution=200,
        time_radius=8,
        rmin=1,
        rmax=16,
        cut_r=6,
        pos_corr1=None,
        pos_corr2=None,
        meta=None,
    ):
        """
        Class to work with K2 Supersampts produced by
        [Cody et al. 2018](https://archive.stsci.edu/prepds/k2superstamp/)

        Parameters and sttributes are the same as `FFIMachine`.
        """
        # init `FFImachine` object
        super().__init__(
            time,
            flux,
            flux_err,
            ra,
            dec,
            sources,
            column,
            row,
            wcs=wcs,
            limit_radius=limit_radius,
            n_r_knots=n_r_knots,
            n_phi_knots=n_phi_knots,
            time_nknots=time_nknots,
            time_resolution=time_resolution,
            time_radius=time_radius,
            rmin=rmin,
            rmax=rmax,
            cut_r=cut_r,
            meta=meta,
        )

        if pos_corr1 is not None and pos_corr1 is not None:
            self.pos_corr1 = np.nan_to_num(np.array(pos_corr1)[None, :])
            self.pos_corr2 = np.nan_to_num(np.array(pos_corr2)[None, :])
            self.time_corrector = "pos_corr"
        else:
            self.time_corrector = "centroid"
        self.poscorr_filter_size = 0
        self.meta["DCT_TYPE"] = "SuperStamp"

    def build_frame_shape_model(self, plot=False, **kwargs):
        """
        Compute shape model for every cadence (frame) using `Machine.build_shape_model()`

        Parameters
        ----------
        plot : boolean
            If `True` will create a video file in the working directory with the PSF
            model at each frame. It uses `imageio` and `imageio-ffmpeg`.
        **kwargs
            Keyword arguments to be passed to `build_shape_model()`
        """
        self.mean_model_frame = []
        images = []
        self._get_source_mask()
        self._get_uncontaminated_pixel_mask()
        org_sm = self.source_mask
        org_usm = self.uncontaminated_source_mask
        for tdx in tqdm(range(self.nt), desc="Building shape model per frame"):
            fig = self.build_shape_model(frame_index=tdx, plot=plot, **kwargs)
            self.mean_model_frame.append(self.mean_model)
            if plot:
                fig.canvas.draw()  # draw the canvas, cache the render
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
                plt.close()
            # we reset the source mask because sources may move slightly, e.g. K2
            self.source_mask = org_sm
            self.uncontaminated_source_mask = org_usm
        if plot:
            if hasattr(self, "meta"):
                gif_name = "./shape_models_%s_%s_c%i.mp4" % (
                    self.meta["MISSION"],
                    self.meta["OBJECT"].replace(" ", ""),
                    self.meta["QUARTER"],
                )
            else:
                gif_name = "./shape_models_%s_q%i.mp4" % (
                    self.tpf_meta["mission"][0],
                    self.tpf_meta["quarter"][0],
                )
            imageio.mimsave(gif_name, images, format="mp4", fps=24)

    def fit_frame_model(self):
        """
        Fits shape model per frame (cadence). It creates 3 attributes:
            * `self.model_flux_frame` has the scene model at every cadence.
            * `self.ws_frame` and `self.werrs_frame` have the flux values of all sources
            at every cadence.
        """
        prior_mu = self.source_flux_estimates  # np.zeros(A.shape[1])
        prior_sigma = (
            np.ones(self.mean_model.shape[0])
            * 5
            * np.abs(self.source_flux_estimates) ** 0.5
        )
        self.model_flux_frame = np.zeros(self.flux.shape) * np.nan
        self.ws_frame = np.zeros((self.nt, self.nsources))
        self.werrs_frame = np.zeros((self.nt, self.nsources))
        f = self.flux
        fe = self.flux_err
        for tdx in tqdm(
            range(self.nt),
            desc=f"Fitting {self.nsources} Sources (per frame model)",
            disable=self.quiet,
        ):
            X = self.mean_model_frame[tdx].copy().T
            sigma_w_inv = X.T.dot(X.multiply(1 / fe[tdx][:, None] ** 2)).toarray()
            sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
            B = X.T.dot((f[tdx] / fe[tdx] ** 2))
            B += prior_mu / (prior_sigma ** 2)
            self.ws_frame[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
            self.werrs_frame[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            self.model_flux_frame[tdx] = X.dot(self.ws_frame[tdx])
            nodata = np.asarray(self.source_mask.sum(axis=1))[:, 0] == 0
            # These sources are poorly estimated
            nodata |= (self.mean_model_frame[tdx].max(axis=1) > 1).toarray()[:, 0]
            self.ws_frame[tdx, nodata] *= np.nan
            self.werrs_frame[tdx, nodata] *= np.nan

    @staticmethod
    def from_file(
        fname,
        magnitude_limit=18,
        dr=2,
        sources=None,
        cutout_size=None,
        cutout_origin=[0, 0],
        **kwargs,
    ):
        """
        Reads data from files and initiates a new SSMachine class. SuperStamp file
        paths are passed as a string (single frame) or a list of paths (multiple
        frames). A samaller cutout of the full SuperSatamp can also be loaded by
        passing argumnts `cutout_size` and `cutout_origin`.

        Parameters
        ----------
        fname : string or list of strings
            Path to the FITS files to be parsed. For only one frame, pass a string,
            for multiple frames pass a list of paths.
        magnitude_limit : float
            Limiting magnitude to query Gaia catalog.
        dr : int
            Gaia data release to be use, default is 2, options are DR2 and EDR3.
        sources : pandas.DataFrame
            DataFrame with sources present in the images, optional. If None, then guery
            Gaia.
        cutout_size : int
            Size in pixels of the cutout, assumed to be squared. Default is 100.
        cutout_origin : tuple of ints
            Origin of the cutout following matrix indexing. Default is [0 ,0].
        **kwargs : dictionary
            Keyword arguments that defines shape model in a `Machine` object.
        Returns
        -------
        SSMachine : Machine object
            A Machine class object built from the SuperStamps files.
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
            poscorr1,
            poscorr2,
            metadata,
        ) = _load_file(fname)
        # do cutout if asked
        if cutout_size is not None:
            flux, flux_err, ra, dec, column, row = _do_image_cutout(
                flux,
                flux_err,
                ra,
                dec,
                column,
                row,
                cutout_size=cutout_size,
                cutout_origin=cutout_origin,
            )

        # we pass only non-empy pixels to the Gaia query and cleaning routines
        valid_pix = np.isfinite(flux).sum(axis=0).astype(bool)
        if sources is None or not isinstance(sources, pd.DataFrame):
            sources = _get_sources(
                ra[valid_pix],
                dec[valid_pix],
                wcs,
                magnitude_limit=magnitude_limit,
                epoch=time.jyear.mean(),
                ngrid=(2, 2) if flux.shape[1] <= 500 else (5, 5),
                dr=dr,
                img_limits=[[row.min(), row.max()], [column.min(), column.max()]],
                square=False,
            )

        return SSMachine(
            time.jd,
            flux,
            flux_err,
            ra.ravel(),
            dec.ravel(),
            sources,
            column.ravel(),
            row.ravel(),
            wcs=wcs,
            meta=metadata,
            pos_corr1=poscorr1,
            pos_corr2=poscorr2,
            **kwargs,
        )

    def fit_lightcurves(
        self,
        plot=False,
        iter_negative=False,
        fit_mean_shape_model=False,
        fit_va=False,
        sap=False,
    ):
        """
        Fit the sources in the data to get its light curves.
        By default it only uses the per cadence PSF model to do the photometry.
        Alternatively it can fit the mean-PSF and the mean-PSF with time model to
        the data, this is the original method implemented in `PSFmachine` and described
        in the paper. Aperture Photometry is also available by creating aperture masks
        that follow the mean-PSF shape.

        This function creates the `lcs` attribuite that contains a collection of light
        curves in the form of `lightkurve.LightCurveCollection`. Each entry in the
        collection is a `lightkurve.KeplerLightCurve` object with the different type
        of photometry (PSF per cadence, SAP, mean-PSF, and mean-PSF velocity-aberration
        corrected). Also each `lightkurve.KeplerLightCurve` object includes its
        asociated metadata.

        The photometry can also be accessed independently from the following attribuites
        that `fit_lightcurves` create:
            * `ws` and `werrs` have the uncorrected PSF flux and flux errors.
            * `ws_va` and `werrs_va` have the PSF flux and flux errors corrected by
            velocity aberration.
            * `sap_flux` and `sap_flux_err` have the flux and flux errors computed
            using aperture mask.
            * `ws_frame` and `werrs_frame` have the flux from PSF at each cadence.

        Parameters
        ----------
        plot : bool
            Whether or not to show some diagnostic plots. These can be helpful
            for a user to see if the PRF and time dependent models are being calculated
            correctly.
        iter_negative : bool
            When fitting light curves, it isn't possible to force the flux to be
            positive. As such, when we find there are light curves that deviate into
            negative flux values, we can clip these targets out of the analysis and
            rerun the model.
            If iter_negative is True, PSFmachine will run up to 3 times, clipping out
            any negative targets each round. This is used when
            `fit_mean_shape_model` is `True`.
        fit_mean_shape_model : bool
            Will do PSF photmetry using the mean-PSF.
        fit_va : bool
            Whether or not to fit Velocity Aberration (which implicitly will try to fit
            other kinds of time variability). `fit_mean_shape_model` must set to `True`
            ortherwise will be ignored. This will try to fit the "long term"
            trends in the dataset. If True, this will take slightly longer to fit.
            If you are interested in short term phenomena, like transits, you may
            find you do not need this to be set to True. If you have the time, it
            is recommended to run it.
        sap : boolean
            Compute or not Simple Aperture Photometry. See
            `Machine.compute_aperture_photometry()` for details.
        """
        # create mean shape model to be used by SAP and mean-PSF
        self.build_shape_model(plot=plot, frame_index="mean")
        # do SAP first
        if sap:
            self.compute_aperture_photometry(
                aperture_size="optimal", target_complete=1, target_crowd=1
            )

        # do mean-PSF photometry and time model if asked
        if fit_mean_shape_model:
            self.build_time_model(plot=plot, downsample=True)
            # fit the OG time model
            self.fit_model(fit_va=fit_va)
            if iter_negative:
                # More than 2% negative cadences
                negative_sources = (self.ws_va < 0).sum(axis=0) > (0.02 * self.nt)
                idx = 1
                while len(negative_sources) > 0:
                    self.mean_model[negative_sources] *= 0
                    self.fit_model(fit_va=fit_va)
                    negative_sources = np.where((self.ws_va < 0).all(axis=0))[0]
                    idx += 1
                    if idx >= 3:
                        break

        # fit shape model at each cadence
        self.build_frame_shape_model()
        self.fit_frame_model()

        self.lcs = []
        for idx, s in self.sources.iterrows():
            meta = {
                "ORIGIN": "PSFMACHINE",
                "APERTURE": "PSF + SAP" if sap else "PSF",
                "LABEL": s.designation,
                "MISSION": self.meta["MISSION"],
                "RA": s.ra,
                "DEC": s.dec,
                "PMRA": s.pmra / 1000,
                "PMDEC": s.pmdec / 1000,
                "PARALLAX": s.parallax,
                "GMAG": s.phot_g_mean_mag,
                "RPMAG": s.phot_rp_mean_mag,
                "BPMAG": s.phot_bp_mean_mag,
            }

            attrs = [
                "channel",
                "module",
                "ccd",
                "camera",
                "quarter",
                "campaign",
                "quarter",
                "row",
                "column",
                "mission",
            ]
            for attr in attrs:
                if attr in self.meta.keys():
                    meta[attr.upper()] = self.meta[attr]

            lc = lk.KeplerLightCurve(
                time=(self.time) * u.d,
                flux=self.ws_frame[:, idx] * (u.electron / u.second),
                flux_err=self.werrs_frame[:, idx] * (u.electron / u.second),
                meta=meta,
                time_format="jd",
            )
            if fit_mean_shape_model:
                lc["flux_NVA"] = (self.ws[:, idx]) * u.electron / u.second
                lc["flux_err_NVA"] = (self.werrs[:, idx]) * u.electron / u.second
                if fit_va:
                    lc["flux_VA"] = (self.ws_va[:, idx]) * u.electron / u.second
                    lc["flux_err_VA"] = (self.werrs_va[:, idx]) * u.electron / u.second
            if sap:
                lc["sap_flux"] = (self.sap_flux[:, idx]) * u.electron / u.second
                lc["sap_flux_err"] = (self.sap_flux_err[:, idx]) * u.electron / u.second

            self.lcs.append(lc)
        self.lcs = lk.LightCurveCollection(self.lcs)
        return

    def plot_image_interactive(self, ax=None, sources=False):
        """
        Function to plot the super stamp and Gaia Sources and interact by changing the
        cadence.

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
            norm=colors.SymLogNorm(linthresh=200, vmin=0, vmax=2000, base=10),
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042)

        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.set_xlim(self.column.min() - 2, self.column.max() + 2)
        ax.set_ylim(self.row.min() - 2, self.row.max() + 2)

        ax.set_aspect("equal", adjustable="box")

        def update(t):
            ax.pcolormesh(
                col_2d,
                row_2d,
                self.flux_2d[t],
                cmap=plt.cm.viridis,
                shading="nearest",
                norm=colors.SymLogNorm(linthresh=200, vmin=0, vmax=2000, base=10),
                rasterized=True,
            )
            ax.set_title(
                "%s %s Ch/CCD %s MJD %f"
                % (
                    self.meta["MISSION"],
                    self.meta["OBJECT"],
                    self.meta["EXTENSION"],
                    self.time[t],
                )
            )
            if sources:
                ax.scatter(
                    self.sources.column,
                    self.sources.row,
                    facecolors="none",
                    edgecolors="r",
                    linewidths=0.5 if self.sources.shape[0] > 1000 else 1,
                    alpha=0.9,
                )
            ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
            fig.canvas.draw_idle()

        interact(update, t=(0, self.flux_2d.shape[0] - 1, 1))

        return


def _load_file(fname):
    """
    Helper function to load K2 SuperStamp files files and parse data. This function
    works with K2 SS files created by Cody et al. 2018, which are single cadence FITS
    file, then a full campaign has many FITS files (e.g. M67 has 3620 files).

    Parameters
    ----------
    fname : string or list of strings
        Path to the FITS files to be parsed. For only one frame, pass a string,
        for multiple frames pass a list of paths.
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
    flux, flux_err = [], []
    times = []
    telescopes = []
    campaigns = []
    channels = []
    quality = []
    wcs_b = True
    poscorr1, poscorr2 = [], []
    for i, f in enumerate(fname):
        if not os.path.isfile(f):
            raise FileNotFoundError("FFI calibrated fits file does not exist: ", f)

        with fits.open(f) as hdul:
            # hdul = fits.open(f)
            header = hdul[0].header
            telescopes.append(header["TELESCOP"])
            campaigns.append(header["CAMPAIGN"])
            quality.append(header["QUALITY"])

            # clusters only have one ext, bulge have multi-extensions
            if len(hdul) > 1:
                img_ext = 1
            else:
                img_ext = 0
            channels.append(hdul[img_ext].header["CHANNEL"])
            hdr = hdul[img_ext].header
            # times.append(Time([hdr["DATE-OBS"], hdr["DATE-END"]], format="isot").mjd.mean())
            poscorr1.append(float(hdr["POSCORR1"]))
            poscorr2.append(float(hdr["POSCORR2"]))
            times.append(Time([hdr["TSTART"], hdr["TSTOP"]], format="jd").mjd.mean())
            flux.append(hdul[img_ext].data)
            if img_ext == 1:
                flux_err.append(hdul[2].data)
            else:
                flux_err.append(np.sqrt(np.abs(hdul[img_ext].data)))

            if header["QUALITY"] == 0 and wcs_b:
                wcs_b = False
                wcs = WCS(hdr)

    # check for integrity of files, same telescope, all FFIs and same quarter/campaign
    if len(set(telescopes)) != 1:
        raise ValueError("All FFIs must be from same telescope")

    # collect meta data, I get everthing from one header.
    attrs = [
        "TELESCOP",
        "INSTRUME",
        "MISSION",
        "DATSETNM",
        "OBSMODE",
        "OBJECT",
    ]
    meta = {k: header[k] for k in attrs if k in header.keys()}
    attrs = [
        "OBJECT",
        "RADESYS",
        "EQUINOX",
        "BACKAPP",
    ]
    meta.update({k: hdr[k] for k in attrs if k in hdr.keys()})
    meta.update({"EXTENSION": channels[0], "QUARTER": campaigns[0], "DCT_TYPE": "FFI"})
    if "MISSION" not in meta.keys():
        meta["MISSION"] = meta["TELESCOP"]

    # mask by quality and sort by times
    qual_mask = lk.utils.KeplerQualityFlags.create_quality_mask(
        np.array(quality), 1 | 2 | 4 | 8 | 32 | 16384 | 32768 | 65536 | 1048576
    )
    times = Time(times, format="mjd", scale="tdb")
    tdx = np.argsort(times)[qual_mask]
    times = times[tdx]
    row_2d, col_2d = np.mgrid[: flux[0].shape[0], : flux[0].shape[1]]
    flux_2d = np.array(flux)[tdx]
    flux_err_2d = np.array(flux_err)[tdx]
    poscorr1 = np.array(poscorr1)[tdx]
    poscorr2 = np.array(poscorr2)[tdx]

    ra, dec = wcs.all_pix2world(np.vstack([col_2d.ravel(), row_2d.ravel()]).T, 0.0).T
    ra_2d = ra.reshape(flux_2d.shape[1:])
    dec_2d = dec.reshape(flux_2d.shape[1:])

    del hdul, header, hdr, ra, dec

    return (
        wcs,
        times,
        flux_2d,
        flux_err_2d,
        ra_2d,
        dec_2d,
        col_2d,
        row_2d,
        poscorr1,
        poscorr2,
        meta,
    )
