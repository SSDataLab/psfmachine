"""Subclass of `Machine` that Specifically work with TPFs"""
import os
import logging
import numpy as np
import lightkurve as lk
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import patches
import urllib.request
import tarfile

from .utils import get_gaia_sources
from .aperture import estimate_source_centroids_aperture, aperture_mask_to_2d
from .machine import Machine
from .version import __version__
from psfmachine import PACKAGEDIR

log = logging.getLogger(__name__)
__all__ = ["TPFMachine"]


class TPFMachine(Machine):
    """Subclass of `Machine` that specifically works with TPFs"""

    def __init__(
        self,
        tpfs,
        time,
        flux,
        flux_err,
        ra,
        dec,
        sources,
        column,
        row,
        limit_radius=24.0,
        time_mask=None,
        n_r_knots=10,
        n_phi_knots=15,
        n_time_knots=10,
        n_time_points=200,
        time_radius=8,
        rmin=1,
        rmax=16,
        pix2obs=None,
        #        pos_corr1=None,
        #        pos_corr2=None,
        focus_mask=None,
        tpf_meta=None,
    ):
        super().__init__(
            time=time,
            flux=flux,
            flux_err=flux_err,
            ra=ra,
            dec=dec,
            sources=sources,
            column=column,
            row=row,
            limit_radius=limit_radius,
            n_r_knots=n_r_knots,
            n_phi_knots=n_phi_knots,
            n_time_knots=n_time_knots,
            n_time_points=n_time_points,
            time_radius=time_radius,
            rmin=rmin,
            rmax=rmax,
        )
        self.tpfs = tpfs

        # combine focus mask and time mask if they exist
        if time_mask is None and focus_mask is None:
            self.time_mask = np.ones(len(self.time), bool)
        elif time_mask is None and focus_mask is not None:
            self.time_mask = focus_mask
        elif time_mask is not None and focus_mask is None:
            self.time_mask = time_mask
        else:
            self.time_mask = time_mask & focus_mask
        self.pix2obs = pix2obs
        #        self.pos_corr1 = pos_corr1
        #        self.pos_corr2 = pos_corr2
        self.tpf_meta = tpf_meta

    def __repr__(self):
        return f"TPFMachine (N sources, N times, N pixels): {self.shape}"

    def fit_lightcurves(
        self,
        plot=False,
        fit_va=True,
        iter_negative=True,
        load_shape_model=False,
        shape_model_file=None,
        sap=True,
    ):
        """
        Fit the sources inside the TPFs passed to `TPFMachine`.
        This function creates the `lcs` attribuite that contains a collection of light
        curves in the form of `lightkurve.LightCurveCollection`. Each entry in the
        collection is a `lightkurve.KeplerLightCurve` object with the different type
        of photometry (SAP, PSF, and PSF velocity-aberration corrected). Also each
        `lightkurve.KeplerLightCurve` object includes its asociated metadata.
        The photometry can also be accessed independently from the following attribuites
        that `fit_lightcurves` create:
            * `ws` and `werrs` have the uncorrected PSF flux and flux errors.
            * `ws_va` and `werrs_va` have the PSF flux and flux errors corrected by
            velocity aberration.
            * `sap_flux` and `sap_flux_err` have the flux and flux errors computed
            using aperture mask.

        Parameters
        ----------
        plot : bool
            Whether or not to show some diagnostic plots. These can be helpful
            for a user to see if the PRF and time dependent models are being calculated
            correctly.
        fit_va : bool
            Whether or not to fit Velocity Aberration (which implicitly will try to fit
            other kinds of time variability). This will try to fit the "long term"
            trends in the dataset. If True, this will take slightly longer to fit.
            If you are interested in short term phenomena, like transits, you may
            find you do not need this to be set to True. If you have the time, it
            is recommended to run it.
        iter_negative : bool
            When fitting light curves, it isn't possible to force the flux to be
            positive.
            As such, when we find there are light curves that deviate into negative flux
            values, we can clip these targets out of the analysis and rerun the model.
            If iter_negative is True, PSFmachine will run up to 3 times, clipping out
            any negative targets each round.
        load_shape_model : bool
            Load PRF shape model from disk or not. Default models were computed from
            FFI of the same channel and quarter.
        shape_model_file : string
            Path to PRF model file to be passed to `load_shape_model(input)`. If None,
            then precomputed models will be download from Zenodo repo.
        sap : boolean
            Compute or not Simple Aperture Photometry. See
            `Machine.compute_aperture_photometry()` for details.
        """
        # use PRF model from FFI or create one with TPF data
        if load_shape_model:
            self.load_shape_model(input=shape_model_file, plot=plot)
        else:
            self.build_shape_model(plot=plot)
        self.build_time_model(plot=plot)
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
        if sap:
            self.compute_aperture_photometry(
                aperture_size="optimal", target_complete=1, target_crowd=1
            )

        self.lcs = []
        for idx, s in self.sources.iterrows():

            meta = self._make_meta_dict(idx, s, sap)

            if fit_va:
                flux, flux_err = (
                    (self.ws_va[:, idx]) * u.electron / u.second,
                    self.werrs_va[:, idx] * u.electron / u.second,
                )
            else:
                flux, flux_err = (
                    (self.ws[:, idx]) * u.electron / u.second,
                    self.werrs[:, idx] * u.electron / u.second,
                )
            lc = lk.KeplerLightCurve(
                time=(self.time) * u.d,
                flux=flux,
                flux_err=flux_err,
                meta=meta,
                time_format="jd",
            )

            if fit_va:
                lc["psf_flux_NVA"] = (self.ws[:, idx]) * u.electron / u.second
                lc["psf_flux_err_NVA"] = (self.werrs[:, idx]) * u.electron / u.second
            if sap:
                lc["sap_flux"] = (self.sap_flux[:, idx]) * u.electron / u.second
                lc["sap_flux_err"] = (self.sap_flux_err[:, idx]) * u.electron / u.second
            self.lcs.append(lc)
        self.lcs = lk.LightCurveCollection(self.lcs)
        return

    def _make_meta_dict(self, idx, s, sap):
        """
        Auxiliar function that creates dictionarywith metadata for a given source in
        the catalog.

        Parameters
        ----------
        idx : int
            Source index.
        s : pandas.Series
            Row entry of the source in the source catalog.
        sap : boolean
            Add or not Simple Aperture Photometry metadata
        """
        ldx = np.where([idx in s for s in self.tpf_meta["sources"]])[0][0]
        mission = self.tpf_meta["mission"][ldx].lower()
        if s.tpf_id is not None:
            if mission == "kepler":
                label, targetid = f"KIC {int(s.tpf_id)}", int(s.tpf_id)
            elif mission == "tess":
                label, targetid = f"TIC {int(s.tpf_id)}", int(s.tpf_id)
            elif mission in ["k2", "ktwo"]:
                label, targetid = f"EPIC {int(s.tpf_id)}", int(s.tpf_id)
            else:
                raise ValueError(f"can not parse mission `{mission}`")
        else:
            label, targetid = s.designation, int(s.designation.split(" ")[-1])

        meta = {
            "ORIGIN": "PSFMACHINE",
            "APERTURE": "PSF + SAP" if sap else "PSF",
            "LABEL": label,
            "TARGETID": targetid,
            "MISSION": mission,
            "RA": s.ra,
            "DEC": s.dec,
            "PMRA": s.pmra / 1000,
            "PMDEC": s.pmdec / 1000,
            "PARALLAX": s.parallax,
            "GMAG": s.phot_g_mean_mag,
            "RPMAG": s.phot_rp_mean_mag,
            "BPMAG": s.phot_bp_mean_mag,
            "SAP": "optimal" if sap else "None",
            "FLFRCSAP": self.FLFRCSAP[idx] if sap else np.nan,
            "CROWDSAP": self.CROWDSAP[idx] if sap else np.nan,
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
            if attr in self.tpf_meta.keys():
                meta[attr.upper()] = self.tpf_meta[attr][ldx]
        return meta

    def to_fits():
        """Save all the light curves to fits files."""
        raise NotImplementedError

    def lcs_in_tpf(self, tpf_number):
        """
        Returns the light curves from a given TPF as a lightkurve.LightCurveCollection.

        Parameters
        ----------
        tpf_number: int
            Index of the TPF
        """
        ldx = self.tpf_meta["sources"][tpf_number]
        return lk.LightCurveCollection([self.lcs[l] for l in ldx])

    def plot_tpf(self, tdx, sap=True):
        """
        Make a diagnostic plot of a given TPF in the stack

        If you passed a stack of TPFs, this function will show an image of that
        TPF, and all the light curves inside it, alongside a diagnostic of which
        source the light curve belongs to.

        Parameters
        ----------
        tdx : int
            Index of the TPF to plot
        sap : boolean
            Overplot the pixel mask used for aperture photometry.
        """
        tpf = self.tpfs[tdx]
        ax_tpf = tpf.plot(aperture_mask="pipeline" if sap else None)
        sources = self.sources.loc[self.tpf_meta["sources"][tdx]]

        img_extent = (
            tpf.column - 0.5,
            tpf.column + tpf.shape[2] - 0.5,
            tpf.row - 0.5,
            tpf.row + tpf.shape[1] - 0.5,
        )

        r, c = np.mgrid[: tpf.shape[1], : tpf.shape[2]]
        r += tpf.row
        c += tpf.column

        # create 2D aperture mask for every source, for potting
        if hasattr(self, "aperture_mask") and sap:
            aperture_mask_2d = aperture_mask_to_2d(
                self.tpfs,
                self.tpf_meta["sources"],
                self.aperture_mask,
                self.column,
                self.row,
            )

        kdx = 0
        for sdx, s in sources.iterrows():
            if hasattr(self, "lcs"):
                lc = self.lcs_in_tpf(tdx)[kdx]
                if not np.isfinite(lc.flux).all():
                    kdx += 1
                    continue

                v = np.zeros(self.mean_model.shape[0])
                v[sdx] = s.phot_g_mean_flux
                m = self.mean_model.T.dot(v)
                mod = np.zeros(tpf.shape[1:]) * np.nan
                for jdx in range(r.shape[1]):
                    for idx in range(r.shape[0]):
                        l = np.where(
                            (self.row == r[idx, jdx]) & (self.column == c[idx, jdx])
                        )[0]
                        if len(l) == 0:
                            continue
                        mod[idx, jdx] = m[l]
                if np.nansum(mod) == 0:
                    kdx += 1
                    continue
                _ = plt.subplots(figsize=(12, 4))
                ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
                lc.errorbar(ax=ax, c="k", lw=0.3, ls="-")
                # plot SAP lc
                if hasattr(lc, "sap_flux") and sap:
                    lc.errorbar(
                        column="sap_flux",
                        ax=ax,
                        c="tab:red",
                        lw=0.3,
                        ls="-",
                        label="SAP",
                    )
                kdx += 1
                ax = plt.subplot2grid((1, 4), (0, 3))
                lk.utils.plot_image(mod, extent=img_extent, ax=ax)
                # Overlay the aperture mask if asked
                if sap:
                    aperture_mask = aperture_mask_2d["%i_%i" % (tdx, sdx)]
                    for i in range(r.shape[0]):
                        for j in range(r.shape[1]):
                            if aperture_mask[i, j]:
                                rect = patches.Rectangle(
                                    xy=(j + tpf.column - 0.5, i + tpf.row - 0.5),
                                    width=1,
                                    height=1,
                                    color="tab:red",
                                    fill=False,
                                    hatch="//",
                                )
                                ax.add_patch(rect)
            col, row = tpf.wcs.all_world2pix([[s.ra, s.dec]], 0)[0]
            if (
                (col < -3)
                | (col > (tpf.shape[2] + 3))
                | (row < -3)
                | (row > (tpf.shape[1] + 3))
            ):
                continue
            ax_tpf.scatter(
                col + tpf.column,
                row + tpf.row,
                facecolor="w",
                edgecolor="k",
            )

    def load_shape_model(self, input=None, plot=False):
        """
        Loads a PRF shape model from a FITs file.
        Not implemented: By default this function will load PRF shapes computed from
        FFI data (Kepler, K2, or TESS).

        Parameters
        ----------
        input : string
            Name of the file containing the shape parameters and weights. The file
            has to be FITS format.
            If None, then previously computed shape model from Kepler's FFI will be
            download from https://zenodo.org/record/5504503/ and used as default.
            The file download from Zenodo happens only the first time that shape models
            of a given mission (e.g. Kepler, K2) are asked. Then, shape models for all
            channels and quarters will be locally available for future use.
            The file is stored in `psfmachine/src/psfmachine/data/`.
        plot : boolean
            Plot or not the mean model.
        """
        # By default we will load PRF model from FFI when this are ok.
        if input is None:
            input = (
                f"{PACKAGEDIR}/data/ffi/ch{self.tpf_meta['channel'][0]:02}/"
                f"{self.tpf_meta['mission'][0]}_ffi_shape_model_"
                f"ch{self.tpf_meta['channel'][0]:02}_"
                f"q{self.tpf_meta['quarter'][0]:02}.fits"
            )
            if not os.path.isfile(input):
                # if file doesnt exist, download file bundle from zenodo:
                tar_file = (
                    f"{PACKAGEDIR}/data/"
                    f"{self.tpf_meta['mission'][0]}_FFI_PRFmodels_v1.0.tar.gz"
                )
                if not os.path.isfile(tar_file):
                    if not os.path.isdir(f"{PACKAGEDIR}/data/"):
                        os.makedirs(f"{PACKAGEDIR}/data/")
                    url = (
                        f"https://zenodo.org/record/5504503/files/"
                        f"{tar_file.split('/')[-1]}?download=1"
                    )
                    log.info(f"Downloading bundle files from: {url}")
                    with urllib.request.urlopen(url) as response, open(
                        tar_file, "wb"
                    ) as out_file:
                        out_file.write(response.read())
                # unpack
                with tarfile.open(tar_file) as f:
                    f.extractall(f"{PACKAGEDIR}/data/ffi/")
        # check if file exists and is the right format
        if not os.path.isfile(input):
            raise FileNotFoundError(f"No shape file: {input}")
        log.info(f"Using shape model from: {input}")
        if not input.endswith(".fits"):
            # should use a custom exception for wrong file format
            raise ValueError("File format not suported. Please provide a FITS file.")

        # create source mask and uncontaminated pixel mask
        self._get_source_mask()
        self._get_uncontaminated_pixel_mask()

        # open file
        hdu = fits.open(input)
        # check if shape parameters are for correct mission, quarter, and channel
        if hdu[1].header["mission"] != self.tpf_meta["mission"][0]:
            raise ValueError(
                "Wrong shape model: file is for mission '%s',"
                % (hdu[1].header["mission"])
                + " it should be '%s'." % (self.tpf_meta["mission"][0])
            )
        if hdu[1].header["quarter"] != self.tpf_meta["quarter"][0]:
            raise ValueError(
                "Wrong shape model: file is for quarter %i,"
                % (hdu[1].header["quarter"])
                + " it should be %i." % (self.tpf_meta["quarter"][0])
            )
        if hdu[1].header["channel"] != self.tpf_meta["channel"][0]:
            raise ValueError(
                "Wrong shape model: file is for channel %i,"
                % (hdu[1].header["channel"])
                + " it should be %i." % (self.tpf_meta["channel"][0])
            )
        # load model hyperparameters and weights
        self.n_r_knots = hdu[1].header["n_rknots"]
        self.n_phi_knots = hdu[1].header["n_pknots"]
        self.rmin = hdu[1].header["rmin"]
        self.rmax = hdu[1].header["rmax"]
        self.cut_r = hdu[1].header["cut_r"]
        self.psf_w = hdu[1].data["psf_w"]
        del hdu

        # create mean model, but PRF shapes from FFI are in pixels! and TPFMachine
        # work in arcseconds
        self._get_mean_model()
        # remove background pixels and recreate mean model
        self._update_source_mask_remove_bkg_pixels()

        if plot:
            return self.plot_shape_model()
        return

    def save_shape_model(self, output=None):
        """Saves the weights of a PRF fit to disk as a FITS file.

        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        """
        # asign a file name
        if output is None:
            output = "./%s_shape_model_ch%02i_q%02i.fits" % (
                self.tpf_meta["mission"][0],
                self.tpf_meta["channel"][0],
                self.tpf_meta["quarter"][0],
            )

        # create data structure (DataFrame) to save the model params
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="psf_w", array=self.psf_w, format="D")]
        )
        # include metadata and descriptions
        table.header["object"] = ("PRF shape", "PRF shape parameters")
        table.header["datatype"] = ("TPF stack", "Type of data used to fit shape model")
        table.header["origin"] = ("PSFmachine.TPFMachine", "Software of origin")
        table.header["version"] = (__version__, "Software version")
        table.header["mission"] = (self.tpf_meta["mission"][0], "Mission name")
        table.header["quarter"] = (
            self.tpf_meta["quarter"][0],
            "Quarter of observations",
        )
        table.header["channel"] = (self.tpf_meta["channel"][0], "Channel output")
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

        return

    def get_source_centroids(self, method="poscor"):
        """
        Compute centroids for sources in pixel coordinates.
        It implements three different methods to calculate centroids:
            * "aperture": computes centroids from moments `
            Machine.estimate_source_centroids_aperture()`. This needs the aperture
            masks to becomputed in advance with `Machine.compute_aperture_photometry`.
            Note that for sources with partial data (i.e. near TPF edges) the this
            method is illdefined.
            * "poscor": uses Gaia RA and Dec coordinates converted to pixel
            space using the TPFs WCS solution and the apply each TPF 'pos_corr'
            correction
            * "scene": uses Gaia coordinates again, but correction is computed from
            the TPF scene jitter using 'Machine._get_centroids()'.

        Parameters
        ----------
        method : string
            What type of corrected centroid will be computed.
            If "aperture", it creates attributes `source_centroids_[column/row]_ap`.
            If "poscor" (default), it creates attributes
            `source_centroids_[column/row]_poscor`.
            If "scene", it creates attributes `source_centroids_[column/row]_scene`.

            Note: "poscor" and "scene" show consistent results below 10th of a pixel.
        """
        # use aperture mask for moments centroids
        if method == "aperture":
            if not hasattr(self, "aperture_mask"):
                raise AttributeError("No aperture masks")
            centroids = estimate_source_centroids_aperture(
                self.aperture_mask, self.flux, self.column, self.row
            )
            self.source_centroids_column_ap = centroids[0] * u.pixel
            self.source_centroids_row_ap = centroids[1] * u.pixel
        # source centroids using pos_corr
        if method == "poscor":
            # get pixel coord from catalog RA, Dec and tpf WCS solution
            col, row = (
                self.tpfs[0]
                .wcs.all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0)
                .T
            )
            col += self.tpfs[0].column
            row += self.tpfs[0].row

            cadno_index = np.in1d(self.tpfs[0].time.jd, self.time)
            self.source_centroids_column_poscor = []
            self.source_centroids_row_poscor = []
            for i in range(self.nsources):
                tpf_idx = [
                    k for k, ss in enumerate(self.tpf_meta["sources"]) if i in ss
                ][0]
                # apply poss_corr from TPF cadences
                self.source_centroids_column_poscor.append(
                    col[i] + self.tpfs[tpf_idx].pos_corr1[cadno_index]
                )
                self.source_centroids_row_poscor.append(
                    row[i] + self.tpfs[tpf_idx].pos_corr2[cadno_index]
                )
            self.source_centroids_column_poscor = (
                np.array(self.source_centroids_column_poscor) * u.pixel
            )
            self.source_centroids_row_poscor = (
                np.array(self.source_centroids_row_poscor) * u.pixel
            )
        # use gaia coordinates and scene centroids
        if method == "scene":
            if not hasattr(self, "ra_centroid"):
                self._get_source_mask()
            centr_ra = np.tile(self.sources.ra.values, (self.nt, 1)).T + np.tile(
                self.ra_centroid.value, (self.nsources, 1)
            )
            centr_dec = np.tile(self.sources.dec.values, (self.nt, 1)).T + np.tile(
                self.dec_centroid.value, (self.nsources, 1)
            )

            self.source_centroids_column_scene, self.source_centroids_row_scene = (
                self.tpfs[0]
                .wcs.all_world2pix(np.array([centr_ra.ravel(), centr_dec.ravel()]).T, 0)
                .T
            )
            self.source_centroids_column_scene = (
                self.source_centroids_column_scene.reshape(self.nsources, self.nt)
                + self.tpfs[0].column
            ) * u.pixel
            self.source_centroids_row_scene = (
                self.source_centroids_row_scene.reshape(self.nsources, self.nt)
                + self.tpfs[0].row
            ) * u.pixel

    @staticmethod
    def from_TPFs(
        tpfs,
        magnitude_limit=18,
        dr=2,
        time_mask=None,
        apply_focus_mask=True,
        query_ra=None,
        query_dec=None,
        query_rad=None,
        **kwargs,
    ):
        """
        Convert TPF input into Machine object:
            * Parse TPFs to extract time, flux, clux erros, and bookkeeping of
            the TPF-pixel correspondance
            * Convert pixel-based coordinates (row, column) into WCS (ra, dec) for
            all pixels
            * Clean pixels with no values, bad cadences, and remove duplicated pixels
            due to overlapping TPFs
            * Query Gaia DR2 data base to find all sources present in the TPF images
            * Clean unresolve sources (within 6`` = 1.5 pixels) and sources off silicon
            with a 1 pixel tolerance (distance from source gaia pocition and TPF edge)

        Parameters
        ----------
        tpfs: lightkurve TargetPixelFileCollection
            Collection of Target Pixel files
        magnitude_limit : float
            Limiting magnitude to query Gaia catalog.
        dr : int
            Gaia data release to be use, default is 2, options are DR2 and EDR3
        time_mask : boolean array
            Mask to be applied to discard cadences if needed.
        apply_focus_mask : boolean
            Mask or not cadances near observation gaps to remove focus change.
        query_ra : numpy.array
            Array of RA to query Gaia catalog. Default is `None` and will use the
            coordinate centers of each TPF.
        query_dec : numpy.array
            Array of Dec to query Gaia catalog. Default is `None` and will use the
            coordinate centers of each TPF.
        query_rad : numpy.array
            Array of radius to query Gaia catalog. Default is `None` and will use
            the coordinate centers of each TPF.
        **kwargs
            Keyword arguments to be passed to `TPFMachine`.

        Returns
        -------
        Machine: Machine object
            A Machine class object built from TPFs.
        """
        if not isinstance(tpfs, lk.collections.TargetPixelFileCollection):
            raise TypeError("<tpfs> must be a of class Target Pixel Collection")

        # CH: all these internal functions should be put in another and from_tpfs
        # should be in another helper module
        attrs = [
            "ra",
            "dec",
            "targetid",
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
        attrs = [attr for attr in attrs if hasattr(tpfs[0], attr)]

        meta = np.asarray(
            [tuple(getattr(tpf, attr) for attr in attrs) for tpf in tpfs]
        ).T

        tpf_meta = {k: m for m, k in zip(meta, attrs)}
        if isinstance(tpfs[0], lk.KeplerTargetPixelFile):
            tpf_meta["tpfmag"] = [tpf.get_header()["kepmag"] for tpf in tpfs]
        elif isinstance(tpfs[0], lk.TessTargetPixelFile):
            tpf_meta["tpfmag"] = [tpf.get_header()["tmag"] for tpf in tpfs]
        else:
            raise ValueError("TPFs not understood")

        if not np.all([isinstance(tpf, lk.KeplerTargetPixelFile) for tpf in tpfs]):
            raise ValueError("Please only pass `lk.KeplerTargetPixelFiles`")
        if len(np.unique(tpf_meta["channel"])) != 1:
            raise ValueError("TPFs span multiple channels.")

        # parse tpfs
        (
            times,
            flux,
            flux_err,
            #            pos_corr1,
            #            pos_corr2,
            column,
            row,
            unw,
            focus_mask,
            qual_mask,
            saturated_mask,
        ) = _parse_TPFs(tpfs, **kwargs)

        if time_mask is not None:
            time_mask = np.copy(time_mask)[qual_mask]
        # use or not the focus mask
        focus_mask = focus_mask if apply_focus_mask else None

        # convert to RA Dec
        locs, ra, dec = _wcs_from_tpfs(tpfs)

        # preprocess arrays
        (
            flux,
            flux_err,
            #            pos_corr1,
            #            pos_corr2,
            unw,
            locs,
            ra,
            dec,
            column,
            row,
        ) = _preprocess(
            flux,
            flux_err,
            # pos_corr1, pos_corr2,
            unw,
            locs,
            ra,
            dec,
            column,
            row,
            tpfs,
            saturated_mask,
        )

        sources = _get_coord_and_query_gaia(
            tpfs, magnitude_limit, dr=dr, ra=query_ra, dec=query_dec, rad=query_rad
        )

        def get_tpf2source():
            tpf2source = []
            for tpf in tpfs:
                tpfra, tpfdec = tpf.get_coordinates(cadence=0)
                dra = np.abs(tpfra.ravel() - np.asarray(sources.ra)[:, None])
                ddec = np.abs(tpfdec.ravel() - np.asarray(sources.dec)[:, None])
                tpf2source.append(
                    np.where(
                        (
                            (dra < 4 * 3 * u.arcsecond.to(u.deg))
                            & (ddec < 4 * 3 * u.arcsecond.to(u.deg))
                        ).any(axis=1)
                    )[0]
                )
            return tpf2source

        tpf2source = get_tpf2source()
        sources = sources[
            np.in1d(np.arange(len(sources)), np.hstack(tpf2source))
        ].reset_index(drop=True)
        #        sources, _ = _clean_source_list(sources, ra, dec)
        tpf_meta["sources"] = get_tpf2source()

        idx, sep, _ = match_coordinates_sky(
            SkyCoord(tpf_meta["ra"], tpf_meta["dec"], unit="deg"),
            SkyCoord(np.asarray(sources[["ra", "dec"]]), unit="deg"),
        )
        match = (sep < 1 * u.arcsec) & (
            np.abs(
                np.asarray(sources["phot_g_mean_mag"][idx])
                - np.asarray(
                    [t if t is not None else np.nan for t in tpf_meta["tpfmag"]]
                )
            )
            < 0.25
        )

        sources["tpf_id"] = None
        sources.loc[idx[match], "tpf_id"] = np.asarray(tpf_meta["targetid"])[match]

        # return a Machine object
        return TPFMachine(
            tpfs=tpfs,
            time=times,
            flux=flux,
            flux_err=flux_err,
            ra=ra,
            dec=dec,
            sources=sources,
            column=column,
            row=row,
            pix2obs=unw,
            focus_mask=focus_mask,
            #            pos_corr1=pos_corr1,
            #            pos_corr2=pos_corr2,
            tpf_meta=tpf_meta,
            time_mask=time_mask,
            **kwargs,
        )


def _parse_TPFs(tpfs, **kwargs):
    """
    Parse TPF collection to extract times, pixel fluxes, flux errors and tpf-index
    per pixel

    Parameters
    ----------
    tpfs: lightkurve TargetPixelFileCollection
        Collection of Target Pixel files

    Returns
    -------
    times: numpy.ndarray
        Array with time values
    flux: numpy.ndarray
        Array with flux values per pixel
    flux_err: numpy.ndarray
        Array with flux errors per pixel
    unw: numpy.ndarray
        Array with TPF index for each pixel
    """

    time = tpfs[0].time.value

    if isinstance(tpfs[0], lk.KeplerTargetPixelFile):
        qual_mask = lk.utils.KeplerQualityFlags.create_quality_mask(
            tpfs[0].quality, 1 | 2 | 4 | 8 | 32 | 16384 | 65536 | 1048576
        )
        qual_mask &= (np.abs(tpfs[0].pos_corr1) < 5) & (np.abs(tpfs[0].pos_corr2) < 5)
        # Cut out 1.5 days after every data gap
        dt = np.hstack([10, np.diff(time)])
        focus_mask = ~np.in1d(
            np.arange(len(time)),
            np.hstack(
                [
                    np.arange(t, t + int(1.5 / np.median(dt)))
                    for t in np.where(dt > (np.median(dt) * 5))[0]
                ]
            ),
        )
        focus_mask = focus_mask[qual_mask]

    elif isinstance(tpfs[0], lk.TessTargetPixelFile):
        qual_mask = lk.utils.TessQualityFlags.create_quality_mask(
            tpfs[0].quality, lk.utils.TessQualityFlags.DEFAULT_BITMASK
        )
        qual_mask &= (np.abs(tpfs[0].pos_corr1) < 5) & (np.abs(tpfs[0].pos_corr2) < 5)
        focus_mask = np.ones(len(tpfs[0].time), bool)[qual_mask]

    cadences = np.array([tpf.cadenceno[qual_mask] for tpf in tpfs])

    # check if all TPFs has same cadences
    if not np.all(cadences[1:, :] - cadences[-1:, :] == 0):
        raise ValueError("All TPFs must have same time basis")
    # extract times
    times = np.asarray(tpfs[0].time.jd)[qual_mask]

    locs = [
        np.mgrid[
            tpf.column : tpf.column + tpf.shape[2],
            tpf.row : tpf.row + tpf.shape[1],
        ].reshape(2, np.product(tpf.shape[1:]))
        for tpf in tpfs
    ]
    locs = np.hstack(locs)
    column, row = locs

    # put fluxes into ntimes x npix shape
    flux = np.hstack(
        [np.hstack(tpf.flux[qual_mask].transpose([2, 0, 1])) for tpf in tpfs]
    )
    flux_err = np.hstack(
        [np.hstack(tpf.flux_err[qual_mask].transpose([2, 0, 1])) for tpf in tpfs]
    )

    sat_mask = []
    for tpf in tpfs:
        # Keplerish saturation limit
        saturated = np.nanmax(tpf.flux, axis=0).T.value > 1.4e5
        saturated = np.hstack(
            (np.gradient(saturated.astype(float))[1] != 0) | saturated
        )
        sat_mask.append(np.hstack(saturated))
    sat_mask = np.hstack(sat_mask)
    # pos_corr1 = np.hstack(
    #     [
    #         np.hstack(
    #             (
    #                 tpf.pos_corr1[qual_mask][:, None, None]
    #                 * np.ones(tpf.flux.shape[1:])[None, :, :]
    #             ).transpose([2, 0, 1])
    #         )
    #         for tpf in tpfs
    #     ]
    # )
    # pos_corr2 = np.hstack(
    #     [
    #         np.hstack(
    #             (
    #                 tpf.pos_corr2[qual_mask][:, None, None]
    #                 * np.ones(tpf.flux.shape[1:])[None, :, :]
    #             ).transpose([2, 0, 1])
    #         )
    #         for tpf in tpfs
    #     ]
    # )
    unw = np.hstack(
        [
            np.zeros((tpf.shape[1] * tpf.shape[2]), dtype=int) + idx
            for idx, tpf in enumerate(tpfs)
        ]
    )
    return (
        times,
        flux,
        flux_err,
        #        pos_corr1,
        #        pos_corr2,
        column,
        row,
        unw,
        focus_mask,
        qual_mask,
        sat_mask,
    )


def _preprocess(
    flux,
    flux_err,
    #    pos_corr1,
    #    pos_corr2,
    unw,
    locs,
    ra,
    dec,
    column,
    row,
    tpfs,
    saturated,
):
    """
    Clean pixels with nan values, bad cadences and removes duplicated pixels.
    """

    # CH this needs to be improved
    def _saturated_pixels_mask(flux, column, row, saturation_limit=1.2e5):
        """Finds and removes saturated pixels, including bleed columns."""
        # Which pixels are saturated
        saturated = np.nanpercentile(flux, 99, axis=0)
        saturated = np.where((saturated > saturation_limit).astype(float))[0]

        # Find bad pixels, including allowence for a bleed column.
        bad_pixels = np.vstack(
            [
                np.hstack([column[saturated] + idx for idx in np.arange(-3, 3)]),
                np.hstack([row[saturated] for idx in np.arange(-3, 3)]),
            ]
        ).T
        # Find unique row/column combinations
        bad_pixels = bad_pixels[
            np.unique(["".join(s) for s in bad_pixels.astype(str)], return_index=True)[
                1
            ]
        ]
        # Build a mask of saturated pixels
        m = np.zeros(len(column), bool)
        for p in bad_pixels:
            m |= (column == p[0]) & (row == p[1])
        return m

    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)

    # Finite pixels
    not_nan = np.isfinite(flux).all(axis=0)
    # Unique Pixels
    _, unique_pix = np.unique(locs, axis=1, return_index=True)
    unique_pix = np.in1d(np.arange(len(ra)), unique_pix)
    # No saturation and bleed columns

    mask = not_nan & unique_pix & ~saturated

    locs = locs[:, mask]
    column = column[mask]
    row = row[mask]
    ra = ra[mask]
    dec = dec[mask]
    flux = flux[:, mask]
    flux_err = flux_err[:, mask]
    #    pos_corr1 = pos_corr1[:, mask]
    #    pos_corr2 = pos_corr2[:, mask]
    unw = unw[mask]

    return (flux, flux_err, unw, locs, ra, dec, column, row)  # pos_corr1, pos_corr2,


def _wcs_from_tpfs(tpfs):
    """
    Extract pairs of row, column coordinates per pixels and convert them into
    World Cordinate System ra, dec.

    Parameters
    ----------
    tpfs: lightkurve TargetPixelFileCollection
        Collection of Target Pixel files

    Returns
    -------
    locs: numpy.ndarray
        2D array with pixel locations (columns, rows) from the TPFs
    ra: numpy.ndarray
        Array with right ascension values per pixel
    dec: numpy.ndarray
        Array with declination values per pixel
    """
    # calculate x,y grid of each pixel and RA, Dec
    locs, ra, dec = [], [], []
    for tpf in tpfs:
        ij = np.mgrid[
            tpf.column : tpf.column + tpf.shape[2],
            tpf.row : tpf.row + tpf.shape[1],
        ].reshape(2, np.product(tpf.shape[1:]))
        ra_, dec_ = tpf.wcs.wcs_pix2world(
            np.vstack([(ij[0] - tpf.column), (ij[1] - tpf.row)]).T, 0.0
        ).T
        locs.append(ij)
        ra.extend(ra_)
        dec.extend(dec_)
    locs = np.hstack(locs)
    ra = np.array(ra)
    dec = np.array(dec)

    return locs, ra, dec


def _get_coord_and_query_gaia(
    tpfs, magnitude_limit=18, dr=3, ra=None, dec=None, rad=None
):
    """
    Calculate ra, dec coordinates and search radius to query Gaia catalog

    Parameters
    ----------
    tpfs:
    magnitude_limit:
    dr: int
        Which gaia data release to use, default is DR2
    ra : float or list of floats
        RAs to do gaia query
    dec : float or list of floats
        Decs to do gaia query
    rad : float or list of floats
        Radius to do gaia query

    Returns
    -------
    sources: pandas.DataFrame
        Catalog with query result
    """
    if not isinstance(tpfs, lk.TargetPixelFileCollection):
        raise ValueError("Please pass a `lk.TargetPixelFileCollection`")

    # find the max circle per TPF that contain all pixel data to query Gaia
    # CH: Sometimes sources are missing from this...worth checking on
    if (ra is None) & (dec is None) & (rad is None):
        ras1, decs1 = np.asarray(
            [
                tpf.wcs.all_pix2world([np.asarray(tpf.shape[::-1][:2]) + 4], 0)[0]
                for tpf in tpfs
            ]
        ).T
        ras, decs = np.asarray(
            [
                tpf.wcs.all_pix2world([np.asarray(tpf.shape[::-1][:2]) // 2], 0)[0]
                for tpf in tpfs
            ]
        ).T
        rads = np.hypot(ras - ras1, decs - decs1)
    elif (ra is not None) & (dec is not None) & (rad is not None):
        ras, decs, rads = ra, dec, rad
    else:
        raise ValueError("Please set all or None of `ra`, `dec`, `rad`")

    # query Gaia with epoch propagation
    sources = get_gaia_sources(
        tuple(ras),
        tuple(decs),
        tuple(rads),
        magnitude_limit=magnitude_limit,
        epoch=Time(tpfs[0].time[len(tpfs[0]) // 2], format="jd").jyear,
        dr=dr,
    )

    ras, decs = [], []
    for tpf in tpfs:
        r, d = np.hstack(tpf.get_coordinates(0)).T.reshape(
            [2, np.product(tpf.shape[1:])]
        )
        ras.append(r)
        decs.append(d)
    ras, decs = np.hstack(ras), np.hstack(decs)
    sources, removed_sources = _clean_source_list(sources, ras, decs)
    return sources


def _clean_source_list(sources, ra, dec):
    """
    Removes sources that are too contaminated and/or off the edge of the image

    Parameters
    ----------
    sources: Pandas Dataframe
        Contains a list with cross-referenced Gaia results
        shape n sources x n Gaia features
    ra: numpy ndarray
        RA pixel position averaged in time
        shape npixel
    dec: numpy ndarray
        Dec pixel position averaged in time
        shape npixel

    Returns
    -------
    sources: Pandas.DataFrame
        Catalog with clean sources
    removed_sources: Pandas.DataFrame
        Catalog with removed sources
    """
    # find sources on the image
    inside = np.zeros(len(sources), dtype=bool)
    # max distance in arcsec from image edge to source ra, dec
    # 4 pixels
    sep = 4 * 4 * u.arcsec.to(u.deg)
    for k in range(len(sources)):
        raok = (sources["ra"][k] > ra - sep) & (sources["ra"][k] < ra + sep)
        decok = (sources["dec"][k] > dec - sep) & (sources["dec"][k] < dec + sep)
        inside[k] = (raok & decok).any()
    del raok, decok

    # Keep track of sources that we removed
    sources.loc[:, "clean_flag"] = 0
    # to avoid pandas SettingWithCopyWarning
    sources.clean_flag.where(inside, 2 ** 0, inplace=True)
    # combine 2 source masks
    clean = sources.clean_flag == 0
    removed_sources = sources[~clean].reset_index(drop=True)
    sources = sources[clean].reset_index(drop=True)

    return sources, removed_sources
