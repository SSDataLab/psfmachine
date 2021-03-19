"""Subclass of `Machine` that Specifically work with TPFs"""
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import sparse
from astropy.coordinates import SkyCoord, match_coordinates_3d, match_coordinates_sky
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, sigma_clipped_stats

from .utils import get_gaia_sources, _make_A_polar, _make_A_cartesian
from .machine import Machine

__all__ = ["TPFMachine"]


class TPFMachine(Machine):
    """Subclass of `Machine` that specifically works with TPFs"""

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
        limit_radius=24.0,
        time_mask=None,
        n_r_knots=10,
        n_phi_knots=15,
        n_time_knots=10,
        n_time_points=200,
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
            rmin=rmin,
            rmax=rmax,
        )

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
        if time_mask is None:
            self.time_mask = focus_mask
        else:
            self.time_mask = time_mask & focus_mask

        self.pix2obs = pix2obs
        #        self.pos_corr1 = pos_corr1
        #        self.pos_corr2 = pos_corr2
        self.tpf_meta = tpf_meta

    def __repr__(self):
        return f"TPFMachine (N sources, N times, N pixels): {self.shape}"

    def fit_lightcurves(self, plot=False, fit_va=True, iter_negative=True):
        """"""

        self.build_shape_model(plot=plot)
        self.build_time_model(plot=plot)
        self.fit_model(fit_va=fit_va)
        if iter_negative:
            negative_sources = np.where((self.ws_va < 0).all(axis=0))[0]
            idx = 1
            while len(negative_sources) > 0:
                self.mean_model[negative_sources] *= 0
                self.fit_model(fit_va=fit_va)
                negative_sources = np.where((self.ws_va < 0).all(axis=0))[0]
                idx += 1
                if idx >= 3:
                    break

        self.lcs = []
        for idx, s in self.sources.iterrows():
            if s.kic is not None:
                label, targetid = f"KIC {int(s.kic)}", int(s.kic)
            else:
                label, targetid = s.designation, int(s.designation.split(" ")[-1])
            ldx = np.where([idx in s for s in self.tpf_meta["sources"]])[0][0]
            vals = [
                "PSFMACHINE",
                "PSF",
                label,
                targetid,
                int(self.tpf_meta["channel"][ldx]),
                int(self.tpf_meta["quarter"][ldx]),
                "kepler",
                s.ra,
                s.dec,
                s.pmra / 1000,
                s.pmdec / 1000,
                s.parallax,
                s.phot_g_mean_mag,
            ]
            keys = [
                "ORIGIN",
                "APERTURE",
                "LABEL",
                "TARGETID",
                "CHANNEL",
                "QUARTER",
                "MISSION",
                "RA",
                "DEC",
                "PMRA",
                "PMDEC",
                "PARALLAX",
                "GMAG",
                "RPMAG",
                "BPMAG",
                "TEFF",
                "RADIUS",
            ]
            meta = {key: val for val, key in zip(vals, keys)}
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
                lc["flux_NVA"] = (self.ws[:, idx]) * u.electron / u.second
                lc["flux_err_NVA"] = (self.werrs[:, idx]) * u.electron / u.second
            self.lcs.append(lc)
            self.lcs = lk.LightCurveCollection(self.lcs)
        return

    def to_fits():
        """Save all the light curves to fits files..."""
        raise NotImplementedError

    def lcs_in_tpf(self, tpf_number):
        ldx = self.tpf_meta["sources"][tpf_number]
        return lk.LightCurveCollection([self.lcs[l] for l in ldx])

    @staticmethod
    def from_TPFs(tpfs, magnitude_limit=18, dr=2, time_mask=None, **kwargs):
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
        tpfs : lightkurve TargetPixelFileCollection
            Collection of Target Pixel files

        Returns
        -------
        Machine : Machine object
            A Machine class object built from TPFs.
        """
        if not isinstance(tpfs, lk.collections.TargetPixelFileCollection):
            raise TypeError("<tpfs> must be a of class Target Pixel Collection")

        # CH: all these internal functions should be put in another and from_tpfs should be in another helper module

        meta = np.asarray(
            [
                (
                    tpf.ra,
                    tpf.dec,
                    tpf.header["kepmag"],
                    tpf.targetid,
                    tpf.channel,
                    tpf.quarter,
                    tpf.row,
                    tpf.column,
                )
                for tpf in tpfs
            ]
        ).T

        keys = ["ra", "dec", "kepmag", "kic", "channel", "quarter", "row", "column"]
        tpf_meta = {k: m for m, k in zip(meta, keys)}

        if not np.all([isinstance(tpf, lk.KeplerTargetPixelFile) for tpf in tpfs]):
            raise ValueError("Please only pass `lk.KeplerTargetPixelFiles`")
        if len(np.unique(tpf_meta["channel"])) != 1:
            raise ValueError(f"TPFs span multiple channels.")

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
        ) = _parse_TPFs(tpfs, **kwargs)

        if time_mask is not None:
            time_mask = np.copy(time_mask)[qual_mask]

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
        )

        sources = _get_coord_and_query_gaia(tpfs, magnitude_limit, dr=dr)

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
        match = (sep < 1 * u.arcsec) & np.asarray(
            np.abs(sources["phot_g_mean_mag"][idx] - tpf_meta["kepmag"]) < 0.25
        )
        sources["kic"] = None
        sources.loc[idx[match], "kic"] = np.asarray(tpf_meta["kic"])[match]

        # return a Machine object
        return TPFMachine(
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
    tpfs : lightkurve TargetPixelFileCollection
        Collection of Target Pixel files

    Returns
    -------
    times : numpy.ndarray
        Array with time values
    flux : numpy.ndarray
        Array with flux values per pixel
    flux_err : numpy.ndarray
        Array with flux errors per pixel
    unw : numpy.ndarray
        Array with TPF index for each pixel
    """

    time = tpfs[0].time.value

    if isinstance(tpfs[0], lk.KeplerTargetPixelFile):
        qual_mask = lk.utils.KeplerQualityFlags.create_quality_mask(
            tpfs[0].quality, 1 | 2 | 4 | 8 | 32 | 16384 | 65536 | 1048576
        )
        qual_mask &= (np.abs(tpfs[0].pos_corr1) < 5) & (np.abs(tpfs[0].pos_corr2) < 5)
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
    saturation_limit=1.5e5,
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
    not_saturated = ~_saturated_pixels_mask(
        flux, column, row, saturation_limit=saturation_limit
    )

    mask = not_nan & unique_pix & not_saturated

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
    tpfs : lightkurve TargetPixelFileCollection
        Collection of Target Pixel files

    Returns
    -------
    locs : numpy.ndarray
        2D array with pixel locations (columns, rows) from the TPFs
    ra : numpy.ndarray
        Array with right ascension values per pixel
    dec : numpy.ndarray
        Array with declination values per pixel
    """
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

    # convert pixel coord to ra, dec using TPF's solution
    ra, dec = (
        tpfs[0]
        .wcs.wcs_pix2world(
            np.vstack([(locs[0] - tpfs[0].column), (locs[1] - tpfs[0].row)]).T,
            0.0,
        )
        .T
    )
    return locs, ra, dec


def _get_coord_and_query_gaia(tpfs, magnitude_limit=18, dr=3):
    """
    Calculate ra, dec coordinates and search radius to query Gaia catalog

    Parameters
    ----------
    tpfs:
    magnitude_limit:
    dr : int
        Which gaia data release to use, default is DR2

    Returns
    -------
    sources : pandas.DataFrame
        Catalog with query result
    """
    if not isinstance(tpfs, lk.TargetPixelFileCollection):
        raise ValueError("Please pass a `lk.TargetPixelFileCollection`")

    # find the max circle per TPF that contain all pixel data to query Gaia
    ras1, decs1 = np.asarray(
        [tpf.wcs.all_pix2world([np.asarray(tpf.shape[1:]) + 2], 0)[0] for tpf in tpfs]
    ).T
    ras, decs = np.asarray(
        [tpf.wcs.all_pix2world([np.asarray(tpf.shape[1:]) // 2], 0)[0] for tpf in tpfs]
    ).T
    rads = np.hypot(ras, decs) - np.hypot(ras1, decs1)
    # query Gaia with epoch propagation
    sources = get_gaia_sources(
        tuple(ras),
        tuple(decs),
        tuple(rads),
        magnitude_limit=magnitude_limit,
        epoch=Time(tpfs[0].time[len(tpfs[0]) // 2], format="jd").jyear,
        dr=dr,
    )
    return sources


def _clean_source_list(sources, ra, dec):
    """
    Removes sources that are too contaminated and/or off the edge of the image

    Parameters
    ----------
    sources : Pandas Dataframe
        Contains a list with cross-referenced Gaia results
        shape n sources x n Gaia features
    ra      : numpy ndarray
        RA pixel position averaged in time
        shape npixel
    dec     : numpy ndarray
        Dec pixel position averaged in time
        shape npixel

    Returns
    -------
    sources : Pandas.DataFrame
        Catalog with clean sources
    removed_sources : Pandas.DataFrame
        Catalog with removed sources
    """
    # find sources on the image
    inside = np.zeros(len(sources), dtype=bool)
    # max distance in arcsec from image edge to source ra, dec
    # 1.25 pixels
    sep = 5 * u.arcsec.to(u.deg)
    for k in range(len(sources)):
        raok = (sources["ra"][k] > ra - sep) & (sources["ra"][k] < ra + sep)
        decok = (sources["dec"][k] > dec - sep) & (sources["dec"][k] < dec + sep)
        inside[k] = (raok & decok).any()
    del raok, decok

    # Keep track of sources that we removed
    sources.loc[:, "clean_flag"] = 0
    sources.loc[:, "clean_flag"][~inside] = 2 ** 0  # outside TPF
    # sources.loc[:, "clean_flag"][unresolved] += 2 ** 1  # close contaminant

    # combine 2 source masks
    clean = sources.clean_flag == 0
    removed_sources = sources[~clean].reset_index(drop=True)
    sources = sources[clean].reset_index(drop=True)

    return sources, removed_sources
