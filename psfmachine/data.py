import numpy as np
import pandas as pd
import fitsio
from tqdm.notebook import tqdm

from scipy import sparse

from astropy.coordinates import SkyCoord, Distance
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u

from .utils import get_sources


class lazy_loader(object):
    def __init__(self, fnames, extensions, name='data', time_offset=2454833):
        """ Class to load FFI data in a lazy way.
        Opening all the FFIs at once isn't necessary, so we
        use a class which lets us read each frame into memory
        by indexing.

        Parameters
        ----------
        fnames : list of str
            List of .fits file names
        extension : int
            The extension of the fits file to use
        name : str
            Name to use for repr
        """
        self.fnames = fnames
        self.name = name

        if len(extensions) == len(fnames):
            self.extensions = np.copy(extensions)
        else:
            raise ValueError('`extensions` must be the same length as `fnames`.')

        hdrs = [fitsio.read_header(self.fnames[idx], ext=self.extensions[idx]) for idx in range(len(self.fnames))]


        self.wcs = WCS(hdrs[0])
        self.shape = (len(self.fnames), hdrs[0]['NAXIS1'], hdrs[0]['NAXIS2'])
        self.time = Time([hdr['TSTART'] + time_offset for hdr in hdrs], format='jd')

        self.fnames = np.asarray(self.fnames)[np.argsort(self.time)]
        self.extensions = np.asarray(self.extensions)[np.argsort(self.time)]
        self.hdrs = [hdrs[s] for s in np.argsort(self.time)]
        self.time = self.time[[np.argsort(self.time)]]


    def __repr__(self):
        return f'{self.name} [{self.shape}]'

    def __getitem__(self, s):
        item = fitsio.read(self.fnames[s], ext=self.extensions[s])
        return item

    def __len__(self):
        return self.shape[0]


class Cube(object):
    """A frame of observations"""

    def __init__(self, cube, cube_err, wcs=None, time=None, magnitude_limit=19):

        self.data = cube
        self.error = cube_err

        if time is None:
            if isinstance(cube, lazy_loader):
                self.time = cube.time
            else:
                raise ValueError('Please pass a time array')
        else:
            if len(time) != len(cube):
                raise ValueError("Please pass one time per input frame.")
            self.time = time

        if wcs is None:
            if isinstance(cube, lazy_loader):
                self.wcs = cube.wcs
            else:
                raise ValueError('Please pass a WCS')
        else:
            self.wcs = wcs

        self.shape = cube.shape
        self.center = np.asarray([cube.shape[1]//2, cube.shape[2]//2])
        self.magnitude_limit = magnitude_limit
        self.sources, self.cs, self.locs = self.get_gaia_sources()
        self.Y, self.X = np.mgrid[:self.shape[1], :self.shape[2]]
        self.X1, self.Y1 = self.Y[::, 0], self.X[0]

    def __repr__(self):
        return "Cube [{}]".format(self.shape)

    @staticmethod
    def from_fnames(filenames=['kplr2009114174833_ffi-cal.fits'], error_filenames=['kplr2009114174833_ffi-uncert.fits'], skygroup=70, **kwargs):
        if not (len(filenames) == len(error_filenames)):
            raise ValueError('Length of input filenames and input error filenames are not the same')

        extensions = np.zeros(len(filenames), dtype=int)
        for idx, fname in enumerate(filenames):
            extensions[idx] = np.where(np.asarray([fitsio.read_header(fname, ext=idx)['SKYGROUP'] for idx in np.arange(1, 85)], int) == skygroup)[0][0] + 1

        data = lazy_loader(filenames, extensions=extensions, name='data')
        error = lazy_loader(error_filenames, extensions=extensions, name='error')
        if not np.in1d(error.time, data.time).all():
            raise ValueError('Not all times are identical between input filenames and input error filenames')
        return Cube(data, error, **kwargs)

    def get_gaia_sources(self):
        """ Get the source table from gaia"""

        # Find the center of the first frame:
        ra, dec = self.wcs.all_pix2world(self.center[:, None].T, 0).T
        ra, dec = ra[0], dec[0]
        r = ((np.max(self.shape[1:])/2)**2 * 2)**0.5
        sources = get_sources(ra, dec, r,
                                      epoch=self.time[0],
                                      magnitude_limit=self.magnitude_limit).reset_index(drop=True)

        # Use gaia space motion to correct for any drifts in time
        dist = Distance(parallax=np.asarray(sources['Plx'])*u.mas, allow_negative=True)
        coords = SkyCoord(ra=np.asarray(sources['RA_ICRS']) * u.deg, dec=np.asarray(sources['DE_ICRS']) * u.deg,
                     pm_ra_cosdec=np.nan_to_num(sources['pmRA']) * u.mas/u.year, pm_dec=np.nan_to_num(sources['pmDE']) * u.mas/u.year,
                     distance=dist,
                     obstime='J2015.05',
                     radial_velocity=np.nan_to_num(sources['RV'])*u.km/u.s
                     )
        cs = coords.apply_space_motion(self.time[0])
        locs = self.wcs.wcs_world2pix(np.atleast_2d((cs.ra.deg, cs.dec.deg)).T, 0)

        # Trim out any sources that are outside the image
        lmask = (locs[:, 0] >= -15) & (locs[:, 0] <= self.shape[1] + 15) & (locs[:, 1] >= -15) & (locs[:, 1] <= self.shape[2] + 15)
        sources, cs, locs = sources[lmask].reset_index(drop=True), cs[lmask], locs[lmask]
        return sources, cs, locs


    def get_masks(self, radius=4):
        dx1, dy1 = (self.X1[:, None] - self.locs[:, 0]), (self.Y1[:, None] - self.locs[:, 1])
        dx1s_m = sparse.csc_matrix((np.abs(dx1) < radius).T)
        dy1s_m = sparse.csc_matrix((np.abs(dy1) < radius))
        dx1s = dx1s_m.multiply(dx1.T).tocsc()
        dy1s = dy1s_m.multiply(dy1).tocsc()

        masks, gaia_flux, dx, dy = [], [], [], []
        for idx in tqdm(range(len(self.locs))):
            xm = dx1s_m[idx]
            ym = dy1s_m[:, idx]
            mask = (xm.multiply(ym)).reshape(np.product(self.shape[1:]))
            masks.append(mask)
            gaia_flux.append(mask.multiply(self.sources.loc[idx, 'FG']))
            dx.append(dx1s[idx].multiply(ym).reshape(np.product(self.shape[1:])))
            dy.append(dy1s[:, idx].multiply(xm).reshape(np.product(self.shape[1:])))
        self.masks, self.gaia_flux, self.dx, self.dy = sparse.vstack(masks, 'csc'), sparse.vstack(gaia_flux, 'csc'), sparse.vstack(dx, 'csc'), sparse.vstack(dy, 'csc')
        self.dx_v = self.dx.data
        self.dy_v = self.dy.data
        self.gaia_flux_v = self.gaia_flux.data
