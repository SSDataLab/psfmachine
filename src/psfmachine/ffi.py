"""Classes for working with FFI data"""
from .utils import get_gaia_sources
from .machine import Machine


def FFIMachine(Machine):
    """Subclass of Machine for working with FFI data"""

    # Probably don't need a very new init function over Machine.
    def __init__(self, channel=channel, quarter=quarter, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.quarter = quarter

    def __repr__(self):
        return f"FFIMachine (N sources, N times, N pixels): {self.shape}"

    @staticmethod
    def from_file(self, fname):
        """Reads data from files and initiates a new class...

        Parameters
        ----------
        fname : str
            Filename"""
        sources = self._get_sources()
        time, flux, flux_err, ra, dec, column, row = self._load_file()
        raise NotImplementedError
        return FFIMachine(
            time=times,
            flux=flux,
            flux_err=flux_err,
            ra=ra,
            dec=dec,
            sources=sources,
            column=column,
            row=row,
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

    def _background_removal_functions(self):
        """kepler-apertures probably used some background removal functions, e.g. mean filter, fine to subclass astropy here"""
        raise NotImplementedError

    def _masking_functions(self):
        """kepler-apertures probably needed some bespoke masking functions?"""
        raise NotImplementedError

    def _get_sources(self):
        """Gets sources in a tiled manner, update and/or use existing query functions?"""
        raise NotImplementedError

    def _load_file(self):
        """Helper function to load file?"""

        # Have to do some checks here that it's the right kind of data.
        #  We could loosen these checks in future.
        if hdu[0].header["MISSION"] is "KEPLER":
            pass
        if hdu[0].header["MISSION"] is "K2":
            pass
        if hdu[0].header["MISSION"] is "TESS":
            pass

        raise NotImplementedError


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
