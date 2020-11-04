

# help functions


class Machine(object):

    def __init__(self, time, flux, flux_err, x, y, ra, dec, radius_limit=20*u.arcsecond):
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
        radius_limit:
            helpful docstring


        Attributes
        ----------

        sources : pd.DataFrame

        source_flux_estiamtes : np.ndarray
            nsources
        """

        # Find sources using Gaia

        #pandas dataframe
        self.sources = # utils.get_sources(...)

        # We need a way to pair down the source list to only sources that are 1) separated 2) on the image
        self._clean_source_list()

        self.source_flux_estimates = # Get from gaia values

        # Get the centroids of the images as a function of time
        self._get_centroids()

        self.dra = # The separation in ra from each source at each pixel shape nsources x npixels
        self.ddec = # The separation in dec from each source at each pixel shape nsources x npixels

        # polar coordinates
        self.r = np.hypot(self.dra, self.ddec) # radial distance from source in arcseconds
        self.phi = np.arctan2(self.ddec, self.dra) # azimuthal angle from source in radians


        self.source_mask = self._get_source_mask() # Mask of shape nsources x number of pixels, one where flux from a source exists
        self.uncontaminated_pixel_mask = self._get_uncontaminated_pixel_mask() # Mask of shape npixels (maybe by nt) where not saturated, not faint, not contaminated etc

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]


    @property
    def shape(self):
        return (self.nsources, self.nt,  self.npixels)

    def __repr__(self):
        return f'Machine {self.shape}'

    def _clean_source_list(self):
        """ Removes sources that are too contaminated or off the edge of the image"""
        clean = # Some source mask

        # Keep track of sources that we removed
        # Flag each source for why they were removed?
        self._removed_sources = self.sources[~clean]

        self.sources = self.sources[clean]

    def _get_source_mask(self):
        """
        Mask of shape nsources x number of pixels, one where flux from a source exists
        """
        return source_mask

    def _get_uncontaminated_pixel_mask(self):
        """
        Mask of shape npixels (maybe by nt) where not saturated, not faint, not contaminated etc
        """
        return uncontaminated_pixel_mask

    def _get_centroids(self):
        """ Find the ra and dec centroid of the image, at each time"""
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
        """ Convert TPF input into object"""

        # Checks that all TPFs have identical time sampling

        # Remove nan pixels

        # Remove bad cadences where the pointing is rubbish

        return Machine(time=, flux=, flux_err=, x=, y=, ra=, dec=)

mac = Machine().from_tpfs()
mac.build_model()
mac.fit_model()

mac.model_flux # Values
mac.model_flux_err # values
