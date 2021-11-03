1.1.0 (2021-10-26)
==================

New Features
------------

psfmachine.aperture
^^^^^^^^^^^^^^^^^^^^^^^^

- Collection of functions that perform aperture photometry. It uses the PSF model to
define the aperture mask and computes photometry.

- The aperture masks are can be created by optimizing two flux metrics, completeness
 and crowdeness.

- Functions that defines the flux metrics, completeness and crowdeness, and compute
object centroids using momentum.

- Many of these functions inputs machine and create new attribute to it.

- Diagnostic functions that plot the flux metrics for a given object.

psfmachine.Machine
^^^^^^^^^^^^^^^^^^

- A new method (`compute_aperture_photometry`) that computes aperture photometry using
the new functionalities in `psfmachine.aperture_utils`.


psfmachine.TPFMachine
^^^^^^^^^^^^^^^^^^^^^

- A new method that computes the centroid of each sources.
