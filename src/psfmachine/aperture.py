"""
Collection of aperture utils lifted from
[Kepler-Apertures](https://github.com/jorgemarpa/kepler-apertures) and adapted to work
with PSFMachine.

Some this functions inputs and operate on a `Machine` object but we move them out of
`mahine.py` to keep the latter smowhow clean and short.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm


def optimize_aperture(
    psf_model,
    target_complete=0.9,
    target_crowd=0.9,
    max_iter=100,
    percentile_bounds=[0, 100],
    quiet=False,
):
    """
    Function to optimize the aperture mask for a given source.

    The optimization is done using scipy Brent's algorithm and it uses a custom
    loss function `goodness_metric_obj_fun` that uses a Leaky ReLU term to
    achive the target value for both metrics.

    Parameters
    ----------
    psf_model : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    target_complete : float
        Value of the target completeness metric.
    target_crowd : float
        Value of the target crowdeness metric.
    max_iter : int
        Numer of maximum iterations to be performed by the optimizer.
    percentile_bounds : tuple
        Tuple of minimun and maximun values for allowed percentile values during
        the optimization. Default is the widest range of [0, 100].

    Returns
    -------
    optimal_percentile : numpy.ndarray
        An array with the percentile value to defines the "optimal" aperture for
        each source.
    """
    # optimize percentile cut for every source
    optimal_percentile = []
    for sdx in tqdm(
        range(psf_model.shape[0]),
        desc="Optimizing apertures per source",
        disable=quiet,
    ):
        optim_params = {
            "percentile_bounds": percentile_bounds,
            "target_complete": target_complete,
            "target_crowd": target_crowd,
            "max_iter": max_iter,
            "psf_models": psf_model,
            "sdx": sdx,
        }
        minimize_result = optimize.minimize_scalar(
            goodness_metric_obj_fun,
            method="Bounded",
            bounds=percentile_bounds,
            options={"maxiter": max_iter, "disp": False},
            args=(optim_params),
        )
        optimal_percentile.append(minimize_result.x)
    return np.array(optimal_percentile)


def goodness_metric_obj_fun(percentile, optim_params):
    """
    The objective function to minimize with scipy.optimize.minimize_scalar called
    during optimization of the photometric aperture.

    Parameters
    ----------
    percentile : int
        Percentile of the normalized flux distribution that defines the isophote.
    optim_params : dictionary
        Dictionary with the variables needed to evaluate the metric:
            * psf_models
            * sdx
            * target_complete
            * target_crowd

    Returns
    -------
    penalty : float
        Value of the objective function to be used for optiization.
    """
    psf_models = optim_params["psf_models"]
    sdx = optim_params["sdx"]
    # Find the value where to cut
    cut = np.nanpercentile(psf_models[sdx].data, percentile)
    # create "isophot" mask with current cut
    mask = (psf_models[sdx] > cut).toarray()[0]

    # Do not compute and ignore if target score < 0
    if optim_params["target_complete"] > 0:
        # compute_FLFRCSAP returns an array of size 1 when doing only one source
        completMetric = compute_FLFRCSAP(psf_models[sdx], mask)[0]
    else:
        completMetric = 1.0

    # Do not compute and ignore if target score < 0
    if optim_params["target_crowd"] > 0:
        crowdMetric = compute_CROWDSAP(psf_models, mask, idx=sdx)
    else:
        crowdMetric = 1.0

    # Once we hit the target we want to ease-back on increasing the metric
    # However, we don't want to ease-back to zero pressure, that will
    # unconstrain the penalty term and cause the optmizer to run wild.
    # So, use a "Leaky ReLU"
    # metric' = threshold + (metric - threshold) * leakFactor
    leakFactor = 0.01
    if (
        optim_params["target_complete"] > 0
        and completMetric >= optim_params["target_complete"]
    ):
        completMetric = optim_params["target_complete"] + leakFactor * (
            completMetric - optim_params["target_complete"]
        )

    if optim_params["target_crowd"] > 0 and crowdMetric >= optim_params["target_crowd"]:
        crowdMetric = optim_params["target_crowd"] + leakFactor * (
            crowdMetric - optim_params["target_crowd"]
        )

    penalty = -(completMetric + crowdMetric)

    return penalty


def plot_flux_metric_diagnose(psf_model, idx=0, ax=None, optimal_percentile=None):
    """
    Function to evaluate the flux metrics for a single source as a function of
    the parameter that controls the aperture size.
    The flux metrics are computed by taking into account the PSF models of
    neighbor sources.

    This function is meant to be used only to generate diagnostic figures.

    Parameters
    ----------
    psf_model : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    idx : int
        Index of the source for which the metrcs will be computed. Has to be a
        number between 0 and psf_models.shape[0].
    ax : matplotlib.axes
        Axis to be used to plot the figure

    Returns
    -------
    ax : matplotlib.axes
        Figure axes
    """
    compl, crowd, cut = [], [], []
    for p in range(0, 101, 1):
        cut.append(p)
        mask = (psf_model[idx] >= np.nanpercentile(psf_model[idx].data, p)).toarray()[0]
        crowd.append(compute_CROWDSAP(psf_model, mask, idx))
        compl.append(compute_FLFRCSAP(psf_model[idx], mask))

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.plot(cut, compl, label=r"FLFRCSAP", c="tab:blue")
    ax.plot(cut, crowd, label=r"CROWDSAP", c="tab:green")
    if optimal_percentile:
        ax.axvline(optimal_percentile, c="tab:red", label="optimal")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Metric")
    ax.legend()
    return ax


def estimate_source_centroids_aperture(aperture_mask, flux, column, row):
    """
    Computes the centroid via 2D moments methods for all sources all times. It needs
    `aperture_mask` to be computed first by runing `compute_aperture_photometry`.

    Parameters
    ----------
    aperture_mask : numpy.ndarray
        Aperture mask, shape is [n_surces, n_pixels]
    flux: numpy.ndarray
        Flux values at each pixels and times in units of electrons / sec
    column : numpy.ndarray
        Data array containing the "columns" of the detector that each pixel is on.
    row : numpy.ndarray
        Data array containing the "rows" of the detector that each pixel is on.
    Returns
    -------
    centroid_col : numpy.ndarray
        Column pixel number of the moments centroid, shape is [nsources, ntimes].
    centroid_row : numpy.ndarray
        Row pixel number of the moments centroid, shape is [nsources, ntimes].
    """
    centroid_col, centroid_row = [], []
    for idx in range(aperture_mask.shape[0]):
        total_flux = np.nansum(flux[:, aperture_mask[idx]], axis=1)
        centroid_col.append(
            np.nansum(
                np.tile(column[aperture_mask[idx]], (flux.shape[0], 1))
                * flux[:, aperture_mask[idx]],
                axis=1,
            )
            / total_flux
        )
        centroid_row.append(
            np.nansum(
                np.tile(row[aperture_mask[idx]], (flux.shape[0], 1))
                * flux[:, aperture_mask[idx]],
                axis=1,
            )
            / total_flux
        )
    return np.array(centroid_col), np.array(centroid_row)


def compute_FLFRCSAP(psf_models, aperture_mask):
    """
    Compute fraction of target flux enclosed in the optimal aperture to total flux
    for a given source (flux completeness).
    Follows definition by Kinemuchi at al. 2012.
    Parameters
    ----------
    psf_models : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    aperture_mask: numpy.ndarray
        Array of boolean indicating the aperture for the target source. It has shape of
        [n_sources, n_pixels].

    Returns
    -------
    FLFRCSAP: numpy.ndarray
        Completeness metric
    """
    return np.array(
        psf_models.multiply(aperture_mask.astype(float)).sum(axis=1)
        / psf_models.sum(axis=1)
    ).ravel()


def compute_CROWDSAP(psf_models, aperture_mask, idx=None):
    """
    Compute the ratio of target flux relative to flux from all sources within
    the photometric aperture (i.e. 1 - Crowdeness).
    Follows definition by Kinemuchi at al. 2012.
    Parameters
    ----------
    psf_models : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    aperture_mask : numpy.ndarray
        Array of boolean indicating the aperture for the target source. It has shape of
        [n_sources, n_pixels].
    idx : int
        Source index for what the metric is computed. Value has to be betweeen 0 and
        psf_model first dimension size.
        If None, it returns the metric for all sources (first dimension of psf_model).

    Returns
    -------
    CROWDSAP : numpy.ndarray
        Crowdeness metric
    """
    ratio = psf_models.multiply(1 / psf_models.sum(axis=0)).tocsr()
    if idx is None:
        return np.array(
            ratio.multiply(aperture_mask.astype(float)).sum(axis=1)
        ).ravel() / aperture_mask.sum(axis=1)
    else:
        return ratio[idx].toarray()[0][aperture_mask].sum() / aperture_mask.sum()


def aperture_mask_to_2d(tpfs, sources, aperture_mask, column, row):
    """
    Convert 1D aperture mask into 2D to match the shape of TPFs. This 2D aperture
    masks are useful to plot them with lightkurve TPF plot.
    Because a sources can be in more than one TPF, having 2D array masks per object
    with the shape of a single TPF is not possible.

    Parameters
    ----------
    tpfs: lightkurve TargetPixelFileCollection
        Collection of Target Pixel files
    tpfs_meta : list
        List of source indices for every TPF in `tpfs`.
    aperture_mask : numpy.ndarray
        Aperture mask, shape is [n_surces, n_pixels]
    column : numpy.ndarray
        Data array containing the "columns" of the detector that each pixel is on.
    row : numpy.ndarray
        Data array containing the "rows" of the detector that each pixel is on.
    Returns
    -------
    aperture_mask_2d : dictionary
        Is a dictionary with key values as 'TPFindex_SOURCEindex', e.g. a source
        (idx=10) with multiple TPF (TPF index 1 and 2) data will look '1_10' and '2_10'.
    """
    aperture_mask_2d = {}
    for k, tpf in enumerate(tpfs):
        # find sources in tpf
        sources_in = sources[k]
        # row_col pix value of TPF
        rc = [
            "%i_%i" % (y, x)
            for y in np.arange(tpf.row, tpf.row + tpf.shape[1])
            for x in np.arange(tpf.column, tpf.column + tpf.shape[2])
        ]
        # iter sources in the TPF
        for sdx in sources_in:
            # row_col value of pixels inside aperture
            rc_in = [
                "%i_%i"
                % (
                    row[aperture_mask[sdx]][i],
                    column[aperture_mask[sdx]][i],
                )
                for i in range(aperture_mask[sdx].sum())
            ]
            # create initial mask
            mask = np.zeros(tpf.shape[1:], dtype=bool).ravel()
            # populate mask with True when pixel is inside aperture
            mask[np.in1d(rc, rc_in)] = True
            mask = mask.reshape(tpf.shape[1:])
            aperture_mask_2d["%i_%i" % (k, sdx)] = mask

    return aperture_mask_2d
