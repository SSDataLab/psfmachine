# PSFMachine

*PRF photometry with Kepler*

<a href="https://github.com/ssdatalab/psfmachine/workflows/tests.yml">
      <img src="https://github.com/ssdatalab/psfmachine/workflows/pytest/badge.svg" alt="Test status"/>
</a> <a href="https://pypi.python.org/pypi/tess-ephem"><img src="https://img.shields.io/pypi/v/tess-ephem", alt="pypi status"></a>

Check out the [documentation](https://ssdatalab.github.io/psfmachine/).
Check out the [paper](#)

`PSFMachine` is an open source Python tool for creating models of instrument effective Point Spread Functions (ePSFs), a.k.a Pixel Response Functions (PRFs). These models are then used to fit a scene in a stack of astronomical images. `PSFMachine` is able to quickly derive photometry from stacks of *Kepler* images and separate crowded sources.

# Installation

```
pip install psfmachine
```

# Example use

Below is an example script that shows how to use `PSFMachine`. Depending on the speed or your computer fitting this sort of model will probably take ~10 minutes to build 200 light curves. You can speed this up by changing some of the input parameters.

```python
import psfmachine as psf
import lightkurve as lk
tpfs = lk.search_targetpixelfile('Kepler-16', mission='Kepler', quarter=12, radius=1000, limit=200, cadence='long').download_all(quality_bitmask=None)
machine = psf.TPFMachine.from_TPFs(tpfs, n_r_knots=10, n_phi_knots=12)
machine.fit_lightcurves()
```

Funding for this project is provided by NASA ROSES grant number 80NSSC20K0874.
