# Network Coherence

Application of the Network Coherence technique to seismic/acoustic networks/arrays to detect weak but coherent seismoacoustic signals. Also included is code to locate non-impulsive acoustic sources using inter-element phase. When using this code we ask you to cite the following paper, which also provides details on the method and some examples:

Scamfer, L. T., Fee, D., Tan, D. (2026). *Analyzing Seismic and Acoustic Signals with Network-based Coherence and Phase*. *The Seismic Record*

## Installation

*Below are install instructions for an example conda environment.*

We recommend using conda (or mamba) and creating a new conda environment using the provided `environment.yml` file:

```bash
git clone https://github.com/uafgeotools/network_coherence
cd network_coherence
conda env create -f environment.yml
conda activate network_coherence_env
```

Information on conda environments (and more) is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Dependencies

Python packages:

* [Python3](https://docs.python.org/3/)
* [ObsPy](http://docs.obspy.org/)
* [RTM](https://uaf-rtm.readthedocs.io/en/master/)
* [waveform_collection](https://uaf-waveform-collection.readthedocs.io/en/master/)
* [Numba](http://numba.pydata.org)
* [multiprocess](https://multiprocess.readthedocs.io/en/latest/)
* [colorcet](https://colorcet.holoviz.org/index.html)
* [pyproj](https://pyproj4.github.io/pyproj/stable/)
* [tqdm](https://tqdm.github.io/)
* [cartopy](https://cartopy.readthedocs.io/stable/)
* [rioxarray](https://corteva.github.io/rioxarray/html/index.html)

## Usage

See the included examples:

* `run_network_coherence.py`
* `run_acoustic_phase_location.py`

## Authors

*(Alphabetical order by last name.)*

David Fee

Logan T. Scamfer

## Acknowledgements and Distribution Statement

This work was supported by the Defense Threat Reduction Agency Nuclear Arms Control Technology program under Contract Number HDTRA121C0030 (Distribution statement: Cleared for release).
