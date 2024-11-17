# Spiking FT

This library provides code for implementing the Spiking Fourier transform (S-FT) for computing the frequency spectrum of radar signals using a spiking neural network. The neuron model and network architecture was introduced in the following paper:

> López-Randulfe, Javier, et al. "Time-coded spiking fourier transform in neuromorphic hardware." IEEE Transactions on Computers 71.11 (2022): 2792-2802.

Currently the library provides implementations compatible with Loihi2 and SpiNNaker2 boards, as well as a NumPy version that works on a normal PC.

# Set-up

To prepare the library, download it to a local folder and install it using pip

```bash
pip install -e .
```

You also need to download and install the *pyRadarSP* library, please follow the instructions [here](https://github.com/KI-ASIC-TUM/pyRadarSP/).

You can now try and run one of the scripts in the examples folder.

```bash
cd examples
python 01c-single_chirp_numpy.py
```

## Important note

Be aware that for running the examples for loihi2 and SpiNNaker2, you will need connection to the board and Lava or py-spinnaker2 libraries, respectively. These are third party libraries and hardware, we are not in charge of maintaining nor providing access for these resources.

## Troubleshooting

This library is currently under development. It is likely
that you will encounter bugs or functionalities not implemented yet.
If you have any feedback on the library or have a specific feature in mind
that you think it should be included, do not hesitate to contact us:

* lopez.randulfe@tum.de
* nico.reeb@tum.de
