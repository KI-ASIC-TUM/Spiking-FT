#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import pyrads.algms.fft
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
try:
    import sft.s_dft_spinnaker
except ModuleNotFoundError:
    pass


experiments = [
    "freq10Hz_narrow",
    "freq25Hz_narrow",
    "freq50Hz_narrow",
    "freq75Hz_narrow",
    "freq100Hz_narrow",
    "freq250Hz_narrow",
    "freq500Hz_narrow",
    "freq750Hz_narrow",
    "freq1000Hz_narrow",
    "freq10Hz_wide",
    "freq25Hz_wide",
    "freq50Hz_wide",
    "freq75Hz_wide",
    "freq100Hz_wide",
    "freq250Hz_wide",
    "freq500Hz_wide",
    "freq750Hz_wide",
    "freq1000Hz_wide",
]


def run_sft(raw_data, sft_params, fft_shape, iter, timesteps=100):
    
    encoder = sft.encoder.Identity(fft_shape)
    s_dft = sft.s_dft_spinnaker.SDFT(encoder.out_data_shape, **sft_params)

    spinn_pipeline = pyrads.pipeline.Pipeline([encoder, s_dft])
    spinn_pipeline(raw_data)
    sft_out = spinn_pipeline.output[-1]
    sft_out = 0.75*timesteps - sft_out
    sft_out = np.sqrt(sft_out[..., 0]**2 + sft_out[..., 1]**2)
    np.save("spinn_{}.npy".format(experiments[iter]), sft_out)
    return sft_out


def run_fft(raw_data, fft_params, fft_shape, iter):
    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([fft_alg])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-1]
    np.save("std_{}.npy".format(experiments[iter]), fft_out)
    return fft_out


def main(timesteps=100):
    fft_params = {
        "n_dims": 1,
        "type": "range",
        "out_format": "modulus",
        "normalize": True,
        "off_bins": 1,
    }

    sft_params = {
        "n_dims": 1,
        "timesteps": timesteps,
    }
    fft_shape = raw_data.shape

    sft_out = []
    fft_out = []
    for i in range(18):
        with open("data/20230718/{}.txt".format(experiments[i]), "r") as f:
            ase_data = f.read()
        ase_list = [int(x) for x in ase_data.split(",")]
        raw_data = np.array(ase_list)
        sft_out.append(run_sft(raw_data, sft_params, fft_shape, i))
        fft_out.append(run_fft(raw_data, fft_params, fft_shape, i))

        fig, axs = plt.subplots(2, figsize=(10,6))
        axs[0].plot(fft_out[-1])
        axs[1].plot(sft_out[-1][1:])
        fig.savefig("{}.eps".format(experiments[i]))
    return


if __name__ == "__main__":
    main()
