# EmulateLSS
Train MLP-based surrogate models for LSS power spectra.

## Dependencies
In order to generate training data, you need `CobayaLSS` (https://github.com/martinjameswhite/CobayaLSS). In particular, you need to use the `provider` branch.

All other requirements are contained in `requirements.txt`

## Generating training data

Any `Cobaya` config file can be converted to generate training data. This can be done by adding an `emulate` section at the end of the config (see, e.g. `configs/lrg_x_planck_aemulus_20xfast_rs_spectra_1e6pts_training_data.yaml`).
The priors in the config specify the parameter space range over which training data will be generated.
The keywords under `emulate` in your config file control how much training data to generate, and where to write the file. In particular, `output_filename` specifies where the training data is written. `nend` is the total number of training points (fast and slow) to generate. You can specify fast parameters in `param_names_fast` and the factor by which you want to oversample fast parameters in `nfast_per_slow`.

Run `generate_training_data.sh`, replacing the config file with your modified version. Note that training data can (and probably should) be generated in parallel via MPI (see an example in `generate_training_data.sh`).

## Training a surrogate model

See `train_emu.sh` for an example of how to train the emulators. You'll need to modify `configs/train_p0_scan_4_128_1.yaml` to point to your new training data file (the `training_filename` keyword). 
This will produce one JSON file per surrogate model. These can be read in with the emulator class in `emulator.py` in order to make predictions using the emulator. The order that parameters must be provided to the `Emulator` class is the same as the order that the Cobaya model used for training assumed.

See `notebooks/train_surrogates.ipynb` for a toy example of this.

## Pre-trained models

We make available a number of pre-trained models in `nn_weights`:

Halofit P_mm : `nn_weights/lrg_x_planck_cleft_priors_buzzard_shape_halofit_pmm_20xfast_rs_spectra_1e6pts_training_data_v1_pmm_emu.json`

CLEFT P_gm : `nn_weights/lrg_x_planck_cleft_priors_buzzard_shape_20xfast_rs_spectra_1e6pts_training_data_v1_pgm_emu.json`

CLEFT P_gg : `nn_weights/lrg_x_planck_cleft_priors_buzzard_shape_20xfast_rs_spectra_1e6pts_training_data_v1_pgg_emu.json`

HEFT (anzu) P_mm : `nn_weights/lrg_x_planck_aemulus_priors_20xfast_rs_spectra_1e6pts_training_data_v1_pmm_emu.json`

HEFT (anzu) P_gm : `nn_weights/lrg_x_planck_aemulus_priors_20xfast_rs_spectra_1e6pts_training_data_v1_pgm_emu.json`

HEFT (anzu) P_gg : `nn_weights/lrg_x_planck_aemulus_priors_20xfast_rs_spectra_1e6pts_training_data_v1_pgg_emu.json`

Lagrangian EFT P_0 : `nn_weights/ptchallenge_cmass2_20xfast_1e6pts_training_data_v2_p0_emu.json`

Lagrangian EFT P_2 : `nn_weights/ptchallenge_cmass2_20xfast_1e6pts_training_data_v2_p2_emu.json`

Lagrangian EFT P_4 : `nn_weights/ptchallenge_cmass2_20xfast_1e6pts_training_data_v2_p4_emu.json`


See `notebooks/pretrained_models.ipynb` for details on how to load and call these models.
