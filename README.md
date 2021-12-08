# EmulateLSS
Train MLP-based surrogate models for LSS power spectra.

## Dependencies
In order to generate training data, you need `CobayaLSS` (https://github.com/martinjameswhite/CobayaLSS). In particular, you need to use the `provider` branch.

All other requirements are contained in `requirements.txt`

## Generating training data

Any `Cobaya` config file can be converted to generate training data. This can be done by adding an `emulate` section at the end of the config (see, e.g. `configs/lrg_x_planck_aemulus_20xfast_rs_spectra_1e6pts_training_data.yaml`).
The priors in the config specify the parameter space range over which training data will be generated.
The keywords under `emulate` in your config file control how much training data to generate, and where to write the file. In particular, output_filename specifies where the training data is written. `nend` is the total number of training points (fast and slow) to generate. You can specify fast parameters in param_names_fast and the factor by which you want to oversample fast parameters in nfast_per_slow.

Run `generate_training_data.sh`, replacing the config file with your modified version. Note that training data can (and probably should) be generated in parallel via MPI (see an example in `generate_training_data.sh`).

## Training a surrogate model

See `train_emu.sh` for an example of how to train the emulators. You'll need to modify configs/train_p{0,2,4}_scan_4_128_1.yaml to point to your new training data file (the training_filename keyword). 
This will produce one JSON file per surrogate model. These can be read in with the emulator class in emulator.py in order to make predictions using the emulator. The order that parameters must be provided to the Emulator class is the same as the order that the Cobaya model used for training assumed.
