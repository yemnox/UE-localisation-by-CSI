# data/

This folder holds the inputs. Its contents are not tracked by git (see `.gitignore`).

```
data/
├── o1_3p5/        DeepMIMO v4 scenario (~21 GB, 4,161 .mat files)
└── csi_o1.mat     generated CSI dataset (~218 MB)
```

## 1. Get the DeepMIMO scenario `o1_3p5`

1. Download the **O1 – 3.5 GHz** scenario (`o1_3p5`, DeepMIMO v4 format) from the
   DeepMIMO website: <https://www.deepmimo.net> (Scenarios → O1).
2. Extract it so that the matrix files sit directly in `data/o1_3p5/`, e.g.
   `data/o1_3p5/power_t006_tx000_r000.mat`.

File naming: `<quantity>_tNNN_tx000_rNNN.mat`, where `tNNN` is the transmitter
(base station) set and `rNNN` the user grid. `matlab/dm_load.m` reads one such file.

Files actually used:

| Script | tx sets | grids | quantities |
|---|---|---|---|
| `generate_csi.m` | t006 | r000 | `rx_pos`, `tx_pos`, `power`, `phase`, `delay`, `aod_az`, `aod_el`, `inter` |
| `plot_rays.m` | t006 | r000 | same + `inter_pos` (~400 MB, slow to load) |
| `explore_o1.m` | t003–t020 | r000–r002 | `tx_pos`, `rx_pos`, and path data for t003 |

To save disk space, you can keep only those files.

## 2. Generate `csi_o1.mat`

In MATLAB, run `matlab/generate_csi.m`. You can run it from any folder because
paths are resolved from the repository root. The configuration block at the top
of the script sets the tx set, user grid, crop box, number of users, array size,
number of subcarriers, bandwidth and number of paths.

Contents of `csi_o1.mat`:

| Variable | Size | Description |
|---|---|---|
| `H` | 30000 × 32 × 32 complex single | CSI, users × antennas × subcarriers |
| `pos` | 30000 × 3 | user position (x, y, z) in m |
| `los` | 30000 × 1 | LoS flag (all 1 for this crop) |
| `bs_pos` | 1 × 3 | gNB position (t006) |
| `cfg` | struct | scenario, tx set, grid, n_ant, n_sub, bw, n_paths, fc |

## Using Colab

Upload `csi_o1.mat` to Google Drive and set `DATA_PATH` in the notebook's first
code cell (the Colab branch).
