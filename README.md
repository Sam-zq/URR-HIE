# URR-HIE

Official implementation accompanying the manuscript **"Range-aware heterogeneous representations for joint meteorological and photovoltaic power forecasting."**

URR-HIE extends a channel-wise Transformer with two representation operations:

- **Upper-range reparameterization (URR):** expands the instance-specific normalization upper bound to provide additional headroom for future upward deviations.
- **Heterogeneous inverted embedding (HIE):** assigns an independent temporal projection to each input variable before channel-wise self-attention.

The attention backbone is unchanged. The implementation is based on the [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library).

## Repository contents

```text
URR-HIE/
|-- dataset/cite30MW/             # Processed data for three PV stations
|-- models/URR-HIE.txt            # URR-HIE model definition
|-- exp/exp_long_term_forecasting.py
|                                  # Modified experiment class; saves test arrays
|-- scripts/                       # Experiment configurations for URR-HIE and baselines
|-- utils/pv_metrics.py            # PV-target evaluation utilities
|-- requirements                   # Python dependencies used by this extension
`-- README.md
```

This repository is an **extension package for TSLib**, rather than a standalone copy of the complete TSLib framework. The official TSLib repository supplies `run.py`, `data_provider/`, and the common experiment infrastructure.

## Data

The processed station files are included in `dataset/cite30MW/`:

| Station | File | Nominal capacity | Sampling interval |
|---|---|---:|---:|
| Station 1 | `solar_station_1.csv` | 50 MW | 15 min |
| Station 2 | `solar_station_2.csv` | 130 MW | 15 min |
| Station 3 | `solar_station_3.csv` | 30 MW | 15 min |

Each CSV file contains a `date` column followed by five meteorological variables and the PV-power target `OT`. The target is the last channel.

The data originate from:

> Y. Chen and J. Xu, "Solar and wind power data from the Chinese State Grid Renewable Energy Generation Forecasting Competition," *Scientific Data*, 9, 577 (2022). https://doi.org/10.1038/s41597-022-01696-6

Please cite the original dataset when using these files.

## Experimental protocol

The paper uses the following common protocol:

- chronological train/validation/test split: 70%/10%/20%;
- Z-score standardization fitted on the training partition only;
- input length: 96 steps (24 h);
- label length: 48 steps;
- prediction lengths: 96, 192, 336, and 720 steps (24, 48, 84, and 180 h);
- multivariate-to-multivariate forecasting (`features=M`);
- URR expansion factor: `alpha = 0.075`;
- maximum training epochs: 10;
- early-stopping patience: 3;
- optimizer: AdamW;
- loss: mean squared error;
- three independent runs for the main benchmark.

The matched URR-HIE and iTransformer settings are:

| Parameter | Value |
|---|---:|
| `seq_len` | 96 |
| `label_len` | 48 |
| `d_model` | 256 |
| `n_heads` | 4 |
| `e_layers` | 2 |
| `d_ff` | 512 |
| dropout | 0.1 |
| learning rate | 5e-4 |
| batch size, Stations 1 and 2 | 32 |
| batch size, Station 3 | 16 |

Baseline-specific settings are provided in `scripts/`.

## Environment

The experiments reported in the manuscript were run with Python 3.9 on an NVIDIA GeForce RTX 2080 GPU with 8 GB of memory. Install a PyTorch build compatible with your CUDA version before installing the remaining dependencies.

```bash
conda create -n urr-hie python=3.9 -y
conda activate urr-hie

# Install PyTorch using the command recommended for your CUDA version:
# https://pytorch.org/get-started/locally/

pip install -r requirements
```

For exact reproduction, record the PyTorch, CUDA, cuDNN, and TSLib commit versions used in your environment.

## Integration with TSLib

Clone the official framework and this repository:

```bash
git clone https://github.com/thuml/Time-Series-Library.git
git clone https://github.com/Sam-zq/URR-HIE.git

export TSLIB_ROOT=/path/to/Time-Series-Library
export URRHIE_ROOT=/path/to/URR-HIE
```

Copy the model into TSLib using a valid Python module name:

```bash
cp "$URRHIE_ROOT/models/URR-HIE.txt" "$TSLIB_ROOT/models/URR_HIE.py"
cp "$URRHIE_ROOT/exp/exp_long_term_forecasting.py" \
   "$TSLIB_ROOT/exp/exp_long_term_forecasting.py"
cp "$URRHIE_ROOT/utils/pv_metrics.py" "$TSLIB_ROOT/utils/pv_metrics.py"
cp -r "$URRHIE_ROOT/dataset/cite30MW" "$TSLIB_ROOT/dataset/"
```

Register the model in the TSLib model dictionary used by `exp/exp_basic.py`:

```python
from models import URR_HIE

# Add to model_dict:
'URR_HIE': URR_HIE,
```

## Training URR-HIE

The following example trains Station 3 for the 96-step horizon. Run it from the TSLib root directory.

```bash
cd "$TSLIB_ROOT"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/cite30MW/ \
  --data_path solar_station_3.csv \
  --model_id solar3_96_96 \
  --model URR_HIE \
  --data custom \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --batch_size 16 \
  --activation gelu \
  --dropout 0.1 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --train_epochs 10 \
  --patience 3 \
  --itr 3 \
  --des Exp
```

For Stations 1 and 2, replace `data_path` with `solar_station_1.csv` or `solar_station_2.csv` and set `batch_size` to 32. To reproduce the remaining horizons, set `pred_len` to 192, 336, or 720.

## Saved outputs

The modified experiment class stores the following arrays under `results/<setting>/`:

- `metrics.npy`: MAE, MSE, RMSE, MAPE, MSPE, and R2;
- `pred.npy`: model predictions;
- `true.npy`: corresponding ground truth;
- `x_test.npy`: historical inputs for the aligned test windows.

Tables III and IV in the manuscript report joint-output standardized-scale metrics averaged over three independent runs. PV-only, persistence, event-conditioned, range-exceedance, and paired-window analyses use the last channel of one retained 96-step reference run. These two evaluation scopes are reported separately and should not be compared as if they were the same metric.

## Reproducing baselines

Configuration files for iTransformer, PatchTST, TimeXer, Crossformer, TimeMixer, TimesNet, and DLinear are provided in `scripts/`. Run each script from the TSLib root directory after checking the station filename, batch size, and prediction horizon.

```bash
bash scripts/iTransformer.sh
bash scripts/PatchTST.sh
```

## Citation

If you use this repository, please cite both the accompanying manuscript and the original dataset. The manuscript citation will be updated after publication.

```bibtex
@misc{fu2026urrhie,
  title  = {Range-aware heterogeneous representations for joint meteorological and photovoltaic power forecasting},
  author = {Fu, Penghui and Li, Xinwei and Yang, Yi},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

Add a `LICENSE` file before archival release. A permissive license such as MIT or BSD-3-Clause is suitable if it is consistent with the licenses of the reused TSLib components.

## Contact

For questions about the implementation or experimental settings, contact Xinwei Li at `lixinwei@hpu.edu.cn`.

