# CT-CQL

Official implementation of our KDD 2026 paper.

## Paper

**Title:** Continuous-Time Counterfactual Quantile Learning for Risk-Sensitive Policy Optimization

**Accepted at:** The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD 2026**), Jeju Island, Republic of Korea, August 9--13, 2026.

**DOI:** [10.1145/3770854.3780271](https://doi.org/10.1145/3770854.3780271)

CT-CQL models full counterfactual outcome distributions with stochastic differential equations governed by the Fokker--Planck equation. The implementation combines adversarial minimax training with a doubly robust AIPW objective for risk-sensitive policy learning under time-varying treatments.

## Repository contents

- `sde_minimax_model.py`: generator, discriminator, treatment-specific drift networks, propensity model, training objective, checkpointing, and distributional prediction.
- `train.py`: training entry point for the tumor-growth simulation datasets.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Eliza-YiHe/CT-CQL.git
cd CT-CQL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation can be platform-specific. If you need CUDA support, install the appropriate PyTorch build for your system before installing the remaining dependencies.

## Data format

The training script expects each pickle file to contain a dictionary with these NumPy-compatible arrays:

- `sequence_lengths`: valid trajectory length for each patient;
- `cancer_volume`: longitudinal outcomes with shape `(n_patients, max_time)`;
- `chemo_application`: binary chemotherapy indicators with the same shape;
- `radio_application`: binary radiotherapy indicators with the same shape;
- `patient_types`: integer patient types in `{1, 2, 3}`.

Place the following simulation files in `data/`:

```text
data/
├── can_sim_chemo_treatment_train.p
├── can_sim_no_treatment_train.p
├── can_sim_radio_chemo_train.p
└── can_sim_radio_treatment_train.p
```

The datasets are not redistributed in this repository. MIMIC-III data must be obtained separately under its applicable access requirements.

## Training

```bash
python train.py
```

By default, datasets are read from `data/` and checkpoints are written to `outputs/`. Override either location with environment variables:

```bash
CT_CQL_DATA_DIR=/path/to/data \
CT_CQL_OUTPUT_DIR=/path/to/checkpoints \
python train.py
```

## Citation

If this code supports your research, please cite:

```bibtex
@inproceedings{he2026ctcql,
  title     = {Continuous-Time Counterfactual Quantile Learning for Risk-Sensitive Policy Optimization},
  author    = {He, Yi and Wu, Anpeng and Xiong, Ruoxuan and Wang, Yingrong and Kuang, Kun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770854.3780271}
}
```

## License

This project is released under the [MIT License](LICENSE).
