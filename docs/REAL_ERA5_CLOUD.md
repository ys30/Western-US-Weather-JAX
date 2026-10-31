# Real ERA5, bounded CPU integration run

Source: [Google Research ARCO-ERA5](https://github.com/google-research/arco-era5). The public dataset is in Google Cloud Storage (us-central1). This repository uses **real** ERA5 only when the manually dispatched workflow successfully downloads and trains on it.

Open GitHub Actions > **Real ERA5 CPU experiment (manual)** > **Run workflow**.
The job installs xarray+zarr+gcsfs, fetches a 6-hourly July 2021 subsection, validates metadata, trains a CNN, and uploads metrics.
The job is manual to avoid unbounded network charges or accidentally fetching an entire global dataset.

Limitations: Google ARCO dataset is time-chunked and a regional request may still read large underlying chunks. GitHub-hosted runners are CPU-only. This is an integration smoke test, NOT a valid generalization result: its short chronological train/test split and few epochs cannot support extreme-weather science.
Do not claim a GPU run until the Colab/GCP notebook reports an actual GPU backend. A failure from public cloud access should be diagnosed from workflow logs rather than treated as weather-model failure.
