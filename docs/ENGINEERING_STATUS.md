# Engineering status and verification gates

## Verified
- GitHub Actions CPU unit and synthetic-training smoke jobs
- NetCDF preflight: coordinate normalization, Kelvin, fixed time intervals
- NetCDF-to-CNN integration test is checked on each CI push

## Not verified
- Training on an actual licensed/downloaded ERA5 extract
- GPU hardware acceleration on Colab or GCP
- Multi-GPU distributed scaling or a skillful operational weather model

## Acceptance gates
1. CPU CI and the NetCDF integration test pass.
2. Real ERA5 training passes data preflight, outputs model weights and a held-out score.
3. GPU runtime explicitly reports GPU, runs real-data training, and saves its metrics.
4. Compare CNN against persistence before adding U-Net.

The synthetic NetCDF integration test is deliberately fabricated data; it is not a real ERA5 scientific benchmark.
