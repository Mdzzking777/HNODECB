AFM04 KAN full functional test logs are written here.

Expected files:

- `log2_04_step2a_kan_full_test_local_driver_YYYYMMDD_HHMMSS.txt`
- `window_per_shard/log2_04_step2a_kan_full_test_local_p1.txt`
- `window_per_shard/log2_04_step2a_kan_full_test_local_p2.txt`
- `window_per_shard/log2_04_step2a_kan_full_test_local_p3.txt`

Visualization outputs are written to:

- `visualization/*.png`

Each shard log records:

- window horizon / window meta
- KAN config and optimizer config
- epoch-by-epoch train / validation loss
- x1 / x3 reconstruction diagnostics
- teacher-forced Fts diagnostics
- checkpoint save events
- final completion summary
