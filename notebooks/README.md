# Colab Training

Use `plantclef_colab_leafscan_bundle.ipynb` first to build and save
`PlantCLEF2015_leafscan_only.tar.gz` to Google Drive. Then use
`plantclef_colab_test_data.ipynb` for the matching test bundle, and
`plantclef_colab_training.ipynb` as the main training entry point.

The notebook is intentionally thin: it installs the project, mounts Google Drive,
checks the expected dataset layout, and launches two training stages:

1. `genus`: global image view, trains `S-CNN (A)`;
2. `species`: local central crop, trains `S-CNN (B)`.

After both checkpoints are ready, the notebook builds a reference index. The desktop
application will later load the two checkpoints plus this index for real inference.

For the paper reproduction path, use `Content=LeafScan`, not `Content=Leaf`.
The expected normalized metadata files are:

```text
data/plantclef2015/leafscan_metadata_split.csv
data/plantclef2015/leafscan_metadata.csv
data/plantclef2015/leafscan_paper60_metadata.csv
data/plantclef2015/test_leafscan_metadata.csv
```

`leafscan_paper60_metadata.csv` is generated in the training notebook by filtering
the official train LeafScan metadata to the 60 species present in the official test
set. The strict paper config samples S-CNN (A) by genus and S-CNN (B) by species
from that pool.

Required columns:

```text
image_path,family,genus,species
```

`image_path` may be absolute or relative to the configured dataset root.

If the full PlantCLEF package is already unpacked with XML annotations, create a
compact LeafScan archive with:

```bash
python scripts/build_plantclef_content_bundle.py \
  --source-root data/plantclef2015/train \
  --output /content/drive/MyDrive/PlantCLEF2015_leafscan_only.tar.gz \
  --content LeafScan
```

Use `plantclef_colab_test_data.ipynb` to build the matching
`PlantCLEF2015_leafscan_test.tar.gz` bundle from the official annotated test package.
