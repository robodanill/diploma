# Colab Training

Use `plantclef_colab_training.ipynb` as the main Colab entry point.

The notebook is intentionally thin: it installs the project, mounts Google Drive,
checks the expected dataset layout, and launches two training stages:

1. `genus`: global image view, trains `S-CNN (A)`;
2. `species`: local central crop, trains `S-CNN (B)`.

After both checkpoints are ready, the notebook builds a reference index. The desktop
application will later load the two checkpoints plus this index for real inference.

Expected normalized metadata file:

```text
data/plantclef2015/metadata.csv
```

Required columns:

```text
image_path,family,genus,species
```

`image_path` may be absolute or relative to `data/plantclef2015`.

If the dataset is unpacked with PlantCLEF XML annotations, create this file with:

```bash
plant-classifier-prepare-plantclef \
  --source-root data/plantclef2015 \
  --output data/plantclef2015/metadata.csv
```

By default only `Content=Leaf` images are included, matching the article's leaf-image
setup. Use `--all-content` only for exploratory experiments.
