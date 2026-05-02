# Colab Training

Use `plantclef_colab_training.ipynb` as the main Colab entry point.

The notebook is intentionally thin: it installs the project, mounts Google Drive,
checks the expected dataset layout, and launches two training stages:

1. `genus`: global image view, trains `S-CNN (A)`;
2. `species`: local central crop, trains `S-CNN (B)`.

Expected normalized metadata file:

```text
data/plantclef2015/metadata.csv
```

Required columns:

```text
image_path,family,genus,species
```

`image_path` may be absolute or relative to `data/plantclef2015`.

