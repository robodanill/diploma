# Honest Demo Artifact Bundle

This branch loads only the matching train-only VGG16 artifact bundle below:

```text
scnn_genus_vgg16.pt
scnn_species_vgg16.pt
reference_index_leafscan_vgg16.pt
```

Download the files from:

```text
/content/drive/MyDrive/diploma_checkpoints/scnn_genus_vgg16.pt
/content/drive/MyDrive/diploma_checkpoints/scnn_species_vgg16.pt
/content/drive/MyDrive/diploma_checkpoints/reference_index_leafscan_vgg16.pt
```

Do not mix these checkpoints with `final_*`, `adapt`, or an index built by
another checkpoint pair. The root-level checkpoint files are synchronized copies
of the matching `_best` checkpoints used when the index was built. The app rejects
`final_scnn_*` checkpoints in this branch.
