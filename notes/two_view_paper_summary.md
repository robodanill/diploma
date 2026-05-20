# Two-view fine-grained classification of plant species - working summary

Source: `1-s2.0-S0925231221014934-main (1).pdf`.

## What they actually evaluate

- The PlantCLEF numbers in the paper match the `Content=LeafScan` subset, not our earlier `Content=Leaf` subset.
- The reported PlantCLEF 2015 result is on the predefined PlantCLEF 2015 test set, not on a validation split created from training images.
- Their PlantCLEF leaf test set has 221 leaf images, 43 genus classes, and 60 species classes.
- For the final PlantCLEF comparison, S-CNN(B) uses the 60 official test species and six training images per species: 360 species-stage training images.
- S-CNN(A) is a genus-level metric, so its six-shot subset must be balanced by genus, not by species.
- The PlantCLEF score is the LifeCLEF `S` metric, which rewards the inverse rank of the first correct match. It is not exactly the same as our current top-k genus retrieval accuracy.

## Data protocol

- They emphasize few-sample metric learning: six training images are selected per class at the active taxonomic level.
- The train pool for the paper-style final run is therefore the 6,527 training images belonging to the 60 species present in the official test set. From that pool, S-CNN(A) should use a genus-balanced subset and S-CNN(B) should use a species-balanced subset.
- PlantCLEF training subset in the paper: 12,605 training images when considering the taxonomic groups used for the method.
- For S-CNN training pair counts on PlantCLEF 2015:
  - Family: 200 positive, 300 negative.
  - Genus: 400 positive, 600 negative.
  - Species: 800 positive, 1,200 negative.
- They intentionally create more negative than positive pairs, following Melekhov et al.
- They do not train references from the test set. References are selected from training samples; test images are separate queries.

## CNN baseline training

- They compare AlexNet, GoogLeNet, and VGG16 as pretrained CNN baselines.
- Fine-tuning details for CNN baselines:
  - SGD.
  - Momentum: 0.9.
  - Learning rate: 0.001.
  - LR decay: 0.5 every 512 iterations.
  - Total iterations: 2,048.
  - Batch size: 32.
- Baseline result on PlantCLEF 2015 global species classification:
  - AlexNet: 71.98%.
  - GoogLeNet: 73.30%.
  - VGG16: 75.09%.

## S-CNN architecture/protocol

- Two separate S-CNN models are trained:
  - S-CNN (A): genus-level metric, global whole-leaf view.
  - S-CNN (B): species-level metric, local center crop view.
- Twin CNN branches are initialized from ImageNet-pretrained CNNs and share weights.
- The original classification layer is replaced by a metric head.
- They compute L1 distance between feature vectors from the last fully connected representation; for VGG16 this is a 4096-dimensional vector.
- Loss is binary cross entropy over image-pair labels: 1 for same category, 0 for different category.

## Preprocessing

- They do not feed raw arbitrary images directly.
- Preprocessing has two steps:
  - segment/filter leaf: grayscale, Otsu threshold, top-hat operation to remove unwanted objects such as stem, bounding box around filtered leaf.
  - resize final leaf image to 224x224.
- Local view is a center crop from the filtered leaf image.
- They test local crop resolutions 32x32, 64x64, and 128x128; 32x32 works best for local species discrimination.

## Hierarchy and fusion

- Best hierarchy is Genus -> Species.
- First stage uses global view; second stage uses local view.
- Family -> Species performs worse.
- Local view should not be used as the first/coarse stage.
- The second stage only evaluates species belonging to genus candidates returned by the first stage.
- They fuse genus and species evidence using the frequency of genus references in the first-stage ranked list as a weight for second-stage species scores.

## References and Rk

- References are selected from training samples.
- Number of reference images per species `Nr` is experimentally varied from 1 to 6.
- Best final PlantCLEF result uses `Nr = 6` and `Rk = 30` genus reference candidates.
- Genus references are built from species references: the paper tries to include at least one reference for each species inside a genus.
- Reported PlantCLEF S-CNN results:
  - first-stage genus score with `Rk=30`: around 0.95-0.98 depending on Nr.
  - final species score: 0.81 for Nr=1, 0.86 for Nr=3, 0.87 for Nr=6.
  - final top-k with Nr=6/Rk=30: top-1 0.87, top-3 0.94, top-5 1.0.

## Why our current pipeline differs

- We were using `Content=Leaf` metadata: locally this is 13,367 train images, 490 genera, and 899 species. The paper's protocol is the much narrower PlantCLEF `LeafScan` subset.
- We initially evaluated references and queries by splitting the same validation metadata, while the paper evaluates training references against the predefined test set.
- We do not yet implement full two-stage species fusion; most current diagnostics are genus-only retrieval.
- We currently do not perform leaf segmentation/Otsu/top-hat/bounding-box preprocessing; we resize raw leaf images.
- Our earlier strict pipeline balanced both S-CNN(A) and S-CNN(B) by species. That overrepresented multi-species genera in S-CNN(A), especially Acer/Quercus-like genera, and does not match the genus-level training protocol.
- Our standalone eval should use train references + PlantCLEF test queries for fairer comparison.

## Next implementation implications

- Add/use real PlantCLEF 2015 test metadata and evaluate against it.
- Build references from train: six references per genus for S-CNN(A), with species coverage inside each genus, and six references per species for S-CNN(B).
- Implement two-stage inference/evaluation with `Rk=30`, S-CNN(A) genus candidates, S-CNN(B) local species scoring, and weighted fusion.
- Add preprocessing closer to the paper: leaf segmentation/bounding box before resize, plus center crop for local view.
- Consider reverting pair counts closer to paper for strict reproduction: genus 400/600, species 800/1200, batch 32, 2048 iterations, rather than very long epochs.
