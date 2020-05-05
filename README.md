# MaskGenerator
End-to-end trainable mask generation


`...D_P` is for the pixel-level Domain Adaptation discriminator

`...D_F` is for the feature-level (latent-vector actually) Domain Adaptation discriminator

Branches

|branch|comment|active|
|:-----|:-----:|:-:|
|megadepth|add input depth information using [MegaDepth model](https://github.com/cc-ai/height_estimation/tree/master/src/MegaDepth) |Yes (Méli)|
|watchdogs|train on Watchdogs dataset| Yes (Méli)|
|discriminative|segmentation (discriminative) rather than generative approach| Yes|
|exp_branch| general functionalities - resume training| Yes|

