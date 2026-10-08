# External software

The implementation uses PyTorch, torchvision, NumPy, SciPy, pandas, scikit-learn, PyArrow, LMDB, Pillow and plotting libraries. CUDA kernels use Triton. Segmentation uses Cellpose; image feature extraction uses DINOv2. Whole-slide and registration workflows may require OpenSlide, libvips and their platform dependencies.

Third-party software and pretrained models remain subject to their respective licenses and access terms. This repository does not bundle complete third-party repositories or redistribute their pretrained weights. Obtain required implementations from their original maintainers and follow the versions and interfaces expected by the selected analysis.

References:

- [PyTorch](https://pytorch.org/)
- [Cellpose](https://github.com/MouseLand/cellpose)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Triton](https://github.com/triton-lang/triton)
- [OpenSlide](https://openslide.org/)
