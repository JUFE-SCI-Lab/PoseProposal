# PoseProposal (Findings 0f CVPR'2026)
Official code repository for the paper:  
[**Learning to Propose Pose for Category-Agnostic Objects\\via Joint Refinement with Co-Matching Supervision**] 
[Junjie Chen, Zezheng Liu, Runxiang Liu, Yuming Fang, Yifan Zuo, Jiebin Yan] 

### Abstract
Perceiving the position, area and structure of category-agnostic objects are fundamental abilities of human vision system, which inspire the research on bounding-box proposal, mask proposal and pose proposal.
In these tasks, pose proposal is relatively under-explored.
Some recent works focus on estimating keypoints for category-agnostic objects with the support of a few annotated images (few-shot) or descriptions (zero-shot).
However, using few/zero-shot support requires non-negligible annotation, preliminary classification and quality guarantee, which are not always available or reliable in inference in extensive applications.
In this paper, we explore a different but more straightforward task, \emph{i.e.}, pose proposal, where a model should directly estimate both keypoints and their links without support.
To solve this novel yet challenging task, we propose a pose refinement framework with co-matching based supervision, which could learn transferable keypoints and links from base classes and directly propose the complete poses of arbitrary objects.
Extensive experiments and in-depth analyses on large-scale benchmark demonstrate the effectiveness of our method.

## Usage

### Install
The installation is similar to [CapeFormer](https://github.com/flyinglynx/CapeFormer), detailed packages could be found in `cape_environment.yml`.

### Data preparation
Please follow the [official guide](https://github.com/luminxu/Pose-for-Everything) to prepare the MP-100 dataset for training and evaluation, and organize the data structure properly. 

Alternatively, we employ an unified annotation file (i.e., `unified_ann_file.json`) and adopt valid_class_ids to set various splits.

### Training and Test
Train:`CUDA_VISIBLE_DEVICES=0 python train_ours.py --config propformer/cfg/fo_e20.py --cfg-options model.metacfg.point_num=100 optimizer.lr=1e-4`

Test:`CUDA_VISIBLE_DEVICES=0 python train_ours.py --config propformer/cfg/fo_e20.py --load-from checkpoint --cfg-options model.metacfg.point_num=80 optimizer.lr=1e-4 --val-only`

## Citation
```bibtex
@inproceedings{FMMP,
  title={Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation},
  author={Junjie Chen, Weilong Chen, Yifan Zuo, Yuming Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

## Acknowledgement
Thanks to:
- MMPose
- Pose-for-Everything
- CapeFormer

## License

This project is released under the [Apache 2.0 license](LICENSE).