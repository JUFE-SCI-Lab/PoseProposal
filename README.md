**PoseProposal (Findings of CVPR 2026)**

Official code repository for the paper:  
**[Learning to Propose Pose for Category-Agnostic Objects via Joint Refinement with Co-Matching Supervision]**  
*Junjie Chen, Zezheng Liu, Runxiang Liu, Yuming Fang, Yifan Zuo, Jiebin Yan*

---

## 📌 Abstract

Perceiving the position, area and structure of category-agnostic objects are fundamental abilities of human vision system, which inspire the research on bounding-box proposal, mask proposal and pose proposal. In these tasks, pose proposal is relatively under-explored. Some recent works focus on estimating keypoints for category-agnostic objects with the support of a few annotated images (few-shot) or descriptions (zero-shot). However, using few/zero-shot support requires non-negligible annotation, preliminary classification and quality guarantee, which are not always available or reliable in inference in extensive applications. In this paper, we explore a different but more straightforward task, *i.e.*, pose proposal, where a model should directly estimate both keypoints and their links without support. To solve this novel yet challenging task, we propose a pose refinement framework with co-matching based supervision, which could learn transferable keypoints and links from base classes and directly propose the complete poses of arbitrary objects. Extensive experiments and in-depth analyses on large-scale benchmark demonstrate the effectiveness of our method.

---

## 🚀 Usage

### 1. Installation

The installation is similar to CapeFormer. Detailed dependencies can be found in `cape_environment.yml`.

```bash
conda env create -f cape_environment.yml
conda activate cape
```

### 2. Data Preparation

Please follow the official guide to prepare the **MP-100 dataset** for training and evaluation, and organize the data structure properly.

Alternatively, we provide a unified annotation file `unified_ann_file.json`. You can use `valid_class_ids` to configure different dataset splits (S1–S5) flexibly.

### 3. Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_ours.py \
  --config propformer/cfg/fo_e20.py \
  --cfg-options model.metacfg.point_num=100 optimizer.lr=1e-4
```

### 4. Testing

```bash
CUDA_VISIBLE_DEVICES=0 python train_ours.py \
  --config propformer/cfg/fo_e20.py \
  --load-from /path/to/checkpoint \
  --cfg-options model.metacfg.point_num=80 optimizer.lr=1e-4 \
  --val-only
```

### 5. Evaluation Metrics

We adopt standard metrics for category-agnostic pose estimation:
- **mAP** (mean Average Precision)

---


## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{chen2026poseproposal,
  title={Learning to Propose Pose for Category-Agnostic Objects via Joint Refinement with Co-Matching Supervision},
  author={Junjie Chen, Zezheng Liu, Runxiang Liu, Yuming Fang, Yifan Zuo, Jiebin Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 🙏 Acknowledgement

This project is built upon the following excellent open-source projects:
- [MMPose](https://github.com/open-mmlab/mmpose)
- [Pose-for-Everything](https://github.com/luminxu/Pose-for-Everything)
- [CapeFormer](https://github.com/dmtrs/CapeFormer)

---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

---

**Now you can copy this entire content into your `README.md` file, commit, and push.**
