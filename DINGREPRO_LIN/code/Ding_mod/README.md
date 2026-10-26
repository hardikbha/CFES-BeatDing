# Decoupling Bias, Aligning Distributions: Synergistic Fairness Optimization for Deepfake Detection

This repository is the official implementation of our paper "Decoupling Bias, Aligning Distributions: Synergistic Fairness Optimization for Deepfake Detection", which has been accepted by **CVPR 2026**.

## 1. Dataset

The FF++, DFD, DFDC, and Celeb-DF datasets used in the paper can be downloaded from this [link](https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/EtMK0nfxMldAikDxesIo6ckBVHMME1iIV1id_ZsbM9hsqg?e=WayYoy).

The additional experimental results mentioned in the appendix are based on this [work](https://github.com/liudan193/Fairness-Benchmark-for-Face-Forgery-Detection-AAAI2025).

## 2. Load Pretrained Weights

Before running the training code, make sure to load the pretrained weights. We provide the corresponding pretrained weights at this [link](https://pan.baidu.com/s/1dC-NPfcN81KNvC7KYGLldQ?pwd=8pv6). Please place the downloaded weights in the `./pretrained` folder.

## 3. Train

To run the training code, you should first train the model using only the cross-entropy loss by running `train_ori_model.py`.

```python
python train_ori_model.py
```

Then, obtain $F_K$, the channels to be decoupled, by running `SFD.py`.

```python
python SFD.py
```

Finally, obtain the final weights by running `GDA.py`.

```python
python GDA.py
```

## 4. Test

For model testing, you can run `test.py`.

```python
python test.py
```

## Citation

Please kindly consider citing our papers in your publications.

```markdown
@article{ding2025fairness,
  title={Fairness-Aware Deepfake Detection: Leveraging Dual-Mechanism Optimization},
  author={Ding, Feng and Yi, Wenhui and Zhou, Yunpeng and He, Xinan and Rao, Hong and Hu, Shu},
  journal={arXiv preprint arXiv:2511.10150},
  year={2025}
}
```

