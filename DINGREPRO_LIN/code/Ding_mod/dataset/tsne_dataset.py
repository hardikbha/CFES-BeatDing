# dataset/tsne_dataset.py
import os
import csv
from typing import Optional, Dict, Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_test_transform():
    """
    Test transform:
        Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize([0.5]*3, [0.5]*3)
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


class IntersecTSNEDataset(Dataset):
    """
    从 CSV 读取:
        img_path: 图片路径（相对或绝对）
        label:    真假标签 (0/1)
        intersec_label: 交叉属性标签（字符串或整数）
    用于 t-SNE，可根据 intersec_label 上色。
    """

    def __init__(
        self,
        csv_path: str,
        img_root: Optional[str] = None,
        transform=None,
    ) -> None:
        """
        Args:
            csv_path: CSV 文件路径，要求表头包含 img_path, label, intersec_label
            img_root: 如果 img_path 是相对路径，这里是图片根目录；如果为 None，则直接使用 img_path
            transform: 图像变换（建议用 get_test_transform()）
        """
        super().__init__()
        self.csv_path = csv_path
        self.img_root = img_root
        self.transform = transform

        self.img_paths = []
        self.labels = []
        self.intersec_labels = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = {"img_path", "label", "intersec_label"}
            if not required_cols.issubset(reader.fieldnames):
                raise ValueError(
                    f"CSV columns must include {required_cols}, "
                    f"but got {reader.fieldnames}"
                )

            for row in reader:
                self.img_paths.append(row["img_path"])
                self.labels.append(int(row["label"]))          # 假设 0/1
                self.intersec_labels.append(row["intersec_label"])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_rel = self.img_paths[idx]
        if self.img_root is not None and not os.path.isabs(img_rel):
            img_path = os.path.join(self.img_root, img_rel)
        else:
            img_path = img_rel

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        intersec_label = self.intersec_labels[idx]

        return {
            "image": img,                 # Tensor [3, 224, 224]
            "label": torch.tensor(label, dtype=torch.long),
            "intersec_label": intersec_label
        }
