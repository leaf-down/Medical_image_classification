import os
import glob
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset


def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(np.asarray(arr), mode=mode)
    if isinstance(size, (list, tuple)):
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def prepare_data_path_recursive(root):
    """递归收集 root 下所有图片路径（大小写无关），返回(路径列表, 文件名列表)"""
    exts = ["bmp", "tif", "tiff", "jpg", "jpeg", "png"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        paths += glob.glob(os.path.join(root, "**", f"*.{ext.upper()}"), recursive=True)
    paths = sorted(list(set(paths)))
    names = [os.path.basename(p) for p in paths]
    return paths, names


class Fusion_dataset(Dataset):
    """
    读取 CT/MRI 配对数据：
      - MRI 视作 IR（红外）分支
      - CT  视作 VIS（可见光）分支
    默认灰度读入，统一 resize 到 256x256，归一化到 [0,1]，并返回 (1,H,W) 张量对。
    """
    def __init__(self, split, data_root="./dataset/CT_MRI_Original", length=0, size=256):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        self.size = size
        self.length_override = max(0, int(length))
        ct_dir = os.path.join(data_root, "CT")
        mri_dir = os.path.join(data_root, "MRI")

        # VIS <- CT, IR <- MRI
        self.filepath_vis, self.filenames_vis = prepare_data_path_recursive(ct_dir)
        self.filepath_ir,  self.filenames_ir  = prepare_data_path_recursive(mri_dir)

        print(f"[Dataset] CT  (VIS) dir: {ct_dir}  -> {len(self.filepath_vis)} files")
        print(f"[Dataset] MRI (IR ) dir: {mri_dir} -> {len(self.filepath_ir)} files")

        if len(self.filepath_vis) == 0 or len(self.filepath_ir) == 0:
            raise RuntimeError(
                "数据为空，请检查 data_root 路径或图片后缀。\n"
                f"  data_root = {data_root}\n"
                f"  ct_dir    = {ct_dir}\n"
                f"  mri_dir   = {mri_dir}\n"
            )

        # 数量不一致时，按最小长度截断（保证索引安全）
        n = min(len(self.filepath_vis), len(self.filepath_ir))
        if n < len(self.filepath_vis) or n < len(self.filepath_ir):
            print(f"[Dataset] ⚠️ CT 与 MRI 数量不一致，已截断到 {n} 对")
            self.filepath_vis = self.filepath_vis[:n]
            self.filepath_ir  = self.filepath_ir[:n]
            self.filenames_vis = self.filenames_vis[:n]
            self.filenames_ir  = self.filenames_ir[:n]

    def __len__(self):
        return self.length_override if self.length_override > 0 else len(self.filepath_vis)

    def __getitem__(self, index):
        if self.length_override > 0:
            index = index % len(self.filepath_vis)

        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        # 以灰度读入
        vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if vis is None:
            raise ValueError(f"Failed to load VIS image: {vis_path}")
        if ir is None:
            raise ValueError(f"Failed to load IR image:  {ir_path}")

        # 统一大小
        vis = imresize(vis, (self.size, self.size), interp='bicubic')
        ir = imresize(ir, (self.size, self.size), interp='bicubic')

        # 归一化到 [0,1] 并加通道维 (1,H,W)
        vis = np.expand_dims(vis.astype(np.float32) / 255.0, axis=0)
        ir = np.expand_dims(ir.astype(np.float32) / 255.0, axis=0)

        # 扩展到 3 通道 (3,H,W)，保持和模型 in_chans=3 一致
        vis = np.repeat(vis, 3, axis=0)
        ir = np.repeat(ir, 3, axis=0)

        return torch.from_numpy(vis), torch.from_numpy(ir)

