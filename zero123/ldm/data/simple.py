import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import webdataset as wds
from datasets import load_dataset
from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)


def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [
        NfpDataset(
            x,
            image_transforms=copy.copy(tforms),
            default_caption="A view from a train window",
        )
        for x in dirs
    ]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2 * self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i + 1) * self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[..., ::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[..., ::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption,
        }
        return data


# end hacky things


def make_transforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ]
    )
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path] * repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [
            FolderData(p, caption_file=c, **kwargs)
            for (p, c) in zip(paths, caption_files)
        ]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)


class NfpDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
    ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_transforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1

    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index + 1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)


def multiview_collate(batch):
    batch_repacked = dict()
    batch_repacked["image_cond"] = torch.cat([sample["image_cond"] for sample in batch])
    batch_repacked["image_target"] = torch.stack(
        [sample["image_target"] for sample in batch]
    )
    batch_repacked["cond_count"] = torch.Tensor(
        [sample["cond_count"] for sample in batch]
    ).int()
    batch_repacked["T"] = torch.stack([sample["T"] for sample in batch])
    # print(batch_repacked["image_cond"].shape)
    # print(batch_repacked["image_target"].shape)
    return batch_repacked


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        paths_dir,
        batch_size,
        total_view,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        **kwargs,
    ):
        super().__init__(self)
        self.root_dir = root_dir
        self.paths_dir = paths_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if "image_transforms" in dataset_config:
            image_transforms = [
                torchvision.transforms.Resize(dataset_config.image_transforms.size)
            ]
        else:
            image_transforms = []
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
            ]
        )
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            paths_dir=self.paths_dir,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=multiview_collate,
        )

    def val_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir,
            paths_dir=self.paths_dir,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms,
        )
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=multiview_collate,
        )

    def test_dataloader(self):
        return wds.WebLoader(
            ObjaverseData(
                root_dir=self.root_dir,
                paths_dir=self.paths_dir,
                total_view=self.total_view,
                validation=self.validation,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=multiview_collate,
        )


class ObjaverseData(Dataset):
    def __init__(
        self,
        root_dir=".objaverse/hf-objaverse-v1/views",
        paths_dir=".objaverse/hf-objaverse-v1/view",
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(paths_dir, "valid_paths.json")) as f:
            self.paths = json.load(f)

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[
                math.floor(total_objects / 100.0 * 99.0) :
            ]  # used last 1% as validation
        else:
            self.paths = self.paths[
                : math.floor(total_objects / 100.0 * 99.0)
            ]  # used first 99% as training
        print("============= length of dataset %d =============" % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(
            np.sqrt(xy), xyz[:, 2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor(
            [
                d_theta.item(),
                math.sin(d_azimuth.item()),
                math.cos(d_azimuth.item()),
                d_z.item(),
            ]
        )
        return d_T

    def load_im(self, path, color):
        """
        replace background pixel with random color in rendering
        """
        try:
            img = plt.imread(path)
        except:
            print(path)

            raise FileNotFoundError
            # sys.exit()
        img[img[:, :, -1] == 0.0] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.0))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        cond_count = random.randint(1, 6)
        indices = random.sample(
            range(total_view), cond_count + 1
        )  # without replacement
        index_target = indices[0]
        if random.random() < 0.1:
            random.shuffle(indices)
        indices_cond = indices[1:]
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)

        color = [1.0, 1.0, 1.0, 1.0]

        try:
            # target_im = torch.stack(
            #     [
            #         self.process_im(
            #             self.load_im(
            #                 os.path.join(filename, "%03d.png" % index_target), color
            #             )
            #         )
            #     ]
            #     * cond_count
            # )
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_ims = torch.stack(
                [
                    self.process_im(
                        self.load_im(
                            os.path.join(filename, "%03d.png" % index_cond), color
                        )
                    )
                    for index_cond in indices_cond
                ]
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % indices_cond[0]))

        except:
            # very hacky solution, sorry about this
            filename = os.path.join(
                self.root_dir, "692db5f2d3a04bb286cb977a7dba903e"
            )  # this one we know is valid
            # target_im = torch.stack(
            #     [
            #         self.process_im(
            #             self.load_im(
            #                 os.path.join(filename, "%03d.png" % index_target), color
            #             )
            #         )
            #     ]
            #     * cond_count
            # )
            target_im = self.process_im(
                self.load_im(os.path.join(filename, "%03d.png" % index_target), color)
            )
            cond_ims = torch.stack(
                [
                    self.process_im(
                        self.load_im(
                            os.path.join(filename, "%03d.png" % index_cond), color
                        )
                    )
                    for index_cond in indices_cond
                ]
            )
            target_RT = np.load(os.path.join(filename, "%03d.npy" % index_target))
            cond_RT = np.load(os.path.join(filename, "%03d.npy" % indices_cond[0]))
            target_im = torch.zeros_like(target_im)
            cond_ims = torch.zeros_like(cond_ims)

        data["image_target"] = target_im
        data["image_cond"] = cond_ims
        data["cond_count"] = cond_count
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class FolderData(Dataset):
    def __init__(
        self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_transforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir / chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


import random


class TransformDataset:
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }

    def __getitem__(self, index):
        data = self.ds[index]

        im = data["image"]
        im = im.permute(2, 0, 1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1, 2, 0)

        data["image"] = im
        data["txt"] = data["txt"] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)


def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split="train",
    image_key="image",
    caption_key="txt",
):
    """Make huggingface dataset with appropriate list of transforms applied"""
    ds = load_dataset(name, split=split)
    tform = make_transforms(image_transforms)

    assert (
        image_column in ds.column_names
    ), f"Didn't find column {image_column} in {ds.column_names}"
    assert (
        text_column in ds.column_names
    ), f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds


class TextOnly(Dataset):
    def __init__(
        self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1
    ):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus * [x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2.0 - 1.0, "c h w -> h w c")
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, "rt") as f:
            captions = f.readlines()
        return [x.strip("\n") for x in captions]


import json
import random


class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir / retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
