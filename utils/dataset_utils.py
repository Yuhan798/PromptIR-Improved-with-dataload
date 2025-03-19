import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'deoverexp': 0, 'delow': 1, 'dehazy': 2, 'decombined': 3}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'deoverexp' in self.de_type:
            self._init_deoverexp_ids()
        if 'delow' in self.de_type:
            self._init_delow_ids()
        if 'dehazy' in self.de_type:
            self._init_dehazy_ids()
        if 'decombined' in self.de_type:
            self._init_decombined_ids()

        random.shuffle(self.de_type)
    
    def _init_deoverexp_ids(self):
        # 从指定的输入文件夹中读取所有文件，并构造对应的 ground truth 路径
        file_list = sorted(os.listdir(self.args.deoverexp_input_dir))
        self.deoverexp_ids = [{
            "input": os.path.join(self.args.deoverexp_input_dir, fname),
            "gt": os.path.join(self.args.deoverexp_gt_dir, fname),
            "de_type": self.de_dict['deoverexp']
        } for fname in file_list]
        print("Total deoverexp Ids: {}".format(len(self.deoverexp_ids)))

    def _init_delow_ids(self):
        file_list = sorted(os.listdir(self.args.delow_input_dir))
        self.delow_ids = [{
            "input": os.path.join(self.args.delow_input_dir, fname),
            "gt": os.path.join(self.args.delow_gt_dir, fname),
            "de_type": self.de_dict['delow']
        } for fname in file_list]
        print("Total delow Ids: {}".format(len(self.delow_ids)))

    def _init_dehazy_ids(self):
        file_list = sorted(os.listdir(self.args.dehazy_input_dir))
        self.dehazy_ids = [{
            "input": os.path.join(self.args.dehazy_input_dir, fname),
            "gt": os.path.join(self.args.dehazy_gt_dir, fname),
            "de_type": self.de_dict['dehazy']
        } for fname in file_list]
        print("Total dehazy Ids: {}".format(len(self.dehazy_ids)))
        
    def _init_decombined_ids(self):
        file_list = sorted(os.listdir(self.args.decombined_input_dir))
        self.decombined_ids = [{
            "input": os.path.join(self.args.decombined_input_dir, fname),
            "gt": os.path.join(self.args.decombined_gt_dir, fname),
            "de_type": self.de_dict['decombined']
        } for fname in file_list]
        print("Total decombined Ids: {}".format(len(self.decombined_ids)))
    
    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[0], img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
    def _merge_ids(self):
        self.sample_ids = []
        if 'deoverexp' in self.de_type:
            self.sample_ids += self.deoverexp_ids
        if 'delow' in self.de_type:
            self.sample_ids += self.delow_ids
        if 'dehazy' in self.de_type:
            self.sample_ids += self.dehazy_ids
        if 'decombined' in self.de_type:
            self.sample_ids += self.decombined_ids
        print("Total sample ids: {}".format(len(self.sample_ids)))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        # 对于新任务，sample 中应包含 "input" 和 "gt" 两个键
        if "input" in sample and "gt" in sample:
            degraded_img = crop_img(np.array(Image.open(sample["input"]).convert('RGB')), base=16)
            clean_img = crop_img(np.array(Image.open(sample["gt"]).convert('RGB')), base=16)
            # 采用同步随机裁剪和数据增强
            degraded_patch, clean_patch = random_augmentation(*self._crop_patch(degraded_img, clean_img))
            name = os.path.basename(sample["input"]).split('.')[0]
        else:
            raise ValueError("样本格式错误，未找到 'input' 和 'gt' 字段。")

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degraded_patch)
        return [name, sample["de_type"]], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DeDataset(Dataset):
    def __init__(self, args, task=None, addnoise = False, sigma = None):
        super(DeDataset, self).__init__()
        # self.ids = []
        self.args = args
        self.task_dict = {'deoverexp': 0, 'delow': 1, 'dehazy': 2, 'decombined': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        
        self.task_idx = self.task_dict[task]
        self.set_dataset(task)
        
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:  # deoverexp
            input_dir = self.args.deoverexp_Test_input_dir
        elif self.task_idx == 1:  # delow
            input_dir = self.args.delow_Test_input_dir
        elif self.task_idx == 2:  # dehazy
            input_dir = self.args.dehazy_Test_input_dir
        elif self.task_idx == 3:  # decombined
            input_dir = self.args.decombined_Test_input_dir

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
        
        self.ids = []
        name_list = sorted(os.listdir(input_dir))
        self.ids = [os.path.join(input_dir, fname) for fname in name_list]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:  # deoverexp
            gt_name = os.path.join(self.args.deoverexp_Test_gt_dir, os.path.basename(degraded_name))
        elif self.task_idx == 1:  # delow
            gt_name = os.path.join(self.args.delow_Test_gt_dir, os.path.basename(degraded_name))
        elif self.task_idx == 2:  # dehazy
            gt_name = os.path.join(self.args.dehazy_Test_gt_dir, os.path.basename(degraded_name))
        elif self.task_idx == 3:  # decombined
            gt_name = os.path.join(self.args.decombined_Test_gt_dir, os.path.basename(degraded_name))
        return gt_name

    def set_dataset(self, task):
        if task not in self.task_dict:
            raise ValueError(f"Unsupported task: {task}. Available tasks: {list(self.task_dict.keys())}")
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise and self.sigma is not None:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img = self.toTensor(clean_img)
        degraded_img = self.toTensor(degraded_img)
        degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
    