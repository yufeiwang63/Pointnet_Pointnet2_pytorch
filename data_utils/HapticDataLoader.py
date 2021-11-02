import os
import numpy as np
import os.path as osp

from tqdm import tqdm
from torch.utils.data import Dataset


class HapticDataset(Dataset):
    def __init__(self, split='train', data_root='data/hapticdata/',
            num_point=4096, block_size=1.0, sample_rate=1.0, 
            transform=None, use_random_center=False):

        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.use_random_center = use_random_center

        data_root = osp.join(data_root, split)
        all_data_file = os.listdir(data_root)
        if "variant.json" in all_data_file:
            all_data_file.remove("variant.json")
        all_data_file = sorted(all_data_file)
        
        self.input_pc = []
        self.pc_contact_labels = []
        self.pc_force_labels = []
        self.pc_normal_labels = []
        self.pc_coord_min, self.pc_coord_max = [], []
        num_point_all = []
        # labelweights = np.zeros(13)

        for data_name in tqdm(all_data_file, total=len(all_data_file)):
            data_path = os.path.join(data_root, data_name)
            data = np.load(data_path)  # xyz + one-hot + contact label, N*5
            points, contact_labels = data[:, 0:4], data[:, 4] 
            forces = data[:, 5]
            forces = forces * (80 * 80) # NOTE: predict lambda here
            normals = data[:, -3:]
            # tmp, _ = np.histogram(labels, range(14))
            # labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.input_pc.append(points)
            self.pc_contact_labels.append(contact_labels)
            self.pc_force_labels.append(forces)
            self.pc_normal_labels.append(normals)
            self.pc_coord_min.append(coord_min), self.pc_coord_max.append(coord_max)
            num_point_all.append(contact_labels.size)

        # labelweights = labelweights.astype(np.float32)
        # labelweights = labelweights / np.sum(labelweights)
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        # print(self.labelweights)
        if num_point < 0:
            self.num_point = int(np.mean(num_point_all))
        else:
            self.num_point = num_point
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / self.num_point)
        # self.num_iter = num_iter
        self.data_num = len(all_data_file)
        data_idxs = []
        # print("sample_prob: ", sample_prob)
        # print("num_iter: ", num_iter)
        for index in range(len(self.input_pc)):
            data_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.data_idxs = np.array(data_idxs)
        pos_label_num = np.sum([np.sum(label) for label in self.pc_contact_labels])
        total_point_num = np.sum(num_point_all)
        self.pos_label_weight = (total_point_num - pos_label_num) / pos_label_num
        # print("Totally {} samples in {} set.".format(len(self.data_idxs), split))

    def get_dataset_statistics(self):
        return self.num_point, self.pos_label_weight, self.data_idxs

    def __getitem__(self, idx):
        data_idx = self.data_idxs[idx]
        # points = self.input_pc[data_idx]   # N * 6
        # labels = self.pc_contact_labels[data_idx]   # N
        # N_points = points.shape[0]
        # data_idx = idx % self.data_num
        points, contact_labels = self.input_pc[data_idx], self.pc_contact_labels[data_idx]
        force_labels = self.pc_force_labels[data_idx]
        normal_labels = self.pc_normal_labels[data_idx]

        if self.use_random_center:
            while (True):
                center = points[np.random.choice(N_points)][:3]
                block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
                point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                if point_idxs.size > 1024:
                    break
            if len(point_idxs) >= self.num_point:
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        else:
            old_points = points.copy()
            points = points - np.mean(points, axis=0)
            selected_point_idxs = [_ for _ in range(points.shape[0])]

        
       

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 4
        current_points = np.zeros((selected_points.shape[0], 7))  # num_point * 7
        current_points[:, 4] = selected_points[:, 0] / self.pc_coord_max[data_idx][0]
        current_points[:, 5] = selected_points[:, 1] / self.pc_coord_max[data_idx][1]
        current_points[:, 6] = selected_points[:, 2] / self.pc_coord_max[data_idx][2]
        if self.use_random_center:
            selected_points[:, 0] = selected_points[:, 0] - center[0]
            selected_points[:, 1] = selected_points[:, 1] - center[1]
            # selected_points[:, 3:6] /= 255.0
        current_points[:, 0:4] = selected_points
        current_contact_labels = contact_labels[selected_point_idxs].reshape(-1, 1)
        current_force_labels = force_labels[selected_point_idxs].reshape(-1, 1)
        current_normal_labels = normal_labels[selected_point_idxs].reshape(-1, 3)
        if self.transform is not None:
            current_points, current_contact_labels = self.transform(current_points, current_contact_labels)
        return current_points, (current_contact_labels, current_force_labels, current_normal_labels), \
            old_points[selected_point_idxs, :3]

    def __len__(self):
        # return self.data_num * self.num_iter
        return len(self.data_idxs)