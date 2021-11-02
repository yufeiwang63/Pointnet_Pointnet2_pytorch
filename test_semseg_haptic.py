"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import os.path as osp
from Pointnet_Pointnet2_pytorch.data_utils.HapticDataLoader import HapticDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import Pointnet_Pointnet2_pytorch.provider as provider
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg_msg]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')

    return parser.parse_args()

def plot(original_points, predictions, gt_label, force_pred, force_target, save_path, do_plot=True):
    original_points = original_points.reshape(-1, 3)
    xyz = original_points

    false_positive = np.logical_and(predictions == 1, gt_label == 0)
    true_positive = np.logical_and(predictions == 1, gt_label == 1)
    false_negative = np.logical_and(predictions == 0, gt_label == 1)
    true_negative = np.logical_and(predictions == 0, gt_label == 0)

    num_point = len(xyz)
    tp = np.sum(true_positive) / num_point
    fp = np.sum(false_positive) / num_point
    tn = np.sum(true_negative) / num_point
    fn = np.sum(false_negative) / num_point
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall)

    if not do_plot:
        return tp, fp, tn, fn, precision, recall, f1

    fig = plt.figure(figsize=(18,12))
    gt_ax = fig.add_subplot(2, 3, 1, projection='3d')
    predict_ax = fig.add_subplot(2, 3, 2, projection='3d')
    error_ax = fig.add_subplot(2, 3, 3, projection='3d')
    predict_ax.set_title("tp {:.3f} fp {:.3f} tn {:.3f} fn {:.3f} \n precision {:.3f} recall {:3f} f1 {:.3f}".format(
        tp, fp, tn, fn, precision, recall, f1
    ))

    # plot contact points
    for ax in [gt_ax, predict_ax, error_ax]:
        ax.scatter3D(xyz[:, 0], xyz[:, 2], xyz[:, 1], color='blue', s=0.05)

    pred_contact = xyz[predictions == 1]
    gt_contact = xyz[gt_label == 1]
    false_positive_contact = xyz[false_positive]
    false_negative_contact = xyz[false_negative]

    predict_ax.scatter3D(pred_contact[:, 0], pred_contact[:, 2], pred_contact[:, 1], color='red')
    gt_ax.scatter3D(gt_contact[:, 0], gt_contact[:, 2], gt_contact[:, 1], color='red')
    error_ax.scatter3D(false_positive_contact[:, 0], false_positive_contact[:, 2], false_positive_contact[:, 1], color='blue')
    error_ax.scatter3D(false_negative_contact[:, 0], false_negative_contact[:, 2], false_negative_contact[:, 1], color='green')


    # plot force
    f_gt_ax = fig.add_subplot(2, 3, 4, projection='3d')
    f_predict_ax = fig.add_subplot(2, 3, 5, projection='3d')
    f_error_ax = fig.add_subplot(2, 3, 6, projection='3d')

    for ax in [f_gt_ax, f_predict_ax, f_error_ax]:
        ax.scatter3D(xyz[:, 0], xyz[:, 2], xyz[:, 1], color='blue', s=0.05)

    # max_force = max(abs(np.max(force_pred)), abs(np.max(force_target)))
    pred_force = force_pred[predictions == 1].flatten()
    gt_force = force_target[gt_label == 1].flatten()
    # [x / max_force / 2 + 0.5, 0, 0, 1 ]
    # pred_force_color = [cm.hot(x) for x in pred_force]
    # gt_force_color = [cm.hot(x) for x in gt_force]
    # print(np.mean(np.array(gt_force_color)))
    # print(np.std(np.array(gt_force_color)))
    min_force = np.nanmin(force_target)
    max_force = np.nanmax(force_target)
    pred_force_color = (pred_force - min_force) / (max_force - min_force)
    gt_force_color = (gt_force - min_force) / (max_force - min_force)
    # for p, c in zip(pred_contact, pred_force_color):
    #     # print(c)
        # for p, c in zip(gt_contact, gt_force_color):
    #     f_gt_ax.scatter3D(p[0], p[2], p[1], color=c)
    
    f_predict_ax.scatter3D(pred_contact[:, 0], pred_contact[:, 2], pred_contact[:, 1], cmap='hot', c=pred_force_color)
    f_gt_ax.scatter3D(gt_contact[:, 0], gt_contact[:, 2], gt_contact[:, 1], cmap='hot', c=gt_force_color)


    force_error_on_gt = (force_pred[gt_label == 1] - force_target[gt_label == 1]).flatten()
    false_postive = np.logical_and(predictions == 1, gt_label == 0)
    force_error_on_pred = force_pred[false_postive].flatten()
    error_color_on_gt = (force_error_on_gt - min_force) / (max_force - min_force)
    error_color_on_pred = (force_error_on_pred - min_force) / (max_force - min_force)

    f_error_ax.scatter3D(gt_contact[:, 0], gt_contact[:, 2], gt_contact[:, 1], cmap='hot', c=error_color_on_gt)
    f_error_ax.scatter3D(xyz[false_postive][:, 0], xyz[false_postive][:, 2], xyz[false_postive][:, 1], cmap='hot', c=error_color_on_pred)

    # for p, c in zip(gt_contact, error_color_on_gt):
    #     f_error_ax.scatter3D(p[0], p[2], p[1], color=c)
    # for p, c in zip(xyz[false_postive], error_color_on_pred):
    #     f_error_ax.scatter3D(p[0], p[2], p[1], color=c)

    # center the axis 
    X = xyz[:, 0]
    Y = xyz[:, 2]
    Z = xyz[:, 1]
    for ax in [gt_ax, predict_ax, error_ax, f_gt_ax, f_predict_ax, f_error_ax]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal')
        # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]) / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range[0], mid_x + max_range[0])
        ax.set_ylim(mid_y + max_range[1], mid_y - max_range[1])
        ax.set_zlim(mid_z - max_range[2], mid_z + max_range[2])

    # rotate the axis for better visual
    for ax in [gt_ax, predict_ax, error_ax, f_gt_ax, f_predict_ax, f_error_ax]:
        ax.view_init(30, 250)

    # plt.show()
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")

    return tp, fp, tn, fn, precision, recall, f1
        
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '../data/haptic-perspective/2021-10-12'
    NUM_CLASSES = 1

    print("start loading training data ...")
    TEST_DATASET = HapticDataset(split='train', data_root=root, num_point=1, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=5,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    log_string("The number of training data is: %d" % len(TEST_DATASET))
    # log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.train()
    criterion = MODEL.get_loss().cuda()

    '''test on dataset'''
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batches = len(testDataLoader)

    with torch.no_grad():
        for i, (points, target, original_xyz) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            points = points.data.numpy()
            original_xyz = original_xyz.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.float().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)
            loss = criterion(seg_pred, target, trans_feat, None)

            pred_choice = (seg_pred.cpu().data.numpy().reshape(-1, 1) > 0).astype(np.float)
            pred_choice = pred_choice.flatten()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += points.shape[0] * points.shape[2]
            loss_sum += loss.item()
            plot(original_xyz, pred_choice, batch_label, os.path.join(visual_dir, '{}.png'.format(i)))

        log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

    images = []
    for _ in range(len(testDataLoader)):
        filename = osp.join(visual_dir, "{}.png".format(_))
        images.append(imageio.imread(filename))

    imageio.mimsave(osp.join(visual_dir, 'pred-visual.gif'), images)

if __name__ == '__main__':
    pass
    # args = parse_args()
    # main(args)
