import pyflex
import numpy as np
from VCD.camera_utils import (
    get_pointcloud_from_depth, get_matrix_world_to_camera, project_to_image,
    flex_get_rgbd, nearest_neighbor_mapping
)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import argparse
import pcl
from softgym.utils.visualization import save_numpy_as_gif
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--save_robot_state', type=int, default=0)
parser.add_argument('--load_robot_state', type=int, default=0)
parser.add_argument('--show_interval', type=int, default=-1)
parser.add_argument('--save_name', type=str, default='2021-10-21')
parser.add_argument('--save_data', type=int, default=0)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--headless', type=int, default=0)
args = parser.parse_args()

fig = plt.figure()
ax = plt.axes(projection='3d')

scene_idx = {
    'fetch': 9,
    'franka': 10,
    'sawyer': 5 # 11
}

headless = args.headless
render = True
camera_width = 720
camera_height = 720
robot_name = 'sawyer'
show = False if args.show_interval < 0 else True
voxel_size = 0.00625
cloth_mass = 0.5
cloth_width = 80
cloth_height = 40
max_force_show = 300
use_table = 1 # was 0
box_scale = 0.05 # was 0.2
box_z = 0.5 # was 0.5
box_color_r = 0
box_color_g = 0
box_color_b = 0
add_cloth = 0 # was 1

env_idx = scene_idx[robot_name]
if robot_name == 'fetch':
    cam_pos = [0.000823, 0.906065, 1.474992]
    cam_angle = [0.000000, -0.3080, 0.000000]
    scene_params = [*cam_pos, *cam_angle, 0.08, 0.5]
elif robot_name == 'sawyer':
    cam_pos = [-0.423, 0.806065, 1.274992]
    cam_angle = [-0.400000, -0.3080, 0.0]
    scene_params = [*cam_pos, *cam_angle, box_scale, box_z, use_table, max_force_show, 
        box_color_r, box_color_g, box_color_g]
    # scene_params = []
robot_params = []

pyflex.init(headless, render, camera_width, camera_height)
pyflex.set_scene(env_idx, scene_params, 0, robot_params)
matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=cam_angle, cam_pos=cam_pos)

if args.save_data:
    save_path = osp.join('data', 'haptic-perspective', args.save_name, args.split)
    if not osp.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
else:
    save_path = None

def get_action(robot_name, t, args):
    if robot_name == 'sawyer':
        if args.save_data:
            if t <= 10:
                action = [0, 0, 0, 0, 0, 0, -0.001]
            elif t <= 50 and t > 10:
                action = [0, -0.005, 0, 0, 0, 0, 0]
            elif t >= 50 and t < 250:
                action = [0.004, 0, 0, 0, 0, 0, 0]
            else:
                action = [0, 0, 0, 0, 0, 0, 0]
        else:
            action = [0, 0, 0, 0, 0, 0, 0]

    if robot_name == 'fetch':
        action = [0, 0, 0, 0, 0, 0, 0]

    return action

def voxelize_pc(pc, voxel_size):
    cloud = pcl.PointCloud(pc)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    pointcloud = sor.filter()
    voxelized_pc = np.asarray(pointcloud)
    return voxelized_pc

def get_obj_mask(rgb):
    # rgb = rgb[:, :, ::-1]
    r_mask = rgb[:, :, 0] < 20
    # g_mask = np.logical_and(rgb[:, :, 1] > 120, rgb[:, :, 1] <= 255)
    g_mask = rgb[:, :, 1] < 20
    b_mask = rgb[:, :, 2] < 20
    mask = np.logical_and(r_mask, np.logical_and(g_mask, b_mask))
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel)    
    # cv2.imshow("object mask", mask)
    # cv2.waitKey()
    return mask

def get_initial_object_pc():
    pyflex.step()
    img = pyflex.render()
    img = img.reshape(camera_height, camera_width, 4)[::-1, :, :3]
    img = img.astype(np.uint8)
    # img = img[:, :, ::-1]
    print(img[camera_width // 2, camera_height // 2])
    # cv2.imshow("image", img[:, :, ::-1])
    # cv2.waitKey()

    obj_mask = get_obj_mask(img) # TODO: get cloth mask
    rgbd = flex_get_rgbd(camera_height, camera_width)
    rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]
    cloth_mask = (obj_mask == 0)
    depth[cloth_mask] = 0

    # cv2.imshow("object depth", depth)
    # cv2.waitKey()

    world_coordinates = get_pointcloud_from_depth(depth, matrix_world_to_camera)
    depth = depth.flatten()
    world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
    pointcloud = world_coords[depth > 0].astype(np.float32)

    voxelized_pc = voxelize_pc(pointcloud, voxel_size)

    return voxelized_pc

    
voxelized_object_pc = get_initial_object_pc() # this is voxelized already

# TODO: get initial object mask
# load pre-stored robot and cloth state
if args.load_robot_state:
    for _ in range(200):
        if robot_name == 'fetch':
            pyflex.step([0, 0, 0, 0, 0, 0, 0])
        elif robot_name == 'sawyer':
            pyflex.step()
        pyflex.render()

        if _ == 10:
            saved_joint_state = np.load('./data/haptic-perspective/robot_state/{}_hold_cloth_joint.npy'.format(robot_name))
            pyflex.set_rigid_joints(saved_joint_state)
        
        if _ == 190:
            saved_particle_pos = np.load('./data/haptic-perspective/robot_state/{}_hold_cloth_particle_pos.npy'.format(robot_name))
            pyflex.set_positions(saved_particle_pos)

# run the simulation
all_images = []
data_cnt = 0
if args.save_data:
    T = 250
else:
    T = 1000000
for _ in range(T):
    
    action = get_action(robot_name, _, args)
    pyflex.step(action)
    pyflex.render()

    if args.save_data:
        # soft-rigid contact
        contact_info = pyflex.get_soft_rigid_contact()
        contact_info = contact_info.reshape((-1, 17))   
        contact_lambda = contact_info[:, 9]
        nonzero_lambda_idx = contact_lambda != 0
        contact_info = contact_info[nonzero_lambda_idx]
        contact_lambda = contact_info[:, 9]
        contact_rigid_pos = contact_info[:, :3] 
        contact_cloth_pos = contact_info[:, 3:6]
        contact_normal = -contact_info[:, 6:9]
        contact_force = contact_lambda * (cloth_mass / (cloth_width * cloth_height))
        contact_normal_force = contact_force[:, None] * contact_normal
        contact_cloth_pos_add_force = contact_cloth_pos + contact_normal_force
        # print("soft contact number: ", contact_info.shape[0])

        # rigid-rigid contact
        # rigid_contact_info = pyflex.get_rigid_rigid_contact()
        # rigid_contact_info = rigid_contact_info.reshape((-1, 13))
        # rigid_contact_lambda = rigid_contact_info[:, 9]
        # nonzero_lambda_idx = rigid_contact_lambda != 0
        # rigid_contact_info = rigid_contact_info[nonzero_lambda_idx]
        # rigid_contact_lambda = rigid_contact_lambda[nonzero_lambda_idx]
        # rigid_contact_pos = rigid_contact_info[:, :3]
        # print("rigid contact number: ", rigid_contact_pos.shape[0])

        rgbd = flex_get_rgbd(camera_height, camera_width)
        rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]

        img = pyflex.render()
        img = img.reshape(camera_height, camera_width, 4)[::-1, :, :3]
        img = img.astype(np.uint8)
        all_images.append(img)
        object_mask = get_obj_mask(img) # TODO: get cloth mask
        # object_mask = np.logical_not(cloth_mask)
        # print(cloth_mask)
        depth[object_mask > 0] = 0
        # cv2.imshow("cloth depth", depth)
        # cv2.waitKey()

        world_coordinates = get_pointcloud_from_depth(depth, matrix_world_to_camera)
        depth = depth.flatten()
        world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
        pointcloud = world_coords[depth > 0].astype(np.float32)
        # print("original pointcloud shape: ", pointcloud.shape)
        # cv2.imshow("rgb", rgb)


        voxelized_pc = voxelize_pc(pointcloud, voxel_size)
        voxelized_pc = np.concatenate([voxelized_object_pc, voxelized_pc], axis=0)
        pos_normalization = np.mean(voxelized_pc, axis=0)
        # contact_rigid_pos = contact_rigid_pos - pos_normalization
        # voxelized_pc = voxelized_pc - pos_normalization # normalize

        ### filter out the gripper-cloth contact
        box_y_upper = np.max(voxelized_object_pc[:, 1])
        box_contact_idx = contact_rigid_pos[:, 1] < box_y_upper + 0.05
        contact_rigid_pos = contact_rigid_pos[box_contact_idx]
        contact_normal = contact_normal[box_contact_idx]
        contact_lambda = contact_lambda[box_contact_idx]
        contact_force = contact_lambda * (cloth_mass / (cloth_width * cloth_height))
        contact_normal_force = contact_force[:, None] * contact_normal

        ### the projection of the contact point is only to the object voxelized pointcloud
        mapping_contact_rigid_to_pc = nearest_neighbor_mapping(contact_rigid_pos, voxelized_object_pc)
        label = np.zeros(voxelized_pc.shape[0], dtype=np.int)
        label[mapping_contact_rigid_to_pc] = 1
        force_label = np.zeros((voxelized_pc.shape[0], 1), dtype=np.float)
        normal_label = np.zeros((voxelized_pc.shape[0], 3), dtype=np.float)
        force_label[mapping_contact_rigid_to_pc] = contact_force[:, None]
        normal_label[mapping_contact_rigid_to_pc] = contact_normal
        pc_cnt_object = voxelized_object_pc.shape[0]
        one_hot_encoding = np.zeros(voxelized_pc.shape[0])
        one_hot_encoding[:pc_cnt_object] = 1 # object: 1; cloth: 0

        if save_path is not None:
            if contact_rigid_pos.shape[0] > 0:
                print("save data point {}".format(data_cnt))
                data_to_store = np.concatenate([
                    voxelized_pc, one_hot_encoding.reshape(-1, 1), label.reshape(-1, 1), 
                    force_label.reshape(-1, 1), normal_label.reshape(-1, 3)
                ], axis=1)

                np.save(osp.join(save_path, 'data_{:06}'.format(data_cnt)), data_to_store)
                data_cnt += 1

        # print("voxelized point cloud shape: ", voxelized_pc.shape)

        if _ > 50 and show and _ % args.show_interval == 0:
            ax = plt.axes(projection='3d' )
            ax.scatter3D(voxelized_pc[:pc_cnt_object, 0], voxelized_pc[:pc_cnt_object, 2], voxelized_pc[:pc_cnt_object, 1], color='blue', s=0.1)
            ax.scatter3D(voxelized_pc[pc_cnt_object:, 0], voxelized_pc[pc_cnt_object:, 2], voxelized_pc[pc_cnt_object:, 1], color='green', s=0.1)
            # ax.scatter3D(contact_rigid_pos[:, 0], contact_rigid_pos[:, 2], contact_rigid_pos[:, 1], color='red')

            ax.scatter3D(voxelized_pc[mapping_contact_rigid_to_pc, 0], 
                voxelized_pc[mapping_contact_rigid_to_pc, 2], 
                voxelized_pc[mapping_contact_rigid_to_pc, 1], color='red')
            print("num soft contact {}, unique nearest neighbor mapping to pc num {}".format(
                contact_rigid_pos.shape[0], len(np.unique(mapping_contact_rigid_to_pc))
            )) 
            contact_num = contact_info.shape[0]

            X = np.concatenate([contact_cloth_pos[:, 0], contact_cloth_pos_add_force[:, 0], voxelized_pc[:, 0]])
            Y = np.concatenate([contact_cloth_pos[:, 2], contact_cloth_pos_add_force[:, 2], voxelized_pc[:, 2]])
            Z = np.concatenate([contact_cloth_pos[:, 1], contact_cloth_pos_add_force[:, 1], voxelized_pc[:, 1]])
            # for idx in range(contact_num):
            #     ax.plot([contact_cloth_pos[idx][0], contact_cloth_pos_add_force[idx][0]], 
            #         [contact_cloth_pos[idx][2], contact_cloth_pos_add_force[idx][2]], 
            #         [contact_cloth_pos[idx][1], contact_cloth_pos_add_force[idx][1]], 'red', linewidth=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_aspect('equal')
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y + max_range, mid_y - max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            # ax.invert_yaxis()

            # plt.savefig("data/haptic-perspective/visual/{}-contact-force-visual-{}-{}.png".format(robot_name, _, args.save_name))
            plt.show()
            plt.close('all')
            # plt.cla()

        # for save
        if args.save_robot_state:
            if _ % 500 == 0:
                current_joint = pyflex.get_rigid_joints()
                np.save('./data/haptic-perspective/robot_state/{}_hold_cloth_joint.npy'.format(robot_name), current_joint)
                current_particle_pos = pyflex.get_positions()
                np.save('./data/haptic-perspective/robot_state/{}_hold_cloth_particle_pos.npy'.format(robot_name), current_particle_pos)
    

# save_numpy_as_gif(np.array(all_images), osp.join('data', '{}-{}.gif'.format(
#     robot_name, save_name
# )))

