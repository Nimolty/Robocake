import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import wandb
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

matplotlib.rcParams["legend.loc"] = 'lower right'


def train_plot_curves(iters, loss, path=''):
    plt.figure(figsize=[16,9])
    plt.plot(iters, loss)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Training Loss', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()


    


def eval_plot_curves(loss_mean, loss_std, colors=['orange', 'royalblue'], 
    alpha_fill=0.3, ax=None, path=''):
    iters, loss_mean_emd, loss_mean_chamfer = loss_mean.T
    _, loss_std_emd, loss_std_chamfer = loss_std.T
    plt.figure(figsize=[16, 9])

    emd_min = loss_mean_emd - loss_std_emd
    emd_max = loss_mean_emd + loss_std_emd

    chamfer_min = loss_mean_chamfer - loss_std_chamfer
    chamfer_max = loss_mean_chamfer + loss_std_chamfer

    plt.plot(iters, loss_mean_emd, color=colors[0], linewidth=6, label='EMD')
    plt.fill_between(iters, emd_max, emd_min, color=colors[0], alpha=alpha_fill)

    plt.plot(iters, loss_mean_chamfer, color=colors[1], linewidth=6, label='Chamfer')
    plt.fill_between(iters, chamfer_max, chamfer_min, color=colors[1], alpha=alpha_fill)

    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Dyanmics Model Evaluation Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)


    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()


def visualize_points(ax, all_points, n_particles):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

    # ax.invert_yaxis()

    # mid_point = [0.5, 0.5, 0.1]
    # r = 0.25
    # ax.set_xlim(mid_point[0] - r, mid_point[0] + r)
    # ax.set_ylim(mid_point[1] - r, mid_point[1] + r)
    # ax.set_zlim(mid_point[2] - r, mid_point[2] + r)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


    return points, shapes


def plt_render(particles_set, n_particle, render_path, physics_params):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    yield_stress = "{:03d}".format(physics_params["yield_stress"])
    E = "{:.4f}".format(physics_params["E"])
    nu = "{:.4f}".format(physics_params["nu"])
    n_frames = particles_set[0].shape[0]
    rows = 3
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 9))
    row_titles = [f"GT_ys_{yield_stress}_E_{E}_nu_{nu}", 'Sample', 'Prediction']
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()

    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (
                states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (
                states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)

    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))
    plt.close()


def plt_render_frames_rm(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    # pdb.set_trace()
    n_frames = particles_set[0].shape[0]
    rows = 2
    cols = 1

    fig, big_axes = plt.subplots(rows, 1, figsize=(3, 9))
    row_titles = ['Sample', 'Prediction']
    views = [(90, 90)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    for step in range(n_frames): # n_frames
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')


def plt_render_robot(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = len(particles_set)
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, rows * 3))
    row_titles = ['Sample', 'Prediction']
    row_titles = row_titles[:rows]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        if rows == 1: 
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)
    
    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))


def visualize_points_helper(ax, all_points, n_particles, p_color='b', alpha=1.0):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c=p_color, s=10)
    shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # ax.invert_yaxis()

    return points, shapes


def plt_render_frames(particles_set, target_shape, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = 1
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 3))
    # plt.gca().invert_yaxis()
    row_titles = ['Simulator']
    # views = [(90, 90)]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        if rows == 1:
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:  
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            visualize_points_helper(ax, target_shape, n_particle, p_color='c', alpha=1.0)
            points, shapes = visualize_points_helper(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    frame_list = [n_frames - 1]
    for g in range(n_frames // (task_params['len_per_grip'] + task_params['len_per_grip_back'])):
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 12)
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 15)
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + task_params['len_per_grip'] - 1)

    for step in frame_list: # range(n_frames):
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')
        

from pdb import set_trace
##### training visualisation #####
def plt_render_training_gif(particles_set, n_particle, render_path):
    n_frames = particles_set[0].shape[0]
    rows = 2
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 6))
    row_titles = ['GT', 'Sample']
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # set_trace()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)
    
    
    set_trace()
    
    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))
    
def plt_render_image_blend(curr_shape, target_shape, n_particle, render_path):
    # curr_shape : [B, N, 3]
    # target_shape : [B, N, 3]
    
    rows = 1
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 3))
    # plt.gca().invert_yaxis()
    row_titles = ['Training']
    # views = [(90, 90)]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for b in range(1):
        for i in range(rows):
            curr_states = curr_shape[b]
            target_states = target_shape[b]
            if rows == 1:
                big_axes.set_title(row_titles[i], fontweight='semibold')
                big_axes.axis('off')
            else:  
                big_axes[i].set_title(row_titles[i], fontweight='semibold')
                big_axes[i].axis('off')
    
            plot_info = []
            for j in range(cols):
                ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
                ax.axis('off')
                ax.view_init(*views[j])
                target_points, target_shapes = visualize_points_helper(ax, target_states, n_particle, p_color='c', alpha=1.0)
                points, shapes = visualize_points_helper(ax, curr_states, n_particle)
                plot_info.append((target_points, target_shapes, points, shapes))
    
            plot_info_all[row_titles[i]] = plot_info


    for step in range(curr_shape.shape[0]): # range(n_frames):
        for i in range(rows):
            curr_states = curr_shape
            target_states = target_shape
            for j in range(cols):
                target_points, target_shapes, points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (curr_states[step, :n_particle, 0], curr_states[step, :n_particle, 2], curr_states[step, :n_particle, 1])
                shapes._offsets3d = (curr_states[step, n_particle+9:, 0], curr_states[step, n_particle+9:, 2], curr_states[step, n_particle+9:, 1])
                target_points._offsets3d = (target_states[step, :n_particle, 0], target_states[step, :n_particle, 2], target_states[step, :n_particle, 1])
                target_shapes._offsets3d = (target_states[step, n_particle+9:, 0], target_states[step, n_particle+9:, 2], target_states[step, n_particle+9:, 1])
                
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{render_path}_{str(step).zfill(3)}.pdf')

def plt_render_image_split(curr_shape, target_shape, n_particle, pstep_idx, vis_dir="visualize"):
    # curr_shape : [B, N, 3]
    # target_shape : [B, N, 3]
    all_shape = [curr_shape, target_shape]
    rows = 2
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 6))
    # plt.gca().invert_yaxis()
    row_titles = ["GT", "Pred"]
    # views = [(90, 90)]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for b in range(1):
        for i in range(rows):
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')
    
            plot_info = []
            for j in range(cols):
                ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
                # ax.axis('off')
                ax.view_init(*views[j])
                points, shapes = visualize_points_helper(ax, all_shape[i][b], n_particle)
                plot_info.append((points, shapes))
    
            plot_info_all[row_titles[i]] = plot_info



    for step in range(all_shape[0].shape[0]): # range(n_frames):
        for i in range(rows):
            # curr_states, target_states = all_shape
            curr_states = all_shape[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (curr_states[step, :n_particle, 0], curr_states[step, :n_particle, 2], curr_states[step, :n_particle, 1])
                shapes._offsets3d = (curr_states[step, n_particle+9:, 0], curr_states[step, n_particle+9:, 2], curr_states[step, n_particle+9:, 1])

                
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{vis_dir}/step_{str(pstep_idx)}_bs_{str(step)}.png')
    plt.close()

from pdb import set_trace
def eval_all_epoch(eval_root_dir, eval_intervals = ["E_00500_02000", "E_03000_06000", "E_07000_10000"], max_epoch=60):
    all_eval_dir = glob.glob(os.path.join(eval_root_dir, "*"))
    loss_dict = {}
    for eval_interval in eval_intervals:
        loss_dict[eval_interval] = {"final_frame_emd": [[] for i in range(max_epoch)], 
                                     "final_frame_emd_interval": [[] for i in range(max_epoch)],
                                     "final_frame_cd": [[] for i in range(max_epoch)], 
                                     "final_frame_cd_interval": [[] for i in range(max_epoch)],
                                     "all_frame_emd" : [[] for i in range(max_epoch)], 
                                     "all_frame_emd_interval": [[] for i in range(max_epoch)],
                                     "all_frame_cd" : [[] for i in range(max_epoch)],  
                                     "all_frame_cd_interval": [[] for i in range(max_epoch)]
                                    } 
        
        for this_eval_dir in tqdm(all_eval_dir):
            this_log_path = os.path.join(this_eval_dir, eval_interval, "eval.log")
            try:
                log_lines = []
                epoch_name = int(this_eval_dir.split('/')[-1].split('_')[3])
                print(epoch_name)
            
                # print(epoch_name)
                with open(this_log_path, "r", encoding="utf-8") as file:
                    for line in file:
                        log_lines.append(line.strip())
                # print(log_lines)
                final_frame_emd = round(float(log_lines[3].split(" ")[6]), 4)
                final_frame_emd_interval = round(float(log_lines[3].split(" ")[8][:-1]), 4)
                final_frame_cd = round(float(log_lines[4].split(" ")[6]), 4)
                final_frame_cd_interval = round(float(log_lines[4].split(" ")[8][:-1]), 4)
                
                all_frame_emd = round(float(log_lines[7].split(" ")[5]), 4)
                # print(all_frame_emd)
                # print(log_lines[7].split(" "))
                all_frame_emd_interval = round(float(log_lines[7].split(" ")[7][:-1]), 4)
                # print(all_frame_emd_interval)
                # print(log_lines)
                all_frame_cd = round(float(log_lines[8].split(" ")[5]), 4)
               #  print(all_frame_cd)
                all_frame_cd_interval = round(float(log_lines[8].split(" ")[7][:-1]), 4)
                #print(all_frame_cd_interval)
                #set_trace()
                # print(all_frame_emd)
                # print(all_frame_emd_interval)
                # print(all_frame_cd)
                # print(all_frame_cd_interval)
                loss_dict[eval_interval]["final_frame_emd"][epoch_name].append(final_frame_emd)
                loss_dict[eval_interval]["final_frame_emd_interval"][epoch_name].append(final_frame_emd_interval)
                loss_dict[eval_interval]["final_frame_cd"][epoch_name].append(final_frame_cd)
                loss_dict[eval_interval]["final_frame_cd_interval"][epoch_name].append(final_frame_cd_interval)
                loss_dict[eval_interval]["all_frame_emd"][epoch_name].append(all_frame_emd)
                loss_dict[eval_interval]["all_frame_emd_interval"][epoch_name].append(all_frame_emd_interval)
                loss_dict[eval_interval]["all_frame_cd"][epoch_name].append(all_frame_cd)
                loss_dict[eval_interval]["all_frame_cd_interval"][epoch_name].append(all_frame_cd_interval)
                # print(loss_dict)
                
            except:
                pass
    # print(loss_dict)
    return_loss_dict = {}
    for eval_interval in eval_intervals:
        return_loss_dict[eval_interval] = {"final_frame_emd": [0.0 for i in range(max_epoch)], 
                                     "final_frame_emd_interval": [0.0 for i in range(max_epoch)],
                                     "final_frame_cd": [0.0 for i in range(max_epoch)], 
                                     "final_frame_cd_interval": [0.0 for i in range(max_epoch)],
                                     "all_frame_emd" : [0.0 for i in range(max_epoch)], 
                                     "all_frame_emd_interval": [0.0 for i in range(max_epoch)],
                                     "all_frame_cd" : [0.0 for i in range(max_epoch)],  
                                     "all_frame_cd_interval": [0.0 for i in range(max_epoch)]
                                    } 
        for k in loss_dict[eval_interval]:
            # print(loss_dict[eval_interval][k])
            for i, stat in enumerate(loss_dict[eval_interval][k]):
            
                if stat:
                    return_loss_dict[eval_interval][k][i] = np.mean(stat)
                else:
                    pass
                    
    # print(return_loss_dict)
    return return_loss_dict
def draw_subplots(return_loss_dicts, eval_intervals = ["E_00500_02000", "E_03000_06000", "E_07000_10000"]):
    for eval_interval in eval_intervals:
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        # print(return_loss_dict[eval_interval])
        # print(range(len(return_loss_dict[eval_interval]["final_frame_emd"])))
        # print(return_loss_dict[eval_interval]["final_frame_emd"])
        for key, return_loss_dict in return_loss_dicts.items():
            axs[0, 0].plot(range(len(return_loss_dict[eval_interval]["final_frame_emd"])), return_loss_dict[eval_interval]["final_frame_emd"],
                          label=f"{key}_final_frame_emd"
                          )
            axs[0, 0].fill_between(range(len(return_loss_dict[eval_interval]["final_frame_emd"])),
                                  # np.array(return_loss_dict[eval_interval]["final_frame_emd"]),
                                  np.array(return_loss_dict[eval_interval]["final_frame_emd"]) - np.array(return_loss_dict[eval_interval]["final_frame_emd_interval"]),
                                  np.array(return_loss_dict[eval_interval]["final_frame_emd"]) + np.array(return_loss_dict[eval_interval]["final_frame_emd_interval"]),
                                  alpha=0.2
                                  )
                                  
            axs[0, 0].set_title('Final Frame EMD')
            axs[0, 0].legend()
            
            
            axs[0, 1].plot(range(len(return_loss_dict[eval_interval]["final_frame_cd"])), return_loss_dict[eval_interval]["final_frame_cd"],
                          label=f"{key}_final_frame_cd"
                          )
            axs[0, 1].fill_between(range(len(return_loss_dict[eval_interval]["final_frame_cd"])),
                                   # return_loss_dict[eval_interval]["final_frame_cd"],
                                   np.array(return_loss_dict[eval_interval]["final_frame_cd"]) - np.array(return_loss_dict[eval_interval]["final_frame_cd_interval"]),
                                   np.array(return_loss_dict[eval_interval]["final_frame_cd"]) + np.array(return_loss_dict[eval_interval]["final_frame_cd_interval"]),                              
                                   alpha=0.2
                                  )
            axs[0, 1].set_title('Final Frame CD')
            axs[0, 1].legend()
            
            
            axs[1, 0].plot(range(len(return_loss_dict[eval_interval]["all_frame_emd"])), return_loss_dict[eval_interval]["all_frame_emd"],
                          label=f"{key}_all_frame_emd"
                          )
            axs[1, 0].fill_between(range(len(return_loss_dict[eval_interval]["all_frame_emd"])),
                                  # np.array(return_loss_dict[eval_interval]["all_frame_emd"]),
                                  np.array(return_loss_dict[eval_interval]["all_frame_emd"]) - np.array(return_loss_dict[eval_interval]["all_frame_emd_interval"]),
                                  np.array(return_loss_dict[eval_interval]["all_frame_emd"]) + np.array(return_loss_dict[eval_interval]["all_frame_emd_interval"]),
                                  alpha=0.2
                                  
                                  )
            axs[1, 0].set_title('All Frame EMD')
            axs[1, 0].legend()
            
            
            axs[1, 1].plot(range(len(return_loss_dict[eval_interval]["all_frame_cd"])), return_loss_dict[eval_interval]["all_frame_cd"],
                          label=f"{key}_all_frame_cd"
                          )
            axs[1, 1].fill_between(range(len(return_loss_dict[eval_interval]["all_frame_cd"])),
                                # np.array(return_loss_dict[eval_interval]["all_frame_cd"]),
                                np.array(return_loss_dict[eval_interval]["all_frame_cd"]) - np.array(return_loss_dict[eval_interval]["all_frame_cd_interval"]),
                                np.array(return_loss_dict[eval_interval]["all_frame_cd"]) + np.array(return_loss_dict[eval_interval]["all_frame_cd_interval"]),
                                alpha=0.2
                                )
            axs[1, 1].set_title('All Frame CD')
            axs[1, 1].legend()
        
    
        plt.tight_layout()
    
    
        plt.savefig(f"{eval_interval}.png")

        
        
    
    

if __name__ == "__main__":
    pass
#    curr_shape = np.random.random((4, 331, 3)) 
#    target_shape = np.random.random((4, 331, 3))
#    # print(pts1.shape)
#    # particles_set = [pts1, pts2]
#    n_particle = 300
#    render_path = f"./plt"
#    # plt_render_training(particles_set, n_particle, render_path)
#    plt_render_image_split(curr_shape, target_shape, n_particle, render_path)
    eval_root_dir_1 = f"/nvme/tianyang/residual_robocake_data/dump_ngrip_fixed_prior_multi/eval_0"
    eval_root_dir_2 = f"/nvme/tianyang/residual_robocake_data/dump_ngrip_fixed/eval_0"
    loss_dict_1 = eval_all_epoch(eval_root_dir_1)
    loss_dict_2 = eval_all_epoch(eval_root_dir_2)
    draw_subplots({"multi_material_prior": loss_dict_1, "multi_material_residual": loss_dict_2})










