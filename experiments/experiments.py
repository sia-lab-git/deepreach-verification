import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def validate(self, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s) 
        fig.savefig(save_path)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def train(
            self, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            double_precision = False, loss_schedules = None, val_dataloader = None
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if adjust_relative_grads:
            new_weight = 1

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            for epoch in range(0, epochs):
                if not epoch % epochs_til_checkpoint and epoch:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = { 
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                        np.array(train_losses))
                    self.validate(
                        save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    if double_precision:
                        model_input = {key: value.double() for key, value in model_input.items()}
                        gt = {key: value.double() for key, value in gt.items()}

                    model_results = self.model({'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks)
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)
                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                        if val_dataloader is not None:
                            print("Running validation set...")
                            self.model.eval()
                            with torch.no_grad():
                                val_losses = []
                                for (model_input, gt) in val_dataloader:
                                    model_results = self.model({'coords': model_input['model_coords']})

                                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).detach()
                                    boundary_values = gt['boundary_values']
                                    dirichlet_masks = gt['dirichlet_masks']

                                    val_loss_components = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks)
                                    val_loss = 0
                                    for loss_name, loss in val_loss_components.items():
                                        val_loss += loss.mean()
                                    val_losses.append(val_loss)

                                writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            self.model.train()

                    total_steps += 1

            torch.save(self.model.state_dict(),
                    os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                    np.array(train_losses))

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint / checkpoint_dt)%sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin, self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[j], dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i, j] = results['max_violation_error']
                        BRT_error_rates_matrix[i, j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j], 
                            sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN 
                        BRT_error_rates_matrix[i, j] = np.NaN 
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[j], dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
                        sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
                        violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i, j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i, j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j], 
                            sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
                            target_validator=ValueThresholdValidator(v_min=0.0, v_max=results['max_violation_error']), 
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN 
                        exBRT_error_rates_matrix[i, j] = np.NaN 
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN 
            
            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)
            
            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y,x),label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' % label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y,x),label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' % label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            if data_step == 'plot_violations':
                # plot violations on slice
                plot_config = dynamics.plot_config()
                slices = plot_config['state_slices']
                slices[plot_config['x_axis_idx']] = None
                slices[plot_config['y_axis_idx']] = None
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=100000, sample_batch_size=100000, 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=slices), 
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                plt.title('violations for slice = %s' % plot_config['state_slices'], fontsize=8)
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][~results['violations']], results['states'][...,  plot_config['y_axis_idx']][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][results['violations']], results['states'][...,  plot_config['y_axis_idx']][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                x_min, x_max = dynamics.state_test_range()[plot_config['x_axis_idx']]
                y_min, y_max = dynamics.state_test_range()[plot_config['y_axis_idx']]
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.savefig(os.path.join(testing_dir, f'violations.png'), dpi=800)
                plt.clf()

                # plot distribution of violations over state variables
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=100000, sample_batch_size=100000, 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim), 
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                for i in range(dynamics.state_dim):
                    plt.title('violations over %s' % plot_config['state_labels'][i])
                    plt.scatter(results['states'][..., i][~results['violations']], results['values'][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                    plt.scatter(results['states'][..., i][results['violations']], results['values'][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                    plt.savefig(os.path.join(testing_dir, f'violations_over_state_dim_{i}.png'), dpi=800)
                    plt.clf()

                # # plot distribution of violations over car distances, for multivehiclecollision
                # for i in range(3):
                #     for j in range(3):
                #         if i == j:
                #             continue
                #         plt.title(f'violations over distance between cars {i}, {j}')
                #         plt.scatter(torch.norm(results['states'][..., i*2:i*2+2] - results['states'][..., j*2:j*2+2], dim=-1)[~results['violations']], results['values'][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                #         plt.scatter(torch.norm(results['states'][..., i*2:i*2+2] - results['states'][..., j*2:j*2+2], dim=-1)[results['violations']], results['values'][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                #         plt.savefig(os.path.join(testing_dir, f'violations_over_cars_{i}_{j}.png'), dpi=800)
                #         plt.clf()

            if data_step == 'run_basic_recovery':
                logs = {}
                
                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float('inf') if dynamics.set_mode == 'reach' else float('-inf')
                algorithm_iters = []
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(
                        model=model, dynamics=dynamics, 
                        tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000), 
                        sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')), 
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float('-inf') if dynamics.set_mode == 'reach' else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()
                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(results['valid_sample_fraction'].item()))
                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level
                
                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    target_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    target_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')), 
                    num_samples=S,
                    batch_size=min(S, 1000000)
                ).item()

                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(logs['theoretically_recoverable_volume']))
                
                # 3. validate theoretical guarantees via mass sampling                   
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(logs['recovered_violation_rate']))
                 
                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(logs['recovered_violation_rate']))

                # 1. for ground truth slices (if available), record (higher-res) grid of learned values
                # plot (with ground truth) learned BRTs, recovered BRTs
                z_res = 5
                plot_config = dataset.dynamics.plot_config()
                if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
                    ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
                    if 'gmat' in ground_truth:
                        ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                        ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                        ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                        ground_truth_values = ground_truth['data']
                        ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
                    elif 'g' in ground_truth:
                        ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                        ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                        ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                        ground_truth_ts = ground_truth['tau'][0]
                        ground_truth_values = ground_truth['data']

                    # idxs to plot
                    x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
                    y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
                    z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
                    t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

                    # indexed ground truth to plot
                    ground_truth_xs = ground_truth_xs[x_idxs]
                    ground_truth_ys = ground_truth_ys[y_idxs]
                    ground_truth_zs = ground_truth_zs[z_idxs]
                    ground_truth_ts = ground_truth_ts[t_idxs]
                    ground_truth_values = ground_truth_values[
                        x_idxs[:, None, None, None], 
                        y_idxs[None, :, None, None], 
                        z_idxs[None, None, :, None],
                        t_idxs[None, None, None, :]
                    ]
                    ground_truth_grids = ground_truth_values

                    xs = ground_truth_xs
                    ys = ground_truth_ys
                    zs = ground_truth_zs
                else:
                    ground_truth_grids = None
                    resolution = 512
                    xs = np.linspace(*dynamics.state_test_range()[plot_config['x_axis_idx']], resolution)
                    ys = np.linspace(*dynamics.state_test_range()[plot_config['y_axis_idx']], resolution)
                    zs = np.linspace(*dynamics.state_test_range()[plot_config['z_axis_idx']], z_res)
                
                xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
                value_grids = np.zeros((len(zs), len(xs), len(ys)))
                for i in range(len(zs)):
                    coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
                    coords[:, 0] = dataset.tMax
                    coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

                    model_results = model({'coords': dataset.dynamics.coord_to_input(coords.cuda())})
                    values = dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
                    value_grids[i] = values.reshape(len(xs), len(ys))
                
                def overlay_ground_truth(image, z_idx):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
                    ground_truth_brts = ground_truth_grid < 0
                    for x in range(ground_truth_brts.shape[0]):
                        for y in range(ground_truth_brts.shape[1]):
                            if not ground_truth_brts[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                                    if not ground_truth_brts[neighbor]:
                                        image[x-thickness:x+1, y-thickness:y+1+thickness] = np.array([50, 50, 50])
                                        break

                def overlay_border(image, set, color):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    for x in range(set.shape[0]):
                        for y in range(set.shape[1]):
                            if not set[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                                    if not set[neighbor]:
                                        image[x-thickness:x+1, y-thickness:y+1+thickness] = color
                                        break

                fig = plt.figure()
                fig.suptitle(plot_config['state_slices'], fontsize=8)
                for i in range(len(zs)):
                    values = value_grids[i]
                    
                    # learned BRT and recovered BRT
                    ax = fig.add_subplot(1, len(zs), (i+1))
                    ax.set_title('%s = %0.2f' % (plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

                    image = np.full((*values.shape, 3), 255, dtype=int)
                    BRT = values < 0
                    recovered_BRT = values < delta_level
                    
                    if dynamics.set_mode == 'reach':
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        image[recovered_BRT] = np.array([155, 241, 249])
                        overlay_border(image, recovered_BRT, np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)
                    else:
                        image[recovered_BRT] = np.array([155, 241, 249])       
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        overlay_border(image, recovered_BRT, np.array([15, 223, 240])) # overlay recovered border over learned BRT
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)

                    ax.imshow(image.transpose(1, 0, 2), origin='lower', extent=(-1., 1., -1., 1.))
                    ax.set_xlabel(plot_config['state_labels'][plot_config['x_axis_idx']])
                    ax.set_ylabel(plot_config['state_labels'][plot_config['y_axis_idx']])
                    ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                    ax.tick_params(labelsize=6)
                    if i != 0:
                        ax.set_yticks([])            

                    # # plot trajectories
                    # n_trajs = 20
                    # slices = plot_config['state_slices']
                    # slices[plot_config['x_axis_idx']] = None
                    # slices[plot_config['y_axis_idx']] = None
                    # slices[plot_config['z_axis_idx']] = zs[i]
                    # results = scenario_optimization(
                    #     model=model, dynamics=dynamics, 
                    #     tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    #     set_type=set_type, control_type=control_type, 
                    #     scenario_batch_size=n_trajs, sample_batch_size=10*n_trajs, 
                    #     sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=slices),
                    #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    #     max_scenarios=n_trajs, max_samples=1000*n_trajs)
                    # ranges = torch.tensor(dynamics.state_test_range())
                    # batch_state_trajs = ((results['batch_state_trajs'] - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) * 2) - 1
                    # ax.add_patch(matplotlib.patches.Rectangle(((-20+150)/300*2-1,-1), 40/300*2, 10/140*2, color=(1/255, 209/255, 32/255), linewidth=0.6, fill=True))
                    # for state_traj in batch_state_trajs:
                    #     # plt.plot(state_traj[:, plot_config['x_axis_idx']], state_traj[:, plot_config['y_axis_idx']], color=(213/255, 122/255, 252/255), linewidth=0.4)
                    #     # specifically for rocketlanding
                    #     cutoff = torch.argmax(1*(state_traj[:, 1] < ((20-10)/(150-10)*2-1))*(torch.abs(state_traj[:, 0]) < 20/150))
                    #     plt.plot(state_traj[:cutoff+1, plot_config['x_axis_idx']], state_traj[:cutoff+1, plot_config['y_axis_idx']], color='black', linewidth=0.4)
                    #     plt.scatter(state_traj[:1, plot_config['x_axis_idx']], state_traj[:1, plot_config['y_axis_idx']], color='black', s=0.08)
                    # # bad_results = scenario_optimization(
                    # #     model=model, dynamics=dynamics, 
                    # #     tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    # #     set_type=set_type, control_type=control_type, 
                    # #     scenario_batch_size=100000, sample_batch_size=100000, 
                    # #     sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=slices),
                    # #     sample_validator=ValueThresholdValidator(v_min=delta_level, v_max=0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0, v_max=delta_level),
                    # #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    # #     max_violations=1, max_scenarios=10*100000, max_samples=1000*100000)
                    # # if bad_results['maxed_violations']:
                    # #     print('found violation trajectory to plot')
                    # #     batch_state_trajs = ((bad_results['batch_state_trajs'] - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) * 2) - 1
                    # #     bad_state_trajs = batch_state_trajs[bad_results['violations'][-100000:]]
                    # #     bad_state_traj = bad_state_trajs[0]
                    # #     plt.plot(bad_state_traj[:, plot_config['x_axis_idx']], bad_state_traj[:, plot_config['y_axis_idx']], color='red', linewidth=0.4)
                    # #     plt.scatter(bad_state_traj[:, plot_config['x_axis_idx']], bad_state_traj[:, plot_config['y_axis_idx']], color='red', s=0.08)

                plt.tight_layout()
                fig.savefig(os.path.join(testing_dir, f'basic_BRTs.png'), dpi=800)
                
            if data_step == 'collect_samples':
                logs = {}

                # 1. record 10M state, learned value, violation
                P = int(1e7)
                logs['P'] = P
                print('collecting training samples')
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=min(P, 100000), sample_batch_size=10*min(P, 10000), 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=P, max_samples=1000*min(P, 10000))
                logs['training_samples'] = {
                    'states': results['states'],
                    'values': results['values'],
                    'violations': results['violations'],
                }
                with open(os.path.join(testing_dir, 'sample_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'train_binner':
                with open(os.path.join(self.experiment_dir, 'sample_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 1. train MLP predictor
                # plot validation of MLP predictor
                def validate_predictor(predictor, epoch):
                    print('validating predictor at epoch', str(epoch))
                    predictor.eval()

                    results = scenario_optimization(
                        model=model, dynamics=dynamics, 
                        tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=100000, sample_batch_size=100000, 
                        sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=100000, max_samples=1000000)
                    
                    inputs = torch.cat((results['states'], results['values'][..., None]), dim=-1)
                    preds = torch.sigmoid(predictor(inputs.cuda())).detach().cpu().numpy()

                    plt.title(f'Predictor Validation at Epoch {epoch}')
                    plt.ylabel('Value')
                    plt.xlabel('Prediction')
                    plt.scatter(preds[~results['violations']], results['values'][~results['violations']], color='blue', label='nonviolations', alpha=0.1)
                    plt.scatter(preds[results['violations']], results['values'][results['violations']], color='red', label='violations', alpha=0.1)
                    plt.legend()
                    plt.savefig(os.path.join(testing_dir, f'predictor_validation_at_epoch_{epoch}.png'), dpi=800)
                    plt.clf()

                    predictor.train()
                
                print('training predictor')
                violation_scale = 5
                violation_weight = 1.5
                states = logs['training_samples']['states']
                values = logs['training_samples']['values']
                violations = logs['training_samples']['violations']
                violation_strengths = torch.where(violations, (torch.max(values[violations]) - values) if dynamics.set_mode == 'reach' else (values - torch.min(values[violations])), torch.tensor([0.0])).cuda()
                violation_scales = torch.exp(violation_scale * violation_strengths / torch.max(violation_strengths))

                plt.title(f'Violation Scales')
                plt.ylabel('Frequency')
                plt.xlabel('Scale')
                plt.hist(violation_scales.cpu().numpy(), range=(0, 10))
                plt.savefig(os.path.join(testing_dir, f'violation_scales.png'), dpi=800)
                plt.clf()

                inputs = torch.cat((states, values[..., None]), dim=-1).cuda()
                outputs = 1.0*violations.cuda()
                # outputs = violation_strengths / torch.max(violation_strengths)

                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.cuda()
                predictor.train()

                lr = 0.00005
                lr_decay = 0.2
                decay_patience = 20
                decay_threshold = 1e-12
                opt = torch.optim.Adam(predictor.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=lr_decay, patience=decay_patience, threshold=decay_threshold)

                pos_weight = violation_weight*((outputs <= 0).sum() / (outputs > 0).sum())

                n_epochs = 1000
                batch_size = 100000
                for epoch in range(n_epochs):
                    idxs = torch.randperm(len(outputs))
                    for batch in range(math.ceil(len(outputs) / batch_size)):
                        batch_idxs = idxs[batch*batch_size : (batch+1)*batch_size]
            
                        BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(weight=violation_scales[batch_idxs], pos_weight=pos_weight)
                        loss = BCEWithLogitsLoss(predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        # MSELoss = torch.nn.MSELoss()
                        # loss = MSELoss(predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        loss.backward()
                        opt.step()
                    print(f'Epoch {epoch}: loss: {loss.item()}')
                    sched.step(loss.item())

                    if (epoch+1)%100 == 0:
                        torch.save(predictor.state_dict(), os.path.join(testing_dir, f'predictor_at_epoch_{epoch}.pth'))
                        validate_predictor(predictor, epoch)
                logs['violation_scale'] = violation_scale
                logs['violation_weight'] = violation_weight
                logs['n_epochs'] = n_epochs
                logs['lr'] = lr
                logs['lr_decay'] = lr_decay
                logs['decay_patience'] = decay_patience
                logs['decay_threshold'] = decay_threshold

                with open(os.path.join(testing_dir, 'train_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'run_binned_recovery':
                logs = {}
                
                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                
                # 1. execute algorithm for each bin of MLP predictor
                epoch = 699
                logs['epoch'] = epoch
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.train()

                bins = [0, 0.8, 0.85, 0.9, 0.95, 1]
                logs['bins'] = bins
                
                binned_delta_levels = []
                for i in range(len(bins)-1):
                    print('bin', str(i))
                    binned_delta_level = float('inf') if dynamics.set_mode == 'reach' else float('-inf')
                    for j in range(M):
                        print('algorithm iter', str(j))
                        results = scenario_optimization(
                            model=model, dynamics=dynamics, 
                            tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                            set_type=set_type, control_type=control_type, 
                            scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000), 
                            sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                            sample_validator=MultiValidator([
                                MLPValidator(mlp=predictor, o_min=bins[i], o_max=bins[i+1], model=model, dynamics=dynamics),
                                ValueThresholdValidator(v_min=float('-inf'), v_max=binned_delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=binned_delta_level, v_max=float('inf')), 
                            ]),
                            violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                            max_scenarios=N, max_samples=100000*min(N, 10000))
                        if not results['maxed_scenarios']:
                            binned_delta_level = float('-inf') if dynamics.set_mode == 'reach' else float('inf')
                            break
                        if results['violation_rate'] == 0:
                            break
                        violation_levels = results['values'][results['violations']]
                        binned_delta_level_arg = np.argmin(violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                        binned_delta_level = violation_levels[binned_delta_level_arg].item()
                        print('violation_rate:', str(results['violation_rate']))
                        print('binned_delta_level:', str(binned_delta_level))
                        print('valid_sample_fraction:', str(results['valid_sample_fraction'].item()))
                    binned_delta_levels.append(binned_delta_level)
                logs['binned_delta_levels'] = binned_delta_levels

                # 2. record solution volume, auto-binned recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    target_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['binned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    target_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')), 
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0
                print('learned_volume', str(logs['learned_volume']))
                print('binned_volume', str(logs['binned_volume']))
                print('theoretically_recoverable_volume', str(logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics, 
                    tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                    set_type=set_type, control_type=control_type, 
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                    sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['binned_violation_rate'] = results['violation_rate']
                else:
                    logs['binned_violation_rate'] = 0           
                print('binned_violation_rate', str(logs['binned_violation_rate']))
                
                with open(os.path.join(testing_dir, 'binned_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_binned_recovery':
                with open(os.path.join(self.experiment_dir, 'binned_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                # print('P:', str(logs['P']))
                print('S:', str(logs['S']))
                print('bins', str(logs['bins']))
                bins = logs['bins']
                print('binned delta levels', str(logs['binned_delta_levels']))
                binned_delta_levels = logs['binned_delta_levels']
                print('learned volume', str(logs['learned_volume']))
                print('binned volume', str(logs['binned_volume']))
                print('theoretically recoverable volume', str(logs['theoretically_recoverable_volume']))
                print('binned violation rate', str(logs['binned_violation_rate']))

                epoch = logs['epoch']
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.eval()

                # 1. for ground truth slices (if available), record (higher-res) grid of learned values and MLP predictions
                # plot (with ground truth) learned BRTs, auto-binned recovered BRTs
                # plot MLP predictor bins
                z_res = 5
                plot_config = dataset.dynamics.plot_config()
                if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
                    ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
                    if 'gmat' in ground_truth:
                        ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                        ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                        ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                        ground_truth_values = ground_truth['data']
                        ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
                    elif 'g' in ground_truth:
                        ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                        ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                        ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                        ground_truth_ts = ground_truth['tau'][0]
                        ground_truth_values = ground_truth['data']

                    # idxs to plot
                    x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
                    y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
                    z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
                    t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

                    # indexed ground truth to plot
                    ground_truth_xs = ground_truth_xs[x_idxs]
                    ground_truth_ys = ground_truth_ys[y_idxs]
                    ground_truth_zs = ground_truth_zs[z_idxs]
                    ground_truth_ts = ground_truth_ts[t_idxs]
                    ground_truth_values = ground_truth_values[
                        x_idxs[:, None, None, None], 
                        y_idxs[None, :, None, None], 
                        z_idxs[None, None, :, None],
                        t_idxs[None, None, None, :]
                    ]
                    ground_truth_grids = ground_truth_values
                    xs = ground_truth_xs
                    ys = ground_truth_ys
                    zs = ground_truth_zs

                else:
                    ground_truth_grids = None
                    resolution = 512
                    xs = np.linspace(*dynamics.state_test_range()[plot_config['x_axis_idx']], resolution)
                    ys = np.linspace(*dynamics.state_test_range()[plot_config['y_axis_idx']], resolution)
                    zs = np.linspace(*dynamics.state_test_range()[plot_config['z_axis_idx']], z_res)

                xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
                value_grids = np.zeros((len(zs), len(xs), len(ys)))
                prediction_grids = np.zeros((len(zs), len(xs), len(ys)))
                for i in range(len(zs)):
                    coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
                    coords[:, 0] = dataset.tMax
                    coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

                    model_results = model({'coords': dataset.dynamics.coord_to_input(coords.cuda())})
                    values = dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
                    value_grids[i] = values.reshape(len(xs), len(ys))

                    inputs = torch.cat((coords[..., 1:], values[:, None]), dim=-1)
                    outputs = torch.sigmoid(predictor(inputs.cuda()).cpu().squeeze(dim=-1))
                    prediction_grids[i] = outputs.reshape(len(xs), len(ys)).detach().cpu()
                
                def overlay_ground_truth(image, z_idx):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
                    ground_truth_BRTs = ground_truth_grid < 0
                    for x in range(ground_truth_BRTs.shape[0]):
                        for y in range(ground_truth_BRTs.shape[1]):
                            if not ground_truth_BRTs[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_BRTs.shape[0] and neighbor[1] < ground_truth_BRTs.shape[1]:
                                    if not ground_truth_BRTs[neighbor]:
                                        image[x-thickness:x+1+thickness, y-thickness:y+1+thickness] = np.array([50, 50, 50])
                                        break

                def overlay_border(image, set, color):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    for x in range(set.shape[0]):
                        for y in range(set.shape[1]):
                            if not set[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                                    if not set[neighbor]:
                                        image[x-thickness:x+1, y-thickness:y+1+thickness] = color
                                        break

                fig = plt.figure()
                fig.suptitle(plot_config['state_slices'], fontsize=8)
                for i in range(len(zs)):
                    values = value_grids[i]
                    predictions = prediction_grids[i]

                    # learned BRT and bin-recovered BRT
                    ax = fig.add_subplot(1, len(zs), (i+1))
                    ax.set_title('%s = %0.2f' % (plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

                    image = np.full((*values.shape, 3), 255, dtype=int)
                    BRT = values < 0
                    bin_recovered_BRT = np.full(values.shape, 0, dtype=bool)
                    # per bin, set accordingly
                    for j in range(len(bins)-1):
                        mask = ((predictions >= bins[j])*(predictions < bins[j+1]))
                        binned_delta_level = binned_delta_levels[j]
                        bin_recovered_BRT[mask*(values < binned_delta_level)] = True
                    
                    if dynamics.set_mode == 'reach':
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        overlay_border(image, bin_recovered_BRT, np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)
                    else:
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        overlay_border(image, bin_recovered_BRT, np.array([15, 223, 240])) # overlay recovered border over learned BRT
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)

                    ax.imshow(image.transpose(1, 0, 2), origin='lower', extent=(-1., 1., -1., 1.))
                    ax.set_xlabel(plot_config['state_labels'][plot_config['x_axis_idx']])
                    ax.set_ylabel(plot_config['state_labels'][plot_config['y_axis_idx']])
                    ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                    ax.tick_params(labelsize=6)
                    if i != 0:
                        ax.set_yticks([])
                plt.tight_layout()
                fig.savefig(os.path.join(testing_dir, f'binned_BRTs.png'), dpi=800)

            if data_step == 'plot_cost_function':
                if os.path.exists(os.path.join(self.experiment_dir, 'cost_logs.pickle')):
                    with open(os.path.join(self.experiment_dir, 'cost_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                else:
                    with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                    S = logs['S']
                    delta_level = logs['delta_level']
                    results = scenario_optimization(
                        model=model, dynamics=dynamics, 
                        tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                        sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['learned_costs'] = results['costs']
                    else:
                        logs['learned_costs'] = None

                    results = scenario_optimization(
                        model=model, dynamics=dynamics, 
                        tMin=dataset.tMin, t=dataset.tMax, dt=dt, 
                        set_type=set_type, control_type=control_type, 
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000), 
                        sample_generator=SliceSampleGenerator(dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['recovered_costs'] = results['costs']
                    else:
                        logs['recovered_costs'] = None

                if logs['learned_costs'] is not None and logs['recovered_costs'] is not None:
                    plt.title(f'Trajectory Costs')
                    plt.ylabel('Frequency')
                    plt.xlabel('Cost')
                    plt.hist(logs['learned_costs'], color=(247/255, 187/255, 8/255), alpha=0.5)
                    plt.hist(logs['recovered_costs'], color=(14/255, 222/255, 241/255), alpha=0.5)
                    plt.axvline(x=0, linestyle='--', color='black')
                    plt.savefig(os.path.join(testing_dir, f'cost_function.png'), dpi=800)

                with open(os.path.join(testing_dir, 'cost_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)
                
            # # error and validation error versus simulation time
            # print('sampling across simulation time')
            # times = np.linspace(self.dataset.tMin, self.dataset.tMax, num=11)

            # BRT_results_over_time = []
            # BRT_error_val_results_over_time = []
            # BRT_value_val_results_over_time = []
            # exBRT_results_over_time = []
            # exBRT_error_val_results_over_time = []
            # exBRT_value_val_results_over_time = []
            # for i in range(len(times)):
            #     print('time', times[i])
            #     # BRT
            #     results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #         violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #     if results['maxed_scenarios']:
            #         BRT_results_over_time.append(
            #             {
            #                 'violation_rate': results['violation_rate'],
            #                 'max_violation_error': results['max_violation_error'],
            #                 'max_violation_value_mag': results['max_violation_value_mag'],
            #             }
            #         )
            #         # # plot BRT, value function error within BRT, BRT error states in 3D
            #         # states = results['states']
            #         # values = results['values']
            #         # true_values = results['true_values']

            #         # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=values, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'BRT_value_function_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=true_values-values, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'BRT_value_function_errors_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # violations = results['violations']     
            #         # fig = px.scatter_3d(x=states[violations, 0], y=states[violations, 1], z=states[violations, 2], color=(true_values-values)[violations], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'BRT_errors_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # # distributions over dv
            #         # coords = torch.cat((torch.full((*states.shape[:-1], 1), times[i]), states), dim=-1)
            #         # model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         # model_results = self.model({'coords': model_coords.cuda()})
            #         # model_dvs = diff_operators.jacobian(model_results['model_out'], model_results['model_in'])[0].detach().cpu()

            #         # # data to plot

            #         # model_dvdts = model_dvs[..., 0, 0]
            #         # model_dvds_norms = torch.norm(model_dvs[..., 0, 1:], dim=-1)

            #         # dvdts = self.dataset.dynamics.unnormalize_dvdt(model_dvdts)
            #         # dvds_norms = torch.norm(self.dataset.dynamics.unnormalize_dvds(model_dvs[..., 0, 1:]), dim=-1)

            #         # values = self.dataset.dynamics.output_to_value(model_results['model_out'][:, 0]).detach().cpu()
            #         # value_errors = results['true_values'] - values
            #         # assert torch.all(values == results['values'])

            #         # xs = {
            #         #     'dvds_norm': dvds_norms,
            #         #     'model_dvds_norm': model_dvds_norms,
            #         #     'dvdt': dvdts,
            #         #     'model_dvdt': model_dvdts,
            #         # }
            #         # ys = {
            #         #     'value_error': value_errors,
            #         #     'value': values,
            #         # }
                    
            #         # # scatter every combination for all sampled states
            #         # for x_name, x in xs.items():
            #         #     for y_name, y in ys.items():
            #         #         plt.scatter(x[~results['violations']], y[~results['violations']], label='nonviolations', s=0.2, color='blue')
            #         #         plt.scatter(x[results['violations']], y[results['violations']], label='violations', s=0.2, color='red')
            #         #         plt.axvline(torch.mean(x), label='BRT mean', linewidth=1, color='black')
            #         #         plt.axvline(torch.mean(x[~results['violations']]), label='nonviolations mean', linewidth=1, color='blue')
            #         #         plt.axvline(torch.mean(x[results['violations']]), label='violations mean', linewidth=1, color='red')
            #         #         plt.axhline(0, linestyle='dashed', linewidth=1, color='green')
            #         #         if y_name == 'value_error':
            #         #             plt.axhline(results['max_violation_error'], label='max violation error', linewidth=1, color='green')
            #         #         elif y_name == 'value':
            #         #             plt.axhline(-results['max_violation_error'], label='level of correction', linewidth=1, color='green')
            #         #         plt.title(f'{num_scenarios} Randomly Sampled BRT States')
            #         #         plt.xlabel(x_name)
            #         #         plt.ylabel(y_name)
            #         #         plt.legend(loc='lower left', fontsize=6)
            #         #         plt.savefig(os.path.join(testing_dir, f'{y_name}_VS_{x_name}_in_BRT_at_t{times[i]:.1f}_for_{checkpoint_toload}.png'), dpi=800)
            #         #         plt.clf()

            #         # # plot value vs dvd(t,s) norm, model norm
            #         # # plot value vs dvdx for x in (t, *s)

            #         # error-corrected
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #             sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=-BRT_results_over_time[-1]['max_violation_error']), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             BRT_error_val_results_over_time.append(
            #                 {
            #                     'violation_rate': results['violation_rate'],
            #                     'max_violation_error': results['max_violation_error'],
            #                     'max_violation_value_mag': results['max_violation_value_mag'],
            #                 }
            #             )
            #         else:
            #             BRT_error_val_results_over_time.append(
            #                 {
            #                     'violation_rate': np.NaN,
            #                     'max_violation_error': np.NaN,
            #                     'max_violation_value_mag': np.NaN,
            #                 }
            #             )

            #         # value-corrected
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #             sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=-BRT_results_over_time[-1]['max_violation_value_mag']), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             BRT_value_val_results_over_time.append(
            #                 {
            #                     'violation_rate': results['violation_rate'],
            #                     'max_violation_error': results['max_violation_error'],
            #                     'max_violation_value_mag': results['max_violation_value_mag'],
            #                 }
            #             )
            #         else:
            #             BRT_value_val_results_over_time.append(
            #                 {
            #                     'violation_rate': np.NaN,
            #                     'max_violation_error': np.NaN,
            #                     'max_violation_value_mag': np.NaN,
            #                 }
            #             )
            #     else:
            #         BRT_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )
            #         BRT_error_val_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )
            #         BRT_value_val_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )
                

            #     # exBRT
            #     results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #     if results['maxed_scenarios']:
            #         exBRT_results_over_time.append(
            #             {
            #                 'violation_rate': results['violation_rate'],
            #                 'max_violation_error': results['max_violation_error'],
            #                 'max_violation_value_mag': results['max_violation_value_mag'],
            #             }
            #         )
            #         # # plot exBRT, value function error within exBRT, exBRT error states in 3D
            #         # states = results['states']
            #         # values = results['values']
            #         # true_values = results['true_values']

            #         # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=values, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'exBRT_value_function_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=values-true_values, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'exBRT_value_function_errors_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # violations = results['violations']     
            #         # fig = px.scatter_3d(x=states[violations, 0], y=states[violations, 1], z=states[violations, 2], color=(values-true_values)[violations], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            #         # fig.write_html(os.path.join(testing_dir, f'exBRT_errors_at_t{times[i]:.1f}_for_{checkpoint_toload}.html'))  

            #         # # distributions over dv
            #         # coords = torch.cat((torch.full((*states.shape[:-1], 1), times[i]), states), dim=-1)
            #         # model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         # model_results = self.model({'coords': model_coords.cuda()})
            #         # model_dvs = diff_operators.jacobian(model_results['model_out'], model_results['model_in'])[0].detach().cpu()

            #         # # data to plot

            #         # model_dvdts = model_dvs[..., 0, 0]
            #         # model_dvds_norms = torch.norm(model_dvs[..., 0, 1:], dim=-1)

            #         # dvdts = self.dataset.dynamics.unnormalize_dvdt(model_dvdts)
            #         # dvds_norms = torch.norm(self.dataset.dynamics.unnormalize_dvds(model_dvs[..., 0, 1:]), dim=-1)

            #         # values = self.dataset.dynamics.output_to_value(model_results['model_out'][:, 0]).detach().cpu()
            #         # value_errors = values - results['true_values']
            #         # assert torch.all(values == results['values'])

            #         # xs = {
            #         #     'dvds_norm': dvds_norms,
            #         #     'model_dvds_norm': model_dvds_norms,
            #         #     'dvdt': dvdts,
            #         #     'model_dvdt': model_dvdts,
            #         # }
            #         # ys = {
            #         #     'value_error': value_errors,
            #         #     'value': values,
            #         # }
                    
            #         # # scatter every combination for all sampled states
            #         # for x_name, x in xs.items():
            #         #     for y_name, y in ys.items():
            #         #         plt.scatter(x[~results['violations']], y[~results['violations']], label='nonviolations', s=0.2, color='blue')
            #         #         plt.scatter(x[results['violations']], y[results['violations']], label='violations', s=0.2, color='red')
            #         #         plt.axvline(torch.mean(x), label='exBRT mean', linewidth=1, color='black')
            #         #         plt.axvline(torch.mean(x[~results['violations']]), label='nonviolations mean', linewidth=1, color='blue')
            #         #         plt.axvline(torch.mean(x[results['violations']]), label='violations mean', linewidth=1, color='red')
            #         #         plt.axhline(0, linestyle='dashed', linewidth=1, color='green')
            #         #         if y_name == 'value_error':
            #         #             plt.axhline(results['max_violation_error'], label='max violation error', linewidth=1, color='green')
            #         #         elif y_name == 'value':
            #         #             plt.axhline(results['max_violation_error'], label='level of correction', linewidth=1, color='green')
            #         #         plt.title(f'{num_scenarios} Randomly Sampled exBRT States')
            #         #         plt.xlabel(x_name)
            #         #         plt.ylabel(y_name)
            #         #         plt.legend(loc='lower left', fontsize=6)
            #         #         plt.savefig(os.path.join(testing_dir, f'{y_name}_VS_{x_name}_in_exBRT_at_t{times[i]:.1f}_for_{checkpoint_toload}.png'), dpi=800)
            #         #         plt.clf()

            #         # # plot value vs dvd(t,s) norm, model norm
            #         # # plot value vs dvdx for x in (t, *s)

            #         # error-corrected
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #             sample_validator=ValueThresholdValidator(v_min=exBRT_results_over_time[-1]['max_violation_error'], v_max=float('inf')), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             exBRT_error_val_results_over_time.append(
            #                 {
            #                     'violation_rate': results['violation_rate'],
            #                     'max_violation_error': results['max_violation_error'],
            #                     'max_violation_value_mag': results['max_violation_value_mag'],
            #                 }
            #             )
            #         else:
            #             exBRT_error_val_results_over_time.append(
            #                 {
            #                     'violation_rate': np.NaN,
            #                     'max_violation_error': np.NaN,
            #                     'max_violation_value_mag': np.NaN,
            #                 }
            #             )

            #         # value-corrected
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #             sample_validator=ValueThresholdValidator(v_min=exBRT_results_over_time[-1]['max_violation_value_mag'], v_max=float('inf')), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             exBRT_value_val_results_over_time.append(
            #                 {
            #                     'violation_rate': results['violation_rate'],
            #                     'max_violation_error': results['max_violation_error'],
            #                     'max_violation_value_mag': results['max_violation_value_mag'],
            #                 }
            #             )
            #         else:
            #             exBRT_value_val_results_over_time.append(
            #                 {
            #                     'violation_rate': np.NaN,
            #                     'max_violation_error': np.NaN,
            #                     'max_violation_value_mag': np.NaN,
            #                 }
            #             )
            #     else:
            #         exBRT_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )
            #         exBRT_error_val_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )
            #         exBRT_value_val_results_over_time.append(
            #             {
            #                 'violation_rate': np.NaN,
            #                 'max_violation_error': np.NaN,
            #                 'max_violation_value_mag': np.NaN,
            #             }
            #         )

            # # BRT
            # BRT_error_rates = [result['violation_rate'] for result in BRT_results_over_time]
            # BRT_error_val_error_rates = [result['violation_rate'] for result in BRT_error_val_results_over_time]
            # BRT_value_val_error_rates = [result['violation_rate'] for result in BRT_value_val_results_over_time]
            
            # # BRT errors
            # BRT_errors = [result['max_violation_error'] for result in BRT_results_over_time]
            # BRT_val_errors = [result['max_violation_error'] for result in BRT_error_val_results_over_time]
            # plt.plot(times, BRT_errors, label='Error', color='red')
            # plt.plot(times, BRT_val_errors, label='Validation', color='blue')
            # for x,y,z in zip(times, BRT_errors, BRT_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # for x,y,z in zip(times, BRT_val_errors, BRT_error_val_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # plt.xticks(ticks=times)
            # plt.title('BRT Errors for Checkpoint ' + str(checkpoint_toload))
            # plt.xlabel('Simulation Time')
            # plt.ylabel('Value Error')
            # plt.legend()
            # plt.savefig(os.path.join(testing_dir, f'BRT_error_time_curves_for_{checkpoint_toload}.png'), dpi=800)
            # plt.clf()

            # # BRT value mags
            # BRT_value_mags = [result['max_violation_value_mag'] for result in BRT_results_over_time]
            # BRT_val_value_mags = [result['max_violation_value_mag'] for result in BRT_value_val_results_over_time]
            # plt.plot(times, BRT_value_mags, label='Violation Value Magnitude', color='red')
            # plt.plot(times, BRT_val_value_mags, label='Validation', color='blue')
            # for x,y,z in zip(times, BRT_value_mags, BRT_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # for x,y,z in zip(times, BRT_val_value_mags, BRT_value_val_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # plt.xticks(ticks=times)
            # plt.title('BRT Violation Value Magnitude for Checkpoint ' + str(checkpoint_toload))
            # plt.xlabel('Simulation Time')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.savefig(os.path.join(testing_dir, f'BRT_violationvalue_time_curves_for_{checkpoint_toload}.png'), dpi=800)
            # plt.clf()

            # # exBRT
            # exBRT_error_rates = [result['violation_rate'] for result in exBRT_results_over_time]
            # exBRT_error_val_error_rates = [result['violation_rate'] for result in exBRT_error_val_results_over_time]
            # exBRT_value_val_error_rates = [result['violation_rate'] for result in exBRT_value_val_results_over_time]
            
            # # exBRT errors
            # exBRT_errors = [result['max_violation_error'] for result in exBRT_results_over_time]
            # exBRT_val_errors = [result['max_violation_error'] for result in exBRT_error_val_results_over_time]
            # plt.plot(times, exBRT_errors, label='Error', color='red')
            # plt.plot(times, exBRT_val_errors, label='Validation', color='blue')
            # for x,y,z in zip(times, exBRT_errors, exBRT_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # for x,y,z in zip(times, exBRT_val_errors, exBRT_error_val_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # plt.xticks(ticks=times)
            # plt.title('exBRT Errors for Checkpoint ' + str(checkpoint_toload))
            # plt.xlabel('Simulation Time')
            # plt.ylabel('Value Error')
            # plt.legend()
            # plt.savefig(os.path.join(testing_dir, f'exBRT_error_time_curves_for_{checkpoint_toload}.png'), dpi=800)
            # plt.clf()

            # # exBRT value mags
            # exBRT_value_mags = [result['max_violation_value_mag'] for result in exBRT_results_over_time]
            # exBRT_val_value_mags = [result['max_violation_value_mag'] for result in exBRT_value_val_results_over_time]
            # plt.plot(times, exBRT_value_mags, label='Violation Value Magnitude', color='red')
            # plt.plot(times, exBRT_val_value_mags, label='Validation', color='blue')
            # for x,y,z in zip(times, exBRT_value_mags, exBRT_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # for x,y,z in zip(times, exBRT_val_value_mags, exBRT_value_val_error_rates):
            #     plt.annotate('%.4f (%s)' % (y, str(z)), xy=(x,y), fontsize=6)
            # plt.xticks(ticks=times)
            # plt.title('exBRT Violation Value Magnitude for Checkpoint ' + str(checkpoint_toload))
            # plt.xlabel('Simulation Time')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.savefig(os.path.join(testing_dir, f'exBRT_violationvalue_time_curves_for_{checkpoint_toload}.png'), dpi=800)
            # plt.clf()

            # # plot distribution of output probabilities after factor change
            # BRT_inputs = torch.load(os.path.join(self.experiment_dir, f'BRT_inputs_at_tMax_for_{checkpoint_toload}.pt'))
            # BRT_outputs = torch.load(os.path.join(self.experiment_dir, f'BRT_outputs_at_tMax_for_{checkpoint_toload}.pt'))

            # plt.hist(BRT_outputs[BRT_outputs > 0], bins=np.linspace(0, 1, 11))
            # plt.title(f'BRT Violation Labels for Violation Predictor')
            # plt.ylabel('Count')
            # plt.xlabel('Probability Label')
            # plt.tight_layout()
            # plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_outputs_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            # plt.clf()
            # while True:
            #     print('factor: ')
            #     factor = float(input())
            #     BRT_inputs = torch.load(os.path.join(self.experiment_dir, f'BRT_inputs_at_tMax_for_{checkpoint_toload}.pt'))
            #     BRT_outputs = torch.load(os.path.join(self.experiment_dir, f'BRT_outputs_at_tMax_for_{checkpoint_toload}.pt'))

            #     # transform
            #     BRT_outputs_transformed = torch.log((BRT_outputs * factor) + 1)
            #     BRT_outputs_transformed  = BRT_outputs_transformed / torch.max(BRT_outputs_transformed)

            #     plt.hist(BRT_outputs_transformed[BRT_outputs > 0], bins=np.linspace(0, 1, 11))
            #     plt.title(f'BRT Violation Labels for Violation Predictor after Transform Factor {factor}')
            #     plt.ylabel('Count')
            #     plt.xlabel('Probability Label')
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_violation_outputs_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            #     plt.clf()
            #     print('saved')

            # # plot guarantees for narrowpassagereachtube
            # BRT_results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            #     sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #     max_scenarios=num_scenarios, max_samples=1000*num_scenarios)

            # exBRT_results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            #     sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #     sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #     violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #     max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            # BRT_level = -BRT_results['max_violation_error']
            # exBRT_level = exBRT_results['max_violation_error']

            # print(BRT_level, exBRT_level)

            # BRT_level = -52.223545
            # exBRT_level = 3.29055


            # times = [self.dataset.tMax]
            # # [th, v, phi, x, y, th, v, phi]
            # slices = np.array([[0.0, 0.0, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0], 
            #             [0.0, 3.0, 0.0, -6.0, 1.4, -np.pi, 2.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 4.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0],
            #             [0.0, 6.5, 0.0,  6.0, 1.4, -np.pi, 6.5, 0.0]])
            # (x_min, x_max), (y_min, y_max) = self.dataset.dynamics.state_test_range()[:2]
            # resolution = 512
            # xs = torch.linspace(x_min, x_max, resolution)
            # ys = torch.linspace(y_min, y_max, resolution)
            # xys = torch.cartesian_prod(xs, ys)

            # fig = plt.figure(figsize=(5*len(times), 5*len(slices)))
            # for i in range(len(times)):
            #     for j in range(len(slices)):
            #         coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = times[i]
            #         coords[:, 1] = xys[:, 0]
            #         coords[:, 2] = xys[:, 1]
            #         coords[:, 3:] = torch.tensor(slices[j])

            #         model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         with torch.no_grad():
            #             values = self.dataset.dynamics.output_to_value(self.model({'coords': model_coords.cuda()})['model_out'][:, 0])
                    
            #         ax = fig.add_subplot(len(times), len(slices), (j+1) + i*len(slices))
            #         ax.set_title('t = %0.2f, slice = %s' % (times[i], slices[j]))


            #         image = np.full((*values.shape, 3), 255, dtype=int)
            #         values = values.detach().cpu().numpy()
            #         image[(values < BRT_level)*(values < 0)] = np.array([255, 0, 0])
            #         image[(values >= BRT_level)*(values < 0)] = np.array([255, 200, 0])
            #         image[(values > exBRT_level)*(values > 0)] = np.array([0, 0, 255])
            #         image[(values <= exBRT_level)*(values > 0)] = np.array([0, 200, 255])
                    
            #         # reshape image
            #         image = image.reshape(resolution, resolution, 3).transpose(1, 0, 2)

            #         ax.imshow(image, origin='lower', extent=(-1., 1., -1., 1.))
            #         ax.set_xticks([])
            #         ax.set_yticks([])
            #         ax.tick_params(labelsize=6)
            #         if j != 0:
            #             ax.set_yticks([])
            # fig.savefig(os.path.join(testing_dir, f'slices_at_tMax_for_{checkpoint_toload}.png'), dpi=800)

        
        
            # # plot learned solution for narrowpassagereachtube
            # times = [4.0]
            # # [th, v, phi, x, y, th, v, phi]
            # slices = np.array([[0.0, 0.0, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0], 
            #             [0.0, 3.0, 0.0, -6.0, 1.4, -np.pi, 2.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 4.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0],
            #             [0.0, 6.5, 0.0,  6.0, 1.4, -np.pi, 6.5, 0.0]])
            # (x_min, x_max), (y_min, y_max) = self.dataset.dynamics.state_test_range()[:2]
            # resolution = 200
            # xs = torch.linspace(x_min, x_max, resolution)
            # ys = torch.linspace(y_min, y_max, resolution)
            # xys = torch.cartesian_prod(xs, ys)

            # fig = plt.figure(figsize=(5*len(times), 5*len(slices)))
            # for i in range(len(times)):
            #     for j in range(len(slices)):
            #         coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = times[i]
            #         coords[:, 1] = xys[:, 0]
            #         coords[:, 2] = xys[:, 1]
            #         coords[:, 3:] = torch.tensor(slices[j])

            #         model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         with torch.no_grad():
            #             values = self.dataset.dynamics.output_to_value(self.model({'coords': model_coords.cuda()})['model_out'][:, 0])
                    
            #         ax = fig.add_subplot(len(times), len(slices), (j+1) + i*len(slices))
            #         ax.set_title('t = %0.2f, slice = %s' % (times[i], slices[j]))
            #         s = ax.imshow(1*(values.detach().cpu().numpy().reshape(resolution, resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            #         if j != 0:
            #             ax.set_yticks([])
            # fig.savefig(os.path.join(testing_dir, f'validation_at_tMax_for_{checkpoint_toload}.png'), dpi=800)

            # # narrowpassagereachtube violations plot
            # times = [
            #     # 0.0, 
            #     # 1.0, 
            #     # 2.0, 
            #     # 3.0, 
            #     4.0,
            # ]
            # # [x, y, th, v, phi, x, y, th, v, phi]
            # slices = np.array([
            #     # [None, None, 0.0, 0.0, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0], 
            #     # [None, None, 0.0, 3.0, 0.0, -6.0, 1.4, -np.pi, 2.0, 0.0], 
            #     # [None, None, 0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 4.0, 0.0], 
            #     [None, None, 0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0],
            #     # [None, None, 0.0, 6.5, 0.0,  6.0, 1.4, -np.pi, 6.5, 0.0],
            # ])
            # fig = plt.figure(figsize=(5*len(times), 5*len(slices)))
            # for i in range(len(times)):
            #     for j in range(len(slices)):
            #         BRT_results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=slices[j]), sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         exBRT_results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=slices[j]), sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)                    
            #         ax = fig.add_subplot(len(times), len(slices), (j+1) + i*len(slices))
            #         ax.set_title('t = %0.2f, slice = %s' % (times[i], slices[j]), fontsize=4)
            #         ax.scatter(BRT_results['states'][:, 0][~BRT_results['violations']], BRT_results['states'][:, 1][~BRT_results['violations']], s=0.05, color=(1, 0, 0), marker='o')
            #         ax.scatter(BRT_results['states'][:, 0][BRT_results['violations']], BRT_results['states'][:, 1][BRT_results['violations']], s=0.05, color=(1, 200/255, 0), marker='o')
            #         ax.scatter(exBRT_results['states'][:, 0][~exBRT_results['violations']], exBRT_results['states'][:, 1][~exBRT_results['violations']], s=0.05, color=(0, 0, 1), marker='o')
            #         ax.scatter(exBRT_results['states'][:, 0][exBRT_results['violations']], exBRT_results['states'][:, 1][exBRT_results['violations']], s=0.05, color=(0, 200/255, 1), marker='o')
            #         x_min, x_max = self.dataset.dynamics.state_test_range()[0]
            #         y_min, y_max = self.dataset.dynamics.state_test_range()[1]
            #         ax.set_xlim(x_min, x_max)
            #         ax.set_ylim(y_min, y_max)
            #         ax.set_aspect(abs((x_max - x_min)/(y_max - y_min)))
            #         if j != 0:
            #             ax.set_yticks([])
            # fig.savefig(os.path.join(testing_dir, f'violations_for_{checkpoint_toload}.png'), dpi=800)

            # # narrowpassagereachtube
            # times = [0., 1.0, 2.0, 3.0, 4.0]
            # # [th, v, phi, x, y, th, v, phi]
            # slices = np.array([[0.0, 0.0, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0], 
            #             [0.0, 3.0, 0.0, -6.0, 1.4, -np.pi, 2.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 4.0, 0.0], 
            #             [0.0, 6.5, 0.0, -6.0, 1.4, -np.pi, 0.0, 0.0],
            #             [0.0, 6.5, 0.0,  6.0, 1.4, -np.pi, 6.5, 0.0]])
            # (x_min, x_max), (y_min, y_max) = self.dataset.dynamics.state_test_range()[:2]
            # resolution = 200
            # xs = torch.linspace(x_min, x_max, resolution)
            # ys = torch.linspace(y_min, y_max, resolution)
            # xys = torch.cartesian_prod(xs, ys)

            # fig = plt.figure(figsize=(5*len(times), 5*len(slices)))
            # for i in range(len(times)):
            #     for j in range(len(slices)):
            #         coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = times[i]
            #         coords[:, 1] = xys[:, 0]
            #         coords[:, 2] = xys[:, 1]
            #         coords[:, 3:] = torch.tensor(slices[j])

            #         model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         with torch.no_grad():
            #             values = self.dataset.dynamics.output_to_value(self.model({'coords': model_coords.cuda()})['model_out'][:, 0])
                    
            #         ax = fig.add_subplot(len(times), len(slices), (j+1) + i*len(slices))
            #         ax.set_title('t = %0.2f, slice = %s' % (times[i], slices[j]))
            #         s = ax.imshow(1*(values.detach().cpu().numpy().reshape(resolution, resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            #         fig.colorbar(s) 
            # fig.savefig(os.path.join(testing_dir, 'validation.png'))



            # # FOR tMAX:
            # # try fitting violations classifier based on state, boundary_fn, value, and value gradients
            
            # # collect training data

            # # # 1,000,000 BRT
            # # BRT_results = scenario_optimization(
            # #     model=self.model, dynamics=self.dataset.dynamics, 
            # #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            # #     set_type=set_type, control_type=control_type, 
            # #     scenario_batch_size=int(1e6), sample_batch_size=int(1e7), 
            # #     sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            # #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            # #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            # #     max_scenarios=int(1e7))
            # # BRT_states = BRT_results['states']
            # # BRT_values = BRT_results['values']
            # # BRT_true_values = BRT_results['true_values']
            # # BRT_violations = BRT_results['violations']

            # # 1,000,000 exBRT
            # exBRT_results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #     sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #     sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #     violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #     max_scenarios=10*num_scenarios, max_samples=100000*num_scenarios)
            # exBRT_states = exBRT_results['states']
            # exBRT_values = exBRT_results['values']
            # exBRT_violations = exBRT_results['violations']

            # # # compute reach_fn
            # # BRT_reach_fn = self.dataset.dynamics.reach_fn(BRT_states)
            # # # exBRT_reach_fn = self.dataset.dynamics.reach_fn(exBRT_states)

            # # compute avoid_fn
            # # BRT_avoid_fn = self.dataset.dynamics.avoid_fn(BRT_states)
            # # exBRT_avoid_fn = self.dataset.dynamics.avoid_fn(exBRT_states.cuda()).cpu()
            
            # # compute boundary_fn
            # # BRT_boundary_fn = self.dataset.dynamics.boundary_fn(BRT_states)
            # exBRT_boundary_fn = self.dataset.dynamics.boundary_fn(exBRT_states.cuda()).cpu()

            # # compute value gradients
            # # BRT_model_dvs = torch.zeros(0, 1, self.dataset.dynamics.state_dim + 1)
            # # batch_size = int(1e5)
            # # idx = 0
            # # while idx < len(BRT_states):
            # #     batch_BRT_model_coords = self.dataset.dynamics.normalize_coord(torch.cat((torch.full(BRT_states.shape[:-1], self.dataset.tMax)[..., None], BRT_states), dim=-1))[idx : idx + batch_size]
            # #     batch_BRT_model_results = self.model({'coords': batch_BRT_model_coords.cuda()})
            # #     batch_BRT_model_dvs = diff_operators.jacobian(batch_BRT_model_results['model_out'], batch_BRT_model_results['model_in'])[0].detach().cpu()
            # #     BRT_model_dvs = torch.cat((BRT_model_dvs, batch_BRT_model_dvs), dim=0)
            # #     idx += batch_size
            # # BRT_dvdts = self.dataset.dynamics.unnormalize_dvdt(BRT_model_dvs[..., 0, 0])
            # # BRT_dvdss = self.dataset.dynamics.unnormalize_dvds(BRT_model_dvs[..., 0, 1:])
            
            # exBRT_dvs = torch.zeros(0, self.dataset.dynamics.state_dim + 1)
            # batch_size = 100000
            # idx = 0
            # while idx < len(exBRT_states):
            #     batch_exBRT_model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(torch.cat((torch.full(exBRT_states.shape[:-1], self.dataset.tMax).unsqueeze(dim=-1), exBRT_states), dim=-1)[idx : idx + batch_size]).cuda()})
            #     exBRT_dvs = torch.cat((exBRT_dvs, self.dataset.dynamics.io_to_dv(batch_exBRT_model_results['model_in'], batch_exBRT_model_results['model_out'].squeeze(dim=-1)).detach().cpu()), dim=0)
            #     idx += batch_size

            # # construct input vectors [*state, boundary_fn, value, *value_gradients] and outputs
            # # BRT_inputs = torch.cat((BRT_states, BRT_reach_fn[..., None], BRT_avoid_fn[..., None], BRT_boundary_fn[..., None], BRT_values[..., None], BRT_dvdts[..., None], BRT_dvdss), dim=-1)
            # # # BRT_inputs = torch.cat((BRT_states, BRT_boundary_fn[..., None], BRT_values[..., None], BRT_dvdts[..., None], BRT_dvdss), dim=-1)
            # # BRT_outputs = BRT_true_values - BRT_values
            # exBRT_inputs = torch.cat((exBRT_states, exBRT_boundary_fn[..., None], exBRT_values[..., None], exBRT_dvs), dim=-1)
            # exBRT_outputs = torch.where(exBRT_violations, exBRT_values, torch.tensor([0.0]))
            # # # assert torch.all(BRT_outputs >= 0)
            # # # # assert torch.all(exBRT_outputs >= 0)

            # # save training data
            # # torch.save(BRT_inputs, os.path.join(self.experiment_dir, f'BRT_inputs_at_tMax_for_{checkpoint_toload}.pt'))
            # # torch.save(BRT_outputs, os.path.join(self.experiment_dir, f'BRT_outputs_at_tMax_for_{checkpoint_toload}.pt'))
            # # quit()
            # torch.save(exBRT_inputs, os.path.join(self.experiment_dir, f'exBRT_inputs_at_tMax_for_{checkpoint_toload}.pt'))
            # torch.save(exBRT_outputs, os.path.join(self.experiment_dir, f'exBRT_outputs_at_tMax_for_{checkpoint_toload}.pt'))

            # # # find error after excluding obstacle-contained states
            # # BRT_inputs = torch.load(os.path.join(self.experiment_dir, f'BRT_inputs_at_tMax_for_{checkpoint_toload}.pt'))
            # # BRT_states = BRT_inputs[..., :10]
            # # BRT_avoid_fn = self.dataset.dynamics.avoid_fn(BRT_states)
            # # BRT_outputs = torch.load(os.path.join(self.experiment_dir, f'BRT_outputs_at_tMax_for_{checkpoint_toload}.pt'))
            # # print('pre')
            # # print(torch.sum(BRT_outputs > 0), 'violations')
            # # print(torch.max(BRT_outputs), 'max error')
            # # print('post')
            # # print(torch.sum(BRT_outputs[BRT_avoid_fn > 0] > 0), 'violations')
            # # print(torch.max(BRT_outputs[BRT_avoid_fn > 0]), 'max error')
            # # quit()

            # # dubins3dwidereachtube
            # times = [1.0]
            # # [x, y, th]
            # slices = np.array([[None, None, -3.14], 
            #             [None, None, -1.66], 
            #             [None, None, -0.06], 
            #             [None, None, 1.42],
            #             [None, None, 3.02]])
            # fig = plt.figure(figsize=(5*len(times), 5*len(slices)))
            # for i in range(len(times)):
            #     for j in range(len(slices)):
            #         BRT_results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=slices[j]), sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #         exBRT_results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=times[i], dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=slices[j]), sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=1000*num_scenarios)                    
            #         ax = fig.add_subplot(len(times), len(slices), (j+1) + i*len(slices))
            #         ax.set_title('t = %0.2f, slice = %s' % (times[i], slices[j]), fontsize=4)
            #         ax.scatter(BRT_results['states'][:, 0][~BRT_results['violations']], BRT_results['states'][:, 1][~BRT_results['violations']], s=0.05, color=(1, 0, 0), marker='o')
            #         ax.scatter(BRT_results['states'][:, 0][BRT_results['violations']], BRT_results['states'][:, 1][BRT_results['violations']], s=0.05, color=(1, 200/255, 0), marker='o')
            #         ax.scatter(exBRT_results['states'][:, 0][~exBRT_results['violations']], exBRT_results['states'][:, 1][~exBRT_results['violations']], s=0.05, color=(0, 0, 1), marker='o')
            #         ax.scatter(exBRT_results['states'][:, 0][exBRT_results['violations']], exBRT_results['states'][:, 1][exBRT_results['violations']], s=0.05, color=(0, 200/255, 1), marker='o')
            #         x_min, x_max = self.dataset.dynamics.state_test_range()[0]
            #         y_min, y_max = self.dataset.dynamics.state_test_range()[1]
            #         ax.set_xlim(x_min, x_max)
            #         ax.set_ylim(y_min, y_max)
            #         ax.set_aspect(abs((x_max - x_min)/(y_max - y_min)))
            #         if j != 0:
            #             ax.set_yticks([])
            # fig.savefig(os.path.join(testing_dir, f'violations_for_{checkpoint_toload}.png'), dpi=800)
            # # quit()


            # # ### START ###
            # # # # train violation error predictors for BRT and exBRT

            # def normalize(inputs, means, stds):
            #     # return (inputs - means) / stds
            #     return inputs

            # # def validate_predictor(BRT_model, epoch, means, stds):
            # #     BRT_model.eval()

            # #     # 100,000 BRT
            # #     BRT_results = scenario_optimization(
            # #         model=self.model, dynamics=self.dataset.dynamics, 
            # #         tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            # #         set_type=set_type, control_type=control_type, 
            # #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            # #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            # #         sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            # #         violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            # #         max_scenarios=num_scenarios, max_samples=100000*num_scenarios)
            # #     BRT_states = BRT_results['states']
            # #     BRT_values = BRT_results['values']
            # #     BRT_true_values = BRT_results['true_values']
            # #     BRT_violations = BRT_results['violations']

            # #     BRT_reach_fn = self.dataset.dynamics.reach_fn(BRT_states.cuda()).cpu()
            # #     BRT_avoid_fn = self.dataset.dynamics.avoid_fn(BRT_states.cuda()).cpu()
            # #     BRT_boundary_fn = self.dataset.dynamics.boundary_fn(BRT_states.cuda()).cpu()

            # #     BRT_model_dvs = torch.zeros(0, 1, self.dataset.dynamics.state_dim + 1)
            # #     batch_size = 100000
            # #     idx = 0
            # #     while idx < len(BRT_states):
            # #         batch_BRT_model_coords = self.dataset.dynamics.normalize_coord(torch.cat((torch.full(BRT_states.shape[:-1], self.dataset.tMax)[..., None], BRT_states), dim=-1))[idx : idx + batch_size]
            # #         batch_BRT_model_results = self.model({'coords': batch_BRT_model_coords.cuda()})
            # #         batch_BRT_model_dvs = diff_operators.jacobian(batch_BRT_model_results['model_out'], batch_BRT_model_results['model_in'])[0].detach().cpu()
            # #         BRT_model_dvs = torch.cat((BRT_model_dvs, batch_BRT_model_dvs), dim=0)
            # #         idx += batch_size
            # #     BRT_dvdts = self.dataset.dynamics.unnormalize_dvdt(BRT_model_dvs[..., 0, 0].cuda()).cpu()
            # #     BRT_dvdss = self.dataset.dynamics.unnormalize_dvds(BRT_model_dvs[..., 0, 1:].cuda()).cpu()

            # #     BRT_inputs = torch.cat((BRT_states, BRT_reach_fn[..., None], BRT_avoid_fn[..., None], BRT_boundary_fn[..., None], BRT_values[..., None], BRT_dvdts[..., None], BRT_dvdss), dim=-1)
            # #     # BRT_inputs = torch.cat((BRT_states, BRT_boundary_fn[..., None], BRT_values[..., None], BRT_dvdts[..., None], BRT_dvdss), dim=-1)

            # #     BRT_preds = torch.sigmoid(BRT_model(normalize(BRT_inputs.cuda(), means.cuda(), stds.cuda()))).detach().cpu().numpy()

            # #     plt.title(f'BRT Violation Predictor Validation')
            # #     # plt.title(f'BRT Violation Predictor Validation')
            # #     plt.ylabel('violation error')
            # #     plt.xlabel('prediction')
            # #     plt.scatter(BRT_preds[~BRT_violations], torch.zeros(BRT_preds.shape)[~BRT_violations], color='blue', label='nonviolations', alpha=0.1)
            # #     plt.scatter(BRT_preds[BRT_violations], (BRT_results['true_values'] - BRT_values)[BRT_violations], color='red', label='violations', alpha=0.1)
            # #     plt.legend()
            # #     plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_violation_error_validation_{epoch}.png'), dpi=800)
            # #     # plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_validation.png'), dpi=800)
            # #     plt.clf()

            # #     plt.title(f'BRT Violation Predictor Validation')
            # #     # plt.title(f'BRT Violation Predictor Validation')
            # #     plt.ylabel('error')
            # #     plt.xlabel('prediction')
            # #     plt.scatter(BRT_preds[~BRT_violations], (BRT_results['true_values'] - BRT_values)[~BRT_violations], color='blue', label='nonviolations', alpha=0.1)
            # #     plt.scatter(BRT_preds[BRT_violations], (BRT_results['true_values'] - BRT_values)[BRT_violations], color='red', label='violations', alpha=0.1)
            # #     plt.legend()
            # #     plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_error_validation_{epoch}.png'), dpi=800)
            # #     # plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_validation.png'), dpi=800)
            # #     plt.clf()

            # #     plt.title(f'BRT Violation Predictor Validation')
            # #     # plt.title(f'BRT Violation Predictor Validation')
            # #     plt.ylabel('value')
            # #     plt.xlabel('prediction')
            # #     plt.scatter(BRT_preds[~BRT_violations], BRT_values[~BRT_violations], color='blue', label='nonviolations', alpha=0.1)
            # #     plt.scatter(BRT_preds[BRT_violations], BRT_values[BRT_violations], color='red', label='violations', alpha=0.1)
            # #     plt.legend()
            # #     plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_value_validation_{epoch}.png'), dpi=800)
            # #     # plt.savefig(os.path.join(self.experiment_dir, f'BRT_violation_predictor_value_validation.png'), dpi=800)
            # #     plt.clf()
            # #     BRT_model.train()

            # def validate_predictor(exBRT_model, epoch, means, stds):
            #     exBRT_model.eval()

            #     # 100,000 exBRT
            #     exBRT_results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 100000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #         max_scenarios=num_scenarios, max_samples=100000*num_scenarios)
            #     exBRT_states = exBRT_results['states']
            #     exBRT_values = exBRT_results['values']
            #     exBRT_violations = exBRT_results['violations']

            #     exBRT_boundary_fn = self.dataset.dynamics.boundary_fn(exBRT_states.cuda()).cpu()

            #     exBRT_dvs = torch.zeros(0, self.dataset.dynamics.state_dim + 1)
            #     batch_size = 100000
            #     idx = 0
            #     while idx < len(exBRT_states):
            #         batch_exBRT_model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(torch.cat((torch.full(exBRT_states.shape[:-1], self.dataset.tMax)[..., None], exBRT_states), dim=-1)[idx : idx + batch_size]).cuda()})
            #         batch_exBRT_dvs = self.dataset.dynamics.io_to_dv(batch_exBRT_model_results['model_in'], batch_exBRT_model_results['model_out'].squeeze(dim=-1)).detach().cpu()
            #         exBRT_dvs = torch.cat((exBRT_dvs, batch_exBRT_dvs), dim=0)
            #         idx += batch_size
                
            #     # exBRT_inputs = torch.cat((exBRT_states, exBRT_reach_fn[..., None], exBRT_avoid_fn[..., None], exBRT_boundary_fn[..., None], exBRT_values[..., None], exBRT_dvdts[..., None], exBRT_dvdss), dim=-1)
            #     exBRT_inputs = torch.cat((exBRT_states, exBRT_boundary_fn[..., None], exBRT_values[..., None], exBRT_dvs), dim=-1)

            #     exBRT_preds = torch.sigmoid(exBRT_model(normalize(exBRT_inputs.cuda(), means.cuda(), stds.cuda()))).detach().cpu().numpy()

            #     plt.title(f'exBRT Violation Predictor Validation')
            #     # plt.title(f'exBRT Violation Predictor Validation')
            #     plt.ylabel('value magnitude')
            #     plt.xlabel('prediction')
            #     plt.scatter(exBRT_preds[~exBRT_violations], exBRT_values[~exBRT_violations], color='blue', label='nonviolations', alpha=0.1)
            #     # plt.scatter(exBRT_preds[~exBRT_violations], torch.zeros(exBRT_preds.shape)[~exBRT_violations], color='blue', label='nonviolations', alpha=0.1)
            #     plt.scatter(exBRT_preds[exBRT_violations], exBRT_values[exBRT_violations], color='red', label='violations', alpha=0.1)
            #     plt.legend()
            #     plt.savefig(os.path.join(self.experiment_dir, f'exBRT_violation_predictor_violation_error_validation_{epoch}.png'), dpi=800)
            #     # plt.savefig(os.path.join(self.experiment_dir, f'exBRT_violation_predictor_validation.png'), dpi=800)
            #     plt.clf()

            #     exBRT_model.train()

            # # # BRT
            # # print('training BRT_model')
            # # BRT_inputs = torch.load(os.path.join(self.experiment_dir, f'BRT_inputs_at_tMax_for_{checkpoint_toload}.pt')).cuda()
            # # means = torch.mean(BRT_inputs, dim=0)
            # # stds = torch.std(BRT_inputs, dim=0)
            # # BRT_outputs = torch.load(os.path.join(self.experiment_dir, f'BRT_outputs_at_tMax_for_{checkpoint_toload}.pt')).cuda()
            # # # BRT_outputs = (BRT_outputs - torch.mean(BRT_outputs)) / torch.std(BRT_outputs)
            # # # BRT_binary_outputs = 1.0*(BRT_outputs > 0)   
            # # BRT_outputs = 1.0*(BRT_outputs > 0)

            # # if torch.any(BRT_outputs > 0):
            # #     # factor = 100000
            # #     weight_factor = 1
            # #     # # transform
            # #     # BRT_outputs_transformed = torch.log((BRT_outputs * factor) + 1)
            # #     # BRT_outputs_transformed  = BRT_outputs_transformed / torch.max(BRT_outputs_transformed)

            # #     # assert torch.all(BRT_outputs_transformed >= 0) and torch.all(BRT_outputs_transformed <= 1)
            # #     BRT_model = MLP(input_size=25)
            # #     # BRT_binary_model = MLP()
            # #     BRT_model.cuda()
            # #     BRT_model.train()
            # #     # BRT_binary_model.train()

            # #     BRT_opt = torch.optim.SGD(BRT_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
            # #     # BRT_binary_opt = torch.optim.SGD(BRT_binary_model.parameters(), lr=0.00001)
            # #     sched = torch.optim.lr_scheduler.ReduceLROnPlateau(BRT_opt, factor=0.2, patience=50, threshold=1e-12)
                
            # #     weight = weight_factor*((~(BRT_outputs > 0)).sum() / (BRT_outputs > 0).sum()).item()
            # #     # weight = ((~(BRT_outputs > 0)).sum() / (BRT_outputs > 0).sum()).item()
            # #     # MSELoss = torch.nn.MSELoss()
            # #     BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=BRT_outputs.device))
            # #     n_epochs = 100000
            # #     batch_size = 1000000
            # #     for epoch in range(n_epochs):
            # #         if epoch > 0 and epoch%10000 == 0:
            # #             torch.save(BRT_model.state_dict(), os.path.join(self.experiment_dir, f'BRT_violation_predictor_at_tMax_for_{checkpoint_toload}_{epoch}.pth'))
            # #             validate_predictor(BRT_model, epoch, means, stds)
            # #         for _ in range(len(BRT_outputs) // batch_size):
            # #             idxs = torch.randperm(len(BRT_outputs))[:batch_size]
            # #             preds = BRT_model(normalize(BRT_inputs[idxs], means, stds)).squeeze(dim=-1)
            # #             # preds_binary = BRT_binary_model(BRT_inputs[idxs])
            # #             loss = BCEWithLogitsLoss(preds, BRT_outputs[idxs])
            # #             # loss_binary = BCEWithLogitsLoss(preds_binary, BRT_binary_outputs[idxs][..., None])
            # #             loss.backward()
            # #             # loss_binary.backward()
            # #             BRT_opt.step()
            # #             # BRT_binary_opt.step()
            # #         print(f'Epoch {epoch}: loss: {loss.item()}')
            # #         sched.step(loss.item())
            # #         # print(f'Epoch {epoch}: loss: {loss.item()} loss_binary: {loss_binary.item()}')
            # #     # torch.save(BRT_model.state_dict(), os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth'))
            # #     torch.save(BRT_model.state_dict(), os.path.join(self.experiment_dir, f'BRT_violation_predictor_at_tMax_for_{checkpoint_toload}_final.pth'))
            # #     # torch.save(BRT_binary_model.state_dict(), os.path.join(self.experiment_dir, 'BRT_binary_model.pth'))
            # # else:
            # #     print('skipping, because no violations in training data')

            # # quit()

            # # exBRT
            # print('training exBRT_model')
            # exBRT_inputs = torch.load(os.path.join(self.experiment_dir, f'exBRT_inputs_at_tMax_for_{checkpoint_toload}.pt')).cuda()
            # exBRT_outputs = torch.load(os.path.join(self.experiment_dir, f'exBRT_outputs_at_tMax_for_{checkpoint_toload}.pt')).cuda()

            # means = torch.mean(exBRT_inputs, dim=0)
            # stds = torch.std(exBRT_inputs, dim=0)
            # # exBRT_binary_outputs = 1.0*(exBRT_outputs > 0)

            # if torch.any(exBRT_outputs > 0):
            #     # assert torch.all(exBRT_outputs >= 0) and torch.all(exBRT_outputs <= 1)
            #     exBRT_outputs = exBRT_outputs / torch.max(exBRT_outputs)
            #     exBRT_model = MLP(input_size=9)
            #     # exBRT_binary_model = MLP()
            #     exBRT_model.cuda()
            #     exBRT_model.train()
            #     # exBRT_binary_model.train()

            #     exBRT_opt = torch.optim.SGD(exBRT_model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
            #     sched = torch.optim.lr_scheduler.ReduceLROnPlateau(exBRT_opt, factor=0.2, patience=50, threshold=1e-12)
            #     # exBRT_binary_opt = torch.optim.SGD(exBRT_binary_model.parameters(), lr=0.00001)
                
            #     weight = ((~(exBRT_outputs > 0)).sum() / (exBRT_outputs > 0).sum()).item()
            #     BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=exBRT_outputs.device))
            #     n_epochs = 100000
            #     batch_size = 1000000
            #     for epoch in range(n_epochs):
            #         if epoch > 0 and epoch%5000 == 0:
            #             torch.save(exBRT_model.state_dict(), os.path.join(self.experiment_dir, f'exBRT_violation_predictor_at_tMax_for_{checkpoint_toload}_{epoch}.pth'))
            #             validate_predictor(exBRT_model, epoch, means, stds)
            #         for _ in range(len(exBRT_outputs) // batch_size):
            #             idxs = torch.randperm(len(exBRT_outputs))[:batch_size]
            #             preds = exBRT_model(normalize(exBRT_inputs[idxs], means, stds)).squeeze(dim=-1)
            #             # preds_binary = exBRT_binary_model(exBRT_inputs[idxs])
            #             loss = BCEWithLogitsLoss(preds, exBRT_outputs[idxs])
            #             # loss_binary = BCEWithLogitsLoss(preds_binary, exBRT_binary_outputs[idxs][..., None])
            #             loss.backward()
            #             # loss_binary.backward()
            #             exBRT_opt.step()
            #             # exBRT_binary_opt.step()
            #         print(f'Epoch {epoch}: loss: {loss.item()}')
            #         sched.step(loss.item())
            #         # print(f'Epoch {epoch}: loss: {loss.item()} loss_binary: {loss_binary.item()}')
            #     # torch.save(exBRT_model.state_dict(), os.path.join(self.experiment_dir, f'exBRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth'))
            #     torch.save(exBRT_model.state_dict(), os.path.join(self.experiment_dir, f'exBRT_violation_predictor_at_tMax_for_{checkpoint_toload}_final.pth'))
            #     validate_predictor(exBRT_model, 'final', means, stds)
            #     # torch.save(exBRT_binary_model.state_dict(), os.path.join(self.experiment_dir, 'BRT_binary_model.pth'))
            # else:
            #     print('skipping, because no violations in training data')

            # quit()

            # # if os.path.exists(os.path.join(self.experiment_dir, f'exBRT_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')):
            # #     exBRT_model = MLP()
            # #     exBRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'exBRT_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')))
            # #     exBRT_model.eval()
                
            # #     # 100,000 exBRT
            # #     exBRT_results = scenario_optimization(
            # #         model=self.model, dynamics=self.dataset.dynamics, 
            # #         tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            # #         set_type=set_type, control_type=control_type, 
            # #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            # #         sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            # #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            # #         max_scenarios=num_scenarios, max_samples=100000*num_scenarios)
            # #     exBRT_states = exBRT_results['states']
            # #     exBRT_values = exBRT_results['values']
            # #     exBRT_true_values = exBRT_results['true_values']
            # #     exBRT_violations = exBRT_results['violations']

            # #     exBRT_reach_fn = self.dataset.dynamics.reach_fn(exBRT_states)
            # #     exBRT_avoid_fn = self.dataset.dynamics.avoid_fn(exBRT_states)
            # #     exBRT_boundary_fn = self.dataset.dynamics.boundary_fn(exBRT_states)
                
            # #     exBRT_model_dvs = torch.zeros(0, 1, self.dataset.dynamics.state_dim + 1)
            # #     batch_size = 100000
            # #     idx = 0
            # #     while idx < len(exBRT_states):
            # #         batch_exBRT_model_coords = self.dataset.dynamics.normalize_coord(torch.cat((torch.full(exBRT_states.shape[:-1], self.dataset.tMax)[..., None], exBRT_states), dim=-1))[idx : idx + batch_size]
            # #         batch_exBRT_model_results = self.model({'coords': batch_exBRT_model_coords.cuda()})
            # #         batch_exBRT_model_dvs = diff_operators.jacobian(batch_exBRT_model_results['model_out'], batch_exBRT_model_results['model_in'])[0].detach().cpu()
            # #         exBRT_model_dvs = torch.cat((exBRT_model_dvs, batch_exBRT_model_dvs), dim=0)
            # #         idx += batch_size
            # #     exBRT_dvdts = self.dataset.dynamics.unnormalize_dvdt(exBRT_model_dvs[..., 0, 0])
            # #     exBRT_dvdss = self.dataset.dynamics.unnormalize_dvds(exBRT_model_dvs[..., 0, 1:])

            # #     exBRT_inputs = torch.cat((exBRT_states, exBRT_reach_fn[..., None], exBRT_avoid_fn[..., None], exBRT_boundary_fn[..., None], exBRT_values[..., None], exBRT_dvdts[..., None], exBRT_dvdss), dim=-1)
                
            # #     exBRT_preds = torch.sigmoid(exBRT_model(exBRT_inputs)).detach().cpu().numpy()

            # #     plt.title('exBRT Violation Predictor Validation')
            # #     plt.ylabel('violation error')
            # #     plt.xlabel('prediction')
            # #     plt.scatter(exBRT_preds[~exBRT_violations], torch.zeros(exBRT_preds.shape)[~exBRT_violations], color='blue', label='nonviolations', alpha=0.1)
            # #     plt.scatter(exBRT_preds[exBRT_violations], (exBRT_values - exBRT_results['true_values'])[exBRT_violations], color='red', label='violations', alpha=0.1)
            # #     plt.legend()
            # #     plt.savefig(os.path.join(self.experiment_dir, 'exBRT_violation_predictor_validation.png'), dpi=800)
            # #     plt.clf()

            # #     plt.title('exBRT Violation Predictor Validation')
            # #     plt.ylabel('value')
            # #     plt.xlabel('prediction')
            # #     plt.scatter(exBRT_preds[~exBRT_violations], exBRT_values[~exBRT_violations], color='blue', label='nonviolations', alpha=0.1)
            # #     plt.scatter(exBRT_preds[exBRT_violations], exBRT_values[exBRT_violations], color='red', label='violations', alpha=0.1)
            # #     plt.legend()
            # #     plt.savefig(os.path.join(self.experiment_dir, 'exBRT_violation_predictor_value_validation.png'), dpi=800)
            # #     plt.clf()

            # # # create slices plot binned by error model
            # # BRT_model = MLP(input_size=25)
            # # BRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')))
            # # BRT_model.eval()

            # # bins = np.linspace(0, 1, 11)
            # # levels = []
            # # for i in range(len(bins) - 1):
            # #     BRT_results = scenario_optimization(
            # #         model=self.model, dynamics=self.dataset.dynamics, 
            # #         tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            # #         set_type=set_type, control_type=control_type, 
            # #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(num_scenarios, 100000), 
            # #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            # #         sample_validator=MultiValidator([
            # #             MLPValidator(mlp=BRT_model, o_min=bins[i], o_max=bins[i+1], model=self.model, dynamics=self.dataset.dynamics),
            # #             ValueThresholdValidator(v_min=float('-inf'), v_max=0.0)
            # #         ]), 
            # #         violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            # #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            # #     if BRT_results['maxed_scenarios']:
            # #         levels.append(-BRT_results['max_violation_error'])
            # #     else:
            # #         levels.append(float('-inf'))

            # # print(levels)
            # # import pickle
            # # with open(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_levels.pickle'), 'wb') as f:
            # #     pickle.dump(levels, f)
  
            # # import pickle
            # # with open(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_levels.pickle'), 'rb') as f:
            # #     levels = pickle.load(f)
            # # print(levels)
            # # bins = np.linspace(0, 1, 11)

            # # BRT_model = MLP(input_size=9)
            # # BRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')))
            # # BRT_model.eval()

            # # create slices plot binned by error model
            # epoch=50000
            # exBRT_model = MLP(input_size=9)
            # exBRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'exBRT_violation_predictor_at_tMax_for_{checkpoint_toload}_{epoch}.pth')))
            # exBRT_model.cuda()
            # exBRT_model.eval()

            # bins = np.linspace(0, 1, 11)
            # levels = []
            # for i in range(len(bins) - 1):
            #     exBRT_results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(num_scenarios, 100000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=MultiValidator([
            #             MLPValidator(mlp=exBRT_model, o_min=bins[i], o_max=bins[i+1], model=self.model, dynamics=self.dataset.dynamics),
            #             ValueThresholdValidator(v_min=0.0, v_max=float('inf'))
            #         ]), 
            #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #     if exBRT_results['maxed_scenarios']:
            #         levels.append(exBRT_results['max_violation_value_mag'])
            #     else:
            #         levels.append(float('inf'))

            # print(levels)

            # # plot slices with ground truth, where error correction is conditioned on mlp

            # # load ground truth
            # ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
            # if 'gmat' in ground_truth:
            #     ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
            #     ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
            #     ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
            #     ground_truth_values = ground_truth['data']
            #     ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
            # elif 'g' in ground_truth:
            #     ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
            #     ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
            #     ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
            #     ground_truth_ts = ground_truth['tau'][0]
            #     ground_truth_values = ground_truth['data']

            # # idxs to plot
            # z_res = 5
            # x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
            # y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
            # z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
            # t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # # indexed ground truth to plot
            # ground_truth_xs = ground_truth_xs[x_idxs]
            # ground_truth_ys = ground_truth_ys[y_idxs]
            # ground_truth_zs = ground_truth_zs[z_idxs]
            # ground_truth_ts = ground_truth_ts[t_idxs]
            # ground_truth_values = ground_truth_values[
            #     x_idxs[:, None, None, None], 
            #     y_idxs[None, :, None, None], 
            #     z_idxs[None, None, :, None],
            #     t_idxs[None, None, None, :]]

            # ground_truth_xys = torch.cartesian_prod(torch.tensor(ground_truth_xs), torch.tensor(ground_truth_ys))
            # plot_config = self.dataset.dynamics.plot_config()
        
            # fig = plt.figure(figsize=(5*len(ground_truth_ts), 5*len(ground_truth_zs)))
            # for i in range(len(ground_truth_ts)):
            #     for j in range(len(ground_truth_zs)):
            #         coords = torch.zeros(ground_truth_xys.shape[0], self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = ground_truth_ts[i]
            #         coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            #         coords[:, 1 + plot_config['x_axis_idx']] = ground_truth_xys[:, 0]
            #         coords[:, 1 + plot_config['y_axis_idx']] = ground_truth_xys[:, 1]
            #         coords[:, 1 + plot_config['z_axis_idx']] = ground_truth_zs[j]

            #         model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
            #         dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).detach().cpu()
            #         values = self.dataset.dynamics.io_to_value(model_results['model_out'].detach(), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            #         boundary_values = self.dataset.dynamics.boundary_fn(coords[..., 1:])

            #         # input vector is [*state, boundary_value, value, dvdt, *dvds]
            #         inputs = torch.cat((coords[..., 1:], boundary_values[:, None], values[:, None], dvs), dim=-1)
            #         outputs = torch.sigmoid(exBRT_model(inputs.cuda()).cpu().squeeze())
                    
            #         ax = fig.add_subplot(len(ground_truth_ts), len(ground_truth_zs), (j+1) + i*len(ground_truth_zs))
            #         ax.set_title('t = %0.2f, %s = %0.2f' % (ground_truth_ts[i], plot_config['state_labels'][plot_config['z_axis_idx']], ground_truth_zs[j]), fontsize=4)

            #         image = np.full((*values.shape, 3), 255, dtype=int)

            #         values = values.numpy()

            #         # per bin, color accordingly
            #         for k in range(len(levels)):
            #             mask = ((outputs >= bins[k])*(outputs < bins[k+1])).numpy()
            #             level = levels[k]
            #             image[mask*(values > level)*(values > 0)] = np.array([0, 0, 255])
            #             image[mask*(values <= level)*(values > 0)] = np.array([0, 200, 255])
            #         image[values < 0] = np.array([255, 0, 0])
                    
            #         # reshape image
            #         image = image.reshape(ground_truth_xs.shape[0], ground_truth_ys.shape[0], 3).transpose(1, 0, 2)

            #         # overlay the true boundary
            #         ground_truth_values_slice = ground_truth_values[:, :, j, i] < 0
            #         for x in range(ground_truth_values_slice.shape[0]):
            #             for y in range(ground_truth_values_slice.shape[1]):
            #                 if not ground_truth_values_slice[x, y]:
            #                     continue
            #                 neighbors = [
            #                     (x, y+1),
            #                     (x, y-1),
            #                     (x+1, y+1),
            #                     (x+1, y),
            #                     (x+1, y-1),
            #                     (x-1, y+1),
            #                     (x-1, y),
            #                     (x-1, y-1),
            #                 ]
            #                 for neighbor in neighbors:
            #                     if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_values_slice.shape[0] and neighbor[1] < ground_truth_values_slice.shape[1]:
            #                         if not ground_truth_values_slice[neighbor]:
            #                             image[y, x] = image[y, x] / 2
            #                             break

            #         ax.imshow(image, origin='lower', extent=(-1., 1., -1., 1.))
            #         ax.set_xticks([-1, 1])
            #         ax.set_yticks([-1, 1])
            #         ax.tick_params(labelsize=6)
            #         if j != 0:
            #             ax.set_yticks([])

            # fig.savefig(os.path.join(testing_dir, f'slices_adjusted_by_mlp_{epoch}_with_ground_truth_at_{self.dataset.tMax}_for_{checkpoint_toload}.png'), dpi=800)

            # # # plot bins of error model in 3D
            # # BRT_model = MLP(input_size=9)
            # # BRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')))
            # # BRT_model.eval()

            # # bins = np.linspace(0, 1, 11)

            # # # plot error model predictions
            # # # 10,000 BRT
            # # results = scenario_optimization(
            # #     model=self.model, dynamics=self.dataset.dynamics, 
            # #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            # #     set_type=set_type, control_type=control_type, 
            # #     scenario_batch_size=min(num_scenarios, 10000), sample_batch_size=min(10*num_scenarios, 100000), 
            # #     sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            # #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            # #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            # #     max_scenarios=num_scenarios/10, max_samples=100000*num_scenarios)
            # # states = results['states']
            # # values = results['values']
            # # true_values = results['true_values']
            # # violations = results['violations']

            # # # reach_fn = self.dataset.dynamics.reach_fn(states)
            # # # avoid_fn = self.dataset.dynamics.avoid_fn(states)
            # # boundary_fn = self.dataset.dynamics.boundary_fn(states)

            # # model_dvs = torch.zeros(0, 1, self.dataset.dynamics.state_dim + 1)
            # # batch_size = 10000
            # # idx = 0
            # # while idx < len(states):
            # #     batch_model_coords = self.dataset.dynamics.normalize_coord(torch.cat((torch.full(states.shape[:-1], self.dataset.tMax)[..., None], states), dim=-1))[idx : idx + batch_size]
            # #     batch_model_results = self.model({'coords': batch_model_coords.cuda()})
            # #     batch_model_dvs = diff_operators.jacobian(batch_model_results['model_out'], batch_model_results['model_in'])[0].detach().cpu()
            # #     model_dvs = torch.cat((model_dvs, batch_model_dvs), dim=0)
            # #     idx += batch_size
            # # dvdts = self.dataset.dynamics.unnormalize_dvdt(model_dvs[..., 0, 0])
            # # dvdss = self.dataset.dynamics.unnormalize_dvds(model_dvs[..., 0, 1:])

            # # # inputs = torch.cat((states, reach_fn[..., None], avoid_fn[..., None], boundary_fn[..., None], values[..., None], dvdts[..., None], dvdss), dim=-1)
            # # inputs = torch.cat((states, boundary_fn[..., None], values[..., None], dvdts[..., None], dvdss), dim=-1)

            # # outputs = torch.sigmoid(BRT_model(inputs).squeeze()).detach().cpu()

            # # binned_outputs = torch.clone(outputs)
            # # for i in range(len(bins)-1):
            # #     binned_outputs[(outputs >= bins[i]) * (outputs < bins[i+1])] = (bins[i] + bins[i+1])/2

            # # for i in range(len(bins)-1):
            # #     mask = (binned_outputs >= bins[i]) * (binned_outputs < bins[i+1])
            # #     fig = px.scatter_3d(x=states[:, 0][mask], y=states[:, 1][mask], z=states[:, 2][mask], color=binned_outputs[mask], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0, range_x=[-1, 1], range_y=[-1, 1], range_z=[-np.pi, np.pi])
            # #     fig.write_html(os.path.join(testing_dir, f'BRT_violation_predictor_bin_{i}_at_tMax_for_{checkpoint_toload}.html'))

            # # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=binned_outputs, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0, range_x=[-1, 1], range_y=[-1, 1], range_z=[-np.pi, np.pi])
            # # fig.write_html(os.path.join(testing_dir, f'BRT_violation_predictor_at_tMax_for_{checkpoint_toload}.html'))
        
            # # # plot bins of error model
            # # BRT_model = MLP(input_size=9)
            # # BRT_model.load_state_dict(torch.load(os.path.join(self.experiment_dir, f'BRT_transformed_{factor}_weight_{weight_factor}_violation_predictor_at_tMax_for_{checkpoint_toload}.pth')))
            # # BRT_model.eval()

            # # bins = np.linspace(0, 1, 11)

            # # # plot error model predictions

            # # # load ground truth
            # # ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
            # # if 'gmat' in ground_truth:
            # #     ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
            # #     ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
            # #     ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
            # #     ground_truth_values = ground_truth['data']
            # #     ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
            # # elif 'g' in ground_truth:
            # #     ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
            # #     ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
            # #     ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
            # #     ground_truth_ts = ground_truth['tau'][0]
            # #     ground_truth_values = ground_truth['data']

            # # # idxs to plot
            # # z_res = 5
            # # x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
            # # y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
            # # z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
            # # t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # # # indexed ground truth to plot
            # # ground_truth_xs = ground_truth_xs[x_idxs]
            # # ground_truth_ys = ground_truth_ys[y_idxs]
            # # ground_truth_zs = ground_truth_zs[z_idxs]
            # # ground_truth_ts = ground_truth_ts[t_idxs]
            # # ground_truth_values = ground_truth_values[
            # #     x_idxs[:, None, None, None], 
            # #     y_idxs[None, :, None, None], 
            # #     z_idxs[None, None, :, None],
            # #     t_idxs[None, None, None, :]]

            # # ground_truth_xys = torch.cartesian_prod(torch.tensor(ground_truth_xs), torch.tensor(ground_truth_ys))
            # # plot_config = self.dataset.dynamics.plot_config()
        
            # # fig = plt.figure(figsize=(5*len(ground_truth_ts), 5*len(ground_truth_zs)))
            # # for i in range(len(ground_truth_ts)):
            # #     for j in range(len(ground_truth_zs)):
            # #         coords = torch.zeros(ground_truth_xys.shape[0], self.dataset.dynamics.state_dim + 1)
            # #         coords[:, 0] = ground_truth_ts[i]
            # #         coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            # #         coords[:, 1 + plot_config['x_axis_idx']] = ground_truth_xys[:, 0]
            # #         coords[:, 1 + plot_config['y_axis_idx']] = ground_truth_xys[:, 1]
            # #         coords[:, 1 + plot_config['z_axis_idx']] = ground_truth_zs[j]

            # #         model_coords = self.dataset.dynamics.normalize_coord(coords)
            # #         model_results = self.model({'coords': model_coords.cuda()})
            # #         model_dvs = diff_operators.jacobian(model_results['model_out'], model_results['model_in'])[0].detach()
            # #         dvdts = self.dataset.dynamics.unnormalize_dvdt(model_dvs[..., 0, 0]).detach().cpu()
            # #         dvdss = self.dataset.dynamics.unnormalize_dvds(model_dvs[..., 0, 1:]).detach().cpu()
            # #         values = self.dataset.dynamics.output_to_value(model_results['model_out'][:, 0]).detach().cpu()
            # #         boundary_values = self.dataset.dynamics.boundary_fn(coords[..., 1:])

            # #         # input vector is [*state, boundary_value, value, dvdt, *dvds]
            # #         inputs = torch.cat((coords[..., 1:], boundary_values[:, None], values[:, None], dvdts[:, None], dvdss), dim=-1)
            # #         outputs = torch.sigmoid(BRT_model(inputs).squeeze())
                    
            # #         ax = fig.add_subplot(len(ground_truth_ts), len(ground_truth_zs), (j+1) + i*len(ground_truth_zs))
            # #         ax.set_title('t = %0.2f, %s = %0.2f' % (ground_truth_ts[i], plot_config['state_labels'][plot_config['z_axis_idx']], ground_truth_zs[j]), fontsize=4)

            # #         image = np.full((*values.shape, 3), 255, dtype=int)

            # #         values = values.numpy()

            # #         # per bin, color accordingly
            # #         for k in range(len(bins) - 1):
            # #             mask = ((outputs >= bins[k])*(outputs < bins[k+1])).numpy()
            # #             # import colorsys
            # #             # h, s, v = 240/360, 1.0-0.8*(k/(len(bins)-2)), 255
            # #             # color = np.array(colorsys.hsv_to_rgb(h, s, v), dtype=int)
            # #             # image[mask] = color
            # #             image[mask] = np.array([255, int(200*(k/(len(bins)-2))), 0])
            # #         image[values > 0] = np.array([0, 200, 255])
            # #         image[values > 0.0287] = np.array([0, 0, 255])
                    
            # #         # reshape image
            # #         image = image.reshape(ground_truth_xs.shape[0], ground_truth_ys.shape[0], 3).transpose(1, 0, 2)

            # #         # overlay the true boundary
            # #         ground_truth_values_slice = ground_truth_values[:, :, j, i] < 0
            # #         for x in range(ground_truth_values_slice.shape[0]):
            # #             for y in range(ground_truth_values_slice.shape[1]):
            # #                 if not ground_truth_values_slice[x, y]:
            # #                     continue
            # #                 neighbors = [
            # #                     (x, y+1),
            # #                     (x, y-1),
            # #                     (x+1, y+1),
            # #                     (x+1, y),
            # #                     (x+1, y-1),
            # #                     (x-1, y+1),
            # #                     (x-1, y),
            # #                     (x-1, y-1),
            # #                 ]
            # #                 for neighbor in neighbors:
            # #                     if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_values_slice.shape[0] and neighbor[1] < ground_truth_values_slice.shape[1]:
            # #                         if not ground_truth_values_slice[neighbor]:
            # #                             image[neighbor[1], neighbor[0]] = image[neighbor[1], neighbor[0]] / 2
            # #                             break

            # #         ax.imshow(image, origin='lower', extent=(-1., 1., -1., 1.))
            # #         ax.set_xticks([-1, 1])
            # #         ax.set_yticks([-1, 1])
            # #         ax.tick_params(labelsize=6)
            # #         if j != 0:
            # #             ax.set_yticks([])

            # # fig.savefig(os.path.join(testing_dir, f'mlp_bins_with_ground_truth_at_{self.dataset.tMax}_for_{checkpoint_toload}.png'), dpi=800)

            # ### STOP ###
            
            # # plot value function in 3D
            # with torch.no_grad():
            #     states = torch.zeros(int(num_scenarios/10), self.dataset.dynamics.state_dim)
            #     for dim in range(self.dataset.dynamics.state_dim):
            #         states[..., dim].uniform_(*self.dataset.dynamics.state_test_range()[dim])
            #     times = torch.full((int(num_scenarios/10), 1), self.dataset.tMax)
            #     coords = torch.cat((times, states), dim=-1)

            #     model_coords = self.dataset.dynamics.normalize_coord(coords)
            #     model_results = self.model({'coords': model_coords.cuda()})

            #     values = self.dataset.dynamics.output_to_value(model_results['model_out'][..., 0])
            #     fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=values.cpu(), opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)
            #     fig.write_html(os.path.join(testing_dir, f'value_function_at_tMax_for_{checkpoint_toload}.html'))

            # # fraction, error, error_rate by BRT boundary levels at tMax

            # # set different seed
            # import random
            # torch.manual_seed(9455728)
            # random.seed(9455728)
            # np.random.seed(9455728)

            # correction_level = -2.5
            # if (correction_level is not np.NaN) and (correction_level <= 1e-12):
            #     level_boundaries = np.linspace(2*correction_level, 0, 11)
            #     print('sampling across BRT boundary at tMax')
            #     print('correction_level', correction_level)
            #     print('level_boundaries', level_boundaries)
                
            #     fractions = np.zeros(len(level_boundaries)-1)
            #     errors = np.zeros(len(level_boundaries)-1)
            #     error_rates = np.zeros(len(level_boundaries)-1)
            #     for i in range(len(errors)):
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #             sample_validator=ValueThresholdValidator(v_min=level_boundaries[i], v_max=level_boundaries[i+1]), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=10000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             fractions[i] = results['valid_sample_fraction']
            #             errors[i] = results['max_violation_error']
            #             error_rates[i] = results['violation_rate']
            #         else:
            #             fractions[i] = np.NaN
            #             errors[i] = np.NaN
            #             error_rates[i] = np.NaN
                
            #     region_centers = level_boundaries[:-1] + ((level_boundaries[1] - level_boundaries[0])/2)

            #     fractions_ax = plt.subplot(311)
            #     plt.plot(region_centers, fractions, color='red', marker='.')
            #     # plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     fractions_ax.set_ylabel('Fraction of Test State Space', fontsize=6)
            #     plt.title(f'Sampling within BRT Boundary Regions at t={self.dataset.tMax}')
            #     for x,y in zip(region_centers, fractions):
            #         plt.annotate('%.6f' % y, xy=(x,y), fontsize=4)

            #     errors_ax = plt.subplot(312, sharex=fractions_ax)
            #     plt.plot(region_centers, errors, color='red', marker='.')
            #     # plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     errors_ax.set_ylabel('BRT Error', fontsize=6)
            #     for x,y in zip(region_centers, errors):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)

            #     error_rates_ax = plt.subplot(313, sharex=fractions_ax)
            #     plt.plot(region_centers, error_rates, color='red', marker='.')
            #     # plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.xticks(level_boundaries, rotation=90)
            #     error_rates_ax.set_ylabel('BRT Error Rate', fontsize=6)

            #     for x,y in zip(region_centers, error_rates):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)
                
            #     plt.xlabel('Value Boundary')
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(testing_dir, f'BRT_boundary_regions_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            #     plt.clf()
            # else:
            #     print('either could not find enough samples within the BRT to perform boundary analysis or the correction level is 0')

            # # fraction, error, error_rate by exBRT boundary levels at tMax
            # correction_level = 4.2997
            # if (correction_level is not np.NaN) and (correction_level >= 1e-12):
            #     level_boundaries = np.linspace(0, 2*correction_level, 11)
            #     print('sampling across exBRT boundary at tMax')
            #     print('correction_level', correction_level)
            #     print('level_boundaries', level_boundaries)
                
            #     fractions = np.zeros(len(level_boundaries)-1)
            #     errors = np.zeros(len(level_boundaries)-1)
            #     error_rates = np.zeros(len(level_boundaries)-1)
            #     for i in range(len(errors)):
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_validator=ValueThresholdValidator(v_min=level_boundaries[i], v_max=level_boundaries[i+1]), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=10000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             fractions[i] = results['valid_sample_fraction']
            #             errors[i] = results['max_violation_error']
            #             error_rates[i] = results['violation_rate']
            #         else:
            #             fractions[i] = np.NaN
            #             errors[i] = np.NaN
            #             error_rates[i] = np.NaN
                
            #     region_centers = level_boundaries[:-1] + ((level_boundaries[1] - level_boundaries[0])/2)

            #     fractions_ax = plt.subplot(311)
            #     plt.plot(region_centers, fractions, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     fractions_ax.set_ylabel('Fraction of Test State Space', fontsize=6)
            #     plt.title(f'Sampling within exBRT Boundary Regions at t={self.dataset.tMax}')
            #     for x,y in zip(region_centers, fractions):
            #         plt.annotate('%.6f' % y, xy=(x,y), fontsize=4)

            #     errors_ax = plt.subplot(312, sharex=fractions_ax)
            #     plt.plot(region_centers, errors, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     errors_ax.set_ylabel('exBRT Error', fontsize=6)
            #     for x,y in zip(region_centers, errors):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)

            #     error_rates_ax = plt.subplot(313, sharex=fractions_ax)
            #     plt.plot(region_centers, error_rates, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.xticks(level_boundaries, rotation=90)
            #     error_rates_ax.set_ylabel('exBRT Error Rate', fontsize=6)

            #     for x,y in zip(region_centers, error_rates):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)
                
            #     plt.xlabel('Value Boundary')
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(testing_dir, f'exBRT_boundary_regions_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            #     plt.clf()
            # else:
            #     print('either could not find enough samples within the exBRT to perform boundary analysis or the correction level is 0')


            # # fraction, error, error_rate by BRT boundary levels at tMax

            # # set different seed
            # import random
            # torch.manual_seed(9455728)
            # random.seed(9455728)
            # np.random.seed(9455728)

            # correction_level = -50.1736
            # if (correction_level is not np.NaN) and (correction_level <= 1e-12):
            #     level_boundaries = np.linspace(2*correction_level, 0, 11)
            #     print('sampling across BRT boundary at tMax')
            #     print('correction_level', correction_level)
            #     print('level_boundaries', level_boundaries)
                
            #     fractions = np.zeros(len(level_boundaries)-1)
            #     errors = np.zeros(len(level_boundaries)-1)
            #     error_rates = np.zeros(len(level_boundaries)-1)
            #     for i in range(len(errors)):
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_validator=ValueThresholdValidator(v_min=level_boundaries[i], v_max=level_boundaries[i+1]), 
            #             violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #             max_scenarios=num_scenarios, max_samples=10000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             fractions[i] = results['valid_sample_fraction']
            #             errors[i] = results['max_violation_error']
            #             error_rates[i] = results['violation_rate']
            #         else:
            #             fractions[i] = np.NaN
            #             errors[i] = np.NaN
            #             error_rates[i] = np.NaN
                
            #     region_centers = level_boundaries[:-1] + ((level_boundaries[1] - level_boundaries[0])/2)

            #     fractions_ax = plt.subplot(311)
            #     plt.plot(region_centers, fractions, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     fractions_ax.set_ylabel('Fraction of Test State Space', fontsize=6)
            #     plt.title(f'Sampling within BRT Boundary Regions at t={self.dataset.tMax}')
            #     for x,y in zip(region_centers, fractions):
            #         plt.annotate('%.6f' % y, xy=(x,y), fontsize=4)

            #     errors_ax = plt.subplot(312, sharex=fractions_ax)
            #     plt.plot(region_centers, errors, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     errors_ax.set_ylabel('BRT Error', fontsize=6)
            #     for x,y in zip(region_centers, errors):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)

            #     error_rates_ax = plt.subplot(313, sharex=fractions_ax)
            #     plt.plot(region_centers, error_rates, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.xticks(level_boundaries, rotation=90)
            #     error_rates_ax.set_ylabel('BRT Error Rate', fontsize=6)

            #     for x,y in zip(region_centers, error_rates):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)
                
            #     plt.xlabel('Value Boundary')
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(testing_dir, f'BRT_boundary_regions_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            #     plt.clf()
            # else:
            #     print('either could not find enough samples within the BRT to perform boundary analysis or the correction level is 0')

            # # fraction, error, error_rate by exBRT boundary levels at tMax
            # correction_level = 4.2997
            # if (correction_level is not np.NaN) and (correction_level >= 1e-12):
            #     level_boundaries = np.linspace(0, 2*correction_level, 11)
            #     print('sampling across exBRT boundary at tMax')
            #     print('correction_level', correction_level)
            #     print('level_boundaries', level_boundaries)
                
            #     fractions = np.zeros(len(level_boundaries)-1)
            #     errors = np.zeros(len(level_boundaries)-1)
            #     error_rates = np.zeros(len(level_boundaries)-1)
            #     for i in range(len(errors)):
            #         results = scenario_optimization(
            #             model=self.model, dynamics=self.dataset.dynamics, 
            #             tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #             set_type=set_type, control_type=control_type, 
            #             scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #             sample_validator=ValueThresholdValidator(v_min=level_boundaries[i], v_max=level_boundaries[i+1]), 
            #             violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #             max_scenarios=num_scenarios, max_samples=10000*num_scenarios)
            #         if results['maxed_scenarios']:
            #             fractions[i] = results['valid_sample_fraction']
            #             errors[i] = results['max_violation_error']
            #             error_rates[i] = results['violation_rate']
            #         else:
            #             fractions[i] = np.NaN
            #             errors[i] = np.NaN
            #             error_rates[i] = np.NaN
                
            #     region_centers = level_boundaries[:-1] + ((level_boundaries[1] - level_boundaries[0])/2)

            #     fractions_ax = plt.subplot(311)
            #     plt.plot(region_centers, fractions, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     fractions_ax.set_ylabel('Fraction of Test State Space', fontsize=6)
            #     plt.title(f'Sampling within exBRT Boundary Regions at t={self.dataset.tMax}')
            #     for x,y in zip(region_centers, fractions):
            #         plt.annotate('%.6f' % y, xy=(x,y), fontsize=4)

            #     errors_ax = plt.subplot(312, sharex=fractions_ax)
            #     plt.plot(region_centers, errors, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.tick_params('x', labelbottom=False)
            #     errors_ax.set_ylabel('exBRT Error', fontsize=6)
            #     for x,y in zip(region_centers, errors):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)

            #     error_rates_ax = plt.subplot(313, sharex=fractions_ax)
            #     plt.plot(region_centers, error_rates, color='red', marker='.')
            #     plt.axvline(correction_level, color='black', linestyle='--')
            #     plt.xticks(level_boundaries, rotation=90)
            #     error_rates_ax.set_ylabel('exBRT Error Rate', fontsize=6)

            #     for x,y in zip(region_centers, error_rates):
            #         plt.annotate('%.4f' % y, xy=(x,y), fontsize=4)
                
            #     plt.xlabel('Value Boundary')
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(testing_dir, f'exBRT_boundary_regions_at_tMax_for_{checkpoint_toload}.png'), dpi=800)
            #     plt.clf()
            # else:
            #     print('either could not find enough samples within the exBRT to perform boundary analysis or the correction level is 0')

            # # plot BRT violations
            # results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=200000, sample_batch_size=200000, 
            #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #     max_violations=num_violations, max_scenarios=10000*num_violations, max_samples=1000000*num_violations)
            # states = results['states']
            # values = results['values']
            # true_values = results['true_values']
            # violations = results['violations']     
            # fig = px.scatter_3d(x=states[violations, 0], y=states[violations, 1], z=states[violations, 2], color=(true_values-values)[violations], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            # fig.write_html(os.path.join(testing_dir, f'BRT_errors_at_tMax_for_{checkpoint_toload}_with_v{int(torch.sum(violations))}s{int(states.shape[0])}.html')) 

            # # plot exBRT violations
            # results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=200000, sample_batch_size=200000, 
            #     sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #     violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #     max_violations=num_violations, max_scenarios=10000*num_violations, max_samples=1000000*num_violations)
            # states = results['states']
            # values = results['values']
            # true_values = results['true_values']
            # violations = results['violations']     
            # fig = px.scatter_3d(x=states[violations, 0], y=states[violations, 1], z=states[violations, 2], color=(values-true_values)[violations], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0)     
            # fig.write_html(os.path.join(testing_dir, f'exBRT_errors_at_tMax_for_{checkpoint_toload}_with_v{int(torch.sum(violations))}s{int(states.shape[0])}.html')) 

            # # plot hi res

            # # load ground truth
            # ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
            # if 'gmat' in ground_truth:
            #     ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
            #     ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
            #     ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
            #     ground_truth_values = ground_truth['data']
            #     ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
            # elif 'g' in ground_truth:
            #     ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
            #     ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
            #     ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
            #     ground_truth_ts = ground_truth['tau'][0]
            #     ground_truth_values = ground_truth['data']

            # # idxs to plot
            # z_res = 5
            # x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
            # y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
            # z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
            # t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # # indexed ground truth to plot
            # ground_truth_xs = np.linspace(-1, 1, 512)
            # ground_truth_ys = np.linspace(-1, 1, 512)
            # ground_truth_zs = ground_truth_zs[z_idxs]
            # ground_truth_ts = ground_truth_ts[t_idxs]

            # # ground_truth_xs = np.linspace(-1, 1, 512)
            # # ground_truth_ys = np.linspace(-1, 1, 512)
            # # ground_truth_zs = np.linspace(-np.pi, np.pi, 5)
            # # ground_truth_ts = np.linspace(0, 1, 3)
            # # x_idxs = np.arange(512)
            # # y_idxs = np.arange(512)
            # # z_idxs = np.arange(5)
            # # t_idxs = np.arange(3)

            # # gather BRT/exBRT levels
            # # time_BRT_levels = []
            # # time_exBRT_levels = []
            # # for t in ground_truth_ts:
            # #     BRT_results = scenario_optimization(
            # #         model=self.model, dynamics=self.dataset.dynamics, 
            # #         tMin=self.dataset.tMin, t=t, dt=dt, 
            # #         set_type=set_type, control_type=control_type, 
            # #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            # #         sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            # #         violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            # #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            # #     exBRT_results = scenario_optimization(
            # #         model=self.model, dynamics=self.dataset.dynamics, 
            # #         tMin=self.dataset.tMin, t=t, dt=dt, 
            # #         set_type=set_type, control_type=control_type, 
            # #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            # #         sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            # #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            # #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            # #     if BRT_results['maxed_scenarios']:
            # #         time_BRT_levels.append(-BRT_results['max_violation_error'])
            # #     else:
            # #         time_BRT_levels.append(float('-inf'))
            # #     if exBRT_results['maxed_scenarios']:
            # #         time_exBRT_levels.append(exBRT_results['max_violation_error'])
            # #     else:
            # #         time_exBRT_levels.append(float('inf'))

            # ground_truth_xys = torch.cartesian_prod(torch.tensor(ground_truth_xs), torch.tensor(ground_truth_ys))
            # plot_config = self.dataset.dynamics.plot_config()
        
            # fig = plt.figure(figsize=(5*len(ground_truth_ts), 5*len(ground_truth_zs)))
            # for i in range(len(ground_truth_ts)):
            #     for j in range(len(ground_truth_zs)):
            #         coords = torch.zeros(ground_truth_xys.shape[0], self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = ground_truth_ts[i]
            #         coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            #         coords[:, 1 + plot_config['x_axis_idx']] = ground_truth_xys[:, 0]
            #         coords[:, 1 + plot_config['y_axis_idx']] = ground_truth_xys[:, 1]
            #         coords[:, 1 + plot_config['z_axis_idx']] = ground_truth_zs[j]

            #         model_coords = self.dataset.dynamics.normalize_coord(coords)
            #         with torch.no_grad():
            #             values = self.dataset.dynamics.output_to_value(self.model({'coords': model_coords.cuda()})['model_out'][:, 0]).cpu().numpy()
                    
            #         ax = fig.add_subplot(len(ground_truth_ts), len(ground_truth_zs), (j+1) + i*len(ground_truth_zs))
            #         ax.set_title('t = %0.2f, %s = %0.2f' % (ground_truth_ts[i], plot_config['state_labels'][plot_config['z_axis_idx']], ground_truth_zs[j]), fontsize=4)

            #         image = np.full((len(ground_truth_xs)*len(ground_truth_ys), 3), 255, dtype=int)

            #         image[values < 0] = np.array([255, 0, 0])
            #         image[values > 0] = np.array([0, 0, 255])

            #         # # BRT
            #         # image[np.logical_and(values < time_BRT_levels[i], values < 0)] = np.array([255, 0, 0])

            #         # # BRT error region
            #         # image[np.logical_and(values >= time_BRT_levels[i], values < 0)] = np.array([255, 200, 0])

            #         # # exBRT
            #         # image[np.logical_and(values > time_exBRT_levels[i], values > 0)] = np.array([0, 0, 255])

            #         # # exBRT error region
            #         # image[np.logical_and(values <= time_exBRT_levels[i], values > 0)] = np.array([0, 200, 255])

            #         # reshape image
            #         image = image.reshape(ground_truth_xs.shape[0], ground_truth_ys.shape[0], 3).transpose(1, 0, 2)

            #         ax.imshow(image, origin='lower', extent=(-1., 1., -1., 1.))
            #         ax.set_xticks([-1, 1])
            #         ax.set_yticks([-1, 1])
            #         ax.tick_params(labelsize=6)
            #         if j != 0:
            #             ax.set_yticks([])


            # fig.savefig(os.path.join(testing_dir, f'ground_truth_at_{self.dataset.tMax}_for_{checkpoint_toload}.png'), dpi=800)




            # # plot slices with ground truth

            # # load ground truth
            # ground_truth = spio.loadmat(os.path.join(self.experiment_dir, 'ground_truth.mat'))
            # if 'gmat' in ground_truth:
            #     ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
            #     ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
            #     ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
            #     ground_truth_values = ground_truth['data']
            #     ground_truth_ts = np.linspace(0, 1, ground_truth_values.shape[3])
            # elif 'g' in ground_truth:
            #     ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
            #     ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
            #     ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
            #     ground_truth_ts = ground_truth['tau'][0]
            #     ground_truth_values = ground_truth['data']

            # # idxs to plot
            # z_res = 5
            # x_idxs = np.linspace(0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
            # y_idxs = np.linspace(0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
            # z_idxs = np.linspace(0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
            # t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # # indexed ground truth to plot
            # ground_truth_xs = ground_truth_xs[x_idxs]
            # ground_truth_ys = ground_truth_ys[y_idxs]
            # ground_truth_zs = ground_truth_zs[z_idxs]
            # ground_truth_ts = ground_truth_ts[t_idxs]
            # ground_truth_values = ground_truth_values[
            #     x_idxs[:, None, None, None], 
            #     y_idxs[None, :, None, None], 
            #     z_idxs[None, None, :, None],
            #     t_idxs[None, None, None, :]]

            # # ground_truth_xs = np.linspace(-1, 1, 512)
            # # ground_truth_ys = np.linspace(-1, 1, 512)
            # # ground_truth_zs = np.linspace(-np.pi, np.pi, 5)
            # # ground_truth_ts = np.linspace(0, 1, 3)
            # # x_idxs = np.arange(512)
            # # y_idxs = np.arange(512)
            # # z_idxs = np.arange(5)
            # # t_idxs = np.arange(3)

            # # gather BRT/exBRT levels
            # time_BRT_levels = []
            # time_exBRT_levels = []
            # for t in ground_truth_ts:
            #     BRT_results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=t, dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #         violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #     exBRT_results = scenario_optimization(
            #         model=self.model, dynamics=self.dataset.dynamics, 
            #         tMin=self.dataset.tMin, t=t, dt=dt, 
            #         set_type=set_type, control_type=control_type, 
            #         scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #         sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
            #         sample_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')), 
            #         violation_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
            #         max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            #     if BRT_results['maxed_scenarios']:
            #         time_BRT_levels.append(-BRT_results['max_violation_value_mag'])
            #     else:
            #         time_BRT_levels.append(float('-inf'))
            #     if exBRT_results['maxed_scenarios']:
            #         time_exBRT_levels.append(exBRT_results['max_violation_value_mag'])
            #     else:
            #         time_exBRT_levels.append(float('inf'))

            # ground_truth_xys = torch.cartesian_prod(torch.tensor(ground_truth_xs), torch.tensor(ground_truth_ys))
            # plot_config = self.dataset.dynamics.plot_config()
        
            # fig = plt.figure(figsize=(5*len(ground_truth_ts), 5*len(ground_truth_zs)))
            # for i in range(len(ground_truth_ts)):
            #     for j in range(len(ground_truth_zs)):
            #         coords = torch.zeros(ground_truth_xys.shape[0], self.dataset.dynamics.state_dim + 1)
            #         coords[:, 0] = ground_truth_ts[i]
            #         coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            #         coords[:, 1 + plot_config['x_axis_idx']] = ground_truth_xys[:, 0]
            #         coords[:, 1 + plot_config['y_axis_idx']] = ground_truth_xys[:, 1]
            #         coords[:, 1 + plot_config['z_axis_idx']] = ground_truth_zs[j]

            #         with torch.no_grad():
            #             model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
            #             values = self.dataset.dynamics.io_to_value(model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).cpu().numpy()
                    
            #         ax = fig.add_subplot(len(ground_truth_ts), len(ground_truth_zs), (j+1) + i*len(ground_truth_zs))
            #         ax.set_title('t = %0.2f, %s = %0.2f' % (ground_truth_ts[i], plot_config['state_labels'][plot_config['z_axis_idx']], ground_truth_zs[j]), fontsize=4)

            #         image = np.full((*values.shape, 3), 255, dtype=int)

            #         # BRT
            #         image[np.logical_and(values < time_BRT_levels[i], values < 0)] = np.array([255, 0, 0])

            #         # BRT error region
            #         image[np.logical_and(values >= time_BRT_levels[i], values < 0)] = np.array([255, 200, 0])

            #         # exBRT
            #         image[np.logical_and(values > time_exBRT_levels[i], values > 0)] = np.array([0, 0, 255])

            #         # exBRT error region
            #         image[np.logical_and(values <= time_exBRT_levels[i], values > 0)] = np.array([0, 200, 255])

            #         # reshape image
            #         image = image.reshape(ground_truth_xs.shape[0], ground_truth_ys.shape[0], 3).transpose(1, 0, 2)

            #         # overlay the true boundary
            #         ground_truth_values_slice = ground_truth_values[:, :, j, i] < 0
            #         for x in range(ground_truth_values_slice.shape[0]):
            #             for y in range(ground_truth_values_slice.shape[1]):
            #                 if not ground_truth_values_slice[x, y]:
            #                     continue
            #                 neighbors = [
            #                     (x, y+1),
            #                     (x, y-1),
            #                     (x+1, y+1),
            #                     (x+1, y),
            #                     (x+1, y-1),
            #                     (x-1, y+1),
            #                     (x-1, y),
            #                     (x-1, y-1),
            #                 ]
            #                 for neighbor in neighbors:
            #                     if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_values_slice.shape[0] and neighbor[1] < ground_truth_values_slice.shape[1]:
            #                         if not ground_truth_values_slice[neighbor]:
            #                             image[y, x] = image[y, x] / 2
            #                             break

            #         ax.imshow(image, origin='lower', extent=(-1., 1., -1., 1.))
            #         ax.set_xticks([-1, 1])
            #         ax.set_yticks([-1, 1])
            #         ax.tick_params(labelsize=6)
            #         if j != 0:
            #             ax.set_yticks([])

            # fig.savefig(os.path.join(testing_dir, f'slices_with_ground_truth_at_{self.dataset.tMax}_for_{checkpoint_toload}.png'), dpi=800)

            # results = scenario_optimization(
            #     model=self.model, dynamics=self.dataset.dynamics, 
            #     tMin=self.dataset.tMin, t=self.dataset.tMax, dt=dt, 
            #     set_type=set_type, control_type=control_type, 
            #     scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000), 
            #     sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=0.0), 
            #     violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
            #     max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
            # if results['maxed_scenarios']:
            #     # plot value function error within BRT in 3D
            #     states = results['states']
            #     values = results['values']
            #     true_values = results['true_values']
            #     errors = true_values-values
            #     # mask = errors < -0.025
            #     mask = torch.full(errors.shape, 1)

            #     # fig = px.scatter_3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], color=values, opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0, range_color=[-0.3, 0.3])     
            #     # fig.write_html(os.path.join(testing_dir, f'BRT_colorbar_value_function_at_t{self.dataset.tMax:.1f}_for_{checkpoint_toload}.html'))  

            #     fig = px.scatter_3d(x=states[:, 0][mask], y=states[:, 1][mask], z=states[:, 2][mask], color=errors[mask], opacity=0.2, size_max=0.000001, color_continuous_scale='Bluered', color_continuous_midpoint=0, range_color=[-0.03, 0.03])     
            #     fig.write_html(os.path.join(testing_dir, f'BRT_colorbar_value_function_errors_at_t{self.dataset.tMax:.1f}_for_{checkpoint_toload}.html'))  

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class VerifyDeepReach(Experiment):
    def init_special(self):
        pass