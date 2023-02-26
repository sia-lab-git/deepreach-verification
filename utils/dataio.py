import torch
from torch.utils.data import Dataset

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), start_time)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.counter_end))
            # make sure we always have training samples at the initial time
            times[-self.num_src_samples:, 0] = start_time
        model_coords = torch.cat((times, model_states), dim=1)        

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_masks = (model_coords[:, 0] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}