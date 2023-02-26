import torch

# uses real units
def init_hji_loss(dynamics, minWith):
    def hji_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_loss