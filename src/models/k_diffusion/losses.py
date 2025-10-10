import torch

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, x):
    bs, n_nodes, n_dims = x.size()
    # Here x is a position tensor, and h is a dictionary with keys
    # 'categorical' and 'integer'.
    nll = generative_model(x)
    # Average over batch.
    nll = nll.mean(0)
    reg_term = torch.tensor([0.]).to(nll.device)
    mean_abs_z = 0.

    return nll, reg_term, mean_abs_z