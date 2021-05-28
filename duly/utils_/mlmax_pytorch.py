import torch as t
import torch.optim as optim
import numpy as np

#device = 'cuda' if t.cuda.is_available() else 'cpu'
device = 'cpu'

# ----------------------------------------------------------------------------------------------

def maximise_wPAk(Fis, kstar, vij_list, dist_indices, Fij_list, Fij_var_list, alpha, alg='BFGS'):
    N = Fis.shape[0]

    Fis = Fis + np.random.normal(0, 1, size=(N,)) * 0.1

    Fis_t = t.from_numpy(Fis).float().to(device)
    Fis_t.requires_grad_()
    PAk_ai_t = t.zeros(N, requires_grad=True, dtype=t.float, device=device)

    vijs_t = t.ones(N, N).float().to(device)
    nijs_t = t.ones(N, N).float().to(device)
    Fijs_t = t.ones(N, N).float().to(device)
    Fijs_var_t = t.ones(N, N).float().to(device)
    mask = t.zeros(N, N).int().to(device)

    for i, (Fijs, Fijs_var, vijs) in enumerate(zip(Fij_list, Fij_var_list, vij_list)):

        k = kstar[i]

        for nneigh in range(k-1):
            j = dist_indices[i, nneigh + 1]

            Fijs_t[i, j] = Fijs[nneigh]
            Fijs_var_t[i, j] = 2 * (Fijs_var[nneigh] + 0.5 * 1e-3)
            vijs_t[i, j] = vijs[nneigh]
            nijs_t[i, j] = float(nneigh + 1)
            mask[i, j] = 1

    def loss_fn():
        deltas = (Fis_t[None, :] - Fis_t[:, None])

        PAk_corr = PAk_ai_t[:, None] * nijs_t

        Fis_corr = Fis_t[:, None] + PAk_corr

        la = t.sum(mask * Fis_corr) - t.sum(mask * (vijs_t * t.exp(Fis_corr)))

        lb = alpha * t.sum(mask * ((deltas - Fijs_t) ** 2 / Fijs_var_t))

        l = la - lb
        return -l

    if alg == 'BFGS':
        lr = 0.5
        n_epochs = 50

        optimiser = optim.LBFGS([Fis_t, PAk_ai_t], lr=lr, max_iter=20, max_eval=None,
                                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)

        for e in range(n_epochs):
            def closure():
                optimiser.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            optimiser.step(closure)


    elif alg == 'GD':
        lr = 1e-7
        n_epochs = 25000

        optimiser = optim.SGD([Fis_t, PAk_ai_t], lr=lr)

        for e in range(n_epochs):

            loss = loss_fn()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            if e % 1000 == 0: print(e, loss.item())

    final_loss = loss_fn()
    return final_loss.data, Fis_t.detach().numpy()

# ----------------------------------------------------------------------------------------------

def maximise(Fis, kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha, alg='BFGS'):
    N = Fis.shape[0]

    Fis = Fis + np.random.normal(0, 1, size=(N,)) * 0.01

    Fis_t = t.from_numpy(Fis).float().to(device)
    Fis_t.requires_grad_()

    Vis_t = t.from_numpy(Vis).float().to(device)
    kstar_t = t.from_numpy(kstar).float().to(device)
    Fijs_t = t.ones(N, N).float().to(device)
    Fijs_var_t = t.ones(N, N).float().to(device)
    mask = t.zeros(N, N).int().to(device)

    for i, (Fijs, Fijs_var) in enumerate(zip(Fij_list, Fij_var_list)):

        k = kstar[i]

        for nneigh in range(k):
            j = dist_indices[i, nneigh + 1]

            Fijs_t[i, j] = Fijs[nneigh]
            Fijs_var_t[i, j] = 2 * (Fijs_var[nneigh] + 1.e-4)
            mask[i, j] = 1

    def loss_fn():

        deltas = (Fis_t[None, :] - Fis_t[:, None])

        la = t.sum(kstar_t * Fis_t - Vis_t * t.exp(Fis_t))

        lb = alpha * t.sum(mask * ((deltas - Fijs_t) ** 2 / Fijs_var_t))  #

        l = la - lb

        # l = lb
        return -l

    if alg == 'BFGS':

        lr = 0.1
        n_epochs = 50

        optimiser = optim.LBFGS([Fis_t], lr=lr, max_iter=20, max_eval=None,
                                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)

        for e in range(n_epochs):
            def closure():
                optimiser.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            optimiser.step(closure)

    elif alg == 'GD':

        lr = 1e-5
        n_epochs = 25000

        optimiser = optim.SGD([Fis_t], lr=lr)

        for e in range(n_epochs):

            loss = loss_fn()

            loss.backward()

            # with t.no_grad():
            #     Fis_t -= lr * Fis_t.grad
            # Fis_t.grad.zero_()

            optimiser.step()
            optimiser.zero_grad()
            if e % 1000 == 0: print(e, loss.item())

    final_loss = loss_fn()
    return final_loss.data, Fis_t.detach().numpy()

# ----------------------------------------------------------------------------------------------

def maximise_wPAk_flatF(Fis, Fis_err, kstar, vij_list, dist_indices, alpha, alg='BFGS', onlyNN=False):
    N = Fis.shape[0]

    Fis = Fis + np.random.normal(0, 1, size=(N,)) * 0.1

    Fis_t = t.from_numpy(Fis).float().to(device)
    Fis_t.requires_grad_()
    Fis_err_t = t.from_numpy(Fis_err).float().to(device)
    PAk_ai_t = t.zeros(N, requires_grad=True, dtype=t.float, device=device)

    vijs_t = t.ones(N, N).float().to(device)
    nijs_t = t.ones(N, N).float().to(device)
    Fijs_t = t.ones(N, N).float().to(device)
    Fijs_var_t = t.ones(N, N).float().to(device)
    mask = t.zeros(N, N).int().to(device)

    if onlyNN is False:
    #keep all neighbours up to k*
        for i, vijs in enumerate(vij_list):

            k = kstar[i]

            for nneigh in range(k-1):
                j = dist_indices[i, nneigh + 1]

                Fijs_t[i, j] = Fis_t[j]-Fis_t[i]
                Fijs_var_t[i, j] = 2 * (Fis_err[i]**2+Fis_err[j]**2)
                vijs_t[i, j] = vijs[nneigh]
                nijs_t[i, j] = float(nneigh + 1)
                mask[i, j] = 1
    else:
    #only correlate to first NN
        for i, vijs in enumerate(vij_list):

            k = kstar[i]

            for nneigh in range(1):
                j = dist_indices[i, nneigh + 1]

                Fijs_t[i, j] = Fis_t[j]-Fis_t[i]
                Fijs_var_t[i, j] = 2 * (Fis_err[i]**2+Fis_err[j]**2)
                vijs_t[i, j] = vijs[nneigh]
                nijs_t[i, j] = float(nneigh + 1)
                mask[i, j] = 1

    def loss_fn():
    #    deltas = (Fis_t[None, :] - Fis_t[:, None])

        PAk_corr = PAk_ai_t[:, None] * nijs_t

        Fis_corr = Fis_t[:, None] + PAk_corr

        la = t.sum(mask * Fis_corr) - t.sum(mask * (vijs_t * t.exp(Fis_corr)))

    #    lb = alpha * t.sum(mask * ((deltas - Fijs_t) ** 2 / Fijs_var_t))
        lb = alpha * t.sum(mask * ((Fijs_t** 2) / Fijs_var_t))

        l = la - lb
        return -l

    if alg == 'BFGS':
        lr = 0.5
        n_epochs = 50

        optimiser = optim.LBFGS([Fis_t, PAk_ai_t], lr=lr, max_iter=20, max_eval=None,
                                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)

        for e in range(n_epochs):
            def closure():
                optimiser.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            optimiser.step(closure)


    elif alg == 'GD':
        lr = 1e-7
        n_epochs = 25000

        optimiser = optim.SGD([Fis_t, PAk_ai_t], lr=lr)

        for e in range(n_epochs):

            loss = loss_fn()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            if e % 1000 == 0: print(e, loss.item())

    final_loss = loss_fn()
    return final_loss.data, Fis_t.detach().numpy()





def optimise_metric_vectors(gammaij, d=2, lr=1e-3, n_epochs=10000, alg='GD', vi_init=None):
    n_metrics = gammaij.shape[0]

    # convert the losses to tensor format
    gammaij_t = t.from_numpy(gammaij).float().to(device)

    # initialise the weights at random
    # t.manual_seed(1)
    # np.random.seed(1)

    if vi_init is not None:
        assert (vi_init.shape[0] == n_metrics and vi_init.shape[1] == d)
        vi_t = t.from_numpy(vi_init).float().to(device)
        vi_t.requires_grad_()
    else:
        vi_t = t.randn(n_metrics, d, device=device, dtype=t.float, requires_grad=True)
        # vi = np.random.normal(0, 1, size=(n_metrics, d))
        # vi0 = np.zeros(d)
        # vi0[0] = 1.
        # vi[0, :] = vi0
        # vi_t = t.from_numpy(vi).float().to(device)
        # vi_t.requires_grad_()

    # define a loss function
    def loss_fn():

        norms_i = t.norm(vi_t, p=2, dim=1)

        vin_t = vi_t / norms_i[:, None]

        vivjn_t = t.mm(vi_t, vin_t.T)

        gammaij_approx_t = norms_i[None, :] - vivjn_t

        l = 0.5 * t.sum((gammaij_t - gammaij_approx_t) ** 2)

        return l

    losses = []

    if alg == 'GD':

        optimiser = optim.SGD([vi_t], lr=lr)

        for e in range(n_epochs):

            loss = loss_fn()

            loss.backward()

            # can set gradient for first vector to zero
            # vi_t.grad[0] = 0

            optimiser.step()

            optimiser.zero_grad()

            if e % 1000 == 0:
                print(e, loss.item())
                losses.append(loss.item())


    elif alg == 'BFGS':
        optimiser = optim.LBFGS([vi_t], lr=lr, max_iter=25, max_eval=None,
                                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)

        for e in range(n_epochs):

            def closure():
                optimiser.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            optimiser.step(closure)

            if e % 100 == 0:
                with t.no_grad():
                    loss = loss_fn()
                print(e, loss.item())
                losses.append(loss.item())

    # compute final gammaij_approx
    with t.no_grad():
        final_loss = loss_fn().item()

        norms_i = t.norm(vi_t, p=2, dim=1)
        vin_t = vi_t / norms_i[:, None]
        vivjn_t = t.mm(vi_t, vin_t.T)
        gammaij_approx_t = norms_i[None, :] - vivjn_t

    return vi_t.detach().numpy(), gammaij_approx_t.detach().numpy(), losses
