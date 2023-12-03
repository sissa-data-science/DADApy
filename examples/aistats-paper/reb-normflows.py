# Import required packages
import torch
import numpy as np
import normflows as nf
import copy 
from dadapy._utils.utils import _align_arrays
import time 

def set_up_model(X):
    # Move model on GPU if available
    enable_gpu = False
    device = torch.device("mps" if torch.backends.mps.is_available() and enable_gpu else 'cpu')

    # Set up model
    K = 64
    torch.manual_seed(0)

    latent_size = X.shape[-1]
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set q0
    q0 = nf.distributions.DiagGaussian(latent_size)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    model = nfm.to(device)

    return model, device

def train_one_epoch(model, optimizer, X, epoch_index, full_dataset=True):
    
    if full_dataset: 
        optimizer.zero_grad()
        loss = model.forward_kld(X)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            
        last_loss = loss.item()

    else:
        running_loss = 0.
        last_loss = 0.
        # define a data loader
        training_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=512,
            shuffle=True,
            drop_last=True,
        )

        for i, data in enumerate(training_loader):
            x = data[0]
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = model.forward_kld(x)

            # Adjust learning weights
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                running_loss = 0.

    return last_loss

def run_nf(X, EPOCHS):

    model, device = set_up_model(X)

    # split X into train and test
    X_train = X[:int(X.shape[0] * 0.5), :]
    X_val = X[-int(X.shape[0] * 0.5):, :]

    X_train = torch.tensor(X_train).to(device)
    X_val = torch.tensor(X_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    best_vloss = 1_000_000.
    epoch_number = 0

    val_losses = []
    for epoch in range(EPOCHS):
        if (epoch+1)%10==0: print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, X_train, epoch_number)

        running_vloss = 0.0
        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            vloss = model.forward_kld(X_val)
            running_vloss += vloss
            val_losses.append(running_vloss)

        avg_vloss = running_vloss / 1.0
        if (epoch+1)%10==0: print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model = copy.deepcopy(model)

    last_log_prob = model.log_prob(torch.tensor(X).to(device)).cpu().detach().numpy()
    best_model.eval()
    best_log_prob = best_model.log_prob(torch.tensor(X).to(device)).cpu().detach().numpy()
    return best_log_prob, last_log_prob


if __name__=="__main__":
    from matplotlib import pyplot as plt
    from utils_rebuttal import den_6d


    # # import data
    data = np.genfromtxt("datasets/6d_double_well-1.2M-last_400k.txt", dtype="float32")
    X_full = data[:, :]
    log_den = np.array([np.log(den_6d(x)) for x in X_full])
    F_full = -log_den
    
    ### TOY TEST ###

    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]
    # X = data[:256, :]

    # log_den = np.array([np.log(den_6d(x)) for x in X])
    # F_true = -log_den

    # EPOCHS = 1000
    # log_prob, last_log_den  = run_nf(X, EPOCHS)
    # F = -log_prob
    # last_F = -last_log_den
    # _, F = _align_arrays(F,np.ones_like(F_true),F_true)
    # _, last_F = _align_arrays(last_F,np.ones_like(F_true),F_true)
    # print("MAE is: ", np.mean(np.abs(F - F_true)) )
    # print("MSE is: ", np.mean((F - F_true)**2))
    # print("MAE is: ", np.mean(np.abs(last_F - F_true)) )
    # print("MSE is: ", np.mean((last_F - F_true)**2))


    #### PROD RUN ####
    EPOCHS = 1000
    nreps = 3 # number of repetitions
    print("Number of repetitions: ",nreps)
    nexp = 8 # number of dataset sizes

    nsample = 5000
    print("Max batch size: ",nsample)
    N = nreps*nsample #X_full.shape[0]
    #print("N: ",N)
    rep_indices = np.arange(N)
    np.random.shuffle(rep_indices)
    rep_indices = np.array_split(rep_indices, nreps)

    # init arrays
    Nsample = np.zeros(nexp, dtype=np.int32)
    MAE_NF = np.zeros((nreps, nexp))
    MSE_NF = np.zeros((nreps, nexp))
    MAE_NF_noreg = np.zeros((nreps, nexp))
    MSE_NF_noreg = np.zeros((nreps, nexp))
    
    time_NF = np.zeros((nreps, nexp))

    # loop over dataset sizes
    for i in reversed(range(0, nexp)):
        print("# -----------------------------------------------------------------")
        # loop over repetitions
        for r in range(nreps):
            Xr = X_full[rep_indices[r]]
            Fr = F_full[rep_indices[r]]

            X_k=Xr[::2**i]
            F_anal_k = Fr[::2**i]
            Nsample[i] = X_k.shape[0]
            
            print()
            print("Batch size: ", X_k.shape[0])
            print("Repetition: ",r)

            sec = time.perf_counter()
            log_den, log_den_noreg = run_nf(X_k, EPOCHS)
            F_k = -log_den
            F_k_noreg = -log_den_noreg
            _, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
            _, F_k_noreg = _align_arrays(F_k_noreg,np.ones_like(F_anal_k),F_anal_k)
            time_NF[r,i] = time.perf_counter() - sec
            MAE_NF[r,i] = np.mean(np.abs(F_k-F_anal_k))
            MSE_NF[r,i] = np.mean((F_k-F_anal_k)**2)
            MAE_NF_noreg[r,i] = np.mean(np.abs(F_k_noreg-F_anal_k))
            MSE_NF_noreg[r,i] = np.mean((F_k_noreg-F_anal_k)**2)

            print("MAE is: ", MAE_NF[r,i])
            print("MSE is: ", MSE_NF[r,i])

            # save arrays to npz file
            np.savez("results/reb-norm_flows.npz", Nsample=Nsample, MAE_NF=MAE_NF, MSE_NF=MSE_NF, MAE_NF_noreg=MAE_NF_noreg, MSE_NF_noreg=MSE_NF_noreg, time_NF=time_NF)