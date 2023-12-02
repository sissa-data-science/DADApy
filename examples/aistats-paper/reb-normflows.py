# Import required packages
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import normflows as nf
import copy 

from matplotlib import pyplot as plt
from tqdm import tqdm
from utils_rebuttal import den_6d


batch_size = 2 ** 9
show_iter = 500
EPOCHS = 100



# import data
data = np.genfromtxt("datasets/6d_double_well-1.2M-last_400k.txt", dtype="float32")
X = data[-80000:, [0, 1, 2, 3, 4, 5]]
log_den = np.array([np.log(den_6d(x)) for x in X])

# data = np.genfromtxt("20D_and_gt_dataset_panel_C_Fig3.txt", dtype="float32")[:10000]
# X = data[:, :-1]
# log_den = np.genfromtxt("20D_and_gt_dataset_panel_C_Fig3.txt", dtype="float32")[:, -1]

# define a data loader
training_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X)),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

validation_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X)),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


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

# Move model on GPU if available
enable_gpu = False
device = torch.device("mps" if torch.backends.mps.is_available() and enable_gpu else 'cpu')

model = nfm.to(device)


def train_one_epoch(epoch_index, writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
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
        if i % show_iter == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
writer = SummaryWriter('runs/fashion_trainer_{}')#.format(timestamp))


best_vloss = 1_000_000.
epoch_number = 0
for epoch in tqdm(range(EPOCHS)):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            x = vdata[0]
            vloss = model.forward_kld(x)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'#.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# # Train model
# max_iter = 1000
# batch_size = 2 ** 9
# show_iter = 200

# best_loss = 999  # high value to ensure that first loss < min_loss
# batch_size_test = 157

# loss_hist = np.array([])


# optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
# for it in tqdm(range(max_iter)):
#     optimizer.zero_grad()

#     x = torch.tensor(get_batch(batch_size, X)).to(device)
    
#     loss = model.forward_kld(x)

#     if ~(torch.isnan(loss) | torch.isinf(loss)):
#         loss.backward()
#         optimizer.step()
    
#     loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

#     if (it + 1) % show_iter == 0:
#         x_test = torch.tensor(get_batch(batch_size_test, X)).to(device)
#         loss = model.forward_kld(x_test)
#         loss_value = loss.to('cpu').data.numpy()
#         print(it, loss_value)

#         log_prob = model.log_prob(torch.tensor(X).to(device)).cpu().detach().numpy()
#         log_prob = log_prob - np.mean(log_prob)
#         print(np.mean(np.abs(log_prob - log_den)))
        
#         torch.save({
#             'epoch': it,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss_value,
#             }, "{}_model.pt".format(it+1))
        
#         if loss_value < best_loss:
#             print("loss improved!")
#             best_loss = loss_value
            
#             best_model = copy.deepcopy(model) 



# plt.figure(figsize=(5, 5))
# plt.plot(loss_hist, label='loss')
# plt.legend()
# plt.show()


# # get predictions on full dataset and allign with true log density
# log_prob = model.log_prob(torch.tensor(X).to(device)).cpu().detach().numpy()
# log_prob = log_prob - np.mean(log_prob)


# plt.scatter(log_prob , log_den)
# plt.plot(log_den, log_den)