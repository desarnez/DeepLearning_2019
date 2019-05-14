# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:52:50 2019

@author: Charles
"""
import matplotlib.pyplot as plt

def legend(model, use_auxLoss):
    if str(model) == 'Dumb':
        legend = str(model)
    elif use_auxLoss is True:
        legend = '{:s}_AuxLoss'.format(str(model))
    else:
        legend = '{:s}_noAuxLoss'.format(str(model))
    return legend

epoch_axis = torch.zeros(iterations)
for n in range(iterations):
    epoch_axis[n] = (n+1)*nb_epochs

fig1 = plt.figure(1)
fig, axes = plt.subplots(2,1, figsize = (10,10), sharex = True)
axes[0].set_title('Evolution of loss')
axes[0].set_ylabel('Loss')
axes[1].set_title('Evolution of test error rate')
axes[1].set_xlabel('Nb Epochs')
axes[1].set_ylabel('Nb of test errors')
fig.suptitle('Evoluttion of errors and loss rates over epochs (average of 10 iterations)')
models = [network.noSharing, network.Sharing, network.Dumb]
i = 0
for p in models:
    for use_auxLoss in [True, False]:
        model = p(10)
        axes[0].plot((loss_history.mean(0)[i, :]/loss_history.mean(0)[i, 0]).numpy(),
            label = legend(model, use_auxLoss))
        axes[1].plot(epoch_axis.numpy(), nb_errors.mean(0)[i].numpy(),
            label = legend(model, use_auxLoss))
        i = i+1
        if p is network.Dumb:
            break

axes[0].legend(loc = "center left", bbox_to_anchor=(1, 0.5))
axes[1].legend(loc = "center left", bbox_to_anchor=(1, 0.5))

fig2, ax = plt.figure(2)
plt.boxplot(loss_history[:,:,-1].numpy())
plt.xticks(range(5), ('Net1', 'Net2', 'Net3', 'Net4', 'net5'))
#ax2 = plt.subplot(221)
#for k in range(3):
#    ax2.plot(range(nb_epochs*iterations), nb_errors[k].numpy())
#ax2.set_ylabel('Loss')
#ax2.set_xlabel('Nb Epochs')
#ax2.legend()
## ax2.plot(nb_errors_history.numpy())
## ax2.set_ylabel('Nb Computing Errors')

