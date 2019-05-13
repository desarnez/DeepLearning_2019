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

fig, axes = plt.subplots(2,1, figsize = (10,10), sharex = True)

axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Nb Epochs')

i = 0
for p in models:
    for use_auxLoss in [True, False]:
        model = p(10)
        axes[0].plot((loss_history[i, :]/loss_history[i, 0]).numpy(),
            label = legend(model, use_auxLoss))
        axes[1].plot(epoch_axis.numpy(), nb_errors[i].numpy(),
            label = legend(model, use_auxLoss))
        i = i+1
        if p is network.Dumb:
            break

axes[0].legend(loc = "center left", bbox_to_anchor=(1, 0.5))
axes[1].legend(loc = "center left", bbox_to_anchor=(1, 0.5))

#ax2 = plt.subplot(221)
#for k in range(3):
#    ax2.plot(range(nb_epochs*iterations), nb_errors[k].numpy())
#ax2.set_ylabel('Loss')
#ax2.set_xlabel('Nb Epochs')
#ax2.legend()
## ax2.plot(nb_errors_history.numpy())
## ax2.set_ylabel('Nb Computing Errors')

