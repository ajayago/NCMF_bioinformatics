Ablation Study - without Zero Inflated Layer:

This folder has the source code needed to initialize an instance of NCMF, fit the dataset, create embedddings and evaluate a link prediction task, without a zero inflated layer after the autoencoder and the matrix completion network.

The loss function for row_losses, col_losses and rec_loss is substituted with RMSELoss in each case.
