require("CMF")
require("dplyr")
require("tidyr")
require("reticulate")

#data_folder <- '../HNE/Data/MIMIC/CMF/' # TODO
data_folder <- '../../datasets/CMF/MIMIC/sampled3/'
test_link_file <- 'sampled3_link.dat.test' # TODO
emb_file <- 'gcmf_s3_emb_MIMIC.dat' # TODO
# Create data for a circular setup with three matrices and three
# object sets of varying sizes.
print('Loading Data...')
X <- list()
#D <- c(2661, 4288, 5546, 592) # TODO dimensions
D <- c(5890, 1321, 596)
inds <- matrix(0, nrow=3, ncol=2) # entity
inds[1,] <- c(1, 2)
inds[2,] <- c(2, 3)
inds[3,] <- c(3, 1)
#inds[4,] <- c(3, 1)
#inds[5,] <- c(3, 2)
#inds[6,] <- c(3, 3)
#inds[7,] <- c(3, 4)
#inds[8,] <- c(4, 1)
#inds[9,] <- c(4, 2)
#inds[10,] <- c(4, 4)

test_link_df <- read.csv(paste0(data_folder, test_link_file), header=FALSE, sep='\t')
colnames(test_link_df) <- c('id_from', 'id_to', 'link_status')

id_idx_map <- read.csv(paste0(data_folder, 'id_idx.csv'), sep=',', header=FALSE)
colnames(id_idx_map) <- c('id', 'idx', 'node_type')
id_idx_map <- id_idx_map %>% mutate(id = id + 1, idx = idx + 1, node_type = node_type + 1) # convert to 1 based indexing

np <- import('numpy', convert=TRUE)
for (i in 1:nrow(inds)) {
  X[[i]] <- np$load(paste0(data_folder, 'X', toString(i - 1), '.npy'))
}

print('Prepare Data...')
test_link_df <- merge(test_link_df, id_idx_map, by.x='id_from', by.y='id')
test_link_df <- test_link_df[c('id_from', 'id_to', 'idx', 'link_status')]
test_link_df <- merge(test_link_df, id_idx_map, by.x='id_to', by.y='id')
test_link_df <- test_link_df[c('idx.x', 'idx.y', 'link_status')]
colnames(test_link_df) <- c('idx_from', 'idx_to', 'link_status')

X[[2]][test_link_df[, 'idx_from'], test_link_df[, 'idx_to']] <- NA # drug-drug target hidden test link

# Convert the data into the right format
triplets <- list()
for(m in 1:nrow(inds)) triplets[[m]] <- matrix_to_triplets(X[[m]])
# Missing entries correspond to missing rows in the triple representation
# so they can be removed from training data by simply taking a subset
# of the rows.
train <- list()
test <- list()
keepForTraining <- c()

for (m in 1:nrow(inds)) {
  keepForTraining[m] <- nrow(triplets[[m]]) * 1
}
for(m in 1:nrow(inds)) {
  subset <- sample(nrow(triplets[[m]]))[1:keepForTraining[m]]
  train[[m]] <- triplets[[m]][subset,]
  test[[m]] <- triplets[[m]][setdiff(1:nrow(triplets[[m]]),subset),]
}

print('Training...')
# Learn the model with the correct likelihoods
K <- 50
#likelihood <- c("poisson", "poisson", "poisson", "poisson", "poisson", "poisson", "poisson", "poisson", "poisson", "poisson")
likelihood <- c("bernoulli", "bernoulli", "bernoulli")
opts <- getCMFopts()
opts$iter.max <- 100 # Less iterations for faster computation default=200
opts$method <- "gCMF" # TODO "gCMF"
model <- CMF(train, inds, K, likelihood, D, test=test, opts=opts)

print('Saving Embedding...')
embs <- list()
for (e in 1:length(D)) {
  id_map <- id_idx_map %>% filter(node_type == e)
  id_map <- id_map[c('id')] - 1
  temp_emb <- as.data.frame(model$U[[e]])
  temp_emb <- cbind(id_map, temp_emb)
  embs[[e]] <- temp_emb
}
embs <- bind_rows(embs)
embs_sorted <- embs[order(embs$id), ]
embs_sorted <- embs_sorted %>% unite('emb', V1:V50, sep=' ', remove=TRUE)
write.table(paste0(''), emb_file, row.names=FALSE, col.names=FALSE, quote=FALSE)
write.table(embs_sorted, emb_file, sep="\t", row.names=FALSE, col.names=FALSE, append=TRUE, quote=FALSE)

# print('Results...')
# # Check the predictions
# # Note that the data created here has no low-rank structure,
# # so we should not expect good accuracy.
# print(test[[1]][1:10,])
# print(model$out[[1]][1:10,])
# 
# # predictions for the test set using the previously learned model
# out <- predictCMF(test, model)
# print(out$error[[1]])
# # ...this should be the same as the output provided by CMF()
# print(model$out[[1]][1:10,])
