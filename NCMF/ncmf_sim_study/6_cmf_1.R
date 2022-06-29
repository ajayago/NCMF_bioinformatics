require("CMF")
require("itertools")


base_dir <- "/data/ragu/bioinf/ncmf-main/NCMF/sample_data_NCMF_v2/ncmf_sim_data/"
base_dir_cmf <- paste(base_dir,"cmf/",sep="")
base_dir_cmf_out <- paste(base_dir_cmf,"out/",sep="")

dir.create(base_dir_cmf_out)

for(dataset_name in list('dt1', 'ds1', 'ds2', 'ds3', 'dn1', 'dn2', 'dn3')){
  cat("\n-------------- \n")
  cat("dataset_name: ",dataset_name,"\n")
  cat("-------------- \n")
  
  out_dir <- paste(base_dir_cmf_out,dataset_name,'/',sep="")
  in_dir <- paste(base_dir_cmf,dataset_name,'/',sep="")
  cat("\n")
  cat("out_dir: ",out_dir)
  cat("\n")  
  cat("in_dir: ",in_dir)
  cat("\n")  
  
  dir.create(out_dir)
  
  fname <- paste(in_dir,"0.csv",sep="")
  X0 = as.matrix(read.csv(fname, sep=","))
  fname <- paste(in_dir,"1.csv",sep="")
  X1 = as.matrix(read.csv(fname, sep=","))
  fname <- paste(in_dir,"2.csv",sep="")
  X2 = as.matrix(read.csv(fname, sep=","))
  fname <- paste(in_dir,"3.csv",sep="")
  X3 = as.matrix(read.csv(fname, sep=","))
  fname <- paste(in_dir,"4.csv",sep="")
  X4 = as.matrix(read.csv(fname, sep=","))
  
  cat("\n")
  cat("dim(X0): ",dim(X0))
  cat("\n")
  cat("dim(X1): ",dim(X1))
  cat("\n")
  cat("dim(X2): ",dim(X2))
  cat("\n")
  cat("dim(X3): ",dim(X3))
  cat("\n")
  cat("dim(X4): ",dim(X4))
  cat("\n")
  
  cat("loading triplets started... \n")
  ptm <- proc.time()
  X0_triplets = matrix_to_triplets(X0)
  X1_triplets = matrix_to_triplets(X1)
  X2_triplets = matrix_to_triplets(X2)
  X3_triplets = matrix_to_triplets(X3)
  X4_triplets = matrix_to_triplets(X4)
  temp <- proc.time() - ptm
  cat("triplets loading completed. Took ",temp['elapsed']/60," mins. \n")
  
  #{'0':200, '1':500, '2':300, '3':400, '4':700, '5':600}
  D <- c(200,500,300,400,700,600)
  inds <- matrix(0,nrow=5,ncol=2)
  #note index starts from 1 and NOT 0
  inds[1,] <- c(1,2) 
  inds[2,] <- c(1,3) 
  inds[3,] <- c(1,4) 
  inds[4,] <- c(5,2)
  inds[5,] <- c(6,3)
  
  #build triplets list of X list
  triplets <- list()
  triplets[[1]] = X0_triplets
  triplets[[2]] = X1_triplets
  triplets[[3]] = X2_triplets
  triplets[[4]] = X3_triplets
  triplets[[5]] = X4_triplets
  
  K <- 128
  cat("K: ",K,"\n")
  #{'0':"real", '1':"real", '2':"binary", '3':"real", '4':"binary"}
  likelihood <- c("gaussian","gaussian","bernoulli","gaussian","bernoulli")
  opts <- getCMFopts()
  opts$method <- "CMF"
  opts$iter.max <- 200
  opts$verbose <- 2
  
  cat("opts: ")
  print(opts)
  
  cat("CMF model building started... \n")
  ptm <- proc.time()
  model <- CMF(triplets,inds,K,likelihood,D,opts=opts)
  temp <- proc.time() - ptm
  cat("CMF model building completed. Took ",temp['elapsed']/60," mins. \n")
  
  #G = {"0":["0","1"],"1":["0","2"],"2":["0","3"],"3":["4","1"],"4":["5","2"]}
  g <- c(
    "1"=list(1,2),
    "3"=list(1,4)
  )
  
  #[0,2]
  target_matrix_index <- list(1,3)
  for(cur_test_mat_idx in target_matrix_index){
    print("cur_test_mat_idx")
    print(cur_test_mat_idx)
    #
    cur_row_key <- paste(as.character(cur_test_mat_idx),"1",sep="")
    cur_col_key <- paste(as.character(cur_test_mat_idx),"2",sep="")
    #
    cur_row_eid <- as.numeric(g[cur_row_key])
    cur_col_eid <- as.numeric(g[cur_col_key])
    #
    row_bias_mu_list <- model$bias[[cur_test_mat_idx]][[1]]$mu
    col_bias_mu_list <- model$bias[[cur_test_mat_idx]][[2]]$mu
    row_bias_vec = matrix(c(row_bias_mu_list),nrow=length(row_bias_mu_list))
    col_bias_vec = matrix(c(col_bias_mu_list),nrow= 1, ncol=length(col_bias_mu_list))
    row_bias_mat <- matrix(rep(row_bias_vec,length(col_bias_mu_list)), ncol = length(col_bias_mu_list)) #repeat row * col wise
    col_bias_mat <- matrix(rep(col_bias_vec,length(row_bias_mu_list)), ncol = length(row_bias_mu_list)) #repeat col * row wise
    #predict
    X_pred <- model$U[[cur_row_eid]] %*% t(model$U[[cur_col_eid]]) + row_bias_mat + t(col_bias_mat)
    #
    fname_out <- paste(out_dir,cur_test_mat_idx,"_pred.csv",sep="")
    print("Persisting pred:")
    print(fname_out)
    write.table(X_pred, file=fname_out, row.names=FALSE, col.names=FALSE, sep=",")
    print("#")
  }
  
  #pred 
}

