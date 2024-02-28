#######################
#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
library(dplyr)
library(ggplot2)
library(mix)
library(sf)
library(missForest)
library(viridis)
library(tidyverse)
library(doParallel)
require(proxy)
#######################
#
# 並列化演算 ----
#
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

WD <- "D:/AI/Conference"
#if (file.exists(WD) == FALSE) dir.create(WD) 
setwd(WD)
Path_figs <- paste0(WD, "Plots/")
Path_save <- paste0(WD, "Data/")


{#固定
  set.seed(100)
{#standard
ML_data <- read.csv("ML_dat.csv",fileEncoding="cp932")
ML_data_adj <- read.csv("ML_dat_adj.csv",fileEncoding="cp932")
ML_N_data <-  read.csv("ML_dat_N.csv",fileEncoding="cp932")
ML_list <- data_valid <- read.csv("ML_list.csv",fileEncoding="cp932")

unique(ML_data$Scientific.name) #322種

###### reading data
ML_data <- read.csv("ML_dat.csv",fileEncoding="cp932")
ML_data <-ML_data%>% pivot_wider(names_from = Item, values_from = Description, values_fill=NA)
ML_data <- ML_data[,3:ncol(ML_data)]
ML_data_x1 <- ML_data[,c("Scientific.name","Temperature")]
ML_data_x2 <- ML_data[,c("Scientific.name","Behavior")]
ML_data_x3 <- ML_data[,c("Scientific.name","Salinity")]
ML_data_x4 <- ML_data[,c("Scientific.name","Habitat")]

ML_data_x1 <-  tidyr::unnest(ML_data_x1, cols = c(Temperature))
ML_data_x2 <-  tidyr::unnest(ML_data_x2, cols = c(Behavior))
ML_data_x3 <-  tidyr::unnest(ML_data_x3, cols = c(Salinity))
ML_data_x4 <-  tidyr::unnest(ML_data_x4, cols = c(Habitat))

ML_data_X <- ML_list %>% left_join(ML_data_x1, by="Scientific.name") %>%
  left_join(ML_data_x2, by="Scientific.name") %>% 
  left_join(ML_data_x3, by="Scientific.name") %>%
  left_join(ML_data_x4, by="Scientific.name")
#unique(ML_data_X$Scientific.name) #OK!

#domain knowledge data
ML_data_adj <- read.csv("ML_dat_adj.csv",fileEncoding="cp932")
#unique(ML_data_adj$Item)

ML_data_adj <-ML_data_adj%>% pivot_wider(names_from = Item, values_from = Description, values_fill=NA)
ML_data_adj <- ML_data_adj[,3:ncol(ML_data_adj)]
ML_data_adj_x1 <- ML_data_adj[,c("Scientific.name","Temperature")]
ML_data_adj_x2 <- ML_data_adj[,c("Scientific.name","Behavior")]
ML_data_adj_x3 <- ML_data_adj[,c("Scientific.name","Salinity")]
ML_data_adj_x4 <- ML_data_adj[,c("Scientific.name","Habitat")]

ML_data_adj_x1 <-  tidyr::unnest(ML_data_adj_x1, cols = c(Temperature))
ML_data_adj_x2 <-  tidyr::unnest(ML_data_adj_x2, cols = c(Behavior))
ML_data_adj_x3 <-  tidyr::unnest(ML_data_adj_x3, cols = c(Salinity))
ML_data_adj_x4 <-  tidyr::unnest(ML_data_adj_x4, cols = c(Habitat))

ML_data_adj_X <- ML_list %>% left_join(ML_data_adj_x1, by="Scientific.name") %>%
  left_join(ML_data_adj_x2, by="Scientific.name") %>% 
  left_join(ML_data_adj_x3, by="Scientific.name") %>%
  left_join(ML_data_adj_x4, by="Scientific.name")
#unique(ML_data_adj_X$Scientific.name) #OK!

#numerical data
ML_N_data <-  read.csv("ML_dat_N.csv",fileEncoding="cp932")
ML_N_data <- ML_N_data %>% group_by(Scientific.name, Item) %>% summarise(Top=mean(Top,na.rm=T),Bottom=mean(Bottom,na.rm=T))%>% ungroup()
ML_N_data$Top[is.nan(ML_N_data$Top)] <- NA; ML_N_data$Bottom[is.nan(ML_N_data$Bottom)] <- NA; 
ML_N_data <- ML_N_data %>% filter(Item==c("Depth","Lat"))
ML_N_data_top <- ML_N_data %>% pivot_wider(names_from = Item, values_from = Top,names_prefix="Top_", values_fill=NA)
ML_N_data_bottom <- ML_N_data %>% pivot_wider(names_from = Item, values_from = Bottom,names_prefix="Bottom_", values_fill=NA)
ML_N_data <- ML_N_data_bottom %>% left_join(ML_N_data_top, by="Scientific.name")
ML_N_data <- ML_N_data %>% select(Scientific.name, Bottom_Depth, Top_Depth, Bottom_Lat, Top_Lat)
ML_N_data <- ML_N_data %>% group_by(Scientific.name) %>% summarise(Depth_B=mean(Bottom_Depth,na.rm=T), Depth_T=mean(Top_Depth,na.rm=T),
                                                                   Lat_B=mean(Bottom_Lat,na.rm=T), Lat_T=mean(Bottom_Lat,na.rm=T))%>% ungroup()
ML_N_data$Depth_B[is.nan(ML_N_data$Depth_B)] <- NA; ML_N_data$Depth_T[is.nan(ML_N_data$Depth_T)] <- NA;
ML_N_data$Lat_B[is.nan(ML_N_data$Lat_B)] <- NA; ML_N_data$Lat_T[is.nan(ML_N_data$Lat_T)] <- NA;

ML_data_all1 <- ML_data_X %>% left_join(ML_N_data, by="Scientific.name")
ML_data_all2 <- ML_data_adj_X %>% left_join(ML_N_data, by="Scientific.name")
}

#data table
#write.table(ML_data_all1, file="ML_data_all1.csv", sep=",")
#write.table(ML_data_all2, file="ML_data_all2.csv", sep=",")

#re-reading
all_data1 <- read.csv("ML_data_all1.csv",fileEncoding="cp932")
all_data2 <- read.csv("ML_data_all2.csv",fileEncoding="cp932")

###############################
##########analysis#############
###############################
dataset <- all_data1
Catch_his_data <- all_data1 %>% mutate(Catch_his=ifelse(LL+PS+Oth>0,1,0)) %>% select(Scientific.name,Catch_his)

##Estimating missing data value
{#missForest
dataset <- dataset[,9:ncol(dataset)]
dataset$Temperature <- as.factor(dataset$Temperature)
dataset$Behavior <- as.factor(dataset$Behavior)
dataset$Salinity <- as.factor(dataset$Salinity)
dataset$Habitat <- as.factor(dataset$Habitat)
dataset_imp <- missForest(dataset,verbose = TRUE)
}

#scaling
dataset_imp$ximp$Depth_B <- scale(dataset_imp$ximp$Depth_B)
dataset_imp$ximp$Depth_T <- scale(dataset_imp$ximp$Depth_T)
dataset_imp$ximp$Lat_B <- scale(dataset_imp$ximp$Lat_B)
dataset_imp$ximp$Lat_T <- scale(dataset_imp$ximp$Lat_T)
dataset <- dataset_imp$ximp

dataset <-data.frame(cbind(Catch_his_data,dataset))
dataset <- dataset[,-c(1)]

################################
# separate data ----
dataset$Catch_his <- factor(dataset$Catch_his)
trainNum <- createDataPartition(dataset$Catch_his, p=0.8, list=FALSE)
## training data
dataset_train <- dataset[trainNum,]
y_train = data.frame(dataset_train[,1]  ) #目的変数
colnames(y_train) <- "Catch_his" 
X_train = dataset_train[,-1] #説明変数（特徴量）
## test data
dataset_test <- dataset[-trainNum,]
y_test = dataset_test[,1]    #目的変数
X_test = dataset_test[,-1]   #説明変数（特徴量）

#
# random forest  ----
#
## モデルの学習（学習データ）
model_plr <- train(
  Catch_his ~ .,
  data = cbind(y_train, X_train),
  method = 'rf', 
  tuneLength = 10,
  trControl = trainControl(method = 'cv', 
                           number = 10)
)
## 学習結果の確認
model_plr
varImp(model_plr, scale = TRUE)
## モデルの検証（テストデータ）
pred_plr <- predict(model_plr, X_test)
confusionMatrix(reference = y_test, 
                data = pred_plr, 
                mode='everything', 
                positive="1")

#
# Neural Network ----
#
model_nnet <- train(
  Catch_his ~ .,
  data = cbind(y_train, X_train),
  method = "nnet",#"SBC"ってやつでできるのかもだけど不明 
  tuneGrid = expand.grid(size=c(1:10), 
                         decay=seq(0.1, 1, 10)),
  trControl = trainControl(method = 'cv',
                           number = 10),
  linout = FALSE,
  na.action = na.omit
)
## 学習結果の確認
model_nnet
varImp(model_nnet, scale = TRUE)
## モデルの検証（テストデータ）
pred_nnet <- predict(model_nnet, X_test)
confusionMatrix(reference = y_test,#as.factor( 
                data = pred_nnet, 
                mode='everything', 
                positive="1")

#
# XGBoost3 ----
#
## モデルの学習（学習データ）
model_xgb3 <- train(
  Catch_his ~ .,
  data = cbind(y_train, X_train),
  method = 'xgbTree', 
  tuneLength = 5,
  trControl = trainControl(method = 'cv', 
                           number = 10)
)
## 学習結果の確認
model_xgb3
varImp(model_xgb3, scale = TRUE)
## モデルの検証（テストデータ）
pred_xgb3 <- predict(model_xgb3, X_test)
confusionMatrix(reference = y_test, 
                data = pred_xgb3, 
                mode='everything', 
                positive="1")

pred_xgb3_AI <- predict(model_xgb3, dataset[,2:ncol(dataset)])

#Validation
all_name_data <- all_data1
#all_name_data <- all_data2

valid_data <- data_valid
valid_df_pul <- cbind(all_name_data$Scientific.name, pred_xgb3_AI,dataset)
colnames(valid_df_pul) <- c("Scientific.name","Catch_his_pred",colnames(dataset))
valid_df_pul$Catch_his_pred <- as.numeric(valid_df_pul$Catch_his_pred) -1 #ここで+1されてしまうので01に戻す
valid_df_pul$Catch_his <- as.numeric(valid_df_pul$Catch_his)-1
valid_df_pul <- valid_df_pul %>% dplyr::group_by(Scientific.name) %>%
  dplyr::summarise(Catch_his = mean(Catch_his), Catch_his_pred = mean(Catch_his_pred))
valid_data <- valid_df_pul %>% left_join(valid_data[,c("Scientific.name","LL","PS","Oth","Type","Source")], by="Scientific.name")
valid_data <- valid_data %>% distinct(Scientific.name, .keep_all=T) #これで2重カウントの魚種は消える
#write.table(valid_data, file="decision1_results.csv", sep=",")
#write.table(valid_data, file="decision1_results_adj.csv", sep=",")

#screening
valid_screaning <- valid_data %>% filter(Catch_his_pred > 0 | Catch_his!=Catch_his_pred)
valid_screaning_name <- valid_screaning$Scientific.name
}

#第2段階のデータの読み込み
{#seed
  set.seed(100)
{#Black mark data
  {#standard
    ML_data <- read.csv("ML_datB2.csv",fileEncoding="cp932")
    ML_N_data <-  read.csv("ML_dat_NB.csv",fileEncoding="cp932")
    ML_list <- data_valid <- read.csv("ML_listB.csv",fileEncoding="cp932")
    
    #ML data
    ML_data <- read.csv("ML_datB2.csv",fileEncoding="cp932")
    unique(ML_data$Item) 
    
    ML_data <-ML_data%>% pivot_wider(names_from = Item, values_from = Description, values_fill=NA)
    ML_data <- ML_data[,3:ncol(ML_data)]
    ML_data_x1 <- ML_data[,c("Scientific.name","Behavior")]
    ML_data_x2 <- ML_data[,c("Scientific.name","Associate")]
    ML_data_x3 <- ML_data[,c("Scientific.name","Habitat")]
    
    ML_data_x1 <-  tidyr::unnest(ML_data_x1, cols = c(Behavior))
    ML_data_x2 <-  tidyr::unnest(ML_data_x2, cols = c(Associate))
    ML_data_x3 <-  tidyr::unnest(ML_data_x3, cols = c(Habitat))

    ML_data_X <- ML_list %>% left_join(ML_data_x1, by="Scientific.name", multiple="all") %>%
      left_join(ML_data_x2, by="Scientific.name", multiple="all") %>% 
      left_join(ML_data_x3, by="Scientific.name", multiple="all")
    unique(ML_data_X$Scientific.name) #OK!
    
    #Numerical data
    ML_N_data <-  read.csv("ML_dat_NB.csv",fileEncoding="cp932")
    ML_N_Mature <- ML_N_data %>% filter(Item==c("Maturity (cm)")) %>% 
      group_by(Scientific.name) %>% summarise(Maturity_size_mean=mean(Maturity_size_mean,na.rm=T))%>% ungroup() #122種で平均サイズcmの成熟度がある
    ML_N_data <- ML_N_data %>% group_by(Scientific.name, Item) %>% summarise(Top=mean(Top,na.rm=T),Bottom=mean(Bottom,na.rm=T))%>% ungroup()
    ML_N_data$Top[is.nan(ML_N_data$Top)] <- NA; ML_N_data$Bottom[is.nan(ML_N_data$Bottom)] <- NA; 

    ML_N_data_top <- ML_N_data %>% pivot_wider(names_from = Item, values_from = Top,names_prefix="Top_", values_fill=NA)
    ML_N_data_bottom <- ML_N_data %>% pivot_wider(names_from = Item, values_from = Bottom,names_prefix="Bottom_", values_fill=NA)
    ML_N_data <- ML_N_data_bottom %>% left_join(ML_N_data_top, by="Scientific.name", multiple="all")
    ML_N_data <- ML_N_data %>% select(Scientific.name, Bottom_Depth, Top_Depth, Bottom_Lat, Top_Lat,
                                      Bottom_Size,Top_Size,Bottom_Temperature,Top_Temperature,
                                      Bottom_Reproductive_age,Top_Reproductive_age)
    ML_N_data <- ML_N_data %>% group_by(Scientific.name) %>% summarise(Depth_B=mean(Bottom_Depth,na.rm=T), Depth_T=mean(Top_Depth,na.rm=T),
                                                                       Lat_B=mean(Bottom_Lat,na.rm=T), Lat_T=mean(Top_Lat,na.rm=T),
                                                                       Size_B=mean(Bottom_Size,na.rm=T), Size_T=mean(Top_Size,na.rm=T),
                                                                       Temp_B=mean(Bottom_Temperature,na.rm=T), Temp_T=mean(Top_Temperature,na.rm=T),
                                                                       Repro_age_B=mean(Bottom_Reproductive_age,na.rm=T), Repro_age_T=mean(Top_Reproductive_age,na.rm=T))
    ML_N_data[is.na(ML_N_data)] <- NA;
    
    ML_data_all2B <- ML_data_X %>% left_join(ML_data_all2B, by="Scientific.name")
    
    #data table
    write.table(ML_data_all2B, file="ML_data_all2B.csv", sep=",")
    
    #re-reading
    all_data2B <- read.csv("ML_data_all2B.csv",fileEncoding="cp932")
    #unique(all_data2B$Scientific.name) 
  }
  dataset <- all_data2B
  {#missForest
    Scientific.name <- dataset[,c("Scientific.name")]
    dataset <- dataset[,13:ncol(dataset)]
    dataset$Behavior <- as.factor(dataset$Behavior)
    dataset$Associate <- as.factor(dataset$Associate)
    dataset$Habitat <- as.factor(dataset$Habitat)
    dataset_imp <- missForest(dataset,verbose = TRUE)
  }
  dataset <- dataset_imp$ximp
}

{#data processing
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="bathydemersal", replacement="1.2")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="benthopelagic", replacement="2")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="demersal", replacement="1")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="epibenthic", replacement="1.7")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="reef-associated", replacement="3")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="epipelagic", replacement="4.1")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="mesopelagic", replacement="4.2")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="bathypelagic", replacement="4.3")
dataset$Behavior <- dataset$Behavior %>% str_replace_all(pattern="pelagic", replacement="4")
dataset$Behavior <- as.numeric(dataset$Behavior)

dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="vegetation", replacement="1")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="benthic invertebrates", replacement="1")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="nekton", replacement="2")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="planktonic organisms", replacement="2.5")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="floating objects", replacement="3")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="large pelgagics", replacement="4")
dataset$Associate <- dataset$Associate %>% str_replace_all(pattern="seabirds", replacement="5")
dataset$Associate <- type.convert(dataset$Associate)

dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="inland", replacement="0")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="estuaries", replacement="1")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="offshore", replacement="1.7")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="inshore", replacement="1.4")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="shore", replacement="1.2")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="coral reef", replacement="1.5")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="reef-associated", replacement="1.5")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="neritic", replacement="1.5")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="coastal", replacement="2")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="continental shelf", replacement="2")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="island", replacement="2.2")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="seamount", replacement="2.2")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="continental slope", replacement="2.3")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="slope", replacement="2.3")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="open water", replacement="2.4")
dataset$Habitat <- dataset$Habitat %>% str_replace_all(pattern="oceanic", replacement="3")
dataset$Habitat <- as.numeric(dataset$Habitat)
unique(dataset$Habitat) 
}
  
dataset <- dataset  %>% select(-Associate)
dataset <- cbind(Scientific.name,dataset)
data_sec <- all_name_data <-  ML_list
}

X_data <- dataset

###data set #non-categorical
X_data <- dataset %>% select(Scientific.name,Behavior,Habitat,Depth_B,Depth_T,Lat_B,Lat_T,Size_B,Size_T,Temp_B,Temp_T,Repro_age_B,Repro_age_T,Maturity_size_mean) %>%
          group_by(Scientific.name) %>% summarise_all(mean,na.rm=T)
X_data <- data_sec %>% left_join(X_data,by="Scientific.name")

#Ovjective value LL
mean_LL <- mean(data_sec$LL,na.rm=T) #mean
sd_LL <- sd(data_sec$LL,na.rm=T) #sd
Y_data <-  as.vector(scale(data_sec$LL))

Xanalysys_data <- X_data[,13:ncol(X_data)] #data set
Xanalysys_data <- data.frame(scale(Xanalysys_data)) 

#calculating similarity for LL
dcos <- proxy::simil(Z_data,method = "cosine") #コサイン類似度を計算
summary(dcos)
cos_mat <- as.matrix(dcos)
#write.table(cos_mat, file="simirality_cosmat_LL_B.csv", sep=",")

##for PS
Xanalysys_data <- X_data[,13:ncol(X_data)] #data set
Xanalysys_data <- data.frame(scale(Xanalysys_data)) 
Y_data <-  as.vector(scale(data_sec$PS))
Z_data <- cbind(Y_data,Xanalysys_data)
dcos <- proxy::simil(Z_data,method = "cosine")
summary(dcos)
cos_mat <- as.matrix(dcos)
#write.table(cos_mat, file="simirality_cosmat_PS_B.csv", sep=",")

##for all
Xanalysys_data <- X_data[,13:ncol(X_data)] #data set
Xanalysys_data <- data.frame(scale(Xanalysys_data)) 
data_sec$Others[is.na(data_sec$Others)] <- 0
all_fishery <- data_sec$LL+data_sec$PS+data_sec$Others
Y_data <-  as.vector(scale(all_fishery))

Z_data <- Xanalysys_data 

dcos <- proxy::simil(Z_data,method = "cosine") 
summary(dcos)
cos_mat <- as.matrix(dcos)
write.table(cos_mat, file="simirality_cosmat_All_B.csv", sep=",")

#qEの指標を用いて相対的なvulneravilityを計算する
asess_data <- read.csv("assess_B.csv",fileEncoding="cp932")
YFT_q <- asess_data$qE[2] #2014の値
ALB_q <- asess_data$qE[5] #2014の値
cos_mat_all_B <- read.csv("simirality_cosmat_All_B.csv",fileEncoding="cp932")
cos_mat_all_B[is.na(cos_mat_all_B)] <- 1
cos_mat_all_B

#yellowfin tuna (Thunnus albacares)
YFT_simil <- cos_mat_all_B[,143]

relative_q_YFT <- c(YFT_simil*YFT_q)

#albacore tuna
ALB_simil <- cos_mat_all_B[,141]
relative_q_ALB <- c(ALB_simil*ALB_q)

relative_q_results <- cbind(relative_q_YFT, relative_q_ALB)
write.table(relative_q_results, file="relative_q_results_cos.csv", sep=",")
