from prep import Prep

data_dir = "/root/app/competition01_gray_128x128/"
data_path = {"labeled":data_dir+"train_val", "unlabeled":data_dir+"test"}
pr = Prep(data_path, batch_size=120, train_ratio=0.9)

dataiter = iter(pr.fetch_test(None))
data, labels = next(dataiter)
print(data.shape)