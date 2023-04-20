from matplotlib import pyplot as plt
import pandas as pd

path = "/root/app/"
csv_name = "competition_hist_0420_192543.csv" 
df = pd.read_csv(path + csv_name)

x = df["Epoch"]
y = df["Acc"]

plt.plot(x, y)
plt.show()