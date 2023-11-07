# import mlflow
# # runs = mlflow.search_runs(filter_string="metrics.Acc > 0.7", search_all_experiments=True)
# runs = mlflow.search_runs(filter_string="metrics.Acc > 0.7", experiment_names=["tmp"])
# print(runs)

for fi in [2 ** i for i in range(7)]: print(fi)