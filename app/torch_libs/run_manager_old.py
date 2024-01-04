from pathlib import Path
from time import time

import torch
import polars as pl


class RunManager:
    def __init__(self, pa_path=None, exc_path=None, exp_path=None, exp_name="tmp", run_id=None):
        # run = RunManager(exc_path=__file__, exp_name="sugoi_research")
        if pa_path is not None:
            pa_path = Path(pa_path).resolve()
            exp_path = pa_path / Path(exp_name)
        elif exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        if run_id is None:
            exp_path.mkdir(parents=True, exist_ok=True)
            run_id = Path(self._get_run_id(exp_path))
        run_path = exp_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.run_path = run_path
        # self.run_id = run_id

        self.start_time = time()

    def _get_run_id(self, exp_path):
        dir_names = list(exp_path.iterdir())
        dir_nums = [int(dir_name.name) for dir_name in dir_names]
        if len(dir_nums) == 0:
            run_id = 0
        else:
            run_id = max(dir_nums) + 1
        return str(run_id)

    # def set_experiment(self, exp_name):
    #     self.exp_path = self.pa_path / Path(exp_name)
    #     self.run_path = self.exp_path / Path(self._get_run_id(self.exp_path))

    def log_text(self, text, file_name):
        with open(self.run_path / Path(str(file_name)), "w") as fh:
            fh.write(text)

    def log_torch_save(self, object, file_name):
        torch.save(object, self.run_path / Path(file_name))

    def log_param(self, name, value):
        if hasattr(self, "df_params"):
            # self.df_params = pl.concat([self.df_params, pl.DataFrame({name: [value]})], how="horizontal")
            self.df_params = self.df_params.hstack(pl.DataFrame({name: [value]}))
        else:
            self.df_params = pl.DataFrame({name: [value]})

    # ひとつひとつかくより、self.run_path渡した方がいい?
    def log_df(self, df, file_name):
        df.write_csv(self.run_path / Path(file_name))

    def log_params(self, stored_dict):
        stored_dict = {k: [v] for k, v in stored_dict.items()}
        if hasattr(self, "df_params"):
            self.df_params = self.df_params.hstack(pl.DataFrame(stored_dict))
        else:
            self.df_params = pl.DataFrame(stored_dict)

    def log_metric(self, name, value, step=None):
        if hasattr(self, "df_metrics"):
            if step is None:
                step = self.df_metrics["step"].max() + 1
            if step in self.df_metrics["step"]:
                self.df_metrics = self._df_set_elem(self.df_metrics, name, "step", step, value)
                # self.df_metrics = self.df_metrics.with_columns(pl.when(self.df_metrics["step"] == step).then(value).otherwise(pl.col(name)).alias(name))
            else:
                df_tmp = pl.DataFrame({"step": [step], name: [value]})
                self.df_metrics = pl.concat([self.df_metrics, df_tmp], how="diagonal")
        else:
            if step is None:
                step = 1
            self.df_metrics = pl.DataFrame({"step": [step], name: [value]})

    def log_metrics(self, stored_dict, step=None):
        if hasattr(self, "df_metrics"):
            for name, value in stored_dict.items():
                if step is None:
                    step = self.df_metrics["step"].max() + 1
                if step in self.df_metrics["step"]:
                    self.df_metrics = self._df_set_elem(self.df_metrics, name, "step", step, value)
                else:
                    df_tmp = pl.DataFrame({"step": [step], name: [value]})
                    self.df_metrics = pl.concat([self.df_metrics, df_tmp], how="diagonal")
        else:
            if step is None:
                step = 1
            self.df_metrics = pl.DataFrame({"step": [step], name: [value]})

    def _df_set_elem(self, df, column, index_column, index, value):
        # indexは存在しなければならない。columnは存在しないとき新たに作られる。
        if column in df.columns:
            df = df.with_columns(pl.when(df[index_column] == index).then(value).otherwise(pl.col(column)).alias(column))
        else:
            df = df.with_columns(pl.when(df[index_column] == index).then(value).otherwise(pl.lit(None)).alias(column))
        return df

    def _fetch_stats(self):
        if hasattr(self, "df_params"):
            stat_params = self.df_params
            if hasattr(self, "df_metrics"):
                stat_metrics = self.df_metrics[-1].select(pl.all().exclude("step"))
                stats = stat_params.hstack(stat_metrics)
            else:
                stats = stat_params
        else:
            if hasattr(self, "df_metrics"):
                stat_metrics = self.df_metrics[-1].select(pl.all().exclude("step"))
                stats = stat_metrics
            else:
                stats = pl.DataFrame()
        return stats

    def ref_stats(self, itv=None, step=None, last_step=None):
        if itv is None or (step - 1) % itv >= itv - 1 or step == last_step:
            if hasattr(self, "df_params"):
                self.df_params.write_csv(self.run_path / Path("param.csv"))
            if hasattr(self, "df_metrics"):
                self.df_metrics.write_csv(self.run_path / Path("metrics.csv"))
            self._fetch_stats().write_csv(self.run_path / Path("stats.csv"))


class RunsManager:
    def __init__(self, runs):
        self.runs = runs

    def log_text(self, text, file_name):
        for i in range(len(self.runs)):
            text_tmp = text[i] if isinstance(text, list) and len(text) == len(self.runs) else text
            self.runs[i].log_text(text_tmp, file_name)

    def log_torch_save(self, object, file_name):
        for i in range(len(self.runs)):
            object_tmp = object[i] if isinstance(object, list) and len(object) == len(self.runs) else object
            self.runs[i].log_torch_save(object_tmp, file_name)

    def log_param(self, name, value):
        for i in range(len(self.runs)):
            value_tmp = value[i] if isinstance(value, list) and len(value) == len(self.runs) else value
            self.runs[i].log_param(name, value_tmp)

    def log_params(self, stored_dict):
        for i in range(len(self.runs)):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            self.runs[i].log_params(new_stored_dict)

    def log_metric(self, name, value, step=None):
        for i in range(len(self.runs)):
            value_tmp = value[i] if isinstance(value, list) and len(value) == len(self.runs) else value
            self.runs[i].log_metric(name, value_tmp, step)

    def log_metrics(self, stored_dict, step=None):
        for i in range(len(self.runs)):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            self.runs[i].log_metrics(new_stored_dict, step)

    def ref_stats(self, itv=None, step=None, last_step=None):
        for i in range(len(self.runs)):
            self.runs[i].ref_stats(itv, step, last_step)


class RunViewer:
    def __init__(self, pa_path=None, exc_path=None, exp_path=None, exp_name="tmp"):
        # run = RunManager(exc_path=__file__, exp_name="sugoi_research")
        if pa_path is not None:
            pa_path = Path(pa_path).resolve()
            exp_path = pa_path / Path(exp_name)
        elif exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        self.pa_path = pa_path
        self.exp_path = exp_path

    def fetch_stats(self):
        dir_names = list(self.exp_path.iterdir())
        run_ids = [int(dir_name.name) for dir_name in dir_names]
        stats_paths = [dir_name / Path("stats.csv") for dir_name in dir_names]

        stats_l = []
        for run_id, stats_path in zip(run_ids, stats_paths):
            try:
                df_stats = pl.read_csv(stats_path)
                df_id = pl.DataFrame({"run_id": run_id})
                df_stats = df_id.hstack(df_stats)
                stats_l.append(df_stats)
            except FileNotFoundError:
                pass
        df = pl.concat(stats_l, how="diagonal").sort(pl.col("run_id"))

        return df
