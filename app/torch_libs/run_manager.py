from pathlib import Path
from time import time

import torch
import polars as pl


class RunManager:
    def __init__(self, pa_path=None, exc_path=None, exp_name="exp_default", exp_path=None):
        """
        ex.)
            run = RunManager(exc_path=__file__, exp_name="exp_nyancat")
        """

        # pa_pathとexp_nameを指定 -> 格納ディレクトリとexp_nameを別々に指定
        if pa_path is not None:
            pa_path = Path(pa_path).resolve()
            exp_path = pa_path / Path(exp_name)

        # exc_path(__file__)とexp_nameを指定 -> コード実行ディレクトリと同じディレクトリに結果を格納
        elif exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)

        # exp_pathを指定 -> 格納ディレクトリとexp_nameを同時に指定
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        # 結果格納用のパスを設定し、適宜ディレクトリを作成
        exp_path.mkdir(parents=True, exist_ok=True)

        runs_path = exp_path / Path("runs")
        runs_path.mkdir(parents=True, exist_ok=True)

        run_id = Path(self._get_run_id(runs_path))
        run_path = runs_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.runs_path = runs_path
        self.run_path = run_path
        # self.run_id = run_id

        # self.start_time = time()

    # def __call__(self, fname):
    #     return self.run_path / Path(fname)

    def _get_run_id(self, runs_path):
        dir_names = list(runs_path.iterdir())
        dir_nums = [int(dir_name.name) for dir_name in dir_names]
        if len(dir_nums) == 0:
            run_id = 0
        else:
            run_id = max(dir_nums) + 1
        return str(run_id)

    def log_text(self, text, fname):
        with open(self.run_path / Path(str(fname)), "w") as fh:
            fh.write(text)

    def log_csv(self, df, fname, *args, **kwargs):
        df.write_csv(self.run_path / Path(fname), *args, **kwargs)
        # df.write_csv(run.run_path / Path("xxx"), ...)

    def log_torch_save(self, object, fname):
        torch.save(object, self.run_path / Path(fname))

    def log_param(self, name, value):
        self.log_params({name: value})

    def log_df(self, df, fname):
        df.write_csv(self.run_path / Path(fname))
        # df.write_csv(self(fname))

    def log_params(self, stored_dict):
        stored_dict = {k: [v] for k, v in stored_dict.items()}
        if hasattr(self, "df_params"):
            self.df_params = self.df_params.hstack(pl.DataFrame(stored_dict))
        else:
            self.df_params = pl.DataFrame(stored_dict)

    def log_metric(self, name, value, step=None):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict, step=None):
        if hasattr(self, "df_metrics"):
            for name, value in stored_dict.items():
                if step is None:
                    step = self.df_metrics["step"].max() + 1
                if step in self.df_metrics["step"]:
                    self.df_metrics = self._df_set_elem(self.df_metrics, name, "step", step, value)
                else:
                    df_tmp = pl.DataFrame({"step": [step], name: [value]})
                    self.df_metrics = pl.concat([self.df_metrics, df_tmp], how="diagonal_relaxed")
        else:
            if step is None:
                step = 1
            stored_dict = {k: [v] for k, v in stored_dict.items()}
            self.df_metrics = pl.DataFrame({"step": [step], **stored_dict})

    def _df_set_elem(self, df, column, index_column_name, index, value):
        # indexは存在しなければならない。columnは存在しないとき新たに作られる。
        if value is not None:
            if column in df.columns:
                df = df.with_columns(pl.when(df[index_column_name] == index).then(value).otherwise(pl.col(column)).alias(column))
            else:
                df = df.with_columns(pl.when(df[index_column_name] == index).then(value).otherwise(pl.lit(None)).alias(column))
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

    # def fetch_stats(self):
    #     dir_names = list(self.exp_path.iterdir())
    #     run_ids = [int(dir_name.name) for dir_name in dir_names]
    #     stats_paths = [dir_name / Path("stats.csv") for dir_name in dir_names]

    #     stats_l = []
    #     for run_id, stats_path in zip(run_ids, stats_paths):
    #         try:
    #             df_stats = pl.read_csv(stats_path)
    #             df_id = pl.DataFrame({"run_id": run_id})
    #             df_stats_wid = df_id.hstack(df_stats)
    #             stats_l.append(df_stats_wid)
    #         except FileNotFoundError:
    #             pass
    #     df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("run_id"))

    #     return df

    # def write_stats(self, fname=None):
    #     df = self.fetch_stats()
    #     if fname is None:
    #         df.write_csv(f"{self.exp_path}.csv")
    #     else:
    #         df.write_csv(str(self.pa_path / Path(fname)))

    # exp_のフォルダに直接格納する場合
    # def write_stats(self, fname="results.csv"):
    #     df = self.fetch_stats()
    #     df.write_csv(str(self.exp_path / Path(fname)))


class RunsManager:
    def __init__(self, runs):
        self.runs = runs

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return_l = []
            for i, run in enumerate(self.runs):
                # new_args = []
                # for arg in args:
                #     if isinstance(arg, list) and len(arg) == len(self.runs):
                #         new_arg = arg[i]
                #     else:
                #         new_arg = arg
                #     new_args.append(new_arg)
                new_args = [arg[i] if isinstance(arg, list) and len(arg) == len(self.runs) else arg for arg in args]

                # new_kwargs = dict()
                # for k, v in kwargs.items():
                #     if isinstance(v, list) and len(v) == len(self.runs):
                #         new_kwargs[k] = v[i]
                #     else:
                #         new_kwargs[k] = v
                new_kwargs = {k: v[i] if isinstance(v, list) and len(v) == len(self.runs) else v for k, v in kwargs.items()}
                return_l.append(getattr(run, attr)(*new_args, **new_kwargs))
            return return_l

        return wrapper

    def __getitem__(self, idx):
        return self.runs[idx]

    # こいつらはこのままの形式だと格納できないから特別
    def log_param(self, name, value):
        self.log_params({name: value})

    def log_params(self, stored_dict):
        for i, run in enumerate(self.runs):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            run.log_params(new_stored_dict)

    def log_metric(self, name, value, step=None):
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, stored_dict, step=None):
        for i, run in enumerate(self.runs):
            new_stored_dict = dict()
            for k, v in stored_dict.items():
                v_tmp = v[i] if isinstance(v, list) and len(v) == len(self.runs) else v
                new_stored_dict[k] = v_tmp
            run.log_metrics(new_stored_dict, step=step)


class RunViewer:
    def __init__(self, pa_path=None, exc_path=None, exp_name="exp_default", exp_path=None):
        # pa_pathとexp_nameを指定 -> 格納ディレクトリとexp_nameを別々に指定
        if pa_path is not None:
            pa_path = Path(pa_path).resolve()
            exp_path = pa_path / Path(exp_name)

        # exc_path(__file__)とexp_nameを指定 -> コード実行ディレクトリと同じディレクトリに結果を格納
        elif exc_path is not None:
            pa_path = Path(exc_path).resolve().parent  # 常に__file__が送られてくるならresolveは不要
            exp_path = pa_path / Path(exp_name)

        # exp_pathを指定 -> 格納ディレクトリとexp_nameを同時に指定
        else:
            exp_path = Path(exp_path).resolve()
            pa_path = exp_path.parent

        runs_path = exp_path / Path("runs")

        self.pa_path = pa_path
        self.exp_path = exp_path
        self.runs_path = runs_path
        
    def ref_results(self, fname):
        dir_names = list(self.runs_path.iterdir())
        run_ids = [int(dir_name.name) for dir_name in dir_names]
        stats_paths = [dir_name / Path("stats.csv") for dir_name in dir_names]

        stats_l = []
        for run_id, stats_path in zip(run_ids, stats_paths):
            try:
                df_stats = pl.read_csv(stats_path)
                df_id = pl.DataFrame({"run_id": run_id})
                df_stats_wid = df_id.hstack(df_stats)
                stats_l.append(df_stats_wid)
            except FileNotFoundError:
                pass
        df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("run_id"))
        self.write_results(df, fname)

        return df
    
    def read_results(self, fname="results.csv", infer_schema_length=100):
        df = pl.read_csv(str(self.exp_path / Path(fname)), infer_schema_length=infer_schema_length)

        return df

    def write_results(self, df, fname="results.csv"):
        df.write_csv(str(self.exp_path / Path(fname)))

    def fetch_results(self, fname="results.csv"):
        try:
            df = self.ref_results(fname)
        except FileNotFoundError:
            df = self.read_results(fname)

        return df


    def fetch_metrics(self, listed=False):
        dir_names = list(self.runs_path.iterdir())
        run_ids = [int(dir_name.name) for dir_name in dir_names]
        stats_paths = [dir_name / Path("metrics.csv") for dir_name in dir_names]

        stats_l = []
        for run_id, stats_path in zip(run_ids, stats_paths):
            try:
                df_stats = pl.read_csv(stats_path)
                df_stats_wid = df_stats.with_columns(pl.lit(run_id).alias("run_id"))
                df_stats_wid = df_stats_wid.select(["run_id"] + df_stats.columns)
                stats_l.append(df_stats_wid)
            except FileNotFoundError:
                # metrics.csvがない場合Errorが発生する。
                pass
        if listed:
            return stats_l
        else:
            df = pl.concat(stats_l, how="diagonal_relaxed").sort(pl.col("step")).sort(pl.col("run_id"))
            return df
