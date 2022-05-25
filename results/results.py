import pandas as pd
import seaborn as sns


def plt_metrics(trainer):
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    # display(metrics.dropna(axis=1, how="all").head())
    sns.relplot(data=metrics, kind="line")