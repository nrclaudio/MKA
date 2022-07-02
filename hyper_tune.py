"""
Created on We Apr 13 21:49:00
@author: Claudio Novella Rausell @nrclaudio
"""

from functools import partial
from random import randint

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import anndata
import scvi
from scvi.model import SCVI
from scvi.data import synthetic_iid
import scanpy as sc
from scvi.model.base import BaseModelClass

from sklearn.metrics import silhouette_samples, silhouette_score

import torch
from pytorch_lightning.callbacks import Callback

import ray
from ray import tune
from ray.tune import loguniform
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneCallback
from ray.tune.suggest.search import SearchAlgorithm
from ray.tune import Callback as RayCallback


sc.set_figure_params(figsize=(4, 4))
scvi.settings.seed = 94705


class ModelSave(Callback):
    def __init__(self, model):
        super()
        self.model = model

    def on_validation_epoch_end(self, trainer, pl_module, outputs=None):
        if trainer.sanity_checking:
            return
        step = f"epoch={trainer.current_epoch}-step={trainer.global_step}"
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            self.model.save( dir_path=checkpoint_dir + "/checkpoint")

        

class _TuneReportMetricFunctionsCallback(TuneCallback):
    def __init__(
        self,
        metrics=None,
        metric_functions=None,
        on="validation_end",
        model=None,
    ):
        super(_TuneReportMetricFunctionsCallback, self).__init__(on)
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics
        self._metric_functions = metric_functions
        self._model = model

    def _handle(self, trainer, pl_module):
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                report_dict[key] = trainer.callback_metrics[metric].item()
        if self._metric_functions:
            for key in self._metric_functions:
                report_dict[key] = self._metric_functions[key](self._model)
        tune.report(**report_dict)

def silhouette_metric_labels_batch(
    model: BaseModelClass,
    labels_key: str,
    batch_key: str,
    sample_size: int = 1000,
) -> float:
    """
    Batch- and label-wise silhouette.
    Parameters
    ----------
    model
        A scvi model.
    labels_key
        The key of the labels.
    batch_key
        The key of the batch.
    sample_size
        Sample size for silhouette. Randomly subsets the data
    Returns
    -------
    Sum of asw for batch and for labels. Scores are scaled such that
    2 is the best score and 0 is the worst.
    Notes
    -----
    This function is influence by the following code:
    https://github.com/theislab/scib/blob/main/scib/metrics/silhouette.py
    """
    model.is_trained_ = True
    adata = model.adata
    # lets just keep annotated cells for computation of scores
    adata = adata[adata.obs[labels_key] != 'Unknown'].copy()
    latent = model.get_latent_representation(adata)
    model.is_traine=False
    # bio conservation
    asw_labels = silhouette_score(
        latent,
        adata.obs[labels_key],
        sample_size=sample_size,
    )
    # normalize into 0-1 range
    asw_labels = (asw_labels + 1) / 2

    sil_all = pd.DataFrame(columns=["group", "silhouette_score"])

    for group in adata.obs[labels_key].unique():
        mask = np.asarray(adata.obs[labels_key] == group)
        adata_group = adata[mask]
        n_batches = adata_group.obs[batch_key].nunique()

        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        sil_per_group = silhouette_samples(latent[mask], adata_group.obs[batch_key])

        # take only absolute value
        sil_per_group = np.abs(sil_per_group)

        # scale it
        sil_per_group = 1 - sil_per_group

        sil_all = sil_all.append(
            pd.DataFrame(
                {
                    "group": [group] * len(sil_per_group),
                    "silhouette_score": sil_per_group,
                }
            )
        )

    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby("group").mean()
    asw_batch = sil_means["silhouette_score"].mean()

    return asw_batch + asw_labels

def trainable(config, adata):
    model_config = {}
    plan_config = {}
    for key in config:
        if key == "model_config":
              model_config = config[key]
        elif key == "plan_config":
              plan_config = config[key]

    if config["hvgs"]["subset"]:
        sc.pp.highly_variable_genes( 
          adata,
          flavor="seurat_v3",
          n_top_genes=config["hvgs"]["nr_hvgs"],
          batch_key="Origin",
          subset=True)

    SCVI.setup_anndata(adata, batch_key="Origin", 
                      continuous_covariate_keys=config["continious_covariates"], 
                      categorical_covariate_keys=config["categorical_covariates"])
    vae = SCVI(adata, **model_config, gene_likelihood = "nb")
    vae.train(plan_kwargs=plan_config, check_val_every_n_epoch=1, 
            max_epochs=config["num_epochs"], callbacks=[ModelSave(vae),
                  _TuneReportMetricFunctionsCallback(
                  metrics=config["metrics"],
                  on="validation_end",
                  model=vae,
                  metric_functions=config["metric_functions"])])

def run_tune(config, adata, num_samples=10, gpus=0.125, cpus=1, num_epochs = 2, local_dir="./ray-tune", name="experiment"):
    scheduler = ASHAScheduler(grace_period=5, reduction_factor=2)
    reporter = CLIReporter(metric_columns=config["metrics"] + list(config["metric_functions"].keys()))
    results = tune.run(
        tune.with_parameters(trainable, adata=adata),
        config=config,
        metric="silhouette_score",
        mode="max",
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"gpu":gpus, "cpu": cpus},
        checkpoint_at_end=True,
        name=name,
        local_dir=local_dir,
        log_to_file=True,
        verbose=3,
        raise_on_failed_trial=False)
    return results

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    print("Running init")
    ray.init(num_cpus=8, _temp_dir="/exports/humgen/cnovellarausell/tmp/")
    print("Done")
    atlas = sc.read_h5ad("/exports/humgen/cnovellarausell/KidneyAtlas/h5ad/atlas_full_raw.h5ad")
    Rename_CT = {
    'PTS1': 'PT',
    'PTS2': 'PT',
    'PTS3': 'PT', 
    'PTS1-2': 'PT',
    'PTS3T2': 'PT',
    'PST': 'PT',
    'PCT': 'PT',
    'ICA': 'IC',
    'ICB': 'IC',
    'Vas-Afferens': 'Endo',
    'Vas-Efferens': 'Endo',
    'Asc-Vasa-Recta': 'Endo',
    'Desc-Vasa-Recta': 'Endo',
    'Glom-Endo': 'Endo',
    'MTAL': 'LOH',
    'CTAL': 'LOH',
    'ATL': 'LOH'    }
    atlas.obs["Celltype_finest_lowres"] = atlas.obs["Celltype_finest"].replace(Rename_CT)
    
    # Set config parameters that will be passed to the trainables
    config = {
        "model_config": { # /dict/ args for model._model()
            "dropout_rate": loguniform(1e-4, 1e-1),
            "n_layers": tune.sample_from(lambda _: np.random.randint(1,3)),
            "n_latent": tune.sample_from(lambda _: np.random.randint(20, 31))
        }, 
        "plan_config": { # /dict/ args for model._train()
            "lr": tune.loguniform(1e-4, 1e-1)
        }, 
        "hvgs": { # /dict/ bool and number of HVGs
            "subset": tune.sample_from(lambda spec: np.random.choice([True, False])), 
            "nr_hvgs": tune.sample_from(lambda spec: np.random.randint(2,8) * 1000 
                                    if spec.config["hvgs"]["subset"] else None)
        }, 
        "continious_covariates":  tune.sample_from(lambda _: np.random.choice([['pct_counts_mt'], None])),   
        "categorical_covariates": tune.sample_from(lambda _: np.random.choice([['Source'], None])),
        "metrics": ["elbo_validation","reconstruction_loss_validation"],
        "metric_functions": {"silhouette_score": partial(silhouette_metric_labels_batch, labels_key="Celltype_finest_lowres", batch_key="Origin")},
        "num_epochs":  tune.sample_from(lambda _: np.random.randint(100, 201)),
        "name": "scvi_experiment_max_sil_1000"}
    
    # Run the analysis
    analysis = run_tune(config, adata=atlas, num_samples=1000, num_epochs=config["num_epochs"], name=config["name"])
    
    # Analyse and save the results of hyperparameter tuning
    df = analysis.dataframe(metric="silhouette_score")
    df.to_csv("hyperparameter_space" + config["name"] + "full" +  ".csv")
    dfs = analysis.trial_dataframes
    # Plot by epoch
    fig = plt.figure()
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.silhouette_score.plot(ax=ax, legend=False)
    fig.savefig("df_trials_100.svg")
    import pickle
    with open('df_trials_1000.pkl', 'wb') as f:
        pickle.dump(df_trials, f)
    best_config = analysis.best_config
    logger.info("Best hyperparameters found were: ", best_config)
    # Get the checkpoint path of the best trial of the experiment
    model_config = {}
    for key in best_config:
        if key in ['dropout_rate', 'n_layers', 'n_latent']:
            model_config[key] = best_config[key]
        if key  in ['lr']:
            plan_config[key] = best_config[key]
    # retrieve, load, train and save best model
    best_checkpoint = analysis.best_checkpoint
    if best_config["hvgs"]["subset"]:
        sc.pp.highly_variable_genes(
            atlas,
            flavor="seurat_v3",
            n_top_genes=best_config["hvgs"]["nr_hvgs"],
            batch_key="Origin",
            subset=True)
    print(best_config)
    best_model = SCVI.load(dir_path = best_checkpoint + "checkpoint", adata=atlas)
    best_model.save("Best_model" + config["name"] + "full")
    best_model.train(plan_kwards = plan_config)
    best_model.save("Best_model_trained" + config["name"] + "full", save_anndata=True)
