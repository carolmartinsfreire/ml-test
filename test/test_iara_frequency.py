"""
IARA
"""
# Consideracoes
#É um script de experimento controlado usado para validar o dataset
# IARA; validar seletores; validar o pipeline completo (processamento → CV → treino → avaliação)
#- roda, treina, mede para ver se faz sentido
#

import argparse

import torch
import torch.utils.data as torch_data
import typing


import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty

#import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.datasets.iara as ml_iara
# Teste rapidamente quais funções estão disponíveis:
#import lps_sp.acoustical.signal as lps_signal
#print(dir(lps_signal))  # Print de todas as funções disponíveis
from torch import nn

#from iara.ml.metrics import Metric, GridCompiler
from lps_ml.utils.metrics import Metric

from lps_sp.acoustical.analysis import SpectralAnalysis, Parameters

import logging







#Roda inferência; converte saída do modelo em classe;



@torch.no_grad()
def evaluate_from_dataloader(
    model: nn.Module,
    dataloader: torch_data.DataLoader,
    metrics: typing.List[Metric],
    device: torch.device | None = None,
) -> dict[Metric, float]:

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)

        if out.ndim == 1:
            preds = (out > 0.5).long()
        else:
            preds = torch.argmax(out, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return Metric.compute_all(metrics, y_true, y_pred)



def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Directory to store iara data.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--debug", action="store_true",
                        help="DEBUGS.")
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "balanced_accuracy", "macro_f1"],
                        help=("Metrics to evaluate. Options: "
                              "accuracy, balanced_accuracy, micro_f1, macro_f1, "
                              "macro_recall, micro_recall, macro_precision, micro_precision, "
                              "detection_probability, sp_index, all"
                             )
                        )

    args = parser.parse_args()




    #Adicao de logging para retirar debugs
    #debugs estao presents em time_processors.py e loader.py

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    fs_out=lps_qty.Frequency.khz(16)
    duration=lps_qty.Time.s(1)
    overlap=lps_qty.Time.s(0.75)


    #TimeProcessor
    dm = ml_db.IARA(
        file_processor=ml_procs.FrequencyProcessor(
            fs_out=fs_out,
            duration=duration,
            overlap=overlap,
            spectral=SpectralAnalysis.LOFAR,
            params=Parameters(),
            pipelines=ml_procs.CPADetector(
                duration,
                lps_qty.Time.s(duration.get_s() * 60)
            )
        ),
        cv = ml_cv.FiveByTwo(),
        data_collection = ml_iara.DC.A,
        selection=ml_iara.CargoShipClassifier.IDENTIFIED.as_selector(),
        batch_size=16,
        data_dir="C:/Users/carol/Documents/Sonat/IARA/src/data/raw/train"
    )
    
    print("OLD")
    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))


    #duplicacao do experimento new selection
    dm = ml_db.IARA(
        file_processor=ml_procs.FrequencyProcessor(
            fs_out=fs_out,
            duration=duration,
            overlap=overlap,
            spectral=SpectralAnalysis.LOFAR,
            params=Parameters(),
            pipelines=ml_procs.CPADetector(
                duration,
                lps_qty.Time.s(duration.get_s() * 60)
            )
        ),
        cv=ml_cv.FiveByTwo(),
        data_collection=ml_iara.DC.A,
        selection=ml_iara.IARA.ship_category_selector("SHIPTYPE"),
        batch_size=16,
        data_dir="C:/Users/carol/Documents/Sonat/IARA/src/data/raw/train"
    )


    print("NEW")
    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))




    model = ml_model.MLP(
        input_shape=dm.get_sample_shape(),
        hidden_channels=[64, 16],
        n_targets=dm.get_n_targets(),
        dropout=0.2,
        lr=1e-5
    )

    checkpoint_cb = lightning_call.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename=f"iara-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_stop_cb = lightning_call.EarlyStopping(monitor="val_loss", patience=4, mode="min")

    logger = lightning_log.TensorBoardLogger(
        "logs",
        name="iara"
    )

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        # reducao de batches para treinamento mais rapido a fim de debugar o codigo (para treinar tudo, comente as duas linhas abaixo)
        # uma mudanca sera feita em relacao ao codigo no tempo, vou colocar para passar por linha de comando os valores de limite de treinamento abaixo
        # TODO
        limit_train_batches=2,
        limit_val_batches=2,
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    # treinamento e eval
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

        # Definicao das metricas para argparse
    METRIC_MAP = {
    "accuracy": Metric.ACCURACY,
    "balanced_accuracy": Metric.BALANCED_ACCURACY,

    "micro_f1": Metric.MICRO_F1,
    "macro_f1": Metric.MACRO_F1,

    "macro_recall": Metric.MACRO_RECALL,
    "micro_recall": Metric.MICRO_RECALL,

    "macro_precision": Metric.MACRO_PRECISION,
    "micro_precision": Metric.MICRO_PRECISION,

    "detection_probability": Metric.DETECTION_PROBABILITY,
    "sp_index": Metric.SP_INDEX,
    }


    if "all" in args.metrics:
        metrics = list(METRIC_MAP.values())
    else:
        metrics = []
        for m in args.metrics:
            key = m.lower()
            if key not in METRIC_MAP:
                raise ValueError(f"Métrica inválida: {m}")
            metrics.append(METRIC_MAP[key])
    
    


    train_metrics = evaluate_from_dataloader(
        model,
        dm.train_dataloader(),
        metrics
    )

    val_metrics = evaluate_from_dataloader(
        model,
        dm.val_dataloader(),
        metrics
    )

    print("Metrics enabled:")
    for m in metrics:
        print(f" - {m.name}")
    print()


    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60, "Results (Train)"))
    for m, v in train_metrics.items():
        print(f"{m.name:25s}: {v:.4f}")

    print()
    print(ml_utils.format_header(60, "Results (Validation)"))
    for m, v in val_metrics.items():
        print(f"{m.name:25s}: {v:.4f}")

    print(ml_utils.format_header(60))



    #train_acc = _evaluate_accuracy(model, dm.train_dataloader())
    #val_acc   = _evaluate_accuracy(model, dm.val_dataloader())
    # test_acc  = _evaluate_accuracy(model, dm.test_dataloader())

    #print(ml_utils.format_header(60))
    #print()
    #print(f"Train accuracy:      {train_acc:.4f}")
    #print(f"Validation accuracy: {val_acc:.4f}")
    # print(f"Test accuracy:       {test_acc:.4f}")
    #print(ml_utils.format_header(60, "Results (Train)"))
    #for m, v in train_metrics.items():
     #   print(f"{m.name:25s}: {v:.4f}")

    #print()
    #print(ml_utils.format_header(60, "Results (Validation)"))
    #for m, v in val_metrics.items():
     #   print(f"{m.name:25s}: {v:.4f}")
    #print(ml_utils.format_header(60))    


if __name__ == "__main__":
    _main()

