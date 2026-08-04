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



#A funcao abaixo passa arrays para a CrossValidationCompiler em metrics.py
#Calculo de metricas agregadas
#INICIO DE MUDANCAS
#desativa mecanismo de calculo de gradientes do pyTorch reduz consumo de RAM
@torch.no_grad()
#recebe modelo e carregadados, retorna duas lists
def get_predictions_from_dataloader(
    model: nn.Module,
    dataloader: torch_data.DataLoader,
    device: torch.device | None = None,
    ) -> tuple[list[int], list[int]]:
    # GPU ou CPU
    if device is None:
        device = next(model.parameters()).device
    #fixa batch para avaliacao
    model.eval()
    #envia pra GPU ou CPU
    model.to(device)

    y_true = []
    y_pred = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
	
	#probabilidades saida bruta
        out = model(x)

	#pos processamento
	#modelo binario
        if out.ndim == 1 or out.shape[-1] == 1:
            preds = (out.squeeze() > 0.5).long()
	#modelo multiclasse
        else:
            preds = torch.argmax(out, dim=1)

	#transforma y e preds da GPU para numpy na CPU e agrega
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred






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

    # Adicao de logging para retirar debugs
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    fs_out = lps_qty.Frequency.khz(16)
    duration = lps_qty.Time.s(1)
    overlap = lps_qty.Time.s(0.75)

   
    # INICIALIZAÇÃO DOS COMPILADORES DE VALIDAÇÃO CRUZADA
   
    # Importa a classe compiladora do arquivo de métricas
    from lps_ml.utils.metrics import CrossValidationCompiler

    cv_compiler_train = CrossValidationCompiler()
    cv_compiler_val = CrossValidationCompiler()

    # Definição das métricas para argparse
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

    # Inicialização do DataModule 
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
        # criar selectio com qt predefinida
        selection=ml_iara.IARA.ship_category_selector("SHIPTYPE"),
        batch_size=16,
        data_dir="C:/Users/carol/Documents/Sonat/IARA/src/data/raw/train"
    )

    print("NEW")
    print(ml_utils.format_header(60, "Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()

    # 
    #  CONFIGURAÇÃO MANUAL DOS FOLDS
    
    dm.setup("fit")
    num_folds = len(dm.folds)
    print(ml_utils.format_header(60, f"Iniciando Validação Cruzada: {num_folds} Folds"))

    
    # LOOP PRINCIPAL DE VALIDAÇÃO CRUZADA (Roda por todos os folds)
    
    for fold_idx in range(num_folds):
        print(f"\n>>> Executando FOLD {fold_idx + 1}/{num_folds} <<<")
        
        # Altera os ponteiros internos para apontar os dados do fold da vez
        dm.set_fold(fold_idx)

        # Cria instancia completamente nova da MLP com pesos aleatorios *** evita que o segundo fold
        # aproveite o primeiro
        # RE-INICIALIZA O MODELO (Zera os pesos neurais para evitar contaminação)
        model = ml_model.MLP(
            input_shape=dm.get_sample_shape(),
            hidden_channels=[64, 16],
            n_targets=dm.get_n_targets(),
            dropout=0.2,
            lr=1e-5
        )

        # Callbacks e Loggers específicos contendo o fold_idx no nome para evitar colisões
        checkpoint_cb = lightning_call.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename=f"iara-fold{fold_idx}-{{epoch:02d}}-{{val_loss:.3f}}",
        )
        early_stop_cb = lightning_call.EarlyStopping(monitor="val_loss", patience=4, mode="min")
        logger = lightning_log.TensorBoardLogger("logs", name=f"iara_fold_{fold_idx}")

        # RE-INICIALIZA O TRAINER DO PYTORCH LIGHTNING
        trainer = lightning.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            #limit_train_batches=2,  # Limitador de amostragem rápida para debug
            #limit_val_batches=2,
            devices="auto",
            logger=logger,
            callbacks=[checkpoint_cb, early_stop_cb],
        )

        # Realiza o treino apenas com a fatia alocada para este fold
        trainer.fit(model, dm)

        
        # 4. EXTRAÇÃO DE ALVOS (Y_TRUE) E PREDIÇÕES (Y_PRED)
        
        # Usando o método nativo embutido na sua classe Metric para gerar arrays completos
        y_true_train, y_pred_train = Metric._infer_from_dataloader(model, dm.train_dataloader())
        y_true_val, y_pred_val = Metric._infer_from_dataloader(model, dm.val_dataloader())

        # 
        # ALIMENTA OS COMPILADORES COM OS DADOS EXTRAÍDOS
        # 
        cv_compiler_train.add(fold_idx, metrics, y_true_train, y_pred_train)
        cv_compiler_val.add(fold_idx, metrics, y_true_val, y_pred_val)

        print(f"Fold {fold_idx + 1} Concluído com Sucesso.")

    # 
    # Resultados
    # 
    print("\n" + ml_utils.format_header(60, "RESULTADOS CONSOLIDADOS DA VALIDAÇÃO CRUZADA"))
    
    print("\n[Métricas de Treino - Todos os Folds]:")
    for m in metrics:
        scores_train = cv_compiler_train.get(m)
        print(f"{m.name:25s}: {CrossValidationCompiler.str_format(scores_train)}")

    print("\n[Métricas de Validação - Todos os Folds]:")
    for m in metrics:
        scores_val = cv_compiler_val.get(m)
        print(f"{m.name:25s}: {CrossValidationCompiler.str_format(scores_val)}")

    
    # MATRIZ DE CONFUSÃO ACUMULADA
     
    print("\n" + ml_utils.format_header(60, "Matriz de Confusão Relativa (%) - Validação"))
    cv_compiler_val.print_cm(relative=True)
    
    print(ml_utils.format_header(60))


if __name__ == "__main__":
    _main()


