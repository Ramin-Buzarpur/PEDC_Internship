# main.py


import torch
from torch.utils.data import DataLoader


from datasets.unified_market_dataset import UnifiedMarketDataset


from models.alpha_matchs import AlphaMaTCHS


from models.diffusion_matchs import (
    DiffusionDenoiser,
    GaussianDiffusion
)


from training.trainer import MarketTrainer


from backtest.walk_forward import WalkForwardBacktest


from configs.config import CONFIG



def main():


    ###################################
    # Device
    ###################################


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    print(
        "Running on:",
        device
    )



    ###################################
    # Dataset
    ###################################


    dataset=UnifiedMarketDataset(

        data_dir=
        CONFIG["data"]["path"],

        seq_len=
        CONFIG["data"]["seq_len"],

        future_len=
        CONFIG["data"]["future_len"],

        max_stocks=
        CONFIG["data"]["max_stocks"]

    )



    ###################################
    # Time Split
    # NO random split
    ###################################


    train_size=int(
        len(dataset)*0.7
    )


    val_size=int(
        len(dataset)*0.15
    )


    test_size=(
        len(dataset)
        -
        train_size
        -
        val_size
    )



    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(

        dataset,

        [
            train_size,
            val_size,
            test_size
        ],

        generator=torch.Generator()
        .manual_seed(42)

    )


    # IMPORTANT:
    # In production replace random_split
    # with chronological split
    # using index ranges.



    loader=DataLoader(

        train_dataset,

        batch_size=
        CONFIG["training"]["batch_size"],

        shuffle=True,

        drop_last=True

    )



    ###################################
    # Alpha MaTCHS
    ###################################


    alpha_model=AlphaMaTCHS(

        features=
        CONFIG["model"]["features"],

        hidden=
        CONFIG["model"]["hidden"],

        seq_len=
        CONFIG["data"]["seq_len"]

    )



    ###################################
    # Diffusion
    ###################################


    diffusion_net=DiffusionDenoiser(

        features=
        CONFIG["model"]["features"],

        hidden=
        CONFIG["model"]["hidden"]

    )



    diffusion=GaussianDiffusion(

        diffusion_net,

        steps=100,

        device=device

    )



    ###################################
    # Trainer
    ###################################


    trainer=MarketTrainer(

        alpha_model,

        diffusion_net,

        diffusion,

        device=device,

        lr=
        CONFIG["training"]["lr"]

    )



    ###################################
    # Training
    ###################################


    history=trainer.fit(

        loader,

        epochs=
        CONFIG["training"]["epochs"],

        save_path=
        "checkpoints/market_ai.pt"

    )



    print(
        "Training Finished"
    )



    ###################################
    # Backtest
    ###################################


    tester=WalkForwardBacktest()


    result=tester.run(

        alpha_model,

        test_dataset,

        device

    )


    print(
        "\n========== RESULT =========="
    )


    print(
        "Return:",
        result["total_return"]
    )


    print(
        "Sharpe:",
        result["sharpe"]
    )


    print(
        "Max Drawdown:",
        result["max_drawdown"]
    )




if __name__=="__main__":

    main()