import numpy as np
import torch



class WalkForwardBacktest:


    def __init__(
        self,
        transaction_cost=0.001,
        annual_factor=252
    ):

        self.transaction_cost = transaction_cost
        self.annual_factor = annual_factor



    def run(
        self,
        model,
        dataset,
        device
    ):

        model.eval()

        daily_returns = []

        predictions = []

        equity_curve = []

        capital = 1.0


        with torch.no_grad():

            for i in range(len(dataset)):


                past, future, graph = dataset[i]


                past = past.unsqueeze(0).to(device)

                graph = graph.to(device)



                output = model(
                    past,
                    graph
                )



                signal = torch.tanh(
                    output["direction"]
                )


                signal = (
                    signal
                    .cpu()
                    .numpy()[0]
                )



                actual_return = future[
                    :,
                    5,
                    -1
                ]


                actual_return = (
                    actual_return
                    .numpy()
                )



                denominator = (
                    np.sum(
                        np.abs(signal)
                    )
                    +
                    1e-8
                )


                weights = (
                    signal /
                    denominator
                )



                gross_return = np.sum(
                    weights *
                    actual_return
                )



                turnover = np.sum(
                    np.abs(weights)
                )



                net_return = (
                    gross_return
                    -
                    turnover *
                    self.transaction_cost
                )


                daily_returns.append(
                    net_return
                )


                predictions.append(
                    signal
                )



                capital *= (
                    1 +
                    net_return
                )


                equity_curve.append(
                    capital
                )



        daily_returns = np.array(
            daily_returns
        )


        equity_curve = np.array(
            equity_curve
        )



        total_return = (
            equity_curve[-1]
            -
            1
        )



        volatility = (
            np.std(
                daily_returns
            )
            +
            1e-8
        )



        sharpe = (
            np.mean(
                daily_returns
            )
            /
            volatility
        ) * np.sqrt(
            self.annual_factor
        )



        running_max = np.maximum.accumulate(
            equity_curve
        )


        drawdown = (
            equity_curve /
            running_max
            -
            1
        )


        max_drawdown = np.min(
            drawdown
        )



        win_rate = np.mean(
            daily_returns > 0
        )



        return {

            "total_return": float(total_return),

            "sharpe": float(sharpe),

            "max_drawdown": float(max_drawdown),

            "win_rate": float(win_rate),

            "daily_returns": daily_returns,

            "equity_curve": equity_curve,

            "predictions": predictions

        }