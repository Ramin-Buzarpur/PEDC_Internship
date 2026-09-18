from pathlib import Path

import torch
import torch.nn.functional as F

from alphamarketai.losses import gaussian_nll


class MarketTrainer:
    def __init__(self, model, diffusion, device, lr=3e-4, weight_decay=1e-5, grad_clip=1.0):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device
        self.grad_clip = grad_clip
        parameters = list(model.parameters()) + list(diffusion.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def _step(self, batch, training=True):
        past, future, graph = (item.to(self.device) for item in batch)
        output = self.model(past, graph)
        target = future[..., 0]
        deterministic = F.smooth_l1_loss(output["return"], target)
        direction = F.binary_cross_entropy_with_logits(output["direction"], (target > 0).float())
        distribution = gaussian_nll(output["mu"], output["sigma"], target)
        condition = output["embedding"].reshape(-1, output["embedding"].shape[-1])
        diffusion_target = future.reshape(-1, future.shape[-1])
        diffusion_loss = self.diffusion.training_loss(diffusion_target, condition)
        loss = deterministic + 0.25 * direction + 0.1 * distribution + diffusion_loss
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.diffusion.parameters()), self.grad_clip)
            self.optimizer.step()
        return {"loss": loss.item(), "forecast": deterministic.item(), "diffusion": diffusion_loss.item()}

    def fit(self, train_loader, epochs, save_path, val_loader=None):
        history = []
        best = float("inf")
        for epoch in range(1, epochs + 1):
            self.model.train(); self.diffusion.train()
            metrics = [self._step(batch, True) for batch in train_loader]
            train_loss = sum(x["loss"] for x in metrics) / len(metrics)
            val_loss = self.evaluate(val_loader) if val_loader is not None else train_loss
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            history.append(row)
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best:
                best = val_loss
                self.save(save_path, epoch, val_loss)
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval(); self.diffusion.eval()
        values = [self._step(batch, False)["loss"] for batch in loader]
        return sum(values) / len(values)

    def save(self, path, epoch, val_loss):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "diffusion": self.diffusion.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epoch": epoch, "val_loss": val_loss}, path)
