from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    data_dir: Optional[str] = None
    seq_len: int = 30
    future_len: int = 5
    max_stocks: int = 12
    synthetic_days: int = 220
    seed: int = 42
    corr_threshold: float = 0.35


@dataclass
class ModelConfig:
    features: int = 10
    hidden: int = 32
    heads: int = 4
    diffusion_steps: int = 20


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "artifacts"

    def validate(self) -> None:
        if self.data.seq_len < 2 or self.data.future_len < 1:
            raise ValueError("seq_len must be >= 2 and future_len must be >= 1")
        if self.model.hidden % self.model.heads:
            raise ValueError("model.hidden must be divisible by model.heads")
        if self.training.batch_size < 1 or self.training.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")

    def as_dict(self):
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
