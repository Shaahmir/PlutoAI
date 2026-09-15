from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:

    def __init__(self, log_dir: Path, enabled: bool):

        self.enabled = enabled
        self.writer = None

        if enabled:
            self.writer = SummaryWriter(
                    log_dir = str(log_dir)
                )
        
    def scalar(self, tag: str, value: float, step: int):

        if self.writer is None:
            return

        self.writer.add_scalar(tag, value, step)

    def flush(self):

        if self.writer is not None:
            self.writer.flush()
    
    def close(self):

        if self.writer is not None:
            self.writer.close()
