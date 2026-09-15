import random
from torch.utils.data import Dataset

class ReplaySampler:

    def __init__(self, dataset: Dataset, seed: int):

        self.dataset = dataset
        self.random = random.Random(seed)

    def sample(self):

        if len(self.dataset) == 0:
            raise RuntimeError("Replay dataset is empty!")

        index = self.random.randrange(
            len(self.dataset)
        )

        item = self.dataset[index]

        return (
            item["input_ids"].tolist(),
            item["labels"].tolist()
        )
