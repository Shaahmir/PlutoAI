import torch
from torch.nn.utils.rnn import pad_sequence
from config import CONFIG

class SFTCollator:

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        input_ids = pad_sequence(
            input_ids_list,
            batch_first = True,
            padding_value = 0
        )

        labels = pad_sequence(
            labels_list,
            batch_first = True,
            padding_value = -100
        )

        attention_mask = pad_sequence(
            [torch.ones_like(x, dtype = torch.bool) for x in input_ids_list],
            batch_first = True,
            padding_value = False
        )

        if "loss_mask" in batch[0]:

            loss_mask_list = [item["loss_mask"] for item in batch]

            loss_mask = pad_sequence(
                loss_mask_list,
                batch_first = True,
                padding_value = False
            )

            labels = labels.masked_fill(
                ~loss_mask,
                -100
            )

        if not bool((labels != -100).any()):
            raise RuntimeError("Entire SFT batch is masked!")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
