import torch

class Collator:

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

        if not batch:
            raise ValueError("Collator received no batch!")

        input_ids = torch.stack(
            [item["input_ids"] for item in batch]
        )

        labels = torch.stack(
            [item["labels"] for item in batch]
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype = torch.bool
        )

        result =  {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

        if "loss_mask" in batch[0]:

            loss_mask = torch.stack(
                [item["loss_mask"] for item in batch]
            ).bool()

            labels = labels.masked_fill(
                ~loss_mask,
                -100
            )

            result["labels"] = labels
        
        return result
