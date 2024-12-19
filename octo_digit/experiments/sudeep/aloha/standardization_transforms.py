from typing import Any, Dict


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory
