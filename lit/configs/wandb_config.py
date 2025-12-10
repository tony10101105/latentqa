from typing import List, Optional
from dataclasses import dataclass


@dataclass
class wandb_config:
    project: str = "latentqa"
    entity: Optional[str] = "tony10101105"
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
