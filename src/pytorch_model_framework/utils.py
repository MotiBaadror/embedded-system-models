import dataclasses
import json

import numpy as np
from torch.types import Device


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, Device):
            return str(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
