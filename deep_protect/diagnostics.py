"""Optional synchronized timing and PyTorch allocator statistics."""
from contextlib import contextmanager
import json
from time import perf_counter

import torch


class RunProfile:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.results = {}

    @contextmanager
    def stage(self, name):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        status = "error"
        try:
            yield
            if self.cuda:
                torch.cuda.synchronize(self.device)
            status = "ok"
        finally:
            record = {"status": status, "seconds": round(perf_counter() - start, 4)}
            if self.cuda:
                record.update(
                    peak_allocated_mib=torch.cuda.max_memory_allocated(self.device) / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(self.device) / 2**20,
                )
            self.results[name] = record
            print(json.dumps({"stage": name, **record}), flush=True)
