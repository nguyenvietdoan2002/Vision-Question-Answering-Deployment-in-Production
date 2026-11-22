"""Tiny model registry: register loaders, lazy-load, unload, list memory."""
import gc

import torch


class ModelRegistry:
    def __init__(self, device: str):
        self.device = device
        self._loaders = {}  # name -> callable that returns a model bundle
        self._loaded = {}   # name -> bundle

    def register(self, name: str, loader_fn):
        self._loaders[name] = loader_fn

    def load(self, name: str):
        if name not in self._loaders:
            raise KeyError(f"unknown model: {name}")
        if name not in self._loaded:
            print(f"Loading {name} on {self.device}...")
            self._loaded[name] = self._loaders[name](self.device)
            print(f"Loaded {name}.")
        return self._loaded[name]

    def get(self, name: str):
        # Lazy-load on first use.
        return self.load(name)

    def unload(self, name: str):
        if name in self._loaded:
            del self._loaded[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def list(self):
        rows = []
        for name in self._loaders:
            loaded = name in self._loaded
            mem_mb = 0.0
            if loaded:
                bundle = self._loaded[name]
                # Sum parameter bytes across tensors in the bundle.
                for obj in bundle.values() if isinstance(bundle, dict) else [bundle]:
                    if hasattr(obj, "parameters"):
                        mem_mb += sum(p.numel() * p.element_size() for p in obj.parameters()) / 1e6
            rows.append({
                "name": name,
                "loaded": loaded,
                "device": self.device if loaded else None,
                "approx_memory_mb": round(mem_mb, 1),
            })
        return rows
