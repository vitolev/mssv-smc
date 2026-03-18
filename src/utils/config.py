import yaml

class Config:
    def __init__(self, d):
        self.__dict__.update({
            k: self._convert(v) for k, v in d.items()
        })

    def _convert(self, v):
        if isinstance(v, dict):
            return Config(v)
        return v

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)