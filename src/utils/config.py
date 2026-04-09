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

    def save_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self._to_dict(), f)

    def _to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v._to_dict()
            else:
                result[k] = v
        return result