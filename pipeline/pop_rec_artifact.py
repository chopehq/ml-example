
import os
from bentoml.utils import cloudpickle
from bentoml.service.artifacts import BentoServiceArtifact


class PopRecArtifact(BentoServiceArtifact):
    def __init__(self, name):
        super(PopRecArtifact, self).__init__(name)
        self._model = None

    def pack(self, model, metadata=None):
        self._model = model
        return self

    def get(self):
        return self._model

    def save(self, directory):
        path = self._file_path(directory)
        with open(path, "wb") as file:
            cloudpickle.dump(self._model, file)

    def load(self, path):
        with open(self._file_path(path), "rb") as file:
            model = cloudpickle.load(file)
        return self.pack(model)

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + ".pkl")
