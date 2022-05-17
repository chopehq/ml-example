from bento_lfm_artifacts import LightFMModelArtifact, LightFMDatasetArtifact
from pop_rec_artifact import PopRecArtifact
from bentoml import BentoService, env, api, artifacts
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.exceptions import BadInput

import numpy as np
from string import punctuation


@env(infer_pip_packages=True)
@artifacts(
    [
        LightFMModelArtifact(name="model"),
        LightFMDatasetArtifact(name="dataset"),
        PopRecArtifact(name="pop_rec"),
    ]
)
class LightFMRecService(BentoService):
    @api(input=JsonInput(), output=JsonOutput(), batch=True)
    def recommend(self, input_data):
        model = self.artifacts.model
        pop_rec = self.artifacts.pop_rec
        dataset = self.artifacts.dataset
        items_map = dataset.mapping()[2]
        users_map = dataset.mapping()[0]
        if isinstance(input_data, dict):
            input_data = [input_data]
        if len(input_data) > 1:
            return [{"message": BadInput(f"input has too many elements")}]
        input_data = input_data[0]
        if not isinstance(input_data, dict):
            return [{"message": BadInput(f"input type is not allowed")}]
        hashed_email = input_data.get("hashed_email")
        if hashed_email is None or any(p in hashed_email for p in punctuation):
            return [{"message": BadInput(f"input {hashed_email} is not accepted")}]
        recommendations = self._recommend(
            model, hashed_email, users_map, items_map, k=3, pop_rec=pop_rec
        )
        result = {
            "meta": input_data,
            "data": {"recommendations": recommendations},
        }
        return [result]

    @staticmethod
    def _recommend(model, user_id, users_map, items_map, k: int, pop_rec):
        items_arr = np.array(list(items_map.keys()))
        n_items = len(items_map)

        _user_id = users_map.get(user_id)
        if _user_id is None:
            return pop_rec[:k]
        scores = model.predict(_user_id, np.arange(n_items))
        top_items = items_arr[np.argsort(-scores)]
        return list(top_items[:k])
