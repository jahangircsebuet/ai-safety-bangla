from typing import List, Dict
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

class RandomUnderSamplerBalancer:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def balance(self, data: List[Dict]) -> List[Dict]:
        """
        Undersamples the safe class to match the number of unsafe examples.
        Returns a shuffled balanced list.
        """
        df = pd.DataFrame(data)
        rus = RandomUnderSampler(
            sampling_strategy={"safe": df["label"].value_counts().get("unsafe", 0),
                               "unsafe": df["label"].value_counts().get("unsafe", 0)},
            random_state=self.seed
        )
        X = df[["prompt", "response"]]
        y = df["label"]
        X_res, y_res = rus.fit_resample(X, y)
        out = pd.DataFrame({
            "prompt": X_res["prompt"].tolist(),
            "response": X_res["response"].tolist(),
            "label": y_res.tolist()
        }).sample(frac=1.0, random_state=self.seed).to_dict(orient="records")
        return out
