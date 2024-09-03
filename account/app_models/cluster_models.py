
from ..project_utils.account_utils import create_module
import joblib


class DimReductionAndClustering:
    def __init__(self,path=None,dr_cfg=None,cl_cfg=None):
        self.dr_alg = create_module(dr_cfg['function'])(**dr_cfg['param'])
        self.dr_alg = joblib.load(path) if path else self.dr_alg
        self.cluster_alg = create_module(cl_cfg['function'])(**cl_cfg['param'])

    def to_latent_space(self, feature, transform_only=False):
        return self.dr_alg.transform(feature) if transform_only else self.dr_alg.fit_transform(feature)

    def cluster_embeddings(self, embeddings):
        labels = self.cluster_alg.fit_predict(embeddings)
        return labels

    def fit(self, embeddings):
        return self.cluster_alg.fit(embeddings)

    def save_weights(self, path):
        joblib.dump(self.dr_alg, path)


