import yaml
from mlrun.datastore import store_manager

from .vector import FeatureVectorSpec
from .mergers import LocalFeatureMerger
from .featureset import FeatureSet
from .model import DataTarget, TargetTypes, FeatureClassKind
from .ingest import write_to_target_store


def store_client(data_prefix='', project=None, secrets=None):
    return FeatureStoreClient(data_prefix, project, secrets)


class FeatureStoreClient:
    def __init__(self, data_prefix='', project=None, secrets=None):
        self._api = None
        self._data_prefix = data_prefix or './store'
        self._data_stores = store_manager.set(secrets)
        self._fs = {}
        self._default_ingest_targets = [TargetTypes.parquet]
        self.project = project

    def _get_target_path(self, kind, name, project=None):
        project = project or self.project or 'default'
        return f'{self._data_prefix}/{project}/{kind}/{name}'

    def ingest(self, featureset: FeatureSet, source, targets=None):
        """Read local DataFrame, file, or URL into the feature store"""
        targets = targets or self._default_ingest_targets
        if not targets:
            raise ValueError('ingestion target(s) were not specified')
        for target in targets:
            target_path = self._get_target_path(target, featureset.metadata.name, featureset.metadata.project)
            target_path = write_to_target_store(target, source, target_path, self._data_stores)
            target = DataTarget(target, target_path)
            featureset.status.update_target(target)
        self.save_object(featureset)

    def start_ingestion_job(self, featureset, source_path, products=None):
        """Start MLRun ingestion job to load data into the feature store"""
        pass

    def start_realtime_ingestion(self, featureset, source_path, products=None):
        """Start real-time Nuclio function which loads data into the feature store"""
        pass

    def get_offline_features(self, features,
                             entity_rows=None,
                             entity_timestamp_column=None,
                             entity_primary_keys=None,
                             watch=True, store_target=None):

        merger = LocalFeatureMerger()
        vector = FeatureVectorSpec(self, features)
        vector.parse_features()
        featuresets, feature_dfs = vector.load_featureset_dfs()
        return merger.merge(entity_rows, entity_timestamp_column, entity_primary_keys, featuresets, feature_dfs)

    def get_online_features(self, features, store_kind, entity_rows=None):
        pass

    def get_featureset(self, name, project=None):
        target = self._get_target_path(FeatureClassKind.FeatureSet, name, project)
        body = self._data_stores.object(url=target + '.yaml').get()
        obj = yaml.load(body, Loader=yaml.FullLoader)
        return FeatureSet.from_dict(obj)

    def save_object(self, featureset):
        """save featureset or other definitions into the DB"""
        target = self._get_target_path(FeatureClassKind.FeatureSet,
                                       featureset.metadata.name,
                                       featureset.metadata.project)
        self._data_stores.object(url=target + '.yaml').put(featureset.to_yaml())

