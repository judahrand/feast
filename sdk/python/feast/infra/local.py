import uuid
from datetime import datetime
from pathlib import Path

from feast.feature_view import FeatureView
from feast.infra.passthrough_provider import PassthroughProvider
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.registry_store import RegistryStore
from feast.repo_config import RegistryConfig
from feast.usage import log_exceptions_and_usage


class LocalProvider(PassthroughProvider):
    """
    This class only exists for backwards compatibility.
    """

    pass


def _table_id(project: str, table: FeatureView) -> str:
    return f"{project}_{table.name}"


class LocalRegistryStore(RegistryStore):
    def __init__(self, registry_config: RegistryConfig, repo_path: Path):
        registry_path = Path(registry_config.path)
        if registry_path.is_absolute():
            self._filepath = registry_path
        else:
            self._filepath = repo_path.joinpath(registry_path)

    @log_exceptions_and_usage(registry="local")
    def get_registry_proto(self):
        registry_proto = RegistryProto()
        if self._filepath.exists():
            registry_proto.ParseFromString(self._filepath.read_bytes())
            return registry_proto
        raise FileNotFoundError(
            f'Registry not found at path "{self._filepath}". Have you run "feast apply"?'
        )

    @log_exceptions_and_usage(registry="local")
    def update_registry_proto(self, registry_proto: RegistryProto):
        self._write_registry(registry_proto)

    def teardown(self):
        try:
            self._filepath.unlink()
        except FileNotFoundError:
            # If the file deletion fails with FileNotFoundError, the file has already
            # been deleted.
            pass

    def _write_registry(self, registry_proto: RegistryProto):
        registry_proto.version_id = str(uuid.uuid4())
        registry_proto.last_updated.FromDatetime(datetime.utcnow())
        file_dir = self._filepath.parent
        file_dir.mkdir(exist_ok=True)
        self._filepath.write_bytes(registry_proto.SerializeToString())
