import struct
from typing import List, Union

import pyarrow as pa
import pyarrow.types


def _serialize_vals(values: Union[pa.Array, pa.ChunkedArray]) -> List[bytes]:
    if hasattr(values, "combine_chunks"):
        values = values.combine_chunks()

    pa_type = values.type
    if pa.types.is_string(pa_type):
        val_bytes = [val.encode("utf8") for val in values.to_pylist()]
    elif pa.types.is_binary(pa_type):
        val_bytes = values.to_pylist()
    elif pa.types.is_int32(pa_type):
        val_bytes = [struct.pack("<i", val) for val in values.to_pylist()]
    elif pa.types.is_int64(pa_type):
        val_bytes = [struct.pack("<l", val) for val in values.to_pylist()]
    else:
        raise ValueError(f"Value type not supported for Firestore: {pa_type}")
    return [struct.pack("<I", len(b)) + b for b in val_bytes]


def serialize_entity_key_prefix(entity_keys: List[str]) -> bytes:
    """
    Serialize keys to a bytestring so it can be used to prefix-scan through items stored in the online store
    using serialize_entity_key.

    This encoding is a partial implementation of serialize_entity_key, only operating on the keys of entities,
    and not the values.
    """
    sorted_keys = sorted(entity_keys)
    output: List[bytes] = []
    for k in sorted_keys:
        output.append(struct.pack("<I", pa.string().id))
        output.append(k.encode("utf8"))
    return b"".join(output)


def serialize_entity_keys(entity_key: Union[pa.Table, pa.RecordBatch]) -> bytes:
    """
    Serialize entity key to a bytestring so it can be used as a lookup key in a hash table.

    We need this encoding to be stable; therefore we cannot just use protobuf serialization
    here since it does not guarantee that two proto messages containing the same data will
    serialize to the same byte string[1].

    [1] https://developers.google.com/protocol-buffers/docs/encoding
    """
    if hasattr(entity_key, "from_batches"):
        entity_key = entity_key.from_batches([entity_key])
    entity_key = entity_key.select(sorted(entity_key.schema.names))

    key_def_bin: List[bytes] = []
    for field in entity_key.schema:
        key_def_bin.append(struct.pack("<I", pa.string().id))
        key_def_bin.append(field.name.encode("utf8"))
        key_def_bin.append(struct.pack("<I", field.type.id))

    key_val_bins: List[List[bytes]] = [[]] * entity_key.num_rows
    for column in entity_key.columns:
        for old, new in zip(key_val_bins, _serialize_vals(column)):
            old.append(new)
    return [b"".join(key_def_bin + key_val_bin) for key_val_bin in key_val_bins]
