import datetime
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# Base feature types
Int32 = Union[
    np.int8, np.int32, np.uint8, np.uint32,
]
Int64 = Union[
    int, np.int64, np.uint64,
]
Float = Union[np.float32]
Double = Union[float, np.float64]
Bool = Union[bool, np.bool_]
String = Union[str, np.str_]
Bytes = Union[bytes, np.str_]
Timestamp = Union[datetime.datetime, np.datetime64, pd.Timestamp]

# Lists of base feature types
Int32List = Union[
    List[Int32],
    npt.NDArray[np.int8],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint64],
]
Int64List = Union[
    List[Int64], npt.NDArray[np.int64], npt.NDArray[np.uint64],
]
FloatList = Union[List[Float], npt.NDArray[np.float32]]
DoubleList = Union[
    List[Double], npt.NDArray[np.float64],
]
BoolList = Union[
    List[Bool], npt.NDArray[np.bool_],
]
StringList = Union[
    List[String], npt.NDArray[np.str_],
]
BytesList = Union[
    List[Bytes], npt.NDArray[np.bytes_],
]
TimestampList = Union[List[Timestamp], npt.NDArray[np.datetime64]]

# General list feature type
ListType = Union[
    Int32List,
    Int64List,
    FloatList,
    DoubleList,
    BoolList,
    StringList,
    BytesList,
    TimestampList,
]

# Any valid feature type
ValidType = Optional[Union[Int32, Int64, Bool, String, Bytes, ListType]]
