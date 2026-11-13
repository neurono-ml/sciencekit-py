"""Dataset interoperability for Python table and array representations.

The `sciencekit` estimators accept any representation listed by
:func:`supported_representations` directly in ``fit``, ``predict``,
``transform`` and ``score``:

* ``numpy.ndarray`` of ``float32``/``float64`` (C-contiguous arrays are
  borrowed zero-copy);
* ``pandas.DataFrame`` / ``pandas.Series`` through the Arrow PyCapsule protocol
  with a documented ``to_numpy`` fallback for older pandas;
* ``scipy.sparse`` CSR matrices (``data``/``indices``/``indptr`` are borrowed)
  with explicit CSC/COO conversion;
* ``pyarrow.Table`` / ``RecordBatch`` / ``RecordBatchReader`` and any object
  exporting ``__arrow_c_stream__`` / ``__arrow_c_array__``;
* ``polars.DataFrame`` (Arrow protocol) and ``polars.LazyFrame`` (collected
  eagerly, batch by batch);
* ``dask.dataframe.DataFrame`` (partitions consumed in order);
* ``pyspark.sql.DataFrame`` (driver-side ``toArrow()`` collect, or the
  distributed :func:`map_in_arrow` path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sciencekit._native import ConversionReport, describe, supported_representations

SPARK_DRIVER_BATCH_SIZE = 10_000
"""Default `mapInArrow` batch size for the distributed PySpark path."""


def map_in_arrow(
    dataframe: Any,
    function: Callable[[Any], Any],
    *,
    schema: Any,
    batch_size: int = SPARK_DRIVER_BATCH_SIZE,
) -> Any:
    """Apply ``function`` to each Arrow batch of a PySpark DataFrame.

    This is the distributed, out-of-core path: partitions are consumed on the
    executors and only one Arrow batch per task is resident. The driver never
    materializes the full frame, unlike the ``toArrow()`` collect path used when
    a PySpark DataFrame is passed straight to ``fit``. Set
    ``spark.sql.execution.arrow.maxRecordsPerBatch`` to ``batch_size`` to bound
    each batch.
    """
    if not hasattr(dataframe, "mapInArrow"):
        raise TypeError(
            "map_in_arrow expects a pyspark.sql.DataFrame exposing mapInArrow(); "
            "install pyspark 4 or use the driver collect path"
        )
    return dataframe.mapInArrow(function, schema)


__all__ = [
    "ConversionReport",
    "SPARK_DRIVER_BATCH_SIZE",
    "describe",
    "map_in_arrow",
    "supported_representations",
]
