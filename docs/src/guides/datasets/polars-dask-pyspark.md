# Polars, Dask and PySpark

## Polars

A `polars.DataFrame` is an Arrow PyCapsule producer, so it reaches the engine
without an intermediate pandas conversion:

```python
import polars as pl
from sciencekit import LinearRegression, describe

frame = pl.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0],
        "y": [3.0, 5.0, 9.0, 11.0, 15.0],
    }
)
report = describe(frame, target="y")
assert report.representation == "polars.dataframe"
assert report.arrow_zero_copy is True

with LinearRegression() as builder:
    with builder.fit(frame, target="y") as model:
        model.predict(frame, columns=["a", "b"])
```

A `polars.LazyFrame` has no Arrow stream until it is collected. The adapter
collects it eagerly and reports that decision:

```python
lazy = frame.lazy()
report = describe(lazy, target="y")
assert report.representation == "polars.lazyframe"
assert report.eager_fallback == "polars.lazyframe.collect"

with LinearRegression() as builder:
    with builder.fit(lazy, target="y") as model:
        assert builder.last_conversion.eager_fallback == "polars.lazyframe.collect"
```

Documented limitation: the lazy plan is executed on the driver before fitting;
Polars' own streaming engine is not interleaved with `sciencekit` fitting.

## Dask

A `dask.dataframe.DataFrame` is consumed partition by partition, in partition
order:

```python
import dask.dataframe as dd
import pandas as pd

frame = pd.DataFrame({"a": [...], "b": [...], "y": [...]})
dask_frame = dd.from_pandas(frame, npartitions=3)

report = describe(dask_frame, target="y")
assert report.representation == "dask.dataframe"
assert report.partitions == 3
assert report.partition_sizes == [2, 2, 1]
assert report.eager_fallback == "dask.partitions"

with LinearRegression() as builder:
    with builder.fit(dask_frame, target="y") as model:
        model.predict(dask_frame, columns=["a", "b"])
```

`partition_sizes` is the observable proof that partitions were materialized in
order. The estimator is not incremental, so the partitions are joined into one
design matrix before the ABI call — resident memory is `O(rows × columns)`, not
the Dask graph's. This is the documented eager fallback; for larger-than-memory
fitting, shard the data and fit on batches with your own accumulation strategy.

## PySpark

The driver path collects a small frame through Arrow and produces exactly the
same result as pandas:

```python
report = describe(spark_frame, target="y")
assert report.representation == "pyspark.dataframe"
assert report.eager_fallback.startswith("pyspark.toArrow")

with LinearRegression() as builder:
    with builder.fit(spark_frame, target="y") as model:
        model.predict(spark_frame, columns=["a", "b"])
```

The conversion order is `toArrow()` (Spark 4 and Spark Connect),
`_collect_as_arrow()`, then `toPandas()`, each recorded in
`eager_fallback`. **The driver path is a collect**: it requires the frame to fit
in driver memory. It never claims out-of-core execution.

For out-of-core work, apply a sciencekit operation per Arrow batch on the
executors:

```python
from sciencekit.datasets import map_in_arrow

output = map_in_arrow(
    spark_frame,
    lambda batch: batch,          # replace with your estimator call per batch
    schema="a double, b double",
)
```

`map_in_arrow` wraps `DataFrame.mapInArrow`; set
`spark.sql.execution.arrow.maxRecordsPerBatch` (default
`sciencekit.datasets.SPARK_DRIVER_BATCH_SIZE = 10_000`) to bound each batch.
Tests skip the real Spark case when `pyspark` is not installed and always cover
the driver path through a duck-typed `toArrow()` producer.
