//! Result formatting: numpy by default, pandas/Arrow/Polars/sparse on request.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::SKPyError;

/// Requested Python representation of a prediction or transform result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKPyResultFormat {
    /// NumPy `ndarray` (default).
    Numpy,
    /// pandas `DataFrame`/`Series`.
    Pandas,
    /// pyarrow `Table` implementing `__arrow_c_stream__`.
    Arrow,
    /// Polars `DataFrame`/`Series`.
    Polars,
    /// SciPy CSR matrix.
    Sparse,
    /// Mirror the representation used for the input.
    Auto,
}

impl SKPyResultFormat {
    /// Parse the public `output` keyword.
    pub(crate) fn parse(name: &str) -> Result<SKPyResultFormat, SKPyError> {
        match name {
            "numpy" => Ok(SKPyResultFormat::Numpy),
            "pandas" => Ok(SKPyResultFormat::Pandas),
            "arrow" => Ok(SKPyResultFormat::Arrow),
            "polars" => Ok(SKPyResultFormat::Polars),
            "sparse" => Ok(SKPyResultFormat::Sparse),
            "auto" => Ok(SKPyResultFormat::Auto),
            other => Err(SKPyError::invalid_argument(format!(
                "unknown output format '{other}'; expected 'numpy', 'pandas', 'arrow', 'polars', 'sparse' or 'auto'"
            ))),
        }
    }

    /// Resolve `Auto` against the representation of the input dataset.
    pub(crate) fn resolve(self, input_representation: &str) -> SKPyResultFormat {
        if self != SKPyResultFormat::Auto {
            return self;
        }
        if input_representation.starts_with("pandas") {
            SKPyResultFormat::Pandas
        } else if input_representation.starts_with("polars") {
            SKPyResultFormat::Polars
        } else if input_representation.starts_with("pyarrow")
            || input_representation.starts_with("arrow")
            || input_representation.starts_with("dask")
            || input_representation.starts_with("pyspark")
        {
            SKPyResultFormat::Arrow
        } else if input_representation.starts_with("scipy") {
            SKPyResultFormat::Sparse
        } else {
            SKPyResultFormat::Numpy
        }
    }
}

/// Format a native result array into the requested Python representation.
pub(crate) fn sk_py_format_result(
    py: Python<'_>,
    array: Py<PyAny>,
    labels: Option<Vec<String>>,
    format: SKPyResultFormat,
) -> PyResult<Py<PyAny>> {
    let array = sk_py_apply_labels(py, array, labels.as_deref())?;
    let dimensional = array.bind(py).getattr("ndim")?.extract::<usize>()?;
    match format {
        SKPyResultFormat::Numpy => Ok(array),
        SKPyResultFormat::Pandas => {
            let pandas = py.import("pandas")?;
            let value = if dimensional == 1 {
                pandas.getattr("Series")?.call1((array,))?.unbind()
            } else {
                pandas.getattr("DataFrame")?.call1((array,))?.unbind()
            };
            Ok(value)
        }
        SKPyResultFormat::Arrow => sk_py_format_arrow(py, array, dimensional),
        SKPyResultFormat::Polars => {
            let polars = py.import("polars")?;
            let value = if dimensional == 1 {
                polars
                    .getattr("Series")?
                    .call1(("prediction", array))?
                    .unbind()
            } else {
                polars.getattr("from_numpy")?.call1((array,))?.unbind()
            };
            Ok(value)
        }
        SKPyResultFormat::Sparse => {
            let sparse = py.import("scipy.sparse")?;
            let matrix = if dimensional == 1 {
                let numpy = py.import("numpy")?;
                let reshaped = array.bind(py).call_method1("reshape", (-1, 1))?;
                let _ = numpy;
                reshaped.unbind()
            } else {
                array
            };
            let value = sparse.getattr("csr_matrix")?.call1((matrix,))?.unbind();
            Ok(value)
        }
        SKPyResultFormat::Auto => Ok(array),
    }
}

fn sk_py_format_arrow(py: Python<'_>, array: Py<PyAny>, dimensional: usize) -> PyResult<Py<PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    let table = pyarrow.getattr("table")?;
    let columns = PyDict::new(py);
    if dimensional == 1 {
        columns.set_item("prediction", pyarrow.getattr("array")?.call1((array,))?)?;
    } else {
        let numpy = py.import("numpy")?;
        let count = array
            .bind(py)
            .getattr("shape")?
            .get_item(1)?
            .extract::<usize>()?;
        let slices = numpy.call_method1("split", (array, count, 1))?;
        for index in 0..count {
            let column = slices.get_item(index)?;
            columns.set_item(
                format!("column_{index}"),
                pyarrow.getattr("array")?.call1((column,))?,
            )?;
        }
    }
    Ok(table.call1((columns,))?.unbind())
}

fn sk_py_apply_labels(
    py: Python<'_>,
    array: Py<PyAny>,
    labels: Option<&[String]>,
) -> PyResult<Py<PyAny>> {
    let Some(labels) = labels else {
        return Ok(array);
    };
    let values: Vec<f64> = array.bind(py).call_method0("tolist")?.extract()?;
    let mapped: Vec<Option<String>> = values
        .iter()
        .map(|value| {
            let index = value.round();
            if index.is_finite() && index >= 0.0 && (index as usize) < labels.len() {
                Some(labels[index as usize].clone())
            } else {
                None
            }
        })
        .collect();
    let numpy = py.import("numpy")?;
    Ok(numpy
        .getattr("array")?
        .call1((mapped,))?
        .call_method1("astype", ("object",))?
        .unbind())
}
