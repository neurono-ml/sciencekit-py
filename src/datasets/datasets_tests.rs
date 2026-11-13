//! Unit tests for the representation catalogue and result formats.

use super::representation::sk_py_supported_representations;
use super::results::SKPyResultFormat;

#[test]
fn supported_representations_cover_every_adapter() {
    let names = sk_py_supported_representations().join(", ");
    for expected in [
        "numpy.ndarray",
        "pandas.DataFrame",
        "scipy.sparse.csr_matrix",
        "pyarrow.Table",
        "polars.DataFrame",
        "dask.dataframe.DataFrame",
        "pyspark.sql.DataFrame",
    ] {
        assert!(names.contains(expected), "missing {expected} in {names}");
    }
}

#[test]
fn auto_result_format_mirrors_the_input_representation() {
    assert_eq!(
        SKPyResultFormat::Auto.resolve("pandas.dataframe"),
        SKPyResultFormat::Pandas
    );
    assert_eq!(
        SKPyResultFormat::Auto.resolve("polars.dataframe"),
        SKPyResultFormat::Polars
    );
    assert_eq!(
        SKPyResultFormat::Auto.resolve("pyarrow.table"),
        SKPyResultFormat::Arrow
    );
    assert_eq!(
        SKPyResultFormat::Auto.resolve("scipy.csr"),
        SKPyResultFormat::Sparse
    );
    assert_eq!(
        SKPyResultFormat::Auto.resolve("numpy.ndarray"),
        SKPyResultFormat::Numpy
    );
}

#[test]
fn unknown_result_format_lists_the_options() {
    let error = SKPyResultFormat::parse("json").expect_err("unknown output format");
    assert!(error.message().contains("'numpy', 'pandas', 'arrow'"));
}
