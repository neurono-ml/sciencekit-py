//! Unit tests for the missing-value policy.

use super::*;

#[test]
fn reject_names_the_column_and_row() {
    let policy = SKPyMissingPolicy::parse("reject", None).expect("reject parses");
    let error = policy.resolve("age", 3).expect_err("reject raises");
    assert!(error.message().contains("'age'"));
    assert!(error.message().contains("row 3"));
}

#[test]
fn fill_requires_a_value_and_returns_it() {
    assert!(SKPyMissingPolicy::parse("fill", None).is_err());
    let policy = SKPyMissingPolicy::parse("fill", Some(-1.0)).expect("fill parses");
    assert_eq!(policy.resolve("age", 0).expect("fill resolves"), -1.0);
}

#[test]
fn nan_policy_returns_nan() {
    let policy = SKPyMissingPolicy::parse("nan", None).expect("nan parses");
    assert!(policy.resolve("age", 0).expect("nan resolves").is_nan());
}

#[test]
fn unknown_policy_lists_the_supported_names() {
    let error = SKPyMissingPolicy::parse("drop", None).expect_err("unknown policy");
    assert!(error.message().contains("'reject', 'nan' or 'fill'"));
}
