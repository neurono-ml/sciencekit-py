//! Process-wide atomic count of live opaque handles.
//!
//! Unlike single-threaded scripting runtimes, Python objects may be allocated
//! on one OS thread and finalized or closed on another. An atomic counter keeps
//! `sciencekit.live_handles()` exact across concurrent threads and garbage
//! collection cycles.

use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_HANDLES: AtomicUsize = AtomicUsize::new(0);

/// Record that one native builder or model handle was created.
pub(crate) fn sk_py_handle_created() {
    LIVE_HANDLES.fetch_add(1, Ordering::SeqCst);
}

/// Record that one native builder or model handle was released.
pub(crate) fn sk_py_handle_freed() {
    let _ = LIVE_HANDLES.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
        Some(current.saturating_sub(1))
    });
}

/// Return how many native handles are currently live in this process.
pub(crate) fn sk_py_live_handles() -> usize {
    LIVE_HANDLES.load(Ordering::SeqCst)
}
