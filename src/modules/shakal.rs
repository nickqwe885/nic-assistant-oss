// Thin re-export — implementation lives in crate::shakal (src/shakal.rs).
// Keeps the modules/ namespace intact; sentinel imports directly from crate::shakal.
#[allow(unused_imports)]
pub use crate::shakal::{active_window_info, CaptureFrame, ShakalProcessor};
