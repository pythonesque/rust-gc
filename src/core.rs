#[path="core/arc.rs"] mod arc;
#[path="core/gc.rs"] mod gc;
#[path="core/layout.rs"] mod layout;
#[path="core/ref.rs"] mod r#ref;

pub use arc::{GcArc};
pub use gc::{GcFwd, GcDead};
pub use layout::{GcArcLayout, GcArcFwdLayout};
pub use r#ref::{GcArcFwd, GcRefInner, Gc};
