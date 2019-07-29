use crate::ghost_cell::{InvariantLifetime/*, Set as GhostSet, Cell as GhostCell*/};

use std::cell::{/*Cell, */UnsafeCell};
use std::sync::atomic;

/// This should provide a zero-cost coercion from Self::AsArc to Self.  This is the only part
/// that needs unsafe, since we are asserting that it's safe to treat data from &T::AsArc
/// as though it were from &T.  A mechanical transformation of every Gc<'gc, 'a, T> pointer in
/// the type to GcArc<T::AsArc> should always satisfy this property, provided that all
/// AsArc implementations follow this pattern.
///
/// Important note: the reverse direction is *not* true.  That is, it's not safe to treat data
/// from &T as though it is from &T::AsArc.  This is because GcArcInner exposes more methods
/// than does GcRefInner (except during collection time) and also has interior mutability.
pub unsafe trait GcArcLayout {
    /// Self with Gc<'gc, 'a, T> replaced by GcArc<'gc, T::AsArc>
    type AsArc;
    type AsArcFwd : GcArcFwdLayout<FromArcFwd=Self::AsArc>;
}

/// We also have an implementation in the other direction, providing a zero-cost coercion from
/// Self::AsArcFwd to Self::AsArc.  This should be done by mechanically transforming every
/// GcArcFwd<'gc, T> pointer in the type to GcArc<T::FromArcFwd>.  This is only sound to do
/// when the forwarding pointer is dead.  However, the reverse direction *always* works.
pub unsafe trait GcArcFwdLayout {
    /// Self with GcArcFwd<'a, T> replaced by GcArc<'a, T::FromArcFwd>
    type FromArcFwd;
}

/// GcFreeze means that pointers to values of this type cannot be modified through a shared
/// reference.  It *is* legal for some other type to modify pointers in it through a shared
/// reference, or for this type to modify pointers to other types through a shared reference.
/// This peculiar requirement is intended to avoid the introduction of cyclic data, which can
/// cause unsoundness.
pub unsafe auto trait GcFreeze {}

/// GcPod means a type has no (cyclic) interior mutability at all.
pub unsafe auto trait GcPod {}

impl<T: ?Sized> !GcPod for UnsafeCell<T> /*where T: GcFreeze*/ { }
impl<T: ?Sized> !GcPod for *mut T { }
impl<T: ?Sized> !GcPod for *const T { }
unsafe impl<'a, T: ?Sized> GcPod for &'a T where &'a T: GcFreeze { }
unsafe impl<'a, T: ?Sized> GcPod for &'a mut T where &'a mut T: GcFreeze { }
/* unsafe impl<'a, T: ?Sized> GcPod for &'a T where T: GcPod { }
unsafe impl<'a, T: ?Sized> GcPod for &'a mut T where T: GcPod { } */

/* /// GcFreeze means a type has no *references* to types with cyclic interior mutability.
/// (These are obviously really the same, but we need to break the cycle with UnsafeCell
///  somehow to make sure people don't auto-impl GcFreeze).
unsafe impl<'a, T: ?Sized> GcFreeze for &'a T where T: GcPod { }
unsafe impl<'a, T: ?Sized> GcFreeze for &'a mut T where T: GcFreeze { } */
// unsafe impl<'a, T: ?Sized> GcPod for T where T: GcFreeze { }
// unsafe impl<'a, T: ?Sized> GcFreeze for &'a T where T: GcPod { }
unsafe impl<'a, T: ?Sized> GcFreeze for &'a T where T: GcPod { }
unsafe impl<'a, T: ?Sized> GcFreeze for &'a mut T where T: GcPod { }
impl<T: ?Sized> !GcFreeze for *mut T { }
impl<T: ?Sized> !GcFreeze for *const T { }

unsafe impl<'id> GcPod for InvariantLifetime<'id> {}

#[cfg(target_has_atomic = "ptr")]
unsafe impl GcPod for atomic::AtomicIsize {}
#[cfg(target_has_atomic = "8")]
unsafe impl GcPod for atomic::AtomicI8 {}
#[cfg(target_has_atomic = "16")]
unsafe impl GcPod for atomic::AtomicI16 {}
#[cfg(GcFreeze = "32")]
unsafe impl GcPod for atomic::AtomicI32 {}
#[cfg(target_has_atomic = "64")]
unsafe impl GcPod for atomic::AtomicI64 {}
#[cfg(target_has_atomic = "128")]
unsafe impl GcPod for atomic::AtomicI128 {}

#[cfg(target_has_atomic = "ptr")]
unsafe impl GcPod for atomic::AtomicUsize {}
#[cfg(target_has_atomic = "8")]
unsafe impl GcPod for atomic::AtomicU8 {}
#[cfg(target_has_atomic = "16")]
unsafe impl GcPod for atomic::AtomicU16 {}
#[cfg(target_has_atomic = "32")]
unsafe impl GcPod for atomic::AtomicU32 {}
#[cfg(target_has_atomic = "64")]
unsafe impl GcPod for atomic::AtomicU64 {}
#[cfg(target_has_atomic = "128")]
unsafe impl GcPod for atomic::AtomicU128 {}

#[cfg(target_has_atomic = "8")]
unsafe impl GcPod for atomic::AtomicBool {}

/* #[cfg(target_has_atomic = "ptr")]
// NOTE: This one acts like a shared reference, but it should probably just be like *mut or *const.
unsafe impl<T> GcFreeze<T> for atomic::AtomicPtr<T> where T: GcFreeze + GcPod {} */

// impl<'a, T: ?Sized> !GcPod for &'a T { }
/* /// If a Cell contains no references to mutable data, it can't be frozen.
unsafe impl<T: ?Sized> GcFreeze for Cell<T> where T: GcPod { }
unsafe impl<'id, T> GcFreeze for GhostCell<'id, T> where T: GcFreeze + GcPod { } */
