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
