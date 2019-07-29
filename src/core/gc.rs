use crate::ghost_cell::{InvariantLifetime};

/// With shared access to a GcFwd<'gc>, we can legally dereference forwarding pointers from
/// Gc<'gc, 'a, T>.  The main purpose of this behavior is to soundly combine tracing and forwarding
/// with a straightforwardly safe API.  Forwarding pointers are effectively weak pointers, meaning
/// we should only be allowed to dereference them as long as the underlying data are still alive,
/// but we don't want to have to actually introduce a weak count for three reasons:
///
///    1. Space overhead for the weak pointer leads to performance degradation in all cases,
///       not just when we are forwarding data.  Almost every efficient [precise] Gc already
///       requires keeping at least a little extra metadata per object, and the more you need the
///       harder it is to fit into available space (especially without performance degradation due
///       to higher overhead to decode the data or tradeoffs in tracking precision / overall
///       collection performance).  Not only that, it makes atomic operations much trickier and
///       often slower if you can't fit enough useful metadata into a single word.
///
///    2. In code without cycles (what the current "immutable" [but really acyclic] Gc is designed
///       for), weak pointers aren't needed in user code.  So the only purpose for its existence
///       is to make sure access to reference counts is still safe if the collection code
///       (maliciously or by accident) ends up dropping Arc objects referenced by forwarding
///       pointers.  Since the intent is that these are only created during tracing, and ideally
///       (in order to avoid a read barrier during normal operations) we fix up all such forwarded
///       references before we resume normal program operations, there is no reason why we should
///       ever *want* to drop such an Arc--tracing should only hit objects transitively referenced
///       by roots!
///
///       From one lens, this opens up an opportunity to explore GCs that don't consider roots
///       anything special and happily drop them during tracing.  Such GCs would allow you to
///       easily mix mutator code with tracing, deallocation, etc., and might be a good fit with
///       techniques allowing for efficient partial reclamation.  That's the world where wanting
///       such a weak reference *might* make sense.  But against the lens of traditional GCs, they
///       really don't, and thus the weak reference would only be a safeguard against errors
///       during collection code (preventing you from shooting yourself in the foot by following a
///       trace pointer and trying to incrementing its strong count after it was deallocated).
///
///       So the second viewpoint (adopted here) is a weak pointer here would only be a safety
///       measure and is not needed or desirable for expressiveness.  That is why our GC is
///       explicitly designed to alleviate this safety issue--forwarding pointers point to
///       GcArcFwd, not GcArc, and the former don't run their destructors when dropped.  As
///       a result, dropping the roots (or a panic, or any other error that might lead to
///       forwarding pointers becoming invalid while they are still accessible) still keeps the
///       underlying pointers alive, at least for long enough that whenever any arena-allocated
///       data (non-strictly outlived by 'a, which is [strictly] outlived by 'gc wherever it
///       contains Gc pointers) is alive, so are the pointer's contents.  The only way to turn a
///       GcArcFwd<'gc, T> into a GcArc<T> is to hand it proof that nobody can read the
///       forwarding pointers from Gc<'gc, 'a, T> anymore--that is exactly what is witnessed by
///       GcDead<'gc>.
///
///       This does require being very careful about how easy it is to get a GcArcFwd<'gc, T>,
///       since it doesn't (currently?) include 'a, and you only need a short-lived reference to
///       one (&'c GcArcFwd<'gc, T> for any 'c : 'a + 'gc) in order to write it to a forwarding
///       pointer for &'a GcRefInner<'gc, T>.  One nice observation is that if you *start* with a
///       reference &GcArc<T>, cloning it and then turning the clone into a GcArcFwd is
///       automatically safe--any live GcArc has strong count at least 1, so the GcArcFwd brings
///       it to at least 2 *and* won't run its destructor until after forwarding pointers can't be
///       read anymore.  Even if the original Arc is subsequently dropped, the referenced memory
///       will still be accessible.  Another observation is that any reference &'a GcArc<T> should
///       be safe to interpret as a Gc<'gc, 'a, T>, because we make it legal to turn such
///       "GcArcInner" references into GcArcFwds using the same the same technique.  Additionally,
///       it is always legal to turn a GcArc<T> *by-value* into a GcArcFwd<'gc, T>, since again
///       you know the object must remain alive until (if ever) GcDead<'gc>.
///
///    3. Point 2 was supposed to be a lot shorter, but point 3 is the real reason we do all this.
///       If we have weak pointers that actually work (meaning, they decrement on drop, or in some
///       other way), then we need to run their destructors when we drop the GcRefInners.  These
///       will generally be arena-allocated.  But a huge amount of the performance benefit of
///       tracing Gc is not having to run destructors!  Otherwise you still have to "trace" the
///       dead objects, too (though for weak pointers the work involved is relatively minimal, but
///       it still involves nonlocal writes and [if the original Arc was actually deallocated]
///       potentially entails running the normal destructor for the Arc as well).  So it is really
///       quite imperative that we avoid this if we want to use garbage collection for
///       performance.
///
///       I firmly believe that except in very specialized cases, destructors are a *mistake* in
///       garbage collection.  I don't think we should (necessarily) go out of our way to break /
///       make unpleasant the usage of types with destructors, but outside of the bare minimum
///       needed to ensure safety [which may just include banning them in some cases] it is
///       unlikely that this set of Gcs will ever make any special effort to support them.  If you
///       need destructors, you should probably avoid tracing Gc!  Other solutions (e.g. epoch
///       based reclamation or more complicated stuff) may be better suited to your use cases.
///
/// While this is alive, we can legally dereference forwarding pointers.  In the future, this
/// should also probably restrict reads of Gc data, which will allow us to reuse the data fields in
/// young data as forwarding pointers and/or introduce read barriers with higher overhead than
/// normal reads during the mutator phase.  There are probably a few other tricks we can use as
/// well.
///
/// This should also probably restrict reads of Gc data.  That would let us reuse data fields in
/// young data as forwarding pointers and/or introduce read barriers with higher overhead than
/// normal reads during the mutator phase.  There are probably a few other tricks we can use as
/// well.  For instance, if we disallow old->young heap references (which *hopefully* is currently
/// disallowed by our typing rules, but this is a good reminder to double check), we can require
/// unique ownership of the forwarding pointer in order to bump a reference count accessed through
/// a GcArcFwd, which would mean we could avoid synchronization on any such pointer; though this
/// would only be safe provided that we either also required access (e.g. shared access) to the
/// forwarding pointer or mutator for GcArc, or could *also* make sure that no GcArcFwd was
/// simultaneously being interpreted as an Arc.  This is not inherently true of GcArcFwd at the
/// moment, but hopefully it could be made true for any GcArcFwd that was actually [transitively]
/// accessed directly through a forwarding pointer.
pub struct GcFwd<'gc> {
    _marker: InvariantLifetime<'gc>,
}

/// 
#[derive(Clone,Copy)]
pub struct GcDead<'gc> {
    _marker: InvariantLifetime<'gc>,
}

/* pub trait Hkt<#[may_dangle]'a> {
    type HktOut : GcArcLayout + Copy;
} */

impl<'a> GcFwd<'a> {
    pub fn new</*T, *//*I, */F, R>(/*init: I, */f: F) -> R
        where
            // T: for <'new_id> Hkt<'new_id>,
            // This is too restrictive, but let's go with it for now.
            /* for <'new_id> <T as Hkt<'new_id>>::HktOut: Copy,
            for <'new_id> <T as Hkt<'new_id>>::HktOut: GcArcLayout, */
            /* T: for <'new_id> <T as Hkt<'new_id>>::HktOut: Default, */
            // I: for <'new_id> FnOnce(&GcFwd<'new_id>) -> TypedArena<GcRefInner<'new_id, <T as Hkt<'new_id>>::HktOut>>,
            // I: for <'new_id> FnOnce(&GcFwd<'new_id>) -> TypedArena<GcRefInner<'new_id, Expr<'new_id>>>,
            // I: for <'new_id> FnOnce(InvariantLifetime<'new_id>) -> <T as Hkt<'new_id>>::HktOut,
            // F: for <'new_id> FnOnce(&'new_id /*mut */<T as Hkt<'new_id>>::HktOut, GcFwd<'new_id>) -> R,
            F: for <'new_id> FnOnce(/*&'new_id /*mut */Allocator/*<GcRefInner<'new_id, <T as Hkt<'new_id>>::HktOut>>*/, */GcFwd<'new_id>) -> R,
            // F: for <'new_id> FnOnce(&'new_id mut /*<T as Hkt<'new_id>>::HktOut*/TypedArena</*<T as Hkt<'new_id>>::HktOut*/GcRefInner<'new_id, Expr<'new_id>>>, GcFwd<'new_id>) -> R,
    {
        // We choose the lifetime; it is definitely unique for each new instance of Set.
        {
            let /*mut data, */fwd : /*TypedArena<GcRefInner<<T as Hkt<'_>>::HktOut>>*//*MemoryArena, */GcFwd<'_>;
            // let (mut data, fwd) : (TypedArena<GcRefInner<<T as Hkt<'_>>::HktOut>>, GcFwd<'_>);
            // let (mut data, fwd) : (T::HktOut, GcFwd);
            // let (mut data, fwd) : (TypedArena</*T::HktOut*/GcRefInner<Expr>>, GcFwd);
            fwd = GcFwd {
                _marker: InvariantLifetime::new(),
            };
            // data = /*init(InvariantLifetime::new())*//*TypedArena*/MemoryArena::new(1024 * 1024);
            // data = init(&fwd);
            // Return the result of running f.  Note that the Set itself can't be returned, because R
            // can't mention the lifetime 'id, so the Set really does only exist within its scope.
            let r = f(/*&mut data*//*&data.allocator(), */fwd);
            r
        }
    }

    /// Kill the ability to read or write forwarding pointers.
    pub fn end(self) -> GcDead<'a> {
        GcDead {
            _marker: InvariantLifetime::new(),
        }
    }
    /* /// Get an immutable reference to the item that lives for as long as the owning set is
    /// immutably borrowed.
    pub fn get<'a, T>(&'a self, item: &'a Cell<'id, T>) -> &'a T {
        unsafe {
            // We know the set and lifetime are both borrowed at 'a, and the set is borrowed
            // immutably; therefore, nobody has a mutable reference to this set.  Therefore, any
            // items in the set that are currently aliased would have been legal to alias at &'a T
            // as well, so we can take out an immutable reference to any of them, as long as we
            // make sure that nobody else can take a mutable reference to any item in the set until
            // we're done.
            &*item.value.get()
        }
    } */
}
