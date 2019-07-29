use core::cell::Cell;
use core::sync::atomic;
use core::sync::atomic::Ordering::{Relaxed};
use core::mem::{self, ManuallyDrop};
use core::ops::{Deref};
use core::ptr::{NonNull};
use core::convert::From;

use crate::gc_core::{GcArc, GcArcLayout, GcArcFwdLayout, GcDead, GcFwd};
use crate::ghost_cell::{InvariantLifetime};

/// Like a GcArc, but without Drop.
#[repr(transparent)]
pub struct GcArcFwd<'gc, T> {
    /// Conceptually speaking, this pointer is "owning", just like with Arc; the main difference
    /// is that we never drop its contents until after GcFwd<'a> is done (at which point we can
    /// move this to another type).  Another way to say this is that the pointer's lifetime is
    /// *at least* GcFwd<'a>, but there's no reasonable way to write this.
    arc: ManuallyDrop<GcArc<T>>,
    /* ptr: NonNull<GcArcInner<T>>,
    phantom: PhantomData<T>, */
    _marker: InvariantLifetime<'gc>,
}

/// The key most important thing here is that GcRefInner and GcArcInner (not exposed by GcArc) have
/// representations such that:
///
///    1. Data access through a Gc pointer to GcRefInner can use the same machine code as data access
///       through a Gc pointer to GcArcInner (it is legal for the pointers to be set up differently
///       on construction though), and
///
///    2. Checking whether a pointer refers to a GcRefInner or a GcArcInner (needed during tracing,
///       for instance) can be done in a way that works on both representations through
///       their respective Gc pointers (an example of an implementation that does this that doesn't
///       impose any restrictions on layout other than the way it is allocated is the use of a
///       "watermark"; pointers with addresses above a known virtual address would always be young
///       pointers / GcRefInner, and below would always be old pointers / GcArcInner (or
///       whatever).  There are also versions of this that work with generations etc., and MICA
///       sort of uses a variant of this as well to implement a very lightweight read barrier, so
///       this has applications beyond traditional GC and a single two-arena approach.
///
/// The main reason we replicate Arc as GcArc is that we can't safely rely on implementation details
/// of Arc's layout going forward.  At the moment we differentiate between GcArc and GcRef using
/// the presence of a nonzero count in the strong field as a tag, so we need to be able to access that
/// field in a uniform way from both an arena-allocated value and a heap-allocated Arc.  With no
/// direct access to ArcInner, it's unclear how to do this without relying on internal
/// implementation details that are explicitly allowed to change (plus, Arc isn't even repr(C)).
///
/// However, we (currently) still try to keep the interface as close to Arc's as possible, so we
/// strive to avoid relying on actually knowing the structure of GcArcInner except in the one place
/// where it's more or less unavoidable (reading the tag field and interpreting it as a strong
/// count wherever it's nonzero).  In the future, if we switch to something like watermarks, we can
/// hopefully considerably improve things.
#[repr(C)]
pub struct GcRefInner<'gc, T/*: ?Sized*/> where T: GcArcLayout {
    /// Always 0
    gc_ref_tag: atomic::AtomicUsize,
    /// TODO: See how this plays with DST stuff... we want to make sure the data are aligned for
    /// both sides of the union, so we don't have to check thet ag in order to downcast in Gc.
    ///
    /// Technically we don't need Gc (and hence GcRefInner) to be non-Sync during the mutator
    /// phase, since we only modify this field during collection, so we could just use
    /// a GhostCell here.  But for now we're trying to be as not-clever as possible.
    ///
    /// (We could also make this atomic, but that only makes sense if we needed to update this
    /// field in a multithreaded way, and currently this design only makes sense if collection
    /// is single-threaded).
    forward: Cell<Option</*GcArcFwd<'a, T::AsArcFwd>*/NonNull</*GcArcInner<T::AsArcFwd>*/T::AsArcFwd>>>,
    data: T,
    _marker: InvariantLifetime<'gc>
}

#[repr(C)]
/* pub union Gc<'gc, 'a, T> where T: GcArcLayout {
    /// FIXME: old should probably be limited by the scope of GcFwd<'a> as well.
    ///  That is, we shouldn't care about being able to access things behind
    /// an old reference after we get a GcDead.  But since currently dereference
    /// of Gc is always legal and doesn't require any capability, at the moment
    /// this is as good as we can get.
    old: &'a GcArcInner<T::AsArc>,
    /// new has a somewhat questionable lifetime--even more than the above, it would be
    /// nice to not promise that new actually dereferenced valid memory as soon as GcDead
    /// occurred.  But currently we just dump all the new pointers in an arena, so it is
    /// actually all valid until the arena is freed.  However, it does sort of imply that
    /// the data behind the pointer are always valid, which means that the forwarding
    /// pointer needs to be extra careful to never be dereferenced unless GcFwd is alive.
    new: &'a GcRefInner<'gc, T>,
} */
pub struct Gc<'gc, 'a, T> where T: GcArcLayout {
    /// FIXME: old should probably be limited by the scope of GcFwd<'a> as well.
    ///  That is, we shouldn't care about being able to access things behind
    /// an old reference after we get a GcDead.  But since currently dereference
    /// of Gc is always legal and doesn't require any capability, at the moment
    /// this is as good as we can get.
    // old: &'a GcArcInner<T::AsArc>,
    /// new has a somewhat questionable lifetime--even more than the above, it would be
    /// nice to not promise that new actually dereferenced valid memory as soon as GcDead
    /// occurred.  But currently we just dump all the new pointers in an arena, so it is
    /// actually all valid until the arena is freed.  However, it does sort of imply that
    /// the data behind the pointer are always valid, which means that the forwarding
    /// pointer needs to be extra careful to never be dereferenced unless GcFwd is alive.
    new: &'a GcRefInner<'gc, T>,
}


impl<'gc, T> GcArcFwd<'gc, T> {
    /// Creating a new GcArcFwd requires only a T.  It is exactly like a GcArc<T>, but won't allow
    /// drops (or reading the data, for the time being at least) until GcDead<'a> is available.
    /// Note that at the moment it doesn't require GcFwd<'a> (evidence that the forwarding pointer
    /// is available) because it doesn't actually *use* a forwarding pointer for anything; it just
    /// implicitly guarantees that its contents will be alive for as long as the forwarding pointer
    /// is.
    #[inline]
    pub fn new(data: T) -> GcArcFwd<'gc, T> /*where T : GcArcFwdLayout*/ {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        /* let x = unsafe {
            let dptr = (&data) as *const _ as *const T::FromArcFwd;
            let value_size = size_of_val(&*dptr);
            let ptr = GcArc::allocate_for_ptr(dptr);

            ptr::copy_nonoverlapping(
                dptr as *const T::FromArcFwd as *const u8,
                &mut (*ptr).data as *mut _ as *mut u8,
                value_size);

            GcArc { ptr: NonNull::new_unchecked(ptr), phantom: PhantomData }
        }; */
        let x = GcArc::new(data);
        /* let x: Box<_> = box GcArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: /*atomic::AtomicUsize::new(1)*/atomic::AtomicPtr::new(ptr::null_mut()),
            data,
        };
        // println!("Initialization, incrementing {:p}", x); */

        GcArcFwd {
            arc: ManuallyDrop::new(/*unsafe { mem::transmute(x) }*/x),
            /* ptr: /*Box::into_raw_non_null(x)*/x.ptr,
            phantom: PhantomData, */
            _marker: InvariantLifetime::new(),
        }
    }

    /* #[inline]
    fn inner(&self) -> &GcArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `GcArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        self.arc.inner()
        // contents.
        /* unsafe { /*self.ptr.as_reF()*/self.inner() } */
    } */

    #[inline]
    /// Transform into a regular GcArc, but only when forwarding pointers cannot be followed
    /// anymore.
    pub fn into_gc_arc(self, _: GcDead<'gc>) -> GcArc<T::FromArcFwd> where T: GcArcFwdLayout {
        // Casting this pointer is safe because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `GcArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        /* GcArc {
            ptr: self.ptr.cast(),
            phantom: PhantomData,
        } */
        /* unsafe {
            mem::transmute(ManuallyDrop::into_inner(self.arc))
        } */
        // ManuallyDrop::into_inner(self.arc)
        unsafe {
            ManuallyDrop::into_inner(mem::transmute::<_, GcArcFwd<T::FromArcFwd>>(self).arc)
        }
        /* GcArc {
            ptr: self.arc.ptr.cast(),
            phantom: PhantomData,
        } */
    }
}

impl<'gc, T> GcRefInner<'gc, T> where T: GcArcLayout {
    pub fn new(data: T) -> Self {
        GcRefInner {
            gc_ref_tag: atomic::AtomicUsize::new(0),
            forward: Cell::new(None),
            data,
            _marker: InvariantLifetime::new(),
        }
    }

    /// Note: we probably should only be allowed to do this when we already know the forwarding
    /// pointer is zero, but there's no safety reason to require this at the moment (that I can
    /// think of, anyway).  Note that we do *not* require GcFwd<'gc> to be alive in order to set the
    /// forwarding pointer, only in order to deference it.
    #[inline]
    pub fn set_forwarding<'b, 'c>(&'b self, forwarding: &'c GcArcFwd<'gc, T::AsArcFwd>) {
        // Whether GcFwd<'gc> is alive or not when we set the poiner, we will only allow *de*referencing it
        // if GcFwd<'gc> is alive, which works fine as long as we know that at least one
        // GcArcFwd<'gc, T::AsArcFwd> exists with this pointer value (and is borrowed for a lifetime
        // at most as long as 'gc).  This is because the only way to drop something pointed to by a
        // GcArcFwd<'gc, T::AsArcFwd> is to first transform it into a GcArc<T>, which requires Dead<'gc>
        // and taking the GcArcFwd by self.  This does mean it's not safe to transform references to
        // &'c GcArc<T>, where 'c is shorter than 'gc, into GcArcFwd<'gc, T::AsArcFwd>, since
        // otherwise you can cause problems--but now that GcArcFwd is not easy to move it is not
        // too likely that we would make the mistake of exposing an API like that.

        // NOTE: In an attempt not to rely on GcArcInner, we stop worrying about.into().
        unsafe {
            let forwarding : *const *const u8 = mem::transmute(&forwarding.arc);
            let forwarding : GcArc<T::AsArcFwd> = mem::transmute(*forwarding);
            // NOTE: If we panic here it's a real problem!  With Arc we could introduce a fake Weak
            // pointer.  This suggests that maybe we should just bite the bullet and expose
            // GcArcInner, at least locally.
            // let forwarding : *const u8 = mem::transmute(forwarding.arc);
            // let forwarding : *const GcArc<T::AsArcFwd> = mem::transmute(&forwarding.arc);
            // let forward : GcArcFwd<T::AsArc> = mem::transmute(*forwarding);
            self.forward.set(Some(/*forwarding.inner().into()*/GcArc::into_raw_non_null(forwarding)));
        }
    }
}


impl<'gc, 'a, T> Gc<'gc, 'a, T> where T: GcArcLayout {
    /// Note: this should only be run during collection, but with the current design it doesn't matter.
    ///
    /// fwd is needed for correctness since we can read from the forwarding pointer.
    #[inline]
    pub fn try_as_arc<'b>(self, _fwd: &'b GcFwd<'gc>) -> Result</*&'a GcArcInner<T::AsArcFwd>*/GcArcFwd<'gc, T::AsArcFwd>, &'a GcRefInner<'gc, T>> {
        unsafe {
            // let old = self.old;
            // let strong = self.new.gc_ref_tag;
            //
            // Note: in general this is treated as a shared strong reference.  According to the
            // https://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html, a Relaxed
            // ordering is okay here because passing the object between threads must already
            // provide any required synchronization; additionally, the destructor calls a Release
            // operation after dropping the reference / decrementing the counter and an Acquire
            // operation before deleting the object.  The Release operation for dropping this
            // particular reference must happen after this Relaxed read, so we're fine (or it never
            // happens in which case the count can never drop below zero after it rises above it;
            // since it started at nonzero before we read this and there's a total order on all
            // Relaxed-or-above operations on the reference count location, we should be okay).
            //
            // TODO: Check with Hai et. al.; we *might* need Acquire here, but it's not clear.
            let old_strong = self./*old.strong*/new.gc_ref_tag.load(Relaxed);
            if old_strong > 0 {
                // This is a GcArc, so old pointer is valid (for at least 'a + 'gc, too).
                let new : & *const GcRefInner<'gc, T> = &(self.new as *const _);
                let old : &GcArc<T::AsArc> = mem::transmute(new);
                // let old : &'a GcArcInner<T::AsArc> = mem::transmute(self.new);
                // let arc : GcArc<T::AsArc> = GcArc::from(/*self.old*/old);
                let arc : GcArc<T::AsArc> = old.clone();
                Ok(arc.into())
            } else {
                // This is a GcRef.
                let new = self.new;
                if let Some(forward) = new.forward.get() {
                    // Since fwd is still going on, it's safe to dereference the forwarding pointer.
                    // let fwd_ref : &GcArcInner<T::AsArc/*Fwd*/> = forward.as_ref();
                    let forward : GcArcFwd<T::AsArcFwd> = mem::transmute(GcArc::from_raw(forward.as_ptr()));
                    // let forward: &GcArcFwd<T::AsArcFwd> = mem::transmute(forward.from_raw());
                    // let forward: &GcArcFwd<T::AsArcFwd> = mem::transmute(/*&fwd_ref*/forward);
                    // Since fwd is still going on, it's safe to treat the forwarding pointer as a GcArcInner.
                    // We then clone it to avoid exposing GcArcInner directly to clients (but maybe
                    // we want to, in which case we'd hand it out with lifetime 'b).
                    /* let forward : &GcArcInner<T::AsArcFwd> = forward.as_ref(); */
                    /* let arc : GcArc<T::AsArcFwd> = GcArc::from(forward);
                    Ok(arc.into()) */
                    Ok(GcArcFwd::clone(&forward))
                } else {
                    // We do not have a forwarding pointer.
                    Err(new)
                }
            }
        }
    }

    /* #[inline]
    fn from(new: &'a GcRefInner<'gc, T>) -> Self {
        Gc { new }
    } */
}

unsafe impl<'gc, T> GcArcFwdLayout for GcArcFwd<'gc, T> {
    type FromArcFwd = GcArc<T>;
}

/// Can always turn a GcArc by-value into a GcArcFwd, if needed.
impl<'gc, T> From<GcArc<T::FromArcFwd>> for GcArcFwd<'gc, T> where T: GcArcFwdLayout {
    #[inline]
    fn from(arc: GcArc<T::FromArcFwd>) -> Self {
        GcArcFwd {
            arc: unsafe { ManuallyDrop::new(mem::transmute(arc)) },
            _marker: InvariantLifetime::new(),
        }
        /* GcArcFwd {
            // Correct because T::AsArc can always be zero-copied to T::AsArcFwd.
            ptr: arc.ptr.cast(),
            phantom: PhantomData,
            _marker: InvariantLifetime::new(),
        } */
    }
}

/// Cloning a GcArcFwd is always legal.
impl<'gc, T/*: ?Sized*/> Clone for GcArcFwd<'gc, T> {
    #[inline]
    fn clone(&self) -> GcArcFwd<'gc, T> {
        /* // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // println!("Incrementing {:p}", self.inner());
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }

        GcArcFwd {
            ptr: self.ptr,
            phantom: PhantomData,
            _marker: InvariantLifetime::new(),
        } */
        GcArcFwd {
            arc: ManuallyDrop::new(GcArc::clone(&self.arc)),
            _marker: InvariantLifetime::new(),
        }
    }
}

/* /// Can always turn a &'a GcArc into a GcArcFwd, if needed.
/// Key: GcArc must provably live long enough that GcFwd<'a>
/// can't still be alive when drop is called.  So this API is
/// a bit questionable / weak compared to some others.
impl<'a, T> From<&'a GcArc<T::AsArc>> for GcArcFwd<'a, T::AsArcFwd> where T: GcArcLayout {
    #[inline]
    fn from(arc: &'a GcArc<T>) -> Self {
        GcArcFwd {
            ptr: arc.ptr,
            phantom: arc.phantom,
            _marker: InvariantLifetime::new(),
        }
    }
} */

/* impl<'a, T> From<&'a GcArcInner<T>> for GcArc<T> {
    fn from(inner: &'a GcArcInner<T>) -> Self {
        // println!("Incrementing {:p}", inner);
        let old_size = inner.strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }

        GcArc { ptr: inner.into(), phantom: PhantomData }
    }
} */

/* impl<'a, 'b, T> From<&'b GcArcInner<T::AsArc>> for GcArcFwd<'a, T::AsArcFwd> where T: GcArcLayout {
    fn from(inner: &'b GcArcInner<T>) -> Self {
        let old_size = inner.strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }

        GcArcFwd {
            ptr: inner.into(),
            phantom: PhantomData,
            _marker: InvariantLifetime::new(),
        }
    }
} */

impl<'gc, 'a, T> Clone for Gc<'gc, 'a, T> where T: GcArcLayout {
    #[inline]
    fn clone(&self) -> Self {
        // Always legal to copy either pointer
        /*unsafe {*/
            Gc { /*old: self.old*/new: self.new }
        /*}*/
    }
}

impl<'gc, 'a, T> Copy for Gc<'gc, 'a, T> where T: GcArcLayout {
}


/// Safe because can go from &T::AsArc to &T, as long as (1) we can differentiate a
/// GcArcInner from a GcRefInner, and (2) the reference outlives 'a + 'gc!
/// Since it lives for at least 'a, (2) is satisfied; (1) is guaranteed by the fact that
/// GcArcInner and GcRefInner have compatible representations (upheld by the two modules
/// in this crate, but should probably make nicer with repr(transparent) or something).
///
/// We used to have a union for the two kinds of references, but got rid of it in hopes of fixing a
/// Miri bug.  Since the bug turned out to be unrelated to the union we might add it back at some
/// point, maybe.
impl<'gc, 'a, T> From<&'a GcArc<T::AsArc>> for Gc<'gc, 'a, T> where T : GcArcLayout {
    #[inline]
    fn from(old: &'a GcArc<T::AsArc>) -> Self {
        unsafe { Gc { new: /*mem::transmute(old)*/ *(old as *const _ as *const &'a GcRefInner<T>) } }
    }
}

/// Note: Trivially safe.
impl<'gc, 'a, T> From<&'a GcRefInner<'gc, T>> for Gc<'gc, 'a, T> where T: GcArcLayout {
    #[inline]
    fn from(new: &'a GcRefInner<'gc, T>) -> Self {
        Gc { new }
    }
}

/* /// Note: not safe because can't go from T::AsArc to T!
impl<'a, T> From<&'a GcArcInner<T::AsArc>> for Gc<'a, T> {
    #[inline]
    fn from(old: Gc<'a, T::AsArc>) -> Self {
        Gc { old }
    }
} */

impl<'gc, 'a, T/*: ?Sized*/> Deref for Gc<'gc, 'a, T> where T: GcArcLayout {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        /*unsafe {*/
            // Shouldn't matter which one you pick!
            &self.new.data
        /*}*/
    }
}
