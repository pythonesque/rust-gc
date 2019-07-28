#![feature(untagged_unions)]
#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(box_into_raw_non_null)]
#![feature(box_syntax)]
#![feature(coerce_unsized)]
#![feature(core_intrinsics)]
#![feature(dispatch_from_dyn)]
#![feature(dropck_eyepatch)]
// #![feature(placement_in_syntax)]
#![feature(raw_vec_internals)]
#![feature(receiver_trait)]
#![feature(rustc_private)]
#![feature(specialization)]
#![feature(unsize)]

extern crate arena;
// extern crate alloc;
extern crate core;

// mod arena;
mod ghost_cell;

use arena::TypedArena;
// use typed_arena::{Arena as TypedArena};
// use copy_arena::{Arena, Allocator};
// use light_arena::{self, MemoryArena, Allocator};


use core::any::Any;
use core::cell::Cell;
use core::sync::atomic;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use core::borrow;
use core::fmt;
use core::cmp::{self, Ordering};
use core::intrinsics::abort;
use core::mem::{self, align_of_val, ManuallyDrop, size_of_val};
use core::ops::{Deref, Receiver, CoerceUnsized, DispatchFromDyn};
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::marker::{Unpin, Unsize, PhantomData};
use core::hash::{Hash, Hasher};
use core::{isize, usize};
use core::convert::From;
use core::slice::from_raw_parts_mut;

use crate::ghost_cell::{InvariantLifetime, Cell as GhostCell, Set as GhostSet};

use std::alloc::{Global, Alloc, Layout, /*box_free, */handle_alloc_error};
use std::boxed::Box;
// use std::rc::is_dangling;
use std::string::String;
use std::vec::Vec;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/* For now, we don't try to support dynamically sized types as data since this makes the use of the union
   incorrect... unless we can manage to turn it into a thin pointer.  We will probably instead do some sort
   of custom thin pointer for slices, since that's the main reason you want this.
*/

#[repr(C)]
struct GcArcInner<T/*: ?Sized*/> {
    strong: atomic::AtomicUsize,
    // No weak pointers for this one, but it keeps the representations the same between
    // GcArcInner and GcRefInner.  We might use it to store object length for vectors at
    // some point.
    weak: /*atomic::AtomicUsize,*//*UnsafeCell<Option<NonNull<GcArcInner<T>>>>*/atomic::AtomicPtr<GcArcInner<T>>,
    data: T,
}

unsafe impl<T: /*?Sized + */Sync + Send> Send for GcArcInner<T> {}
unsafe impl<T: /*?Sized + */Sync + Send> Sync for GcArcInner<T> {}

#[repr(C)]
pub struct GcArc<T/*: ?Sized*/> {
    ptr: NonNull<GcArcInner<T>>,
    phantom: PhantomData<T>,
}

impl<T> GcArc<T> {
    #[inline]
    pub fn new(data: T) -> GcArc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = box GcArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: /*atomic::AtomicUsize::new(1)*/atomic::AtomicPtr::new(ptr::null_mut()),
            data,
        };
        // println!("Initialization, incrementing {:p}", x);

        GcArc { ptr: Box::into_raw_non_null(x), phantom: PhantomData }
    }

    pub fn pin(data: T) -> Pin<GcArc<T>> {
        unsafe { Pin::new_unchecked(GcArc::new(data)) }
    }

    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // See `drop` for why all these atomics are like this
        // println!("Decrementing if unique {:p}", this.inner());
        if this.inner().strong.compare_exchange(1, 0, Release, Relaxed).is_err() {
            return Err(this);
        }

        atomic::fence(Acquire);

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);

            // Make a weak pointer to clean up the implicit strong-weak reference
            /* let _weak = Weak { ptr: this.ptr };
            mem::forget(this); */
            atomic::fence(Acquire);
            Global.dealloc(this.ptr.cast(), Layout::for_value(this.ptr.as_ref()));

            Ok(elem)
        }
    }
}

impl<T/*: ?Sized*/> GcArc<T> {
    pub fn into_raw(this: Self) -> *const T {
        let ptr: *const T = &*this;
        mem::forget(this);
        ptr
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // Align the unsized value to the end of the GcArcInner.
        // Because it is ?Sized, it will always be the last field in memory.
        let align = align_of_val(&*ptr);
        let layout = Layout::new::<GcArcInner<()>>();
        let offset = (layout.size() + layout.padding_needed_for(align)) as isize;

        // Reverse the offset to find the original GcArcInner.
        let fake_ptr = ptr as *mut GcArcInner<T>;
        let arc_ptr = set_data_ptr(fake_ptr, (ptr as *mut u8).offset(-offset));

        GcArc {
            ptr: NonNull::new_unchecked(arc_ptr),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn into_raw_non_null(this: Self) -> NonNull<T> {
        // safe because GcArc guarantees its pointer is non-null
        unsafe { NonNull::new_unchecked(GcArc::into_raw(this) as *mut _) }
    }

    /* pub fn downgrade(this: &Self) -> Weak<T> {
        // This Relaxed is OK because we're checking the value in the CAS
        // below.
        let mut cur = this.inner().weak.load(Relaxed);

        loop {
            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                cur = this.inner().weak.load(Relaxed);
                continue;
            }

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and GcArc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match this.inner().weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => {
                    // Make sure we do not create a dangling Weak
                    debug_assert!(!is_dangling(this.ptr));
                    return Weak { ptr: this.ptr };
                }
                Err(old) => cur = old,
            }
        }
    }

    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.inner().weak.load(SeqCst);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX { 0 } else { cnt - 1 }
    } */

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(SeqCst)
    }

    #[inline]
    fn inner(&self) -> &GcArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `GcArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        // println!("{:?}", mem::size_of::<atomic::AtomicUsize>() as isize);
        // ptr::drop_in_place(((self.ptr.as_ptr() as *mut atomic::AtomicUsize).offset(1) as *mut atomic::AtomicPtr<GcArcInner<T>>).offset(1) as *mut T);
        // mem::size_of::<atomic::AtomicUsize>() as isize).offset(mem::size_of::<atomic::AtomicPtr<GcArcInner<T>>> as isize)));
        // ptr::drop_in_place((&mut self.ptr.as_mut().data) as *mut _);
        ptr::drop_in_place(&mut self.ptr.as_mut().data);

        /* if self.inner().weak.fetch_sub(1, Release) == 1 {*/
            atomic::fence(Acquire);
            Global.dealloc(self.ptr.cast(), Layout::for_value(self.ptr.as_ref()))
        /*} */
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T/*: ?Sized*/> GcArc<T> {
    // Allocates an `GcArcInner<T>` with sufficient space for an unsized value
    unsafe fn allocate_for_ptr(ptr: *const T) -> *mut GcArcInner<T> {
        // Calculate layout using the given value.
        // Previously, layout was calculated on the expression
        // `&*(ptr as *const GcArcInner<T>)`, but this created a misaligned
        // reference (see #54908).
        let layout = Layout::new::<GcArcInner<()>>()
            .extend(Layout::for_value(&*ptr)).unwrap().0
            .pad_to_align().unwrap();

        let mem = Global.alloc(layout)
            .unwrap_or_else(|_| handle_alloc_error(layout));

        // Initialize the GcArcInner
        let inner = set_data_ptr(ptr as *mut T, mem.as_ptr() as *mut u8) as *mut GcArcInner<T>;
        debug_assert_eq!(Layout::for_value(&*inner), layout);

        ptr::write(&mut (*inner).strong, atomic::AtomicUsize::new(1));
        ptr::write(&mut (*inner).weak, /*atomic::AtomicUsize::new(1)*/atomic::AtomicPtr::new(ptr::null_mut()));
        // println!("Initialization, incrementing {:p}", inner);


        inner
    }

    /* fn from_box(v: Box<T>) -> GcArc<T> {
        unsafe {
            let box_unique = Box::into_unique(v);
            let bptr = box_unique.as_ptr();

            let value_size = size_of_val(&*bptr);
            let ptr = Self::allocate_for_ptr(bptr);

            // Copy value as bytes
            ptr::copy_nonoverlapping(
                bptr as *const T as *const u8,
                &mut (*ptr).data as *mut _ as *mut u8,
                value_size);

            // Free the allocation without dropping its contents
            box_free(box_unique);

            GcArc { ptr: NonNull::new_unchecked(ptr), phantom: PhantomData }
        }
    } */
}

// Sets the data pointer of a `?Sized` raw pointer.
//
// For a slice/trait object, this sets the `data` field and leaves the rest
// unchanged. For a sized raw pointer, this simply sets the pointer.
unsafe fn set_data_ptr<T: ?Sized, U>(mut ptr: *mut T, data: *mut U) -> *mut T {
    ptr::write(&mut ptr as *mut _ as *mut *mut u8, data as *mut u8);
    ptr
}

impl<T/*: ?Sized*/> Clone for GcArc<T> {
    #[inline]
    fn clone(&self) -> GcArc<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
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

        GcArc { ptr: self.ptr, phantom: PhantomData }
    }
}

impl<T/*: ?Sized*/> Deref for GcArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T/*: ?Sized*/> Receiver for GcArc<T> {}

impl<T: Clone> GcArc<T> {
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T {
        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the GcArcInner itself could be
        // deallocated.
        // println!("Decrementing if unique {:p}", this.inner());
        if this.inner().strong.compare_exchange(1, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists; clone
            *this = GcArc::new((**this).clone());
        } /*else if this.inner().weak.load(Relaxed) != 1 {
            // Relaxed suffices in the above because this is fundamentally an
            // optimization: we are always racing with weak pointers being
            // dropped. Worst case, we end up allocated a new Arc unnecessarily.

            // We removed the last strong ref, but there are additional weak
            // refs remaining. We'll move the contents to a new Arc, and
            // invalidate the other weak refs.

            // Note that it is not possible for the read of `weak` to yield
            // usize::MAX (i.e., locked), since the weak count can only be
            // locked by a thread with a strong reference.

            // Materialize our own implicit weak pointer, so that it can clean
            // up the GcArcInner as needed.
            let weak = Weak { ptr: this.ptr };

            // mark the data itself as already deallocated
            unsafe {
                // there is no data race in the implicit write caused by `read`
                // here (due to zeroing) because data is no longer accessed by
                // other threads (due to there being no more strong refs at this
                // point).
                let mut swap = GcArc::new(ptr::read(&weak.ptr.as_ref().data));
                mem::swap(this, &mut swap);
                mem::forget(swap);
            }
        }*/ else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            // println!("Was unique, incrementing {:p}", this.inner());
            this.inner().strong.store(1, Release);
        }

        // As with `get_mut()`, the unsafety is ok because our reference was
        // either unique to begin with, or became one upon cloning the contents.
        unsafe {
            &mut this.ptr.as_mut().data
        }
    }
}

impl<T/*: ?Sized*/> GcArc<T> {
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the GcArc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe {
                Some(&mut this.ptr.as_mut().data)
            }
        } else {
            None
        }
    }

    fn is_unique(&mut self) -> bool {
        // lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release).  If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        /*if self.inner().weak.compare_exchange(1, usize::MAX, Acquire, Relaxed).is_ok() {*/
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = self.inner().strong.load(Acquire) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            /* self.inner().weak.store(1, Release); // release the lock */
            unique
        /*} else {
            false
        }*/
    }
}

unsafe impl<#[may_dangle] T/*: ?Sized*/> Drop for GcArc<T> {
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        // println!("Decrementing {:p}", self.inner());
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of a GcArc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        atomic::fence(Acquire);

        unsafe {
            self.drop_slow();
        }
    }
}


trait GcArcEqIdent<T: /*?Sized + */PartialEq> {
    fn eq(&self, other: &GcArc<T>) -> bool;
    fn ne(&self, other: &GcArc<T>) -> bool;
}

impl<T: /*?Sized + */PartialEq> GcArcEqIdent<T> for GcArc<T> {
    #[inline]
    default fn eq(&self, other: &GcArc<T>) -> bool {
        **self == **other
    }
    #[inline]
    default fn ne(&self, other: &GcArc<T>) -> bool {
        **self != **other
    }
}

impl<T: /*?Sized + */Eq> GcArcEqIdent<T> for GcArc<T> {
    #[inline]
    fn eq(&self, other: &GcArc<T>) -> bool {
        GcArc::ptr_eq(self, other) || **self == **other
    }

    #[inline]
    fn ne(&self, other: &GcArc<T>) -> bool {
        !GcArc::ptr_eq(self, other) && **self != **other
    }
}

impl<T: /*?Sized + */PartialEq> PartialEq for GcArc<T> {
    #[inline]
    fn eq(&self, other: &GcArc<T>) -> bool {
        GcArcEqIdent::eq(self, other)
    }

    #[inline]
    fn ne(&self, other: &GcArc<T>) -> bool {
        GcArcEqIdent::ne(self, other)
    }
}

impl<T: /*?Sized + */PartialOrd> PartialOrd for GcArc<T> {
    fn partial_cmp(&self, other: &GcArc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    fn lt(&self, other: &GcArc<T>) -> bool {
        *(*self) < *(*other)
    }

    fn le(&self, other: &GcArc<T>) -> bool {
        *(*self) <= *(*other)
    }

    fn gt(&self, other: &GcArc<T>) -> bool {
        *(*self) > *(*other)
    }

    fn ge(&self, other: &GcArc<T>) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: /*?Sized + */Ord> Ord for GcArc<T> {
    fn cmp(&self, other: &GcArc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: /*?Sized + */Eq> Eq for GcArc<T> {}

impl<T: /*?Sized + */fmt::Display> fmt::Display for GcArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: /*?Sized + */fmt::Debug> fmt::Debug for GcArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T/*: ?Sized*/> fmt::Pointer for GcArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: Default> Default for GcArc<T> {
    fn default() -> GcArc<T> {
        GcArc::new(Default::default())
    }
}

impl<T: /*?Sized + */Hash> Hash for GcArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T> From<T> for GcArc<T> {
    fn from(t: T) -> Self {
        GcArc::new(t)
    }
}

/*impl<T: Clone> From<&[T]> for GcArc<[T]> {
    #[inline]
    fn from(v: &[T]) -> GcArc<[T]> {
        <Self as GcArcFromSlice<T>>::from_slice(v)
    }
}

impl From<&str> for GcArc<str> {
    #[inline]
    fn from(v: &str) -> GcArc<str> {
        let arc = GcArc::<[u8]>::from(v.as_bytes());
        unsafe { GcArc::from_raw(GcArc::into_raw(arc) as *const str) }
    }
}

impl From<String> for GcArc<str> {
    #[inline]
    fn from(v: String) -> GcArc<str> {
        GcArc::from(&v[..])
    }
}*/

/* impl<T/*: ?Sized*/> From<Box<T>> for GcArc<T> {
    #[inline]
    fn from(v: Box<T>) -> GcArc<T> {
        GcArc::from_box(v)
    }
} */

/*impl<T> From<Vec<T>> for GcArc<[T]> {
    #[inline]
    fn from(mut v: Vec<T>) -> GcArc<[T]> {
        unsafe {
            let arc = GcArc::copy_from_slice(&v);

            // Allow the Vec to free its memory, but not destroy its contents
            v.set_len(0);

            arc
        }
    }
}*/

impl<T/*: ?Sized*/> borrow::Borrow<T> for GcArc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T/*: ?Sized*/> AsRef<T> for GcArc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T/*: ?Sized*/> Unpin for GcArc<T> { }

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

pub struct GcFwd<'gc> {
    _marker: InvariantLifetime<'gc>,
}

#[derive(Clone,Copy)]
pub struct GcDead<'gc> {
    _marker: InvariantLifetime<'gc>,
}

pub trait Hkt<#[may_dangle]'a> {
    type HktOut : GcArcLayout + Copy;
}

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
            let (/*mut data, */fwd) : (/*TypedArena<GcRefInner<<T as Hkt<'_>>::HktOut>>*//*MemoryArena, */GcFwd<'_>);
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

unsafe impl<'gc, T> GcArcFwdLayout for GcArcFwd<'gc, T> {
    type FromArcFwd = GcArc<T>;
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

    #[inline]
    fn inner(&self) -> &GcArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `GcArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        self.arc.inner()
        // contents.
        /* unsafe { /*self.ptr.as_reF()*/self.inner() } */

    }

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
        GcArc {
            ptr: self.arc.ptr.cast(),
            phantom: PhantomData,
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

/// Can always turn a GcArc into a GcArcFwd, if needed.
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
    forward: Cell<Option</*GcArcFwd<'a, T::AsArcFwd>*/NonNull<GcArcInner<T::AsArcFwd>>>>,
    data: T,
    _marker: InvariantLifetime<'gc>
}

impl<'a, T> From<&'a GcArcInner<T>> for GcArc<T> {
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
}

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

impl<'gc, T> GcRefInner<'gc, T> where T: GcArcLayout {
    pub fn new(data: T) -> Self {
        GcRefInner {
            gc_ref_tag: atomic::AtomicUsize::new(0),
            forward: Cell::new(None),
            data,
            _marker: InvariantLifetime::new(),
        }
    }
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


/* pub struct GcHkt<T>(T);

impl<'a, T> Hkt<'a> for GcHkt<T>
    where
        T: GcArcLayout + 'a,
        <T as GcArcLayout>::AsArc: 'a,
{
    type HktOut = Gc<'a, T>;
} */

/* pub struct GcRefInnerHkt<T>(T);

impl<'a, T> Hkt<'a> for GcRefInnerHkt<T>
    where
        <T as Hkt<'a>>::HktOut: GcArcLayout/* + 'a*/,
        T: Hkt<'a>,
        // <T as GcArcLayout>::AsArcFwd: 'a,
{
    type HktOut = GcRefInner<'a, <T as Hkt<'a>>::HktOut>;
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

/// Safe because can go from T::AsArc to &T!
impl<'gc, 'a, T> From<&'a GcArc<T::AsArc>> for Gc<'gc, 'a, T> where T : GcArcLayout {
    #[inline]
    fn from(old: &GcArc<T::AsArc>) -> Self {
        unsafe { Gc { new: mem::transmute(old.inner()) } }
    }
}

impl<'gc, 'a, T> Gc<'gc, 'a, T> where T: GcArcLayout {
    /// Note: this should only be run during collection, but with the current design it doesn't matter.
    #[inline]
    pub fn try_as_arc<'b>(self, fwd: &'b GcFwd<'gc>) -> Result</*&'a GcArcInner<T::AsArcFwd>*/GcArcFwd<'gc, T::AsArcFwd>, &'a GcRefInner<'gc, T>> {
        unsafe {
            // let old = self.old;
            // let strong = self.new.gc_ref_tag;
            let old_strong = self./*old.strong*/new.gc_ref_tag.load(Relaxed);
            if old_strong > 0 {
                // This is a GcArc, so old pointer is valid.
                let old : &'a GcArcInner<T::AsArc> = mem::transmute(self.new);
                let arc : GcArc<T::AsArc> = GcArc::from(/*self.old*/old);
                Ok(arc.into())
            } else {
                // This is a GcRef.
                let new = self.new;
                if let Some(ref forward) = new.forward.get() {
                    // Since fwd is still going on, it's safe to dereference the forwarding pointer.
                    // let fwd_ref : &GcArcInner<T::AsArc/*Fwd*/> = forward.as_ref();
                    let forward: &GcArcFwd<T::AsArcFwd> = mem::transmute(/*&fwd_ref*/forward);
                    // Since fwd is still going on, it's safe to treat the forwarding pointer as a GcArcInner.
                    // We then clone it to avoid exposing GcArcInner directly to clients (but maybe
                    // we want to, in which case we'd hand it out with lifetime 'b).
                    /* let forward : &GcArcInner<T::AsArcFwd> = forward.as_ref(); */
                    /* let arc : GcArc<T::AsArcFwd> = GcArc::from(forward);
                    Ok(arc.into()) */
                    Ok(GcArcFwd::clone(forward))
                } else {
                    // We do not have a forwarding pointer.
                    Err(new)
                }
            }
        }
    }
}

impl<'gc, T> GcRefInner<'gc, T> where T: GcArcLayout {
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
        self.forward.set(Some(forwarding.inner().into()));
    }
}

/* impl<'gc, 'a, T> Gc<'gc, 'a, T> where T: GcArcLayout {
    #[inline]
    fn from(new: &'a GcRefInner<'gc, T>) -> Self {
        Gc { new }
    }
} */

pub struct TypedArenaHkt<T>(T);

/* impl<'a, T> Hkt<'a> for TypedArenaHkt<T>
    where
        T: Hkt<'a>,
        // T: Hkt<'a> + 'a,
        // <T as Hkt<'a>>::HktOut : 'a
{
    type HktOut = TypedArena<<T as Hkt<'a>>::HktOut>;
} */

    /// Just trying to keep representations the same.
    #[repr(C)]
    #[derive(Clone,Copy)]
    struct App<T>(T, T);

    /* #[repr(C)]
    #[derive(Clone,Copy)]
    union ExprData<T> {
        rel: u64,
        abs: T,
        app: App<T>,
    }

    #[repr(C)]
    #[derive(Clone,Copy)]
    enum ExprTag {
        Rel = 0,
        Abs = 1,
        App = 2,
    }

    #[repr(C)]
    struct ExprVar<T> {
        tag: ExprTag,
        data: ExprData<T>,
    } */
    #[repr(C)]
    #[derive(Clone,Copy)]
    enum ExprVar<T> {
        Rel(u64),
        Abs(T),
        App(App<T>),
    }

    /* impl<T> Clone for ExprVar<T> where T: Copy {
        fn clone(&self) -> Self {
            Self { tag: self.tag, data: self.data }
        }
    }

    impl<T> Copy for ExprVar<T> where T: Copy {} */

    /// Just trying to keep representations the same.
    #[repr(C)]
    #[derive(Clone,Copy)]
    struct Expr<'gc, 'a>(ExprVar<Gc<'gc, 'a, Expr<'gc, 'a>>>);
    impl<'gc, 'a> Deref for Expr<'gc,'a> { type Target = ExprVar<Gc<'gc, 'a, Expr<'gc, 'a>>>; fn deref(&self) -> &Self::Target { &self.0 } }

    #[repr(C)]
    #[derive(Clone,Copy)]
    struct ExprRef<'gc, 'a>(ExprVar<&'a GcRefInner<'gc, ExprRef<'gc, 'a>>>);
    impl<'gc, 'a> Deref for ExprRef<'gc,'a> { type Target = ExprVar<&'a GcRefInner<'gc, ExprRef<'gc, 'a>>>; fn deref(&self) -> &Self::Target { &self.0 } }

    #[repr(C)]
    struct ExprArcFwd<'gc>(ExprVar<GcArcFwd<'gc, ExprArcFwd<'gc>>>);
    impl<'gc> Deref for ExprArcFwd<'gc> { type Target = ExprVar<GcArcFwd<'gc, ExprArcFwd<'gc>>>; fn deref(&self) -> &Self::Target { &self.0 } }

    /* #[repr(C)]
    struct ExprArcFwd<'gc>(ExprVar<GcArcFwd<'gc, ExprArc>>);
    impl<'gc> Deref for ExprArcFwd<'gc> { type Target = ExprVar<GcArcFwd<'gc, ExprArc>>; fn deref(&self) -> &Self::Target { &self.0 } } */

    #[repr(C)]
    struct ExprArc(ExprVar<GcArc<ExprArc>>);
    impl<'gc, 'a> Deref for ExprArc { type Target = ExprVar<GcArc<ExprArc>>; fn deref(&self) -> &Self::Target { &self.0 } }

    impl<T> ExprVar<T> {
        /* #[inline]
        fn Rel(rel: u64) -> Self {
            /*ExprVar {
                tag: ExprTag::Rel,
                data: ExprData { rel },
            }*/
            ExprVar::Rel(rel)
        }

        #[inline]
        fn Abs(abs: T) -> Self {
            /*ExprVar {
                tag: ExprTag::Abs,
                data: ExprData { abs },
            }*/
            ExprVar::Abs(abs)
        }

        #[inline]
        fn App(app: App<T>) -> Self {
            /*ExprVar {
                tag: ExprTag::App,
                data: ExprData { app },
            }*/
            ExprVar::App(app)
        } */

        #[inline]
        fn match_own<R>(self, f_rel: impl FnOnce(u64) -> R,
                         f_abs: impl FnOnce(T) -> R,
                         f_app: impl FnOnce(App<T>) -> R) -> R {
            /*unsafe {
                match self.tag {
                    ExprTag::Rel => f_rel(self.data.rel),
                    ExprTag::Abs => f_abs(self.data.abs),
                    ExprTag::App => f_app(self.data.app),
                }
            }*/
            match self {
                ExprVar::Rel(rel) => f_rel(rel),
                ExprVar::Abs(abs) => f_abs(abs),
                ExprVar::App(app) => f_app(app),
            }
        }

        #[inline]
        fn match_shr<R>(&self, f_rel: impl FnOnce(u64) -> R,
                         f_abs: impl FnOnce(&T) -> R,
                         f_app: impl FnOnce(&App<T>) -> R) -> R {
            /* unsafe {
                match self.tag {
                    ExprTag::Rel => f_rel(self.data.rel),
                    ExprTag::Abs => f_abs(&self.data.abs),
                    ExprTag::App => f_app(&self.data.app),
                }
            } */
            match self {
                ExprVar::Rel(rel) => f_rel(*rel),
                ExprVar::Abs(abs) => f_abs(abs),
                ExprVar::App(app) => f_app(app),
            }
        }
    }

    /* impl Drop for ExprArc {
        #[inline]
        fn drop(&mut self) {
            unsafe {
                match self.0.tag {
                    ExprTag::Rel => ptr::drop_in_place(&mut self.0.data.rel),
                    ExprTag::Abs => ptr::drop_in_place(&mut self.0.data.abs),
                    ExprTag::App => ptr::drop_in_place(&mut self.0.data.app),
                }
            }
        }
    } */

    /* enum Expr<'gc, 'a> {
        Rel(u64),
        Abs(Gc<'gc, 'a, Expr<'gc, 'a>>),
        App(App<Gc<'gc, 'a, Expr<'gc, 'a>>>),
    }

    /// Just trying to keep representations the same.
    #[repr(C)]
    enum ExprRef<'gc, 'a> {
        Rel(u64),
        Abs(&'a GcRefInner<'gc, ExprRef<'gc, 'a>>),
        App(App<&'a GcRefInner<'gc, ExprRef<'gc, 'a>>>),
    }

    /// Just trying to keep representations the same.
    #[repr(C)]
    enum ExprArc {
        Rel(u64),
        Abs(GcArc<ExprArc>),
        App(App<GcArc<ExprArc>>),
    }

    /// Just trying to keep representations the same.
    #[repr(C)]
    enum ExprArcFwd<'gc> {
        Rel(u64),
        Abs(GcArcFwd<'gc, ExprArcFwd<'gc>>/*OwnFwd<'gc, ExprArcFwd<'gc>>*/),
        App(App<GcArcFwd<'gc, ExprArcFwd<'gc>>/*OwnFwd<'gc, ExprArcFwd<'gc>>*/>),
    } */

    /* union OwnFwd<'gc, T> where T: GcArcFwdLayout {
        old: GcArcFwd<'gc, T>,
        new: GcArc<T::FromArcFwd>,
    } */

    /// Should have the exact same layout.
    unsafe impl<'gc, 'a> GcArcLayout for Expr<'gc, 'a> {
        type AsArc = ExprArc;
        type AsArcFwd = ExprArcFwd<'gc>;
    }

    /// Should have the exact same layout.
    unsafe impl<'gc, 'a> GcArcLayout for ExprRef<'gc, 'a> {
        type AsArc = ExprArc;
        type AsArcFwd = ExprArcFwd<'gc>;
    }

    unsafe impl<'gc> GcArcFwdLayout for ExprArcFwd<'gc> {
        type FromArcFwd = ExprArc;
    }

pub fn gc_example() {
    type Idx = usize;


    /* struct ExprHkt;

    impl<'a> Hkt<'a> for ExprHkt {
        type HktOut = Expr<'a>;
    } */

    /* struct ExprHkt2;

    impl<'a> Hkt<'a> for ExprHkt2 {
        type HktOut = TypedArena<GcRefInner<'a, Expr<'a>>>;
    } */

    // We don't have a single collection function.  Instead, our types are designed to require us
    // to collect GC'd data manually.  The three kinds of data we have are "transitively old"
    // (ExprArc), "transitively young" (ExprRef), and "young, but could refer to old pointers"
    // (ExprRef).  These allow for different optimizations and semantics.
    let mut old_stack = Vec::<ExprArc>::new();
    /* fn initialize<'b, 'a>(_: &'b GcFwd<'a>) -> /*TypedArena<GcRefInner<'a, Expr<'a>>>*//*<TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt<'a>>::HktOut*//*<ExprHkt2 as Hkt<'a>>::HktOut*/TypedArena<GcRefInner<'a, /*Expr<'a>*/<ExprHkt as Hkt<'a>>::HktOut>> {
        /*<TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt<'a>>::HktOut::default()*/
        TypedArena::default()
    } */

    /* fn run<'a>(my_arena: &'a mut TypedArena<GcRefInner<'a, Expr<'a>>>, gc: GcFwd<'a>) -> () {
        let mut new_stack = Vec::<ExprRef>::new();
        let var = (&*my_arena.alloc(GcRefInner::new(Expr::Rel(0)))).into();
        let id = my_arena.alloc(GcRefInner::new(Expr::Abs(var)));
        my_arena.alloc(GcRefInner::new(Expr::Rel(1)));

        // Do some tracing.

    } */

    /* fn constrain<F, R>(f: F) -> F
    where
        F: for<'a> FnOnce(&'a mut /*<TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt<'a>>::HktOut*/TypedArena<GcRefInner<'a, Expr<'a>>>, GcFwd<'a>) -> R
    {
        f
    }
    // let foo : <TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt>::HktOut = TypedArena::<GcRefInner<Expr>>::default();
        /*for <'a> |GcFwd<'a>| -> &'a TypedArena<GcRefInner<'a, Expr<'a>>>*/
    // let k
    let nt = constrain(move |my_arena, fwd| {
        // let my_arena: &mut TypedArena::<GcRefInner<Expr>> = my_arena;
        /* let my_arena = TypedArena::<GcRefInner<Expr>>::default(); */

        // Mutator part.
        let mut new_stack = Vec::<ExprRef>::new();
        let var = (&*my_arena.alloc(GcRefInner::new(Expr::Rel(0)))).into();
        let id = my_arena.alloc(GcRefInner::new(Expr::Abs(var)));
        my_arena.alloc(GcRefInner::new(Expr::Rel(1)));

        // Do some tracing.

    });
    /* fn constrain2<F, R>(f: F) -> /*R*/()
        where
            for<'a> F: FnOnce(&'a mut <ExprHkt2 as Hkt<'a>>::HktOut/*<TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt<'a>>::HktOut*//*TypedArena<GcRefInner<'a, Expr<'a>>>*/, GcFwd<'a>) -> R

    {
        GcFwd::new::</*TypedArenaHkt<GcRefInnerHkt<ExprHkt>>*/ExprHkt2, _, _, ()>(/*|_: &_| TypedArena::default()*/(initialize as (for<'a> fn(&GcFwd<'a>) -> <ExprHkt2 as Hkt<'a>>::HktOut)), /*f*/(run as (for<'a> fn(&'a mut <ExprHkt2 as Hkt<'a>>::HktOut, GcFwd<'a>))))
    } */ */
    fn expr_to_string<T>(/*fwd: &GcFwd<'gc>, */expr: /*&*//*Expr<'gc, 'a>*/&ExprVar<T>) -> String
        where T: Deref, T::Target: Deref<Target=ExprVar<T>>, /*T::Target: AsRef<ExprVar<T>>*//*: Borrow<<Target=ExprVar<T>>*/
    {
        /*match expr {
            Expr::Rel(idx) => idx.to_string(),
            Expr::Abs(body) => format!("(λ. {:?})", expr_to_string(fwd, *body)),
            Expr::App(App(fun, arg)) => format!("({:?} {:?})", expr_to_string(fwd, *fun), expr_to_string(fwd, *arg)),
        }*/
        expr.match_shr(
            |idx| idx.to_string(),
            |body| format!("(λ. {})", expr_to_string(&**body)),
            |App(fun, arg)| format!("({} {})", expr_to_string(&*fun), expr_to_string(&**arg))
        )
    }

    /* fn expr_arc_to_string(expr: /*&*/&ExprArc) -> String {
        match expr {
            ExprArc::Rel(idx) => idx.to_string(),
            ExprArc::Abs(body) => format!("(λ. {:?})", expr_arc_to_string(body)),
            ExprArc::App(App(fun, arg)) => format!("({:?} {:?})", expr_arc_to_string(fun), expr_arc_to_string(arg)),
        }
    } */
    fn trace_root<'a, 'gc>(fwd: &GcFwd<'gc>, expr: /*&*/Expr<'gc, 'a>) -> ExprArcFwd<'gc> {
        /* match expr {
            Expr::Rel(idx) => ExprArcFwd::Rel(idx),
            Expr::Abs(body) => ExprArcFwd::Abs(
                /*OwnFwd */{ /*old: */body.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(trace_root(fwd, *body));
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                }) }),
            Expr::App(App(fun, arg)) => ExprArcFwd::App(App(
                /*OwnFwd */{ /*old: */fun.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(trace_root(fwd, *fun));
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                }) },
                /*OwnFwd */{ /*old: */arg.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(trace_root(fwd, *arg));
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                }) })),
        } */
        ExprArcFwd(expr.0.match_shr(
            |idx| ExprVar::Rel(idx),
            |body| ExprVar::Abs(
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/body.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, **body)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ }),
            |App(fun, arg)| ExprVar::App(App(
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/fun.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, **fun)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ },
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/arg.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, **arg)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ })),
        ))
    }


    let root = GcFwd::new/*::<ExprHkt, _, _>*/(/*|_| (TypedArena::default()), */move |/*my_arena, */fwd| {
        let my_arena: TypedArena::<GcRefInner<Expr>> = TypedArena::default()/*with_capacity(1000)*/;

        // Mutator part.
        let mut new_stack = Vec::<ExprRef>::new();
        let var = (&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Rel(0))))).into();
        let id = (&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Abs(var))))).into();
        let r2 = Gc::from(&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Rel(1)))));

        let v = ExprVar::App(App(id, id));

        // Use the values to make sure the issue isn't in the arena itself.

        /* fn expr_to_string<'a, 'gc>(fwd: &GcFwd<'gc>, expr: /*&*/Expr<'gc, 'a>) -> String {
            match expr {
                Expr::Rel(idx) => idx.to_string(),
                Expr::Abs(body) => format!("(λ. {:?})", expr_to_string(fwd, *body)),
                Expr::App(App(fun, arg)) => format!("({:?} {:?})", expr_to_string(fwd, *fun), expr_to_string(fwd, *arg)),
            }
        } */

        println!("{}", expr_to_string(/*&fwd, */&v));

        // Do some tracing.
        /* fn trace<'a, 'gc>(fwd: &GcFwd<'gc>, expr: Gc<'gc, 'a, Expr<'gc, 'a>>) -> GcArcFwd<'gc, ExprArcFwd<'gc>> {
            expr.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                let expr_new = GcArcFwd::new(match *expr {
                    Expr::Rel(idx) => ExprArcFwd::Rel(idx),
                    Expr::Abs(body) => ExprArcFwd::Abs(trace(fwd, body)),
                    Expr::App(App(fun, arg)) => ExprArcFwd::App(App(trace(fwd, fun), trace(fwd, arg))),
                });
                expr_ref.set_forwarding(&expr_new);
                expr_new
            })
        } */

        /* let root = match Gc::from(id).try_as_arc(&fwd) {
            Ok(old_id) => old_id,
            Err(new_id) => {
                if let Expr::Abs(r) = *Gc::from(new_id) {
                    let var = match r.try_as_arc(&fwd) {
                        Ok(var_id) => var_id,
                        Err(new_id) => {
                            let new_var_id = if let Expr::Rel(i) = *Gc::from(new_id) {
                                GcArcFwd::new(ExprArcFwd::Rel(i))
                            } else {
                                panic!()
                            };
                            new_id.set_forwarding(&new_var_id);
                            new_var_id
                        }
                    };
                    let new_abs = GcArcFwd::new(ExprArcFwd::Abs(var));
                    new_id.set_forwarding(&new_abs);
                    new_abs
                } else { panic!(); }
            }
        }*/;
        let root = trace_root(&fwd, /*&*/Expr(v));
        let end = fwd.end();
        // root.into_gc_arc(end)
        /* if let ExprArcFwd::App(App(fun, arg)) = root {
            ExprArc::App(App(fun.into_gc_arc(end), arg.into_gc_arc(end)))
            /* unsafe {
                ExprArc::App(App(fun.new, arg.new))
            } */
        } else { panic!() } */
        root.0.match_own(
            |_| panic!(),
            |_| panic!(),
            |App(fun, arg)| ExprArc(ExprVar::App(App(fun.into_gc_arc(end), arg.into_gc_arc(end)))),
        )

        /* unsafe { mem::transmute(root) } */
        // ExprArc::Rel(0)
    });
    println!("Passed root out of GC context: {}", /*expr_arc_to_string*/expr_to_string(&root));

    let root2 = GcFwd::new(move |fwd| {
        // let root = GcArc::new(root);
        let root = &root;
        // NOTE: This has one extra increment compared to the "preemptively initialize GcArc<T>"
        // version when the GcArc is actually used, because we can't use the GcArc "in place."
        // This is because we don't know for sure whether it will actually be allocated in the
        // arena yet, so we need something to keep it alive--and then during the arena sweep it
        // increments again.  To make this work nicely seems tricky (surely you need *some* way
        // to differentiate between the two cases, and would any of those ways be lower overhead
        // than a count?), so maybe we just accept this limitation for now.
        let my_arena: TypedArena::<GcRefInner<Expr>> = TypedArena::default()/*with_capacity(1000)*/;
        let root = GcRefInner::new(Expr(match root.0 {
            ExprVar::Rel(rel) => ExprVar::Rel(rel),
            ExprVar::Abs(ref abs) => ExprVar::Abs(Gc::from(abs)),
            ExprVar::App(App(ref fun, ref arg)) => ExprVar::App(App(Gc::from(fun), Gc::from(arg))),
        }));

        let root = Gc::from(&*my_arena.alloc(root)); // GcRefInner::new(Gc::from(&**root)))); */
        // let root = Gc::from(root);
        // let r2 = ExprVar::Rel(1);
        let v = /*Gc::from(&*my_arena.alloc(GcRefInner::new(Expr(*/ExprVar::App(App(root, root))/*))))*/;

        println!("{}", expr_to_string(/*&fwd, */&v));

        let root = trace_root(&fwd, /*&*/Expr(v/*r2*/));
        let end = fwd.end();
        GcArcFwd::new(root).into_gc_arc(end)
    });
    // GcFwd::new::</*TypedArenaHkt<GcRefInnerHkt<*/ExprHkt/*>>*/, _, _, _>(/*|_: &_| TypedArena::default()*/initialize, nt);
    // constrain2(nt);
    /*GcFwd::new::<TypedArenaHkt<GcRefInnerHkt<ExprHkt>>, _, _, _>(initialize, /*move |my_arena : &'r mut <TypedArenaHkt<GcRefInnerHkt<ExprHkt>> as Hkt<'r>>::HktOut, fwd|*/
                                                                 move |my_arena, fwd| {
        let my_arena: &mut TypedArena::<GcRefInner<Expr>> = my_arena;
        /* let my_arena = TypedArena::<GcRefInner<Expr>>::default(); */

        // Mutator part.
        let mut new_stack = Vec::<ExprRef>::new();
        let var = (&*my_arena.alloc(GcRefInner::new(Expr::Rel(0)))).into();
        let id = my_arena.alloc(GcRefInner::new(Expr::Abs(var)));
        my_arena.alloc(GcRefInner::new(Expr::Rel(1)));

        // Do some tracing.
    });*/

    /* Idea: we just buzz along during the mutator phase, and can freely access (or drop, currently) a GcArc anywhere,
       but we cannot set a forwarding pointer.  The tracing phase will allow setting the forwarding pointer (currently
       only one at a time though) and freely reading GcRefInner data (currently GcArcInner data too, which is probably
       wrong in other contexts), but does not allow dropping GcArcs.  At least, not GcArcs that *might* be forwarded.
       The challenge here is that the lifetime of the GcArc isn't explicit.  But we have to make sure the reference
       lasts at least as long as the forwarding pointers are being accessed.  We may be able to resolve this by creating
       a Forwarding<'fwd> which "represents" the lifetime during which forwarding pointers are being accessed.  In the
       current model, forwarding pointers can be accessed as long as a Gc<'a, T> is alive, but we could change things so
       forwarding pointers can only be accessed when Forwarding<'fwd> is alive.  The trouble is that there's no way to
       tell (without doing *something* at runtime, like checking for a watermark or adding a version) whether a forwarding
       pointer was set following an "old" Gc, so it probably needs to be known when you create Gc<'a, T>.  Then the followup
       would be, how do you know when it's safe to start again?

       ...

       Take 2: A Gc<'a, 'gc, T> holds either &'a GcArcInner<T::AsArc> or &'a GcRefInner<'gc, T>.

       In general the 'gc part does not kick in until we begin forwarding.  We should be able to take a "plain" Gc<'a, T>
       for anything that doesn't have any forwarding pointers set.

       When we start scanning a root, it is a Gc<'a, T>.  But in the process of manipulating roots, *all* other Gc<'a, T>
       could be turned into Gc<'a, 'gc, T>.  The trick (in the immutable case) is that Gc<'a, T> doesn't care about anything
       but the data pointer in the first place.  So changing its type shouldn't matter much to non-mutators.  However, it
       SHOULD matter to mutators: how can they make sure that all the living Gc<'a, T> get captured by a single 'gc?  The
       easiest way is to make 'gc = 'a.  Then forwarding references aren't &'gc GcArcInner<T>, they're
       GcFwd<'gc, GcArcInner<T>> = GcFwd<'a, GcArcInner<T>>--meaning not that these references "have lifetime 'a", but that
       they are alive as long as the forwarding pointer Forward<'a> is alive.  Once Forward<'a> is dead, these references
       will once again be inaccessible, even if 'a is.  In the immutable case, that's fine, because all the old references
       are already accessible throughout.  Then we just have to make sure that we go from Mutator<'a> to Trace<'a> to
       Dropper<'a>, monotonically--in fact, we can even just go straight from Forward<'a> to Dead<'a> in the current
       model.  Ideally we'd go from Forward<'a> to nothing, but part of the point is that we need to make sure the roots
       stay alive until we are done forwarding.  Once you have a "GcArcForward<'a, T>" you should be able to turn it
       into an owned GcArc<T> by combining it with Dead<'a> (which says that all forwarding references for Gc<'a, T> are
       inaccessible, meaning they aren't holding onto forwarding references with lifetime Gc<'a> any longer).  The only
       reason not to do things like this is that it might make it hard to treat values in a single vector of GcArc<T>s
       uniformly if you want them available outside the scope of 'a; you presumably can't just reinterpret them as being
       forwarded at 'a.  Then again, maybe you can?  If I reinterpret something as &'b mut Vec<GcArcForward<'a, T>>, and
       a panic happens... well, let's think:
       - if I actually swap it for an owned Vec<GcArcForward<'a, T>>, nothing bad will happen because GcArcForward<'a, T>
         doesn't run a destructor.  In general if the Vec<GcArc<T>> outlives 'a, nothing bad can possibly happen because
         *the destructor can only run when 'a is not in scope.*  Additionally, reads for GcArc<T> and GcArcForward<'a, T>
         are (in the immutable case) identical.  The challenge is when the Vec does not outlive 'a.  In that case the
         destructor would be problematic.  So let's provide an API that takes (e.g.) &'b mut Vec<Gc<T>> where 'b : 'a, and
         does this reinterpretation trick to &'a mut Vec<GcArc<'a, T>>.  Can this be generalized?  No idea.  We should
         also be able to do the "reverse" and say that Vec<GcArc<'a, T>> ∗ GcDead<'a> -> Vec<Gc<T>>, since Forward<'a> is
         dead at that point and that's the "actual" lifetime we care about--so this provides a witness basically that the
         forwarding pointer at 'a can't be used by anyone anymore.

         In general it is fine to interpret a write of a &'b mut GcArcForward<'a, T> as a write to a
         &'b mut GcArc<T> because *if the destructor for GcArc runs*, it will be when 'a is not in scope.

         Assigning a lifetime other than 'a would mostly be about making this less irritating to do.  When 'a came into
         scope you would reinterpret Vec<Gc> as...

         ...hold on a sec.

         Is it sufficient to just make Vec<Pin<GcArc<T>>>?  What does it buy us?
         Well, if we insist on only pinned stuff... it means we can rely on the destructor being called.  But the destructor
         being called "correctly" would still be problematic for forwarding pointers.  Once again we need to make sure the
         vector can't get dropped until we're done with 'a.

         &'a Pin<GcArc<T>> -> Pin<&'a GcArcInner<T>>.

         Here we guarantee that the memory storing the reference to GcArc<T> cannot be invalidated.

         What extra advantage does "not invalidated until Drop is called" confer?  Well, none really in this case, since the
         objects are *always* valid, but one can imagine a situation where that wasn't the case.  In this case particularly,
         our new objects are never pointing at stuff in the arena, so even if the arena is allowed to reuse space after 'a
         is done (without dropping) it doesn't really matter.  I guess having a Pin<&'a GcArcInner<T>> would be more useful
         if we let you hold onto references into the arena after the scope expired if we leaked the arena, though...

         Suppose we ask for &'a Pin<GcArc<T>>.  Then we know
         Well, if GcArc<T> is Unpin,
       - conversely, if
       if you can always drop them, then it seems difficult to safely include

       Or even from Forward<'a> to nothing, because once we've turned all the roots into things

       and turn it into a GcForward<'a, 'gc, T>, the understanding
       The only reason we need / want to know what
       'gc is ahead

       The 'gc represents the lifetime during which


       the forwarding pointer.
       we should freeze the stack (which will prevent roots from being dropped). */
    println!("Passed root out of GC context: {}", /*expr_arc_to_string*/expr_to_string(&root2));
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        // super::gc_example();
        {
            /* let x = {
                let var = super::GcArc::new(super::ExprArc::Rel(0));
                let id = super::GcArc::new(super::ExprArc::Abs(var));
                let r2 = /*super::GcArc::new(*/super::ExprArc::Rel(1)/*)*/;

                let id_ = super::GcArc::clone(&id);
                let v = super::ExprArc::App(super::App(id, id_));
            }; */
        }
        {
            let x = super::gc_example();
        }
        assert_eq!(2 + 2, 4);

    }
}
