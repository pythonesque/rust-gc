// use core::any::Any;
use core::sync::atomic;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use core::borrow;
use core::fmt;
use core::cmp::{/*self, */Ordering};
use core::intrinsics::abort;
use core::mem::{self, align_of_val, /*size_of_val*/};
use core::ops::{Deref, Receiver, /*CoerceUnsized, DispatchFromDyn*/};
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::marker::{Unpin, /*Unsize, */PhantomData};
use core::hash::{Hash, Hasher};
use core::{isize, usize};
use core::convert::From;
// use core::slice::from_raw_parts_mut;

use std::alloc::{Global, Alloc, Layout, /*box_free, *//*handle_alloc_error*/};
use std::boxed::Box;
// use std::rc::is_dangling;
// use std::string::String;
// use std::vec::Vec;

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
    /* unsafe fn allocate_for_ptr(ptr: *const T) -> *mut GcArcInner<T> {
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
    } */

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
