#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(box_into_raw_non_null)]
#![feature(box_syntax)]
#![feature(cfg_target_has_atomic)]
#![feature(coerce_unsized)]
#![feature(core_intrinsics)]
#![feature(dispatch_from_dyn)]
#![feature(dropck_eyepatch)]
#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
// #![feature(placement_in_syntax)]
#![feature(raw_vec_internals)]
#![feature(receiver_trait)]
#![feature(rustc_private)]
#![feature(specialization)]
#![feature(unsize)]
#![feature(untagged_unions)]

extern crate arena;
// extern crate alloc;
extern crate core;

// mod arena;
pub mod ghost_cell;
#[path = "core.rs"]
mod gc_core;

use arena::TypedArena;
// use typed_arena::{Arena as TypedArena};
// use copy_arena::{Arena, Allocator};
// use light_arena::{self, MemoryArena, Allocator};

use core::ops::{Deref};
use core::convert::From;

use std::cell::{Cell};

pub use gc_core::{
    Gc,
    GcArc, GcArcFwd, GcArcFwdLayout, GcArcLayout,
    GcDead, GcFreeze, GcFwd, GcPod, GcRefInner,
};

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

// pub struct TypedArenaHkt<T>(T);

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
    #[derive(Clone, Copy, Debug)]
    struct Idx(Option</*&'static Idx*/&'static /*ExprArc*/str>);
    /// The Rust guide confirms that repr(C) on enums will have a nice uniform representation,
    /// so it is likely that we can rely on the standard Rust representation in most cases.  In any
    /// case pointer tagging will often be desirable to support interesting representations.  We
    /// are *not* (currently?) guaranteed that the tag values remain the same between two enums
    /// with the same shape (I don't think), but I would hope that it is at least true for
    /// instantiations of the same generic enum, with the generic type T only being replaced with
    /// word-size references to difference (Sized) types!  But if it is not we can always fall back
    /// on the explicit union representation.
    ///
    /// An important note is that it is *not* guaranteed that #[repr(Rust)] enums that wrap repr(C)
    /// enums retain the same tags.  This isn't necessarily the same as them not having a
    /// reliable layout where they are used the same way, but it's still worth noting since it
    /// means we should be careful about transmuting something like ExprVar which actually uses
    /// Rust Options (or other values in that vein).  Perhaps explicit projections really are a
    /// better way to go long-term (but hopefully not since it puts more burden on the optimizer),
    /// *or* we should just bite the bullet and always use unions, eating the disastrous ergonomic
    /// cost.
    ///
    /// See https://github.com/rust-lang/rfcs/blob/master/text/2195-really-tagged-unions.md
    #[repr(C)]
    #[derive(Clone)]
    enum ExprVar<T> {
        Rel(Cell<Idx>),
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
    #[derive(Clone)]
    struct Expr<'gc, 'a>(ExprVar<Gc<'gc, 'a, Expr<'gc, 'a>>>);
    impl<'gc, 'a> Deref for Expr<'gc,'a> { type Target = ExprVar<Gc<'gc, 'a, Expr<'gc, 'a>>>; fn deref(&self) -> &Self::Target { &self.0 } }

    #[repr(C)]
    #[derive(Clone)]
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
        fn match_own<R>(self, f_rel: impl FnOnce(Cell<Idx>) -> R,
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
        fn match_shr<R>(&self, f_rel: impl FnOnce(&Cell<Idx>) -> R,
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
                ExprVar::Rel(rel) => f_rel(rel),
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

    fn expr_to_string<T>(/*fwd: &GcFwd<'gc>, */expr: /*&*//*Expr<'gc, 'a>*/&ExprVar<T>) -> String
        where T: Deref, T::Target: Deref<Target=ExprVar<T>>, /*T::Target: AsRef<ExprVar<T>>*//*: Borrow<<Target=ExprVar<T>>*/
    {
        /*match expr {
            Expr::Rel(idx) => idx.to_string(),
            Expr::Abs(body) => format!("(λ. {:?})", expr_to_string(fwd, *body)),
            Expr::App(App(fun, arg)) => format!("({:?} {:?})", expr_to_string(fwd, *fun), expr_to_string(fwd, *arg)),
        }*/
        expr.match_shr(
            |idx| format!("{:?}", idx.get()),
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
    fn trace_root<'a, 'gc>(fwd: &GcFwd<'gc>, expr: &Expr<'gc, 'a>) -> ExprArcFwd<'gc> {
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
            |idx| ExprVar::Rel(idx.clone()),
            |body| ExprVar::Abs(
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/body.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, &*body)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ }),
            |App(fun, arg)| ExprVar::App(App(
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/fun.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, &*fun)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ },
                /*OwnFwd */{ /*old: *//*unsafe { mem::transmute(*/arg.try_as_arc(&fwd).unwrap_or_else(|expr_ref| {
                    let expr_new = GcArcFwd::new(/*mem::transmute(*/trace_root(fwd, &*arg)/*)*/);
                    expr_ref.set_forwarding(&expr_new);
                    expr_new
                })/*) }*/ })),
        ))
    }

pub fn gc_example() {
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

    // let mut old_stack = Vec::<ExprArc>::new();
    let root = GcFwd::new/*::<ExprHkt, _, _>*/(/*|_| (TypedArena::default()), */move |/*my_arena, */fwd| {
        let my_arena: TypedArena::<GcRefInner<Expr>> = TypedArena::default()/*with_capacity(1000)*/;

        // Mutator part.
        // let mut new_stack = Vec::<ExprRef>::new();
        let var = (&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Rel(Cell::new(/*Some("foo")*/Idx(None))))))).into();
        let id = (&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Abs(var))))).into();
        // let r2 = Gc::from(&*my_arena.alloc(GcRefInner::new(Expr(ExprVar::Rel(1)))));

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
        let root = trace_root(&fwd, &Expr(v));
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
        //
        // NOTE: a way around this is to only log references that we actually have to share or
        // can't reconstruct otherwise.  So for instance, instead of tracing from Expr, we could
        // choose to only trace from id.  This would let us still construct root without copies,
        // and we would skip the need for bumping the reference counts for id until we actually ran
        // the arena.  Again though we would have extra bumps.  It seems really hard to avoid this
        // unless you can somehow prove to Rust that you *will* be tracing something in the
        // future--and if you can do that, you can just GcArc it from the getgo, right?
        //
        // NOTE: There is an approach where we decrement ahead of time, but even then we only save
        // a single atomic decrement (replacing it with a comparison).  Not clear how beneficial
        // that is to begin with, but it might be good in some cases.  Anyway it doesn't work with
        // our current system.
        let my_arena: TypedArena::<GcRefInner<Expr>> = TypedArena::default()/*with_capacity(1000)*/;
        let root = GcRefInner::new(Expr(match &root.0 {
            ExprVar::Rel(rel) => ExprVar::Rel(rel.clone()),
            ExprVar::Abs(ref abs) => ExprVar::Abs(Gc::from(abs)),
            ExprVar::App(App(ref fun, ref arg)) => ExprVar::App(App(Gc::from(fun), Gc::from(arg))),
        }));

        let root = Gc::from(&*my_arena.alloc(root)); // GcRefInner::new(Gc::from(&**root)))); */
        // let root = Gc::from(root);
        // let r2 = ExprVar::Rel(1);
        let v = /*Gc::from(&*my_arena.alloc(GcRefInner::new(Expr(*/ExprVar::App(App(root, root))/*))))*/;

        println!("{}", expr_to_string(/*&fwd, */&v));

        let root = trace_root(&fwd, &Expr(v/*r2*/));
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

    /* #[bench]
    fn large_garbage() {
        for
        let root = GcArena::new();
    } */
}
