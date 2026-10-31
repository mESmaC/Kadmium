//! Fault injection at the actual driver call boundary, without NPU hardware.
use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct State {
    submit: u32,
    wait: u32,
    init_stages: u32,
    allocations: usize,
    fail_allocation: usize,
    calls: Vec<&'static str>,
}
thread_local! { static STATE: RefCell<State> = RefCell::default(); }
fn record(call: &'static str) {
    STATE.with_borrow_mut(|s| s.calls.push(call));
}
unsafe extern "system" fn submit(_: Handle, _: u32, _: *const Handle, _: Handle) -> Status {
    record("submit");
    STATE.with_borrow(|s| s.submit)
}
unsafe extern "system" fn wait(_: Handle, _: u64) -> Status {
    record("wait");
    STATE.with_borrow(|s| s.wait)
}
unsafe extern "system" fn destroy_list(_: Handle) -> Status {
    record("list");
    0
}
unsafe extern "system" fn destroy_queue(_: Handle) -> Status {
    record("queue");
    0
}
unsafe extern "system" fn destroy_graph(_: Handle) -> Status {
    record("graph");
    0
}
unsafe extern "system" fn destroy_context(_: Handle) -> Status {
    record("context");
    0
}
unsafe extern "system" fn free(_: Handle, _: Handle) -> Status {
    record("free");
    0
}

fn kernel() -> IntelMatMul {
    STATE.with_borrow_mut(|s| *s = State::default());
    let mut api = Api::mock();
    api.queue_execute = submit;
    api.queue_sync = wait;
    api.list_destroy = destroy_list;
    api.queue_destroy = destroy_queue;
    api.context_destroy = destroy_context;
    api.free = free;
    let graph = GraphApi {
        create: None,
        destroy: Some(destroy_graph),
        properties: None,
        argument: None,
        set_argument: None,
        initialize: None,
        execute: None,
        native_binary: None,
        device_properties: None,
    };
    assert!(graph.validate().is_err());
    IntelMatMul {
        context: Rc::new(Context {
            api,
            graph,
            build_log: None,
            initialization: None,
            device: ptr::without_provenance_mut(1),
            handle: ptr::without_provenance_mut(2),
            ordinal: 0,
            info: DeviceInfo {
                name: "fault injection".into(),
                vendor_id: 0x8086,
                device_id: 0,
                graph_extension_version: 0x10000,
                compiler_version: (0, 0),
                arithmetic: Arithmetic::ApproximateFp16,
            },
        }),
        shape: MatMulShape::new(1, 1, 1).unwrap(),
        graph: ptr::without_provenance_mut(3),
        queue: ptr::without_provenance_mut(4),
        list: ptr::without_provenance_mut(5),
        input: ptr::without_provenance_mut(6),
        output: ptr::without_provenance_mut(7),
        payload: Vec::new(),
        in_flight: false,
        poisoned: false,
    }
}

#[test]
fn completed_work_releases_resources_in_dependency_order() {
    let mut k = kernel();
    k.run_closed_list().unwrap();
    assert!(!k.in_flight && !k.poisoned);
    drop(k);
    STATE.with_borrow(|s| {
        assert_eq!(
            s.calls,
            [
                "submit", "wait", "list", "queue", "graph", "free", "free", "context"
            ]
        )
    });
}

#[test]
fn failed_submission_poisons_and_waits_before_cleanup() {
    let mut k = kernel();
    STATE.with_borrow_mut(|s| s.submit = 0x70000001);
    assert!(k.run_closed_list().is_err());
    assert!(k.in_flight && k.poisoned);
    assert!(matches!(k.execute(&[1.]), Err(Error::Poisoned)));
    drop(k);
    STATE.with_borrow(|s| {
        assert_eq!(
            s.calls,
            [
                "submit", "wait", "list", "queue", "graph", "free", "free", "context"
            ]
        )
    });
}

#[test]
fn timeout_never_reuses_or_frees_potentially_live_device_memory() {
    let mut k = kernel();
    let context = Rc::clone(&k.context);
    STATE.with_borrow_mut(|s| s.wait = 1); // ZE_RESULT_NOT_READY
    assert!(k.run_closed_list().is_err());
    assert!(matches!(k.execute(&[1.]), Err(Error::Poisoned)));
    drop(k);
    STATE.with_borrow(|s| assert_eq!(s.calls, ["submit", "wait", "wait"]));
    assert_eq!(Rc::strong_count(&context), 2); // observer + quarantined loader
    // SAFETY: fake handles have no device work. Reclaim the one deliberately
    // forgotten reference to keep this test from leaking its mock context.
    unsafe {
        Rc::decrement_strong_count(Rc::as_ptr(&context));
    }
}

#[test]
fn partial_preparation_only_destroys_acquired_resources() {
    let mut k = kernel();
    k.queue = ptr::null_mut();
    k.list = ptr::null_mut();
    k.input = ptr::null_mut();
    k.output = ptr::null_mut();
    drop(k);
    STATE.with_borrow(|s| assert_eq!(s.calls, ["graph", "context"]));
}

unsafe extern "system" fn create_graph(
    _: Handle,
    _: Handle,
    _: *const GraphDesc,
    out: *mut Handle,
) -> Status {
    // SAFETY: test invokes these mocks with the live out-pointers from prepare.
    unsafe {
        *out = ptr::without_provenance_mut(3);
    }
    0
}
unsafe extern "system" fn properties(_: Handle, out: *mut GraphProperties) -> Status {
    unsafe {
        (*out).count = 2;
    }
    0
}
unsafe extern "system" fn properties2(_: Handle, out: *mut GraphProperties2) -> Status {
    unsafe {
        (*out).count = 2;
        (*out).init_stages = STATE.with_borrow(|s| s.init_stages);
    }
    0
}
unsafe extern "system" fn argument(_: Handle, index: u32, out: *mut ArgumentProperties) -> Status {
    unsafe {
        (*out).kind = index;
        (*out).dims = [1, 1, 0, 0, 0];
        (*out).device_precision = 1;
        (*out).device_layout = 0xc1;
    }
    0
}
unsafe extern "system" fn set_argument(_: Handle, _: u32, _: *const std::ffi::c_void) -> Status {
    0
}
unsafe extern "system" fn host_init(_: Handle) -> Status {
    record("host init");
    0
}
unsafe extern "system" fn command_init(
    _: Handle,
    _: Handle,
    _: Handle,
    _: u32,
    _: *const Handle,
) -> Status {
    record("command init");
    0
}
unsafe extern "system" fn execute(
    _: Handle,
    _: Handle,
    _: Handle,
    _: Handle,
    _: u32,
    _: *const Handle,
) -> Status {
    record("record execute");
    0
}
unsafe extern "system" fn queue_create(
    _: Handle,
    _: Handle,
    _: *const QueueDesc,
    out: *mut Handle,
) -> Status {
    unsafe {
        *out = ptr::without_provenance_mut(4);
    }
    0
}
unsafe extern "system" fn list_create(
    _: Handle,
    _: Handle,
    _: *const ListDesc,
    out: *mut Handle,
) -> Status {
    unsafe {
        *out = ptr::without_provenance_mut(5);
    }
    0
}
unsafe extern "system" fn close(_: Handle) -> Status {
    record("close");
    0
}
unsafe extern "system" fn reset(_: Handle) -> Status {
    record("reset");
    0
}
unsafe extern "system" fn alloc(
    _: Handle,
    _: *const FlagsDesc,
    size: usize,
    _: usize,
    out: *mut Handle,
) -> Status {
    let fail = STATE.with_borrow_mut(|s| {
        s.allocations += 1;
        s.allocations == s.fail_allocation
    });
    if fail || size != 4 {
        return 0x70000002;
    }
    // This fixture has a scalar shape, so one initialized f32 is sufficient.
    unsafe {
        *out = Box::into_raw(Box::new(0f32)).cast();
    }
    0
}
unsafe extern "system" fn free_alloc(_: Handle, allocation: Handle) -> Status {
    record("free");
    // SAFETY: exactly the scalar allocation returned by alloc, freed once.
    unsafe {
        drop(Box::from_raw(allocation.cast::<f32>()));
    }
    0
}

fn preparable_kernel(stages: u32) -> IntelMatMul {
    let mut k = kernel();
    k.graph = ptr::null_mut();
    k.queue = ptr::null_mut();
    k.list = ptr::null_mut();
    k.input = ptr::null_mut();
    k.output = ptr::null_mut();
    STATE.with_borrow_mut(|s| s.init_stages = stages);
    let c = Rc::get_mut(&mut k.context).unwrap();
    c.graph.create = Some(create_graph);
    c.graph.properties = Some(properties);
    c.graph.argument = Some(argument);
    c.graph.set_argument = Some(set_argument);
    c.graph.initialize = Some(command_init);
    c.graph.execute = Some(execute);
    c.initialization = Some((properties2, host_init));
    c.api.queue_create = queue_create;
    c.api.list_create = list_create;
    c.api.list_close = close;
    c.api.list_reset = reset;
    c.api.alloc_host = alloc;
    c.api.free = free_alloc;
    k
}

#[test]
fn honors_all_driver_reported_initialization_stages() {
    for stages in 0..=3 {
        let mut k = preparable_kernel(stages);
        k.prepare().unwrap();
        STATE.with_borrow(|s| {
            assert_eq!(s.calls.contains(&"host init"), stages & 2 != 0);
            assert_eq!(s.calls.contains(&"command init"), stages & 1 != 0);
            assert_eq!(s.calls.contains(&"submit"), stages & 1 != 0);
            assert_eq!(s.calls.contains(&"reset"), stages & 1 != 0);
            assert!(s.calls.ends_with(&["record execute", "close"]));
        });
        drop(k);
    }
    let mut legacy = preparable_kernel(0);
    Rc::get_mut(&mut legacy.context).unwrap().initialization = None;
    legacy.prepare().unwrap();
    STATE.with_borrow(|s| {
        assert!(s.calls.contains(&"command init") && !s.calls.contains(&"host init"))
    });
}

#[test]
fn unsupported_init_stage_fails_before_allocating() {
    let mut k = preparable_kernel(4);
    assert!(matches!(k.prepare(), Err(Error::Unsupported(_))));
    drop(k);
    STATE.with_borrow(|s| {
        assert_eq!(s.allocations, 0);
        assert_eq!(s.calls, ["graph", "context"]);
    });
}

#[test]
fn failed_second_allocation_releases_first_allocation_and_graph() {
    let mut k = preparable_kernel(0);
    STATE.with_borrow_mut(|s| s.fail_allocation = 2);
    assert!(k.prepare().is_err());
    drop(k);
    STATE.with_borrow(|s| assert_eq!(s.calls, ["graph", "free", "context"]));
}
