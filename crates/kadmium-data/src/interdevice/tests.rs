use super::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

fn link(source: Endpoint, destination: Endpoint, rate: f64) -> Link {
    Link {
        source,
        destination,
        method: Method::Buffered,
        bytes_per_second: rate,
        latency_ns: 100,
        alignment: 1,
        max_chunk_bytes: 1024,
    }
}

fn topology() -> Topology {
    Topology {
        version: PROTOCOL_VERSION,
        system: "test-system/adapter-v1".into(),
        links: vec![
            link(Endpoint::Storage(0), Endpoint::Ram(0), 1e9),
            link(Endpoint::Ram(0), Endpoint::Vram(0), 10e9),
            link(Endpoint::Vram(0), Endpoint::Ram(0), 5e9),
            link(Endpoint::Ram(0), Endpoint::Storage(0), 0.5e9),
        ],
        staging: vec![Endpoint::Ram(0)],
    }
}

fn request(bytes: u64) -> Request {
    Request {
        version: PROTOCOL_VERSION,
        source: Span {
            region: RegionId(1),
            offset: 0,
        },
        destination: Span {
            region: RegionId(2),
            offset: 0,
        },
        bytes,
        priority: Priority::Demand,
    }
}

fn region(id: u64, endpoint: Endpoint, bytes: u64) -> Region {
    Region {
        id: RegionId(id),
        endpoint,
        bytes,
        writable: true,
    }
}

fn plan(t: &Topology, r: &Request) -> io::Result<Plan> {
    t.plan(
        r,
        region(1, Endpoint::Storage(0), 16384),
        region(2, Endpoint::Vram(0), 16384),
        8192,
        4096,
    )
}

#[test]
fn route_uses_costs_for_this_direction_and_system() {
    let mut t = topology();
    let mut direct = link(Endpoint::Storage(0), Endpoint::Vram(0), 0.1e9);
    direct.method = Method::DirectStorage;
    t.links.push(direct);
    assert_eq!(plan(&t, &request(8192)).unwrap().legs.len(), 2);
    t.links.last_mut().unwrap().bytes_per_second = 20e9;
    assert_eq!(
        plan(&t, &request(8192)).unwrap().legs[0].method,
        Method::DirectStorage
    );
    // A directed path does not fabricate its reverse, nor a missing peer-GPU path.
    assert!(
        t.plan(
            &request(8),
            region(1, Endpoint::Vram(1), 8),
            region(2, Endpoint::Vram(0), 8),
            8,
            8
        )
        .is_err()
    );
}

#[test]
fn alignment_tails_use_buffered_fallback() {
    let mut t = topology();
    let mut direct = link(Endpoint::Storage(0), Endpoint::Vram(0), 100e9);
    direct.method = Method::DirectStorage;
    direct.alignment = 4096;
    direct.max_chunk_bytes = 8192;
    t.links.push(direct);
    assert_eq!(plan(&t, &request(8192)).unwrap().legs.len(), 1);
    assert_eq!(plan(&t, &request(8193)).unwrap().legs.len(), 2);
    let mut unaligned = request(4096);
    unaligned.source.offset = 1;
    assert_eq!(plan(&t, &unaligned).unwrap().legs.len(), 2);
    t.links.drain(..4);
    assert!(plan(&t, &unaligned).is_err());
}

#[test]
fn malformed_and_overflowing_requests_are_rejected() {
    let t = topology();
    let mut r = request(1);
    r.source.offset = u64::MAX;
    assert!(plan(&t, &r).is_err());
    r = request(1);
    r.version = 2;
    assert!(plan(&t, &r).is_err());
    assert!(
        t.plan(
            &request(1),
            region(1, Endpoint::Storage(0), 1),
            Region {
                writable: false,
                ..region(2, Endpoint::Vram(0), 1)
            },
            1,
            1
        )
        .is_err()
    );
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut malformed = t.clone();
        malformed.links[0].bytes_per_second = rate;
        assert!(plan(&malformed, &request(1)).is_err());
    }
    let mut malformed = t.clone();
    malformed.links[0].alignment = 3;
    assert!(plan(&malformed, &request(1)).is_err());
    assert!(
        t.plan(
            &request(1),
            region(1, Endpoint::Storage(0), 1),
            region(2, Endpoint::Vram(0), 1),
            1,
            0
        )
        .is_err()
    );
}

#[test]
fn zero_length_and_request_serialization() {
    let r = request(0);
    let decoded: Request = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(decoded, r);
    assert!(plan(&topology(), &r).unwrap().legs.is_empty());
}

#[derive(Default)]
struct State {
    buffers: BTreeMap<u64, (Region, Vec<u8>)>,
    next: u64,
    starts: Vec<CopyOp>,
    events: Vec<&'static str>,
    fences: usize,
    leased: BTreeMap<u64, usize>,
    fail_start: Option<usize>,
    fail_poll: Option<usize>,
    fail_allocate: bool,
}

struct TestAllocation {
    state: Rc<RefCell<State>>,
    region: Region,
}
impl Lease for TestAllocation {
    fn region(&self) -> Region {
        self.region
    }
}
impl Drop for TestAllocation {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        assert_eq!(
            state.leased.get(&self.region.id.0).copied().unwrap_or(0),
            0,
            "staging freed while its DMA fence is live"
        );
        state.events.push("free");
        state.buffers.remove(&self.region.id.0);
    }
}

struct TestFence {
    state: Rc<RefCell<State>>,
    operation: CopyOp,
    pending: bool,
    fail: bool,
}
impl Fence for TestFence {
    fn poll(&mut self) -> io::Result<Poll<()>> {
        if self.fail {
            return Err(io::Error::other("injected poll failure"));
        }
        if self.pending {
            self.pending = false;
            return Ok(Poll::Pending);
        }
        let mut state = self.state.borrow_mut();
        let op = self.operation;
        let src = &state.buffers[&op.source.region.0].1;
        let bytes = src[op.source.offset as usize..(op.source.offset + op.bytes) as usize].to_vec();
        let dst = &mut state.buffers.get_mut(&op.destination.region.0).unwrap().1;
        dst[op.destination.offset as usize..(op.destination.offset + op.bytes) as usize]
            .copy_from_slice(&bytes);
        Ok(Poll::Ready(()))
    }
}
impl Drop for TestFence {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        state.events.push("drain");
        state.fences -= 1;
        for id in [
            self.operation.source.region.0,
            self.operation.destination.region.0,
        ] {
            *state.leased.get_mut(&id).unwrap() -= 1;
        }
    }
}

struct TestBackend {
    topology: Topology,
    state: Rc<RefCell<State>>,
}
impl Backend for TestBackend {
    type Allocation = TestAllocation;
    type Completion = TestFence;
    fn topology(&self) -> &Topology {
        &self.topology
    }
    fn describe(&self, id: RegionId) -> io::Result<Region> {
        self.state
            .borrow()
            .buffers
            .get(&id.0)
            .map(|entry| entry.0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "unknown region"))
    }
    fn allocate(&mut self, endpoint: Endpoint, bytes: u64) -> io::Result<TestAllocation> {
        let mut state = self.state.borrow_mut();
        if state.fail_allocate {
            return Err(io::Error::other("injected allocation failure"));
        }
        state.next += 1;
        let region = region(state.next, endpoint, bytes);
        state
            .buffers
            .insert(region.id.0, (region, vec![0; bytes as usize]));
        Ok(TestAllocation {
            state: self.state.clone(),
            region,
        })
    }
    fn start(&mut self, operation: CopyOp) -> io::Result<TestFence> {
        let mut state = self.state.borrow_mut();
        state.starts.push(operation);
        if state.fail_start == Some(state.starts.len()) {
            return Err(io::Error::other("injected start failure"));
        }
        state.fences += 1;
        for id in [operation.source.region.0, operation.destination.region.0] {
            *state.leased.entry(id).or_default() += 1;
        }
        Ok(TestFence {
            state: self.state.clone(),
            operation,
            pending: true,
            fail: state.fail_poll == Some(state.starts.len()),
        })
    }
}

fn fixture() -> (TransferEngine<TestBackend>, Rc<RefCell<State>>) {
    let mut state = State {
        next: 10,
        ..State::default()
    };
    state
        .buffers
        .insert(1, (region(1, Endpoint::Storage(0), 16), (0..16).collect()));
    state
        .buffers
        .insert(2, (region(2, Endpoint::Vram(0), 16), vec![0; 16]));
    let state = Rc::new(RefCell::new(state));
    let limits = Limits {
        max_active: 2,
        max_prefetch_active: 1,
        max_queued: 2,
        chunk_bytes: 4,
        staging_bytes: 8,
    };
    let engine = TransferEngine::new(
        TestBackend {
            topology: topology(),
            state: state.clone(),
        },
        limits,
    )
    .unwrap();
    (engine, state)
}

fn drain<B: Backend>(engine: &mut TransferEngine<B>) -> Vec<Completion> {
    let mut results = Vec::new();
    for _ in 0..1000 {
        results.extend(engine.progress());
        if engine.is_idle() {
            return results;
        }
    }
    panic!("executor failed to make progress");
}

#[test]
fn delayed_dma_holds_staging_through_both_legs_and_all_chunks() {
    let (mut engine, state) = fixture();
    engine.submit(request(15)).unwrap();
    engine.progress(); // start read
    assert_eq!(state.borrow().starts.len(), 1);
    engine.progress(); // read still pending
    assert_eq!(state.borrow().starts.len(), 1);
    engine.progress(); // start upload
    assert_eq!(state.borrow().starts.len(), 2);
    engine.progress(); // upload pending; no reuse for next read
    assert_eq!(state.borrow().starts.len(), 2);
    assert_eq!(engine.stats().staging_bytes, 4);
    let results = drain(&mut engine);
    assert!(matches!(results[0].outcome, Outcome::Complete));
    assert_eq!(results[0].completed_bytes, 15);
    assert_eq!(
        &state.borrow().buffers[&2].1[..15],
        &(0..15).collect::<Vec<_>>()
    );
    assert_eq!(engine.stats().physical_bytes, 30);
    assert_eq!(engine.stats().logical_bytes, 15);
    assert_eq!(engine.stats().operations, 8);
    assert_eq!(engine.stats().peak_staging_bytes, 4);
    assert_eq!(engine.stats().staging_bytes, 0);
}

#[test]
fn failure_and_cancellation_drain_before_releasing_staging() {
    for mode in 0..5 {
        let (mut engine, state) = fixture();
        match mode {
            0 => state.borrow_mut().fail_start = Some(2),
            1 => state.borrow_mut().fail_poll = Some(2),
            2 => state.borrow_mut().fail_allocate = true,
            _ => {}
        }
        let id = engine.submit(request(8)).unwrap();
        if mode < 3 {
            let results = drain(&mut engine);
            assert!(matches!(results[0].outcome, Outcome::Failed(_)));
            assert_eq!(engine.stats().failed, 1);
        } else {
            engine.progress();
            engine.progress();
            engine.progress(); // upload owns staging
            if mode == 3 {
                assert!(matches!(
                    engine.cancel(id).unwrap().outcome,
                    Outcome::Cancelled
                ));
                assert!(engine.cancel(id).is_none());
                assert_eq!(engine.stats().staging_bytes, 0);
            }
        }
        drop(engine);
        let state = state.borrow();
        assert_eq!(state.fences, 0);
        assert_eq!(state.buffers.len(), 2);
        if mode != 2 {
            assert_eq!(state.events.last(), Some(&"free"));
        }
    }
}

#[test]
fn demand_precedes_prefetch_and_conflicting_requests_backpressure() {
    let (mut engine, state) = fixture();
    let mut prefetch = request(4);
    prefetch.priority = Priority::Prefetch;
    let prefetched = engine.submit(prefetch.clone()).unwrap();
    assert_eq!(
        engine.submit(request(4)).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    let mut demand = request(4);
    demand.destination.offset = 8;
    demand.source.offset = 8;
    engine.submit(demand).unwrap();
    assert_eq!(
        engine.submit(request(0)).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    engine.progress();
    assert_eq!(state.borrow().starts[0].source.offset, 8);
    let results = drain(&mut engine);
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|c| c.id == prefetched));
    assert!(engine.stats().peak_staging_bytes <= 8);
}

#[test]
fn in_place_overlap_rejected_but_disjoint_ranges_allowed() {
    let mut r = request(4);
    r.destination.region = r.source.region;
    let region = region(1, Endpoint::Ram(0), 16);
    assert!(plan::validate_request(&r, region, region).is_err());
    r.destination.offset = 8;
    assert!(plan::validate_request(&r, region, region).is_ok());
}

fn host() -> HostBackend {
    HostBackend::new(
        "portable-test",
        HostCosts {
            ram_bytes_per_second: 10e9,
            read_bytes_per_second: 1e9,
            write_bytes_per_second: 0.5e9,
            io_latency_ns: 1000,
        },
    )
    .unwrap()
}

#[test]
fn reverse_device_spill_and_peer_route_preserve_bytes() {
    for peer in [false, true] {
        let (_, state) = fixture();
        let source = Endpoint::Vram(0);
        let destination = if peer {
            Endpoint::Vram(1)
        } else {
            Endpoint::Storage(0)
        };
        state.borrow_mut().buffers.get_mut(&1).unwrap().0.endpoint = source;
        state.borrow_mut().buffers.get_mut(&2).unwrap().0.endpoint = destination;
        let mut t = topology();
        if peer {
            let mut direct = link(source, destination, 20e9);
            direct.method = Method::Peer;
            t.links.push(direct);
        }
        // Use an adapter exposing this device topology.
        let limits = Limits {
            chunk_bytes: 4,
            staging_bytes: 16,
            ..Limits::default()
        };
        let mut engine = TransferEngine::new(
            TestBackend {
                topology: t,
                state: state.clone(),
            },
            limits,
        )
        .unwrap();
        engine.submit(request(13)).unwrap();
        assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Complete));
        assert_eq!(
            &state.borrow().buffers[&2].1[..13],
            &(0..13).collect::<Vec<_>>()
        );
        assert_eq!(engine.stats().physical_bytes, if peer { 13 } else { 26 });
        assert_eq!(engine.stats().peak_staging_bytes, if peer { 0 } else { 4 });
    }
}

#[test]
fn prefetch_reserves_queue_and_memory_for_later_demand() {
    let (mut engine, state) = fixture();
    let mut prefetch = request(8);
    prefetch.priority = Priority::Prefetch;
    engine.submit(prefetch).unwrap();
    let mut other = request(8);
    other.destination.offset = 8;
    other.priority = Priority::Prefetch;
    assert_eq!(
        engine.submit(other.clone()).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    engine.progress();
    // There is only one prefetch slot. Demand still gets a chunk and can start while the
    // speculative source remains leased; both requests share a read-only source range.
    other.priority = Priority::Demand;
    engine.submit(other).unwrap();
    engine.progress();
    assert_eq!(state.borrow().starts.len(), 2);
    assert_eq!(engine.stats().staging_bytes, 8);
    assert_eq!(drain(&mut engine).len(), 2);
    assert_eq!(engine.stats().staging_bytes, 0);
}

#[test]
fn queued_cancel_zero_length_and_failed_retry_leave_no_reservations() {
    let (mut engine, state) = fixture();
    let queued = engine.submit(request(8)).unwrap();
    assert!(matches!(
        engine.cancel(queued).unwrap().outcome,
        Outcome::Cancelled
    ));
    assert!(engine.is_idle());
    assert!(state.borrow().starts.is_empty());
    let empty = engine.submit(request(0)).unwrap();
    let result = drain(&mut engine);
    assert_eq!(result[0].id, empty);
    assert!(matches!(result[0].outcome, Outcome::Complete));
    assert!(state.borrow().starts.is_empty());
    state.borrow_mut().fail_poll = Some(2);
    let failed = engine.submit(request(8)).unwrap();
    assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Failed(_)));
    let retry = engine.submit(request(8)).unwrap();
    assert!(retry.0 > failed.0);
    assert!(engine.cancel(failed).is_none());
    assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Complete));
    assert_eq!(
        &state.borrow().buffers[&2].1[..8],
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn tiny_staging_budget_fails_admission_instead_of_stalling_forever() {
    let (_, state) = fixture();
    let mut t = topology();
    for link in &mut t.links {
        link.alignment = 4;
    }
    let limits = Limits {
        max_active: 2,
        max_prefetch_active: 1,
        max_queued: 2,
        chunk_bytes: 4,
        staging_bytes: 7,
    };
    let mut engine = TransferEngine::new(TestBackend { topology: t, state }, limits).unwrap();
    assert_eq!(
        engine.submit(request(8)).unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert!(engine.is_idle());
}

#[test]
fn ram_self_copy_is_nonoverlapping_and_zero_staging_works() {
    let mut backend = host();
    let ram = backend.register_ram((0..16).collect()).unwrap();
    let r = Request {
        source: Span {
            region: ram.region().id,
            offset: 0,
        },
        destination: Span {
            region: ram.region().id,
            offset: 8,
        },
        ..request(5)
    };
    let mut engine = TransferEngine::new(
        backend,
        Limits {
            staging_bytes: 0,
            ..Limits::default()
        },
    )
    .unwrap();
    engine.submit(r).unwrap();
    assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Complete));
    assert_eq!(&ram.read_bytes().unwrap()[8..13], &[0, 1, 2, 3, 4]);
}

#[test]
fn portable_ram_copy_and_stale_registration() {
    let mut backend = host();
    let source = backend.register_ram((0..16).collect()).unwrap();
    let dest = backend.register_ram(vec![0; 16]).unwrap();
    let stale = backend.register_ram(vec![1]).unwrap();
    let stale_id = stale.region().id;
    drop(stale);
    assert!(backend.describe(stale_id).is_err());
    let r = Request {
        source: Span {
            region: source.region().id,
            offset: 3,
        },
        destination: Span {
            region: dest.region().id,
            offset: 5,
        },
        ..request(7)
    };
    let mut engine = TransferEngine::new(
        backend,
        Limits {
            chunk_bytes: 3,
            ..Limits::default()
        },
    )
    .unwrap();
    engine.submit(r).unwrap();
    assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Complete));
    assert_eq!(&dest.read_bytes().unwrap()[5..12], &[3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(engine.stats().physical_bytes, 7);
    assert_eq!(engine.stats().peak_staging_bytes, 0);
}

#[test]
fn portable_file_ram_file_roundtrip_and_truncation_failure() {
    use std::fs::{File, OpenOptions};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "kadmium-kip-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let input = dir.join("input");
    let output = dir.join("output");
    let bytes: Vec<u8> = (0..10003).map(|i| (i % 251) as u8).collect();
    std::fs::write(&input, &bytes).unwrap();
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&output)
        .unwrap();
    file.set_len(bytes.len() as u64).unwrap();
    let mut backend = host();
    let source = backend
        .register_file(File::open(&input).unwrap(), 0, false)
        .unwrap();
    let dest = backend.register_file(file, 1, true).unwrap();
    let r = Request {
        source: Span {
            region: source.region().id,
            offset: 0,
        },
        destination: Span {
            region: dest.region().id,
            offset: 0,
        },
        ..request(bytes.len() as u64)
    };
    let mut engine = TransferEngine::new(
        backend,
        Limits {
            chunk_bytes: 1024,
            staging_bytes: 4096,
            ..Limits::default()
        },
    )
    .unwrap();
    engine.submit(r.clone()).unwrap();
    assert!(matches!(drain(&mut engine)[0].outcome, Outcome::Complete));
    dest.sync_all().unwrap();
    assert_eq!(std::fs::read(&output).unwrap(), bytes);
    assert_eq!(engine.stats().physical_bytes, 20006);
    assert_eq!(engine.stats().peak_staging_bytes, 1024);
    // A source truncated after registration must fail; no zero padding can masquerade as data.
    OpenOptions::new()
        .write(true)
        .open(&input)
        .unwrap()
        .set_len(0)
        .unwrap();
    engine.submit(r).unwrap();
    let completion = drain(&mut engine).remove(0);
    assert!(
        matches!(completion.outcome, Outcome::Failed(ref e) if e.kind() == io::ErrorKind::UnexpectedEof)
    );
    drop((engine, source, dest));
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
