//! The host memory envelope a worker's caches live under, after InferSSD's process arbiter
//! and ledger (`engine/ggml/src/ggml-ssd/host/`).
//!
//! A worker holds memory the owner never sees in a planner's offer: the expert cache, the
//! delta-link chunk stores (128 MiB per connection and direction), activation blobs in flight,
//! a resident `.ktb` bank. InferSSD's rule is that every full new allocation asks first, that an
//! allocation which does not fit first takes room back from the caches, and that the envelope
//! keeps a reserve the caches never eat into. The ledger has two terms on every platform:
//!
//! - the **envelope**: a finite limit on this process's memory (an explicit budget, or the
//!   process's use plus the memory available when the ledger was made) less a 512 MiB reserve,
//!   against the bytes the ledger has granted;
//! - the **system term**: the memory the OS reports available right now, less a floor (the
//!   reserve plus a learned margin), so a worker beside a game or a browser gives way.
//!
//! An admission fits when neither term is short. A [`AllocationTicket`] debits the pending bytes
//! until the allocation is committed (first touch done) or cancelled. The margin grows, never
//! shrinks, to the memory an interval saw vanish that nothing tracked (driver growth, page
//! tables, another process): InferSSD's `end_interval` rule.
//!
//! Nothing here allocates. The probe reads `/proc/meminfo` on Linux, `GlobalMemoryStatusEx` on
//! Windows and `vm_stat` on macOS; everywhere else the system term is unreadable and only the
//! envelope applies.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks a shared ledger, recovering it if a panic poisoned the lock: the ledger's counters are
/// updated one at a time, so a poisoned ledger is still consistent.
pub fn lock_host_ledger(ledger: &Mutex<HostMemoryLedger>) -> MutexGuard<'_, HostMemoryLedger> {
    ledger.lock().unwrap_or_else(PoisonError::into_inner)
}
/// Memory the caches never take: InferSSD's cgroup reserve, kept on every platform.
pub const HOST_RESERVE_BYTES: u64 = 512 << 20;

/// What the OS reports about memory, as the ledger reads it.
pub trait HostMemoryProbe: Send {
    /// Memory available to this process right now without swapping, if readable.
    fn available_bytes(&self) -> Option<u64>;
    /// This process's resident memory, if readable.
    fn process_bytes(&self) -> Option<u64>;
}

/// The platform probe.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemMemoryProbe;

impl HostMemoryProbe for SystemMemoryProbe {
    fn available_bytes(&self) -> Option<u64> {
        platform::available_bytes()
    }

    fn process_bytes(&self) -> Option<u64> {
        platform::process_bytes()
    }
}

/// A probe with fixed answers, for tests and for a worker that reports its own figures.
#[derive(Debug, Clone, Copy, Default)]
pub struct FixedMemoryProbe {
    pub available: Option<u64>,
    pub process: Option<u64>,
}

impl HostMemoryProbe for FixedMemoryProbe {
    fn available_bytes(&self) -> Option<u64> {
        self.available
    }

    fn process_bytes(&self) -> Option<u64> {
        self.process
    }
}

/// One measurement of both terms for an admission of `extra` bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HostRoom {
    pub envelope_bytes: u64,
    /// Granted bytes: committed, pending and external.
    pub known_bytes: u64,
    pub envelope_shortage: u64,
    pub envelope_surplus: u64,
    /// `None` when the platform probe cannot read it; the term then never limits.
    pub system_available: Option<u64>,
    pub floor: u64,
    pub system_shortage: u64,
    pub system_surplus: u64,
}

impl HostRoom {
    pub fn fits(&self) -> bool {
        self.envelope_shortage == 0 && self.system_shortage == 0
    }

    pub fn shortage(&self) -> u64 {
        self.envelope_shortage.max(self.system_shortage)
    }

    /// Room both terms agree on.
    pub fn surplus(&self) -> u64 {
        match self.system_available {
            Some(_) => self.envelope_surplus.min(self.system_surplus),
            None => self.envelope_surplus,
        }
    }
}

/// A granted allocation whose bytes are pending until committed or cancelled. Dropping a ticket
/// without either is reported as a leak in the ledger's statistics, so a caller that forgets
/// shows up.
#[derive(Debug)]
#[must_use = "commit or cancel the ticket once the allocation is done"]
pub struct AllocationTicket {
    pub id: u64,
    pub bytes: u64,
    pub observation: HostRoom,
}

/// A step that gives memory back when an admission is short: a cache's reclaim. Asked for a
/// number of bytes, it returns how many it freed; those bytes leave the ledger's held total,
/// so the cache must have taken them under a ticket it committed.
pub type Reclaimer = Box<dyn FnMut(u64) -> u64 + Send>;

/// Counters an operator reads.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HostLedgerStats {
    pub grants: u64,
    pub refusals: u64,
    pub reclaim_runs: u64,
    pub reclaimed_bytes: u64,
    pub margin_updates: u64,
    pub interval_overlaps: u64,
    pub leaked_tickets: u64,
}

/// The envelope, its terms and its reclaim chain.
pub struct HostMemoryLedger {
    envelope_bytes: u64,
    reserve_bytes: u64,
    margin_bytes: u64,
    held_bytes: u64,
    pending: BTreeMap<u64, u64>,
    external_bytes: u64,
    next_ticket: u64,
    probe: Box<dyn HostMemoryProbe>,
    reclaimers: Vec<(String, Reclaimer)>,
    stats: HostLedgerStats,
    interval: Option<(u64, u64)>,
    interval_overlap: bool,
}

impl std::fmt::Debug for HostMemoryLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMemoryLedger")
            .field("envelope_bytes", &self.envelope_bytes)
            .field("reserve_bytes", &self.reserve_bytes)
            .field("margin_bytes", &self.margin_bytes)
            .field("held_bytes", &self.held_bytes)
            .field("pending", &self.pending.len())
            .field("external_bytes", &self.external_bytes)
            .field("reclaimers", &self.reclaimers.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl HostMemoryLedger {
    /// An envelope of exactly `envelope_bytes` (a job limit, a cgroup `memory.max`, an owner's
    /// budget) with the default reserve.
    pub fn with_envelope(envelope_bytes: u64, probe: impl HostMemoryProbe + 'static) -> Self {
        Self {
            envelope_bytes,
            reserve_bytes: HOST_RESERVE_BYTES,
            margin_bytes: 0,
            held_bytes: 0,
            pending: BTreeMap::new(),
            external_bytes: 0,
            next_ticket: 1,
            probe: Box::new(probe),
            reclaimers: Vec::new(),
            stats: HostLedgerStats::default(),
            interval: None,
            interval_overlap: false,
        }
    }

    /// The envelope InferSSD derives when no limit is given: the process's use plus the memory
    /// available now, fixed at creation. With an unreadable probe the envelope is `fallback`.
    pub fn from_probe(probe: impl HostMemoryProbe + 'static, fallback_bytes: u64) -> Self {
        let envelope = match (probe.process_bytes(), probe.available_bytes()) {
            (Some(process), Some(available)) => process.saturating_add(available),
            (None, Some(available)) => available,
            _ => fallback_bytes,
        };
        Self::with_envelope(envelope, probe)
    }

    /// The platform probe, an envelope from it, and `fallback_bytes` where the platform cannot
    /// be read.
    pub fn for_this_host(fallback_bytes: u64) -> Self {
        Self::from_probe(SystemMemoryProbe, fallback_bytes)
    }

    pub fn with_reserve(mut self, reserve_bytes: u64) -> Self {
        self.reserve_bytes = reserve_bytes;
        self
    }

    /// A starting margin (InferSSD carries the largest learned margin forward between runs).
    pub fn with_margin(mut self, margin_bytes: u64) -> Self {
        self.margin_bytes = self.margin_bytes.max(margin_bytes);
        self
    }

    /// Adds a reclaim step; steps run in registration order, each asked for what is still short.
    pub fn register_reclaimer(&mut self, name: &str, reclaimer: Reclaimer) {
        self.reclaimers.push((name.to_string(), reclaimer));
    }

    pub fn reclaimer_names(&self) -> Vec<String> {
        self.reclaimers.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn envelope_bytes(&self) -> u64 {
        self.envelope_bytes
    }

    pub fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }

    pub fn margin_bytes(&self) -> u64 {
        self.margin_bytes
    }

    pub fn held_bytes(&self) -> u64 {
        self.held_bytes
    }

    pub fn pending_bytes(&self) -> u64 {
        self.pending.values().sum()
    }

    pub fn external_bytes(&self) -> u64 {
        self.external_bytes
    }

    pub fn stats(&self) -> HostLedgerStats {
        self.stats
    }

    fn known_bytes(&self) -> u64 {
        self.held_bytes
            .saturating_add(self.pending_bytes())
            .saturating_add(self.external_bytes)
    }

    /// The floor the system term keeps free: the reserve plus the learned margin.
    pub fn floor_bytes(&self) -> u64 {
        self.reserve_bytes.saturating_add(self.margin_bytes)
    }

    /// Measures both terms for an admission of `extra` bytes, without changing anything.
    pub fn measure(&self, extra: u64) -> HostRoom {
        let known = self.known_bytes();
        let required = self.reserve_bytes.saturating_add(extra);
        let envelope_room = self.envelope_bytes.saturating_sub(known);
        let (envelope_shortage, envelope_surplus) = if envelope_room >= required {
            (0, envelope_room - required)
        } else {
            (required - envelope_room, 0)
        };
        let floor = self.floor_bytes();
        let need = floor.saturating_add(extra);
        let system_available = self.probe.available_bytes();
        let (system_shortage, system_surplus) = match system_available {
            Some(available) if available >= need => (0, available - need),
            Some(available) => (need - available, 0),
            None => (0, 0),
        };
        HostRoom {
            envelope_bytes: self.envelope_bytes,
            known_bytes: known,
            envelope_shortage,
            envelope_surplus,
            system_available,
            floor,
            system_shortage,
            system_surplus,
        }
    }

    /// Room for a new allocation right now, in bytes: what a heartbeat reports as free.
    pub fn room_bytes(&self) -> u64 {
        self.measure(0).surplus()
    }

    /// Asks for `bytes`. When short, the reclaim chain runs for the shortage and the terms are
    /// measured again; a refusal returns the last measurement. A grant debits the bytes as
    /// pending until the ticket is committed or cancelled.
    pub fn take(&mut self, bytes: u64) -> Result<AllocationTicket, HostRoom> {
        let mut room = self.measure(bytes);
        if !room.fits() && !self.reclaimers.is_empty() {
            let mut short = room.shortage();
            self.stats.reclaim_runs += 1;
            for (_, reclaimer) in self.reclaimers.iter_mut() {
                if short == 0 {
                    break;
                }
                let freed = reclaimer(short);
                // What a reclaimer gives back was held under a committed ticket.
                self.held_bytes = self.held_bytes.saturating_sub(freed);
                self.stats.reclaimed_bytes = self.stats.reclaimed_bytes.saturating_add(freed);
                short = short.saturating_sub(freed);
            }
            room = self.measure(bytes);
        }
        if !room.fits() {
            self.stats.refusals += 1;
            return Err(room);
        }
        let id = self.next_ticket;
        self.next_ticket += 1;
        self.pending.insert(id, bytes);
        self.interval_overlap = true;
        self.stats.grants += 1;
        Ok(AllocationTicket {
            id,
            bytes,
            observation: room,
        })
    }

    /// The allocation is made and touched: its bytes move from pending to held.
    pub fn commit(&mut self, ticket: AllocationTicket) {
        if self.pending.remove(&ticket.id).is_some() {
            self.held_bytes = self.held_bytes.saturating_add(ticket.bytes);
        }
    }

    /// The allocation was not made after all.
    pub fn cancel(&mut self, ticket: AllocationTicket) {
        self.pending.remove(&ticket.id);
    }

    /// Held bytes freed by their owner.
    pub fn release(&mut self, bytes: u64) {
        self.held_bytes = self.held_bytes.saturating_sub(bytes);
    }

    /// Memory the process holds outside the ledger's tickets (a compute buffer another layer
    /// owns, a checkpoint being written), published as a total.
    pub fn set_external_bytes(&mut self, bytes: u64) {
        self.external_bytes = bytes;
        self.interval_overlap = true;
    }

    /// Tickets granted but neither committed nor cancelled.
    pub fn open_tickets(&self) -> usize {
        self.pending.len()
    }

    /// Opens a margin-learning interval: a snapshot of what the OS says is available and what
    /// the ledger knows is granted. A grant or an external change before `end_interval` makes
    /// the interval unusable (the pending bytes are not an observation of resident memory).
    pub fn begin_interval(&mut self) {
        self.interval = self
            .probe
            .available_bytes()
            .map(|available| (available, self.known_bytes()));
        self.interval_overlap = !self.pending.is_empty();
    }

    /// Closes the interval. If memory vanished that nothing tracked, the margin rises to that
    /// amount; it never falls. Returns the untracked bytes when the margin moved.
    pub fn end_interval(&mut self) -> Option<u64> {
        let (available_start, known_start) = self.interval.take()?;
        if self.interval_overlap || !self.pending.is_empty() {
            self.stats.interval_overlaps += 1;
            self.interval_overlap = false;
            return None;
        }
        let available_end = self.probe.available_bytes()?;
        let lhs = available_start.saturating_add(known_start);
        let rhs = available_end.saturating_add(self.known_bytes());
        let untracked = lhs.saturating_sub(rhs);
        if untracked <= self.margin_bytes {
            return None;
        }
        self.margin_bytes = untracked;
        self.stats.margin_updates += 1;
        Some(untracked)
    }

    /// A ticket that was dropped without commit or cancel, reported by the owner's wrapper.
    pub fn note_leaked_ticket(&mut self, id: u64) {
        if self.pending.remove(&id).is_some() {
            self.stats.leaked_tickets += 1;
        }
    }
}

mod platform {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn available_bytes() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn process_bytes() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }

    #[cfg(target_os = "windows")]
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[cfg(target_os = "windows")]
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[cfg(target_os = "windows")]
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    /// The smaller of available physical memory and available commit, as InferSSD's Windows
    /// ledger reads it.
    #[cfg(target_os = "windows")]
    pub fn available_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the struct is the documented MEMORYSTATUSEX layout with its length set.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then(|| status.avail_phys.min(status.avail_page_file))
    }

    #[cfg(target_os = "windows")]
    pub fn process_bytes() -> Option<u64> {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        // SAFETY: the pseudo handle of the current process, and the documented
        // PROCESS_MEMORY_COUNTERS layout with its size set.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (ok != 0).then_some(counters.pagefile_usage.max(counters.working_set_size) as u64)
    }

    /// Free, inactive, speculative and purgeable pages, as `vm_stat` reports them.
    #[cfg(target_os = "macos")]
    pub fn available_bytes() -> Option<u64> {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut page_size = 16_384u64;
        let mut pages = 0u64;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
                && let Some(num) = rest.split_whitespace().next()
                && let Ok(size) = num.parse::<u64>()
            {
                page_size = size;
            }
            for key in [
                "Pages free:",
                "Pages inactive:",
                "Pages speculative:",
                "Pages purgeable:",
            ] {
                if let Some(rest) = line.strip_prefix(key)
                    && let Ok(count) = rest.trim().trim_end_matches('.').parse::<u64>()
                {
                    pages = pages.saturating_add(count);
                }
            }
        }
        Some(pages.saturating_mul(page_size))
    }

    #[cfg(target_os = "macos")]
    pub fn process_bytes() -> Option<u64> {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(kb.saturating_mul(1024))
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "macos"
    )))]
    pub fn available_bytes() -> Option<u64> {
        None
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "macos"
    )))]
    pub fn process_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    const MIB: u64 = 1 << 20;

    #[test]
    fn envelope_keeps_the_reserve_and_tickets_debit_until_committed() {
        let mut ledger = HostMemoryLedger::with_envelope(
            1024 * MIB,
            FixedMemoryProbe {
                available: None,
                process: None,
            },
        );
        // 1024 MiB less the 512 MiB reserve: 512 MiB of room.
        assert_eq!(ledger.room_bytes(), 512 * MIB);
        let ticket = ledger.take(300 * MIB).expect("fits");
        assert_eq!(ledger.pending_bytes(), 300 * MIB);
        assert_eq!(ledger.room_bytes(), 212 * MIB);
        let refused = ledger.take(300 * MIB).unwrap_err();
        assert_eq!(refused.envelope_shortage, 88 * MIB);
        assert_eq!(refused.system_available, None);
        ledger.commit(ticket);
        assert_eq!(ledger.held_bytes(), 300 * MIB);
        assert_eq!(ledger.pending_bytes(), 0);
        ledger.release(100 * MIB);
        assert_eq!(ledger.held_bytes(), 200 * MIB);
        let cancelled = ledger.take(10 * MIB).unwrap();
        ledger.cancel(cancelled);
        assert_eq!(ledger.held_bytes(), 200 * MIB);
        assert_eq!(ledger.stats().grants, 2);
        assert_eq!(ledger.stats().refusals, 1);
    }

    #[test]
    fn system_term_limits_under_a_readable_probe_and_the_reclaim_chain_makes_room() {
        let mut ledger = HostMemoryLedger::with_envelope(
            64 * 1024 * MIB,
            FixedMemoryProbe {
                available: Some(700 * MIB),
                process: Some(100 * MIB),
            },
        );
        // Available 700 less the 512 floor: 188 MiB of system room.
        assert_eq!(ledger.room_bytes(), 188 * MIB);
        let freed = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&freed);
        ledger.register_reclaimer(
            "expert-cache",
            Box::new(move |short| {
                log.lock().unwrap().push(short);
                short
            }),
        );
        // The chain is asked for the shortage; the probe does not move (fixed), so the envelope
        // term is satisfied by the reclaim's accounting only when it actually frees held bytes.
        let refused = ledger.take(400 * MIB).unwrap_err();
        assert_eq!(refused.system_shortage, 212 * MIB);
        assert_eq!(freed.lock().unwrap().as_slice(), &[212 * MIB]);
        assert_eq!(ledger.stats().reclaim_runs, 1);
        assert_eq!(ledger.reclaimer_names(), vec!["expert-cache".to_string()]);
    }

    #[test]
    fn reclaim_that_frees_held_bytes_lets_the_envelope_admit() {
        let held = Arc::new(Mutex::new(0u64));
        let mut ledger = HostMemoryLedger::with_envelope(
            1000 * MIB,
            FixedMemoryProbe {
                available: None,
                process: None,
            },
        );
        let t = ledger.take(400 * MIB).unwrap();
        ledger.commit(t);
        *held.lock().unwrap() = 400 * MIB;
        // Room: 1000 - 512 - 400 = 88 MiB. A 200 MiB ask is 112 MiB short.
        let cache = Arc::clone(&held);
        ledger.register_reclaimer(
            "cache",
            Box::new(move |short| {
                let mut h = cache.lock().unwrap();
                let give = short.min(*h);
                *h -= give;
                give
            }),
        );
        // The reclaimer frees 112 MiB of held bytes and the ledger admits in the same call.
        let granted = ledger.take(200 * MIB).unwrap();
        assert_eq!(*held.lock().unwrap(), 288 * MIB);
        assert_eq!(granted.observation.envelope_shortage, 0);
        assert_eq!(ledger.held_bytes(), 288 * MIB);
        ledger.commit(granted);
        assert_eq!(ledger.held_bytes(), 488 * MIB);
        assert_eq!(ledger.stats().reclaimed_bytes, 112 * MIB);
        // More than the envelope can ever hold is refused even after the chain runs dry.
        let refused = ledger.take(900 * MIB).unwrap_err();
        assert!(refused.envelope_shortage > 0);
        assert_eq!(*held.lock().unwrap(), 0);
    }

    #[test]
    fn margin_learns_untracked_growth_and_never_shrinks() {
        let probe = Arc::new(Mutex::new(FixedMemoryProbe {
            available: Some(4000 * MIB),
            process: Some(500 * MIB),
        }));
        struct Shared(Arc<Mutex<FixedMemoryProbe>>);
        impl HostMemoryProbe for Shared {
            fn available_bytes(&self) -> Option<u64> {
                self.0.lock().unwrap().available
            }
            fn process_bytes(&self) -> Option<u64> {
                self.0.lock().unwrap().process
            }
        }
        let mut ledger = HostMemoryLedger::from_probe(Shared(Arc::clone(&probe)), 0);
        assert_eq!(ledger.envelope_bytes(), 4500 * MIB);
        ledger.begin_interval();
        // 300 MiB vanished that nothing tracked.
        probe.lock().unwrap().available = Some(3700 * MIB);
        assert_eq!(ledger.end_interval(), Some(300 * MIB));
        assert_eq!(ledger.margin_bytes(), 300 * MIB);
        assert_eq!(ledger.floor_bytes(), 812 * MIB);
        // Memory coming back does not lower it.
        ledger.begin_interval();
        probe.lock().unwrap().available = Some(4200 * MIB);
        assert_eq!(ledger.end_interval(), None);
        assert_eq!(ledger.margin_bytes(), 300 * MIB);
        // A grant inside the interval skips it.
        ledger.begin_interval();
        let t = ledger.take(MIB).unwrap();
        ledger.commit(t);
        probe.lock().unwrap().available = Some(1000 * MIB);
        assert_eq!(ledger.end_interval(), None);
        assert_eq!(ledger.stats().interval_overlaps, 1);
        assert_eq!(ledger.margin_bytes(), 300 * MIB);
    }

    #[test]
    fn the_platform_probe_snapshot_sets_the_envelope_where_it_reads() {
        let probe = SystemMemoryProbe;
        if let Some(available) = probe.available_bytes() {
            assert!(available > 0);
            // Reuse these readings: a second live probe can observe less available memory.
            let snapshot = FixedMemoryProbe {
                available: Some(available),
                process: probe.process_bytes(),
            };
            let expected = available.saturating_add(snapshot.process.unwrap_or(0));
            let ledger = HostMemoryLedger::from_probe(snapshot, 0);
            assert_eq!(ledger.envelope_bytes(), expected);
        }
    }
}
