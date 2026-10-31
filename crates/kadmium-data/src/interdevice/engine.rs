use super::*;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_active: usize,
    pub max_queued: usize,
    /// Speculation cannot occupy all execution slots. Zero disables prefetch admission.
    pub max_prefetch_active: usize,
    pub chunk_bytes: u64,
    /// Includes all active staging allocations, retained through the second copy's fence.
    /// Caller-owned source/destination memory and adapter metadata are separate.
    pub staging_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_active: 4,
            max_queued: 128,
            max_prefetch_active: 2,
            chunk_bytes: 4 << 20,
            staging_bytes: 16 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferId(pub u64);

#[derive(Debug)]
pub enum Outcome {
    Complete,
    Cancelled,
    Failed(io::Error),
}

#[derive(Debug)]
pub struct Completion {
    pub id: TransferId,
    /// Whole chunks whose final leg completed. Failed/cancelled destinations may additionally
    /// contain a partially written chunk and must not be published as a valid tensor.
    pub completed_bytes: u64,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferStats {
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub logical_bytes: u64,
    /// Includes each successful hop through RAM, so staging traffic is visible.
    pub physical_bytes: u64,
    pub operations: u64,
    pub staging_bytes: u64,
    pub peak_staging_bytes: u64,
}

struct Queued {
    id: TransferId,
    request: Request,
    plan: Plan,
}

struct Active<B: Backend> {
    // Field order matters: drain the fence before returning its staging memory.
    fence: Option<B::Completion>,
    staging: Option<B::Allocation>,
    queued: Queued,
    done: u64,
    leg: usize,
}

impl<B: Backend> Active<B> {
    fn chunk(&self) -> u64 {
        self.queued
            .plan
            .chunk_bytes
            .min(self.queued.request.bytes - self.done)
    }

    fn operation(&self) -> CopyOp {
        let request = &self.queued.request;
        let source = Span {
            region: request.source.region,
            offset: request.source.offset + self.done,
        };
        let destination = Span {
            region: request.destination.region,
            offset: request.destination.offset + self.done,
        };
        let (source, destination) = match &self.staging {
            Some(staging) => {
                let intermediate = Span {
                    region: staging.region().id,
                    offset: 0,
                };
                if self.leg == 0 {
                    (source, intermediate)
                } else {
                    (intermediate, destination)
                }
            }
            None => (source, destination),
        };
        CopyOp {
            source,
            destination,
            bytes: self.chunk(),
            method: self.queued.plan.legs[self.leg].method,
        }
    }
}

/// Bounded demand-first executor. Call `progress` when a device event or host loop wakes;
/// it polls each active fence once and starts at most one operation per active request.
/// Different requests can overlap; a chunk's two legs are strictly ordered. No detached
/// workers, hidden polling thread, unbounded completion history or speculative eviction.
pub struct TransferEngine<B: Backend> {
    active: Vec<Active<B>>,
    queue: VecDeque<Queued>,
    backend: B,
    limits: Limits,
    next_id: u64,
    stats: TransferStats,
}

impl<B: Backend> TransferEngine<B> {
    pub fn new(backend: B, limits: Limits) -> io::Result<Self> {
        backend.topology().validate()?;
        if limits.max_active == 0
            || limits.max_queued == 0
            || limits.chunk_bytes == 0
            || limits.max_prefetch_active >= limits.max_active
        {
            return Err(invalid(
                "invalid transfer limits (reserve at least one demand slot)",
            ));
        }
        Ok(Self {
            active: Vec::new(),
            queue: VecDeque::new(),
            backend,
            limits,
            next_id: 1,
            stats: TransferStats::default(),
        })
    }

    pub fn stats(&self) -> TransferStats {
        self.stats
    }
    pub fn is_idle(&self) -> bool {
        self.active.is_empty() && self.queue.is_empty()
    }

    /// Backpressure is WouldBlock. Regions must remain registered until terminal completion.
    /// Overlapping writes/read-writes are refused until the older request terminates; reads
    /// can share a source. Dependencies are expressed by submitting after completion.
    pub fn submit(&mut self, request: Request) -> io::Result<TransferId> {
        if self.queue.len() >= self.limits.max_queued {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "transfer queue is full",
            ));
        }
        if request.priority == Priority::Prefetch
            && (self.limits.max_prefetch_active == 0
                || self.queue.len() >= self.limits.max_queued - 1)
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "prefetch disabled or remaining queue slot reserved for demand",
            ));
        }
        let source = self.backend.describe(request.source.region)?;
        let destination = self.backend.describe(request.destination.region)?;
        let plan = self.backend.topology().plan(
            &request,
            source,
            destination,
            self.limits.chunk_bytes,
            self.limits.staging_bytes / self.limits.max_active as u64,
        )?;
        for other in self
            .queue
            .iter()
            .chain(self.active.iter().map(|a| &a.queued))
        {
            let old = &other.request;
            if plan::overlaps(request.destination, request.bytes, old.source, old.bytes)
                || plan::overlaps(
                    request.destination,
                    request.bytes,
                    old.destination,
                    old.bytes,
                )
                || plan::overlaps(request.source, request.bytes, old.destination, old.bytes)
            {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "conflicting region is leased",
                ));
            }
        }
        let id = TransferId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| invalid("transfer IDs exhausted"))?;
        self.queue.push_back(Queued { id, request, plan });
        Ok(id)
    }

    fn finish(&mut self, index: usize, outcome: Outcome) -> Completion {
        let active = self.active.remove(index);
        let completion = Completion {
            id: active.queued.id,
            completed_bytes: active.done,
            outcome,
        };
        let reserved = if active.staging.is_some() {
            active.queued.plan.chunk_bytes
        } else {
            0
        };
        // Drain before releasing the allocation, on success, failure and cancellation alike.
        drop(active);
        self.stats.staging_bytes -= reserved;
        self.record(&completion);
        completion
    }

    fn record(&mut self, completion: &Completion) {
        match completion.outcome {
            Outcome::Complete => {
                self.stats.completed = self.stats.completed.saturating_add(1);
                self.stats.logical_bytes = self
                    .stats
                    .logical_bytes
                    .saturating_add(completion.completed_bytes);
            }
            Outcome::Failed(_) => self.stats.failed = self.stats.failed.saturating_add(1),
            Outcome::Cancelled => self.stats.cancelled = self.stats.cancelled.saturating_add(1),
        }
    }

    /// Cancellation drains in-flight DMA before returning. Completed chunks are not rolled
    /// back. Unknown/already terminal IDs return None and cannot cancel a newer transfer.
    pub fn cancel(&mut self, id: TransferId) -> Option<Completion> {
        if let Some(index) = self.queue.iter().position(|q| q.id == id) {
            self.queue.remove(index);
            let completion = Completion {
                id,
                completed_bytes: 0,
                outcome: Outcome::Cancelled,
            };
            self.record(&completion);
            Some(completion)
        } else {
            self.active
                .iter()
                .position(|a| a.queued.id == id)
                .map(|index| self.finish(index, Outcome::Cancelled))
        }
    }

    pub fn progress(&mut self) -> Vec<Completion> {
        let mut completions = Vec::new();
        let mut index = 0;
        while index < self.active.len() {
            let active = &mut self.active[index];
            let result = active.fence.as_mut().map(Fence::poll);
            match result {
                Some(Err(error)) => {
                    completions.push(self.finish(index, Outcome::Failed(error)));
                    continue;
                }
                Some(Ok(Poll::Ready(()))) => {
                    active.fence.take();
                    self.stats.physical_bytes =
                        self.stats.physical_bytes.saturating_add(active.chunk());
                    active.leg += 1;
                    if active.leg == active.queued.plan.legs.len() {
                        active.done += active.chunk();
                        active.leg = 0;
                    }
                    if active.done == active.queued.request.bytes {
                        completions.push(self.finish(index, Outcome::Complete));
                        continue;
                    }
                }
                _ => {}
            }
            index += 1;
        }

        // Priority applies at admission, never by recycling a live DMA source. Speculative
        // staging is capped to a proportional share, leaving room for one demand allocation.
        while self.active.len() < self.limits.max_active {
            let prefetch_count = self
                .active
                .iter()
                .filter(|a| a.queued.request.priority == Priority::Prefetch)
                .count();
            let prefetch_bytes: u64 = self
                .active
                .iter()
                .filter(|a| a.queued.request.priority == Priority::Prefetch && a.staging.is_some())
                .map(|a| a.queued.plan.chunk_bytes)
                .sum();
            let prefetch_budget = self.limits.staging_bytes / self.limits.max_active as u64
                * self.limits.max_prefetch_active as u64;
            let eligible = |q: &Queued| {
                let bytes = if q.plan.staging.is_some() {
                    q.plan.chunk_bytes
                } else {
                    0
                };
                bytes <= self.limits.staging_bytes - self.stats.staging_bytes
                    && (q.request.priority == Priority::Demand
                        || (prefetch_count < self.limits.max_prefetch_active
                            && bytes <= prefetch_budget.saturating_sub(prefetch_bytes)))
            };
            let candidate = self
                .queue
                .iter()
                .position(|q| q.request.priority == Priority::Demand && eligible(q))
                .or_else(|| self.queue.iter().position(eligible));
            let Some(index) = candidate else {
                break;
            };
            let queued = self.queue.remove(index).expect("queue index exists");
            if queued.request.bytes == 0 {
                let completion = Completion {
                    id: queued.id,
                    completed_bytes: 0,
                    outcome: Outcome::Complete,
                };
                self.record(&completion);
                completions.push(completion);
                continue;
            }
            let staging = match queued.plan.staging {
                Some(endpoint) => match self.backend.allocate(endpoint, queued.plan.chunk_bytes) {
                    Ok(allocation)
                        if allocation.region().endpoint == endpoint
                            && allocation.region().bytes == queued.plan.chunk_bytes
                            && allocation.region().writable =>
                    {
                        Some(allocation)
                    }
                    result => {
                        let error = result.err().unwrap_or_else(|| {
                            invalid("backend returned an incompatible staging allocation")
                        });
                        let completion = Completion {
                            id: queued.id,
                            completed_bytes: 0,
                            outcome: Outcome::Failed(error),
                        };
                        self.record(&completion);
                        completions.push(completion);
                        continue;
                    }
                },
                None => None,
            };
            if staging.is_some() {
                self.stats.staging_bytes += queued.plan.chunk_bytes;
                self.stats.peak_staging_bytes =
                    self.stats.peak_staging_bytes.max(self.stats.staging_bytes);
            }
            self.active.push(Active {
                fence: None,
                staging,
                queued,
                done: 0,
                leg: 0,
            });
        }

        let mut index = 0;
        while index < self.active.len() {
            let active = &mut self.active[index];
            if active.fence.is_none() {
                match self.backend.start(active.operation()) {
                    Ok(fence) => {
                        active.fence = Some(fence);
                        self.stats.operations = self.stats.operations.saturating_add(1);
                    }
                    Err(error) => {
                        completions.push(self.finish(index, Outcome::Failed(error)));
                        continue;
                    }
                }
            }
            index += 1;
        }
        completions
    }
}

impl<B: Backend> Drop for TransferEngine<B> {
    fn drop(&mut self) {
        self.active.clear(); // drain fences while the backend is still alive
    }
}
