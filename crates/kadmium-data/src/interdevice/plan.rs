use super::*;

/// Directional, adapter-confirmed connectivity. Costs are explicit per-system estimates,
/// not benchmark claims. Use measured link costs where available, including NUMA penalties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub source: Endpoint,
    pub destination: Endpoint,
    pub method: Method,
    pub bytes_per_second: f64,
    pub latency_ns: u64,
    /// Power-of-two offset/length constraint, 1 for buffered tails.
    pub alignment: u64,
    pub max_chunk_bytes: u64,
}

impl Link {
    fn valid(&self) -> bool {
        self.bytes_per_second.is_finite()
            && self.bytes_per_second > 0.0
            && self.alignment.is_power_of_two()
            && self.max_chunk_bytes >= self.alignment
    }

    fn accepts(&self, source: u64, destination: u64, bytes: u64) -> bool {
        source.is_multiple_of(self.alignment)
            && destination.is_multiple_of(self.alignment)
            && bytes.is_multiple_of(self.alignment)
    }

    fn estimate_ns(&self, bytes: u64, chunks: u64) -> f64 {
        self.latency_ns as f64 * chunks as f64
            + bytes as f64 / self.bytes_per_second * 1_000_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Topology {
    pub version: u16,
    /// Diagnostic/calibration binding, e.g. host + adapter/driver version. No global profile.
    pub system: String,
    pub links: Vec<Link>,
    /// RAM endpoints on which the backend can allocate staging buffers.
    pub staging: Vec<Endpoint>,
}

impl Topology {
    pub fn validate(&self) -> io::Result<()> {
        if self.version != PROTOCOL_VERSION
            || self.system.is_empty()
            || self.links.iter().any(|link| !link.valid())
            || self.staging.iter().any(|e| !matches!(e, Endpoint::Ram(_)))
        {
            return Err(invalid("invalid KIP topology or protocol version"));
        }
        Ok(())
    }

    /// Cheapest supported direct route or two-leg RAM route for this exact range. Alignment
    /// is checked on each leg; an unaligned direct-storage tail can use a buffered route.
    pub fn plan(
        &self,
        request: &Request,
        source: Region,
        destination: Region,
        chunk_limit: u64,
        staging_limit: u64,
    ) -> io::Result<Plan> {
        self.validate()?;
        validate_request(request, source, destination)?;
        if chunk_limit == 0 {
            return Err(invalid("chunk limit must be positive"));
        }
        if request.bytes == 0 {
            return Ok(Plan {
                legs: Vec::new(),
                staging: None,
                chunk_bytes: 0,
                estimated_ns: 0.0,
            });
        }
        let mut best: Option<Plan> = None;
        let mut consider = |legs: Vec<Link>, staging: Option<Endpoint>| {
            let alignment = legs.iter().map(|l| l.alignment).max().unwrap_or(1);
            let mut chunk = legs
                .iter()
                .map(|l| l.max_chunk_bytes)
                .min()
                .unwrap_or(0)
                .min(chunk_limit)
                .min(request.bytes);
            if staging.is_some() {
                chunk = chunk.min(staging_limit);
            }
            chunk = chunk / alignment * alignment;
            if chunk == 0 {
                return;
            }
            let chunks = request.bytes.div_ceil(chunk);
            let estimated_ns: f64 = legs
                .iter()
                .map(|l| l.estimate_ns(request.bytes, chunks))
                .sum();
            if estimated_ns.is_finite()
                && best.as_ref().is_none_or(|p| estimated_ns < p.estimated_ns)
            {
                best = Some(Plan {
                    legs,
                    staging,
                    chunk_bytes: chunk,
                    estimated_ns,
                });
            }
        };
        for first in self.links.iter().filter(|l| l.source == source.endpoint) {
            if first.destination == destination.endpoint
                && first.accepts(
                    request.source.offset,
                    request.destination.offset,
                    request.bytes,
                )
            {
                consider(vec![first.clone()], None);
            }
            if !self.staging.contains(&first.destination)
                || !first.accepts(request.source.offset, 0, request.bytes)
            {
                continue;
            }
            for second in self.links.iter().filter(|l| {
                l.source == first.destination
                    && l.destination == destination.endpoint
                    && l.accepts(0, request.destination.offset, request.bytes)
            }) {
                consider(vec![first.clone(), second.clone()], Some(first.destination));
            }
        }
        best.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "no compatible route within staging budget",
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub legs: Vec<Link>,
    pub staging: Option<Endpoint>,
    pub chunk_bytes: u64,
    /// Modelled serial service time, not a hardware measurement or overlap guarantee.
    pub estimated_ns: f64,
}

pub(super) fn validate_request(
    request: &Request,
    source: Region,
    destination: Region,
) -> io::Result<()> {
    if request.version != PROTOCOL_VERSION
        || source.id != request.source.region
        || destination.id != request.destination.region
        || !destination.writable
    {
        return Err(invalid("invalid version, region or read-only destination"));
    }
    for (span, region) in [(request.source, source), (request.destination, destination)] {
        if span
            .offset
            .checked_add(request.bytes)
            .is_none_or(|end| end > region.bytes)
        {
            return Err(invalid("transfer range exceeds registered region"));
        }
    }
    if overlaps(
        request.source,
        request.bytes,
        request.destination,
        request.bytes,
    ) {
        return Err(invalid("overlapping in-place transfers are not supported"));
    }
    Ok(())
}

pub(super) fn overlaps(a: Span, a_len: u64, b: Span, b_len: u64) -> bool {
    a.region == b.region
        && a_len != 0
        && b_len != 0
        && a.offset < b.offset.saturating_add(b_len)
        && b.offset < a.offset.saturating_add(a_len)
}
