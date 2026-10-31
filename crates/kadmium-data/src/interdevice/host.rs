use super::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::sync::{Arc, Mutex, Weak};

enum Storage {
    Ram(Vec<u8>),
    File(File),
}

struct Registered {
    region: Region,
    storage: Mutex<Storage>,
}

/// Keeps a registered file/buffer alive. Clones refer to the same region ID, so executor
/// conflict checks still apply. Do not separately register aliases of one writable file.
#[derive(Clone)]
pub struct HostRegion(Arc<Registered>);

impl Lease for HostRegion {
    fn region(&self) -> Region {
        self.0.region
    }
}

impl HostRegion {
    /// Snapshot RAM bytes after completion; file-backed regions return Unsupported.
    pub fn read_bytes(&self) -> io::Result<Vec<u8>> {
        let storage = self
            .0
            .storage
            .lock()
            .map_err(|_| io::Error::other("region lock poisoned"))?;
        match &*storage {
            Storage::Ram(bytes) => Ok(bytes.clone()),
            Storage::File(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "not a RAM buffer",
            )),
        }
    }

    /// Storage completion means visibility, not crash durability. Call this after successful
    /// transfers if the caller needs a durable file before publishing a checkpoint.
    pub fn sync_all(&self) -> io::Result<()> {
        let storage = self
            .0
            .storage
            .lock()
            .map_err(|_| io::Error::other("region lock poisoned"))?;
        match &*storage {
            Storage::File(file) => file.sync_all(),
            Storage::Ram(_) => Ok(()),
        }
    }
}

/// Explicit priors or measurements for this system. The portable backend does buffered I/O;
/// filesystem page-cache hits and the physical storage medium affect observed throughput.
#[derive(Debug, Clone, Copy)]
pub struct HostCosts {
    pub ram_bytes_per_second: f64,
    pub read_bytes_per_second: f64,
    pub write_bytes_per_second: f64,
    pub io_latency_ns: u64,
}

/// Portable, synchronous buffered adapter. It advertises no GPU, pinned memory, direct I/O,
/// io_uring or GPUDirect capability. Requests still use bounded chunks and the same lifecycle
/// as asynchronous adapters. Region tables hold weak references and prune expired entries.
pub struct HostBackend {
    topology: Topology,
    costs: HostCosts,
    regions: BTreeMap<u64, Weak<Registered>>,
    next_id: u64,
}

impl HostBackend {
    pub fn new(system: impl Into<String>, costs: HostCosts) -> io::Result<Self> {
        if [
            costs.ram_bytes_per_second,
            costs.read_bytes_per_second,
            costs.write_bytes_per_second,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid("host costs must be positive finite rates"));
        }
        let topology = Topology {
            version: PROTOCOL_VERSION,
            system: system.into(),
            links: vec![Link {
                source: Endpoint::Ram(0),
                destination: Endpoint::Ram(0),
                method: Method::Buffered,
                bytes_per_second: costs.ram_bytes_per_second,
                latency_ns: 0,
                alignment: 1,
                max_chunk_bytes: 64 << 20,
            }],
            staging: vec![Endpoint::Ram(0)],
        };
        topology.validate()?;
        Ok(Self {
            topology,
            costs,
            regions: BTreeMap::new(),
            next_id: 1,
        })
    }

    fn register(
        &mut self,
        endpoint: Endpoint,
        bytes: u64,
        writable: bool,
        storage: Storage,
    ) -> io::Result<HostRegion> {
        let id = RegionId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| invalid("region IDs exhausted"))?;
        let registered = Arc::new(Registered {
            region: Region {
                id,
                endpoint,
                bytes,
                writable,
            },
            storage: Mutex::new(storage),
        });
        self.regions.retain(|_, region| region.strong_count() > 0);
        self.regions.insert(id.0, Arc::downgrade(&registered));
        Ok(HostRegion(registered))
    }

    pub fn register_ram(&mut self, bytes: Vec<u8>) -> io::Result<HostRegion> {
        self.register(
            Endpoint::Ram(0),
            bytes.len() as u64,
            true,
            Storage::Ram(bytes),
        )
    }

    /// Takes ownership of an already opened file. No creation, truncation or resizing occurs.
    /// `storage_id` distinguishes physical volumes; the caller supplies that discovery.
    /// Keep files stable and exclude external writers for the lifetime of their registration.
    pub fn register_file(
        &mut self,
        file: File,
        storage_id: u32,
        writable: bool,
    ) -> io::Result<HostRegion> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(invalid("only regular files can be registered"));
        }
        let endpoint = Endpoint::Storage(storage_id);
        let region = self.register(endpoint, metadata.len(), writable, Storage::File(file))?;
        if !self
            .topology
            .links
            .iter()
            .any(|link| link.source == endpoint)
        {
            for (source, destination, rate) in [
                (endpoint, Endpoint::Ram(0), self.costs.read_bytes_per_second),
                (
                    Endpoint::Ram(0),
                    endpoint,
                    self.costs.write_bytes_per_second,
                ),
            ] {
                self.topology.links.push(Link {
                    source,
                    destination,
                    method: Method::Buffered,
                    bytes_per_second: rate,
                    latency_ns: self.costs.io_latency_ns,
                    alignment: 1,
                    max_chunk_bytes: 64 << 20,
                });
            }
        }
        Ok(region)
    }

    fn get(&self, id: RegionId) -> io::Result<Arc<Registered>> {
        self.regions
            .get(&id.0)
            .and_then(Weak::upgrade)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "region is not registered in this session",
                )
            })
    }
}

pub struct HostFence;
impl Fence for HostFence {
    fn poll(&mut self) -> io::Result<Poll<()>> {
        Ok(Poll::Ready(()))
    }
}

impl Backend for HostBackend {
    type Allocation = HostRegion;
    type Completion = HostFence;

    fn topology(&self) -> &Topology {
        &self.topology
    }
    fn describe(&self, region: RegionId) -> io::Result<Region> {
        Ok(self.get(region)?.region)
    }
    fn allocate(&mut self, endpoint: Endpoint, bytes: u64) -> io::Result<HostRegion> {
        if endpoint != Endpoint::Ram(0) {
            return Err(invalid("host adapter can only allocate RAM(0)"));
        }
        let len =
            usize::try_from(bytes).map_err(|_| invalid("allocation does not fit address space"))?;
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(len).map_err(|_| {
            io::Error::new(io::ErrorKind::OutOfMemory, "cannot allocate staging RAM")
        })?;
        buffer.resize(len, 0);
        self.register_ram(buffer)
    }

    fn start(&mut self, operation: CopyOp) -> io::Result<HostFence> {
        let source = self.get(operation.source.region)?;
        let destination = self.get(operation.destination.region)?;
        plan::validate_request(
            &Request {
                version: PROTOCOL_VERSION,
                source: operation.source,
                destination: operation.destination,
                bytes: operation.bytes,
                priority: Priority::Demand,
            },
            source.region,
            destination.region,
        )?;
        if operation.method != Method::Buffered
            || !self.topology.links.iter().any(|link| {
                link.source == source.region.endpoint
                    && link.destination == destination.region.endpoint
                    && operation.bytes <= link.max_chunk_bytes
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unsupported host operation",
            ));
        }
        let len = usize::try_from(operation.bytes)
            .map_err(|_| invalid("copy does not fit address space"))?;
        let ram_offset = |offset| {
            usize::try_from(offset).map_err(|_| invalid("RAM offset does not fit address space"))
        };
        let mut src = source
            .storage
            .lock()
            .map_err(|_| io::Error::other("source lock poisoned"))?;
        if source.region.id == destination.region.id {
            if let Storage::Ram(bytes) = &mut *src {
                let start = ram_offset(operation.source.offset)?;
                bytes.copy_within(
                    start..start + len,
                    ram_offset(operation.destination.offset)?,
                );
                return Ok(HostFence);
            }
            return Err(invalid("file copies need a RAM staging route"));
        }
        let mut dst = destination
            .storage
            .lock()
            .map_err(|_| io::Error::other("destination lock poisoned"))?;
        match (&*src, &mut *dst) {
            (Storage::Ram(source), Storage::Ram(destination)) => {
                let s = ram_offset(operation.source.offset)?;
                let d = ram_offset(operation.destination.offset)?;
                destination[d..d + len].copy_from_slice(&source[s..s + len]);
            }
            (Storage::File(file), Storage::Ram(destination)) => {
                let d = ram_offset(operation.destination.offset)?;
                read_exact_at(file, operation.source.offset, &mut destination[d..d + len])?;
            }
            (Storage::Ram(source), Storage::File(file)) => {
                let s = ram_offset(operation.source.offset)?;
                write_all_at(file, operation.destination.offset, &source[s..s + len])?;
            }
            _ => return Err(invalid("file copies need a RAM staging route")),
        }
        Ok(HostFence)
    }
}

fn read_exact_at(file: &File, mut offset: u64, mut bytes: &mut [u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::FileExt;
            file.read_at(bytes, offset)
        };
        #[cfg(windows)]
        let result = {
            use std::os::windows::fs::FileExt;
            file.seek_read(bytes, offset)
        };
        #[cfg(not(any(unix, windows)))]
        let result = {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = file;
            file.seek(SeekFrom::Start(offset))
                .and_then(|_| file.read(bytes))
        };
        match result {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "registered source was truncated",
                ));
            }
            Ok(n) => {
                offset += n as u64;
                bytes = &mut bytes[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn write_all_at(file: &File, mut offset: u64, mut bytes: &[u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::FileExt;
            file.write_at(bytes, offset)
        };
        #[cfg(windows)]
        let result = {
            use std::os::windows::fs::FileExt;
            file.seek_write(bytes, offset)
        };
        #[cfg(not(any(unix, windows)))]
        let result = {
            use std::io::{Seek, SeekFrom, Write};
            let mut file = file;
            file.seek(SeekFrom::Start(offset))
                .and_then(|_| file.write(bytes))
        };
        match result {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "file write made no progress",
                ));
            }
            Ok(n) => {
                offset += n as u64;
                bytes = &bytes[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
