//! KIP/1: node-local, capability-negotiated byte transfers between RAM, VRAM and storage.
//!
//! The planner and executor are backend independent. [`HostBackend`] performs real buffered
//! file/RAM copies; GPU adapters implement [`Backend`] and must own DMA leases through their
//! completion fences. No hardware capability is inferred from a device name.

mod engine;
mod host;
mod plan;

pub use engine::*;
pub use host::*;
pub use plan::*;

use serde::{Deserialize, Serialize};
use std::io;
use std::task::Poll;

pub const PROTOCOL_VERSION: u16 = 1;

/// IDs are local to one backend session. RAM IDs can distinguish NUMA nodes; unified-memory
/// adapters expose a RAM endpoint unless they actually need a separate device copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Endpoint {
    Ram(u32),
    Vram(u32),
    Storage(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub id: RegionId,
    pub endpoint: Endpoint,
    pub bytes: u64,
    pub writable: bool,
}

/// Byte offsets, never virtual addresses or paths, cross the protocol boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub region: RegionId,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Demand,
    Prefetch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub version: u16,
    pub source: Span,
    pub destination: Span,
    pub bytes: u64,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    Buffered,
    Dma,
    Peer,
    DirectStorage,
}

/// One physical operation. A logical request can have many chunks and two legs per chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyOp {
    pub source: Span,
    pub destination: Span,
    pub bytes: u64,
    pub method: Method,
}

/// Dropping a fence MUST drain/cancel the underlying operation before releasing any source,
/// destination, pin or stream lease, including after a poll error. Ready means destination
/// bytes are visible to subsequent copies/compute (not necessarily durable on storage).
pub trait Fence {
    fn poll(&mut self) -> io::Result<Poll<()>>;
}

/// A staging allocation owns its registration until dropped. Its ID cannot be recycled.
pub trait Lease {
    fn region(&self) -> Region;
}

/// Adapter contract for one system. Topology is an immutable session snapshot; a changed
/// device/driver/topology requires a new session. Start must acquire resource leases before
/// returning. On start failure, no unowned work may remain. Unsupported fast paths must not
/// be advertised. A DMA adapter must pin/register RAM itself, or fail before issuing DMA.
pub trait Backend {
    type Allocation: Lease;
    type Completion: Fence;

    fn topology(&self) -> &Topology;
    fn describe(&self, region: RegionId) -> io::Result<Region>;
    fn allocate(&mut self, endpoint: Endpoint, bytes: u64) -> io::Result<Self::Allocation>;
    fn start(&mut self, operation: CopyOp) -> io::Result<Self::Completion>;
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests;
