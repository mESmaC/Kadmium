//! # `kadmium-grid`
//!
//! The generic distributed-compute substrate underneath Kadmium's AI workloads.
//!
//! Kadmium's premise is that a mesh of unequal machines can behave like one computer assembled
//! from parts: nodes are the components, and you add compute nodes for more compute and storage
//! nodes for more storage. This crate is the part of the operating system that makes the parts
//! act as one machine:
//!
//! | In the one-machine picture | In this crate | Heritage |
//! | :--- | :--- | :--- |
//! | A process | [`WorkUnit`], one piece of computation with inputs, placement constraints, a cost estimate, a deadline and a redundancy policy; each attempt to run it is a [`WorkInstance`] | BOINC workunits and results |
//! | A job with dependencies | [`WorkGraph`], a DAG whose ready set is the wavefront the scheduler hands out | HTCondor DAGMan, grid workflows |
//! | The disk map | [`ReplicaCatalog`], which node holds which content in which memory tier, plus [`PlacementRule`]s that diff against it into [`StagingJob`]s | Rucio replica rules, FTS3 staging |
//! | The process scheduler | [`WorkQueue`] with its [`Matchmaker`]: capability, locality, politeness, trust and deadline matching, pull or push dispatch, straggler duplication, retries | BOINC scheduler, pilot-job late binding, ClassAd matchmaking |
//! | The audit trail | Quorum validation per trust tier and the [`CreditLedger`] that credits only validated work | BOINC redundancy and credit |
//! | The spec sheet | [`PoolInventory`], the mesh summed up as one machine | Grid information systems |
//!
//! AI workloads sit on top as work graphs: a pipeline decode step is a chain, expert
//! parallelism is a fan-out and fan-in, a Dream-RSI replay batch is a bag with no edges, and a
//! training step is a chain with a reduce at the end. The scheduler never looks inside a unit's
//! `params`; the executor registered for its [`AppKind`] does. This crate has no sockets and no
//! kernels: it depends only on `kadmium-core` for the capability vector, and `kadmium-runtime`
//! brings executors and the wire protocol.

pub mod catalog;
pub mod credit;
pub mod graph;
pub mod scheduler;
pub mod work;

pub use catalog::*;
pub use credit::*;
pub use graph::*;
pub use scheduler::*;
pub use work::*;

/// Errors raised by the grid substrate.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GridError {
    #[error("unknown work unit `{0}`")]
    UnknownUnit(WorkUnitId),
    #[error("work unit `{0}` already exists")]
    DuplicateUnit(WorkUnitId),
    #[error("unknown work instance `{0}`")]
    UnknownInstance(InstanceId),
    #[error("adding `{prereq}` as a prerequisite of `{unit}` would create a cycle")]
    DependencyCycle {
        unit: WorkUnitId,
        prereq: WorkUnitId,
    },
    #[error("invalid transition for `{unit}`: {from:?} -> {to:?}")]
    InvalidTransition {
        unit: WorkUnitId,
        from: WorkState,
        to: WorkState,
    },
    #[error("parameter encoding error: {0}")]
    Params(String),
}
