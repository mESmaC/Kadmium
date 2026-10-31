# Guide: Sub-Cluster Planning, Companion Modes & CloudShield Simulations

Kadmium provides pool-partitioning data structures, Companion mode state, and fixed cloud-arbitrage/failover scenarios. These are planning and reference features; they do not provide managed cloud connectivity or production fault-tolerance guarantees.

---

## 1. Sub-cluster pool slicing

[`SubClusterPartitioner::partition_pool`](../../crates/kadmium-sharding/src/lib.rs) uses stratified balanced round-robin partitioning to distribute node classes across slices and computes a shard plan for each one.

```bash
kadmium slice --pool-size 100 --slices 5
```

The default demonstration creates five 20-node slices:

| Slice ID | Purpose | Example planning role |
| :--- | :--- | :--- |
| `slice-01-prod-deploy` | `ProductionDeployment` | Inference placement |
| `slice-02-staging-test` | `StagingAndTesting` | Canary placement |
| `slice-03-train-diloco` | `DistributedTrainingLab` | Training placement |
| `slice-04-rl-dream-rsi` | `RlAndDreamRsiArena` | RL/replay placement |
| `slice-05-batch-eval` | `BatchAndEvalPool` | Batch/evaluation placement |

A `ComputeSubCluster` includes an API namespace, elastic-borrowing policy, member nodes, and shard plan. The current implementation does **not** elect per-slice coordinators, create isolated QUIC network namespaces, authenticate tenants, or perform online elastic re-slicing.

`GET /api/v1/slices` returns the five-slice, 100-node demonstration outside live-only mode. In live-only mode it returns no slices when no model is active, or one live partition when a model is active.

## 2. Desktop Companion modes

### `SelfHostedOpenSource`

The default mode points the Companion at the local coordinator. No cloud account is required. Hardware probing, planners, local reference execution, and simulations run locally.

### `KadmiumCloudManaged`

`POST /api/v1/companion/mode` changes the local `companion_mode` value and returns updated Companion status. The request can store a workspace slug, endpoint, feature toggles, and spend-cap value.

This mode switch is currently **local configuration/status only**. It does not contact `kadmium.ai`, establish a WAN relay, provision spot capacity, enable failover, or enforce a hard spending limit. Displayed cloud-burn values are sample state, not billing records.

## 3. Cloud arbitrage and failover scenarios

Run the fixed scenario report with:

```bash
kadmium cloud-shield
```

[`HybridCloudArbitragePlanner`](../../crates/kadmium-sharding/src/lib.rs) compares illustrative local and spot offers. The current fixture totals **$1.82/hr versus $13.70/hr**, an **86.7% modelled saving**. These are hard-coded example prices, not live order books or a quote users can purchase.

The failover routines are also simulations. Current sample timings are:

| Scenario | Simulated time |
| :--- | ---: |
| Local to cloud | 340 ms |
| Cloud to local or multicloud | 185 ms |
| Failback | 210 ms |
| Companion CloudShield watcher report | 340 ms |

No provider API is called, no cloud instance is started, and no live requests are failed over. Values such as zero dropped requests and hourly burn are scenario outputs, not measured service-level guarantees. Managed procurement, secure multi-tenancy, live migration, billing, and production CloudShield remain planned work.
