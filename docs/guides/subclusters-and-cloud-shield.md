# Guide: Sub-Cluster Pool Slicing, Desktop Companion App & `kadmium.ai` CloudShield

As your Kadmium pool grows—from a few home machines to a 100-node university lab, startup cluster, or hybrid cloud fleet—you often need to **section off isolated slices of the compute pool** and protect production workloads against local or cloud outages.

---

## 1. Sub-Cluster Pool Slicing (`SubClusterPartitioner`)

Suppose you have **100 heterogeneous nodes** in your master pool and want **5 isolated sub-clusters of 20 nodes each**—for example, testing a new model on one slice while serving production traffic on another and running Dream-RSI on a third.

[`SubClusterPartitioner::partition_pool`](../../crates/kadmium-sharding/src/lib.rs) uses **Stratified Balanced Round-Robin Partitioning** so every sub-cluster receives a balanced mix of high-VRAM GPUs, unified-memory nodes, NAS seed storage, and CPU/SBC workers:

| Slice ID | Purpose (`SlicePurpose`) | Nodes | Example Workload |
| :--- | :--- | :---: | :--- |
| `slice-01-prod-deploy` | `ProductionDeployment` | `20` | High-availability OpenAI/Anthropic inference gateway |
| `slice-02-staging-test` | `TestingAndStaging` | `20` | Canary testing new `.ktb` quantizations & prompts |
| `slice-03-diloco-train` | `DistributedTraining` | `20` | Sharded-Island DiLoCo + `DeMo` fine-tuning |
| `slice-04-rl-dream-rsi` | `RlAndDreamRsi` | `20` | Disaggregated GRPO rollouts + Dream-RSI replay |
| `slice-05-batch-etl` | `BatchDataProcessing` | `20` | Offline synthetic dataset filtering & embedding |

### Key Isolation Guarantees
- **Hierarchical Sub-Coordinator:** Each slice elects its own local coordinator leader so a heavy training sync in `slice-03-diloco-train` never adds jitter to `slice-01-prod-deploy`.
- **Isolated QUIC Stream Namespace (`isolated_network_namespace: true`):** P2P wire frames (`"KDMF"`) are scoped per slice.
- **Elastic Re-Slicing:** Nodes can be reassigned across slices at epoch boundaries without restarting the daemon.

Try slicing a 100-node pool from the CLI:
```bash
kadmium slice --pool-size 100 --slices 5
```

---

## 2. Dual-Mode Desktop Companion App (`SelfHostedOpenSource` $\leftrightarrow$ `KadmiumCloudManaged`)

The **Kadmium Desktop Companion App** (and underlying [`kadmium-gateway`](../../crates/kadmium-gateway/src/lib.rs) daemon) is designed so the exact same binary serves both **100% free open-source self-hosters** and **`kadmium.ai` paid cloud users**.

### Mode A: `SelfHostedOpenSource` (Default — `$0.00/hr`)
- Connects strictly to your local coordinator (`http://127.0.0.1:7848` or LAN mDNS).
- Zero telemetry or cloud account required.
- Full access to asymmetric sharding, `.ktb` transcoding, CERN FTS3 LAN staging, `<200ms` Polite Gaming Auto-Yield, Dream-RSI, and sub-cluster slicing.

### Mode B: `KadmiumCloudManaged` (Opt-In `kadmium.ai` Hybrid Cloud)
Switchable at runtime via `POST /api/v1/companion/mode` or the Desktop Companion system-tray toggle:
- Connects your local cluster to your `kadmium.ai` workspace (`https://api.kadmium.ai`).
- Enables zero-config WAN QUIC hole-punching (`relay.kadmium.ai:4433`), **Multi-Datacenter Spot Arbitrage**, and **Bidirectional Downtime Protection (`CloudShield`)** with a hard `max_hourly_spend_cap_usd` guardrail.

---

## 3. Multi-Datacenter Spot Arbitrage & Bidirectional Downtime Protection (`CloudShield`)

Because Kadmium's runtime already tolerates heterogeneous latency (`1F1B` + `DeMo`), mid-epoch node churn (`ShardPlanner::plan_responsibility_shift`), and untrusted nodes (`Verde` refereed verification), it unlocks two capabilities that single-cloud providers cannot match:

### A. Multi-Datacenter Spot Arbitrage (`~62%` Cost Reduction)
Instead of renting an expensive monolithic $8 \times \text{H100}$ node on AWS (`$26.40/hr`), [`HybridCloudArbitragePlanner::compute_arbitrage_plan`](../../crates/kadmium-sharding/src/lib.rs):
1. Keeps Layer $0$ (ingress) and Layer $N-1$ (egress) pinned on your **`TrustedCore` local hardware (`$0.00/hr`)** for complete prompt/output privacy.
2. Bids on the cheapest spot GPU capacity across **RunPod, Vast.ai, Lambda Cloud, AWS Spot, and GCP Spot** for middle transformer layers (`$0.69/hr` blended overflow cost).
3. Continuously verifies remote spot GPU outputs via `Verde` $O(\log_2 L)$ refereed bisection.

### B. Bidirectional Downtime Protection (`Local <-> Cloud` Failover Shield)
[`HybridCloudArbitragePlanner::simulate_bidirectional_failover_shield`](../../crates/kadmium-sharding/src/lib.rs) protects your workloads in **both** directions:

1. **`LocalToCloud` Failover (Local Outage $\to$ Cloud Shield):**
   - Your Home NAS asynchronously trickle-syncs encrypted `.ktb` weight chunks and periodic LoRA/KV checkpoints to warm object storage via CERN FTS3 (`$0.04/hr` standby).
   - If your home loses power, your ISP drops, or your gaming PC overheats, `kadmium.ai` spins up spot GPU capacity and shifts the affected layers to the cloud in **`380ms`** with **`0` dropped requests**.
2. **`CloudToLocal` Failover (Cloud Outage / Spot Preemption $\to$ Local Cluster):**
   - If a cloud region suffers an outage or a spot GPU provider sends a preemption notice, Kadmium shifts the cloud layers back onto your local Mac Mini, Gaming PC, and laptops (activating FreeToken $q^*$ + Colibri NVMe expert streaming) in **`310ms`** with **`0` dropped requests**.

Run the full arbitrage and bidirectional failover simulation anytime via:
```bash
kadmium cloud-shield
```
