//! # `kadmium_network::raid_striper`
//!
//! BitTorrent Multi-Seeder Swarming, `RAID-0` Sequence/Channel Striping, `RAID-5` Linear-Algebraic
//! Parity Tensor Striping, and Multi-Path Interface Bonding (`DDR / RAID` Connection Striping).
//!
//! ## Core Problems Solved
//! 1. **10:1 Asymmetric Residential / Edge Upload Bottleneck:**
//!    Typical consumer and campus nodes have `500–1,000 Mbps` download (`62–125 MB/s`) but only
//!    `20–35 Mbps` upload (`2.5–4.3 MB/s`). Striping a prefill activation (`33 MB`), MoE expert
//!    batch, or `.ktb` shard across $k$ parallel seeders/workers aggregates their upload pipes
//!    ($k \times 25\text{ Mbps} = 200\text{–}400\text{ Mbps}$) into the receiver's gigabit downlink.
//! 2. **Residential Tail-Latency Stragglers & Gaming Preemption (`RAID-5` Linear Parity):**
//!    In naive `RAID-0` striping across $k$ nodes, a single Wi-Fi spike or `< 200 ms` gaming yield
//!    stalls the entire pass. By adding a $(k+1)$-th **Parity Worker** that evaluates the linear
//!    operator on the elementwise sum of the input stripes:
//!    $$X_{\text{parity}} = \sum_{j=1}^k X_j \implies Y_{\text{parity}} = X_{\text{parity}} W = \sum_{j=1}^k Y_j$$
//!    the coordinator reconstructs any missing or slow stripe $m$ instantaneously as soon as
//!    **any $k$ out of $k+1$ stripes arrive**:
//!    $$Y_m = Y_{\text{parity}} - \sum_{j \neq m} Y_j$$
//! 3. **Multi-Homed Physical Interface Striping (`MultiPathInterfaceStriper`):**
//!    Bonds multiple network interfaces on a single machine (`WiredLanQuic`, `WirelessLanQuic`,
//!    `Usb3AdbMultiplexedTunnel`, `WanDirectHolepunch`) using Deficit-Round-Robin (`DRR`) weighted
//!    by inverse RTT and effective uplink bandwidth.

use crate::{PeerLinkMetrics, TransportKind};
use kadmium_core::{HardwareNode, NodeId};
use serde::{Deserialize, Serialize};

/// A single data or parity stripe of a 2D activation tensor ($S \times d$) or `.ktb` page block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStripePacket {
    /// Stripe index ($0 \dots k-1$ for data stripes; $k$ for the `RAID-5` linear parity stripe).
    pub stripe_index: usize,
    /// True if this packet holds the linear-algebraic parity sum $X_{\text{parity}} = \sum_{j=1}^k X_j$.
    pub is_parity_stripe: bool,
    /// Assigned worker / seeder node ID.
    pub worker_node_id: NodeId,
    /// Number of token rows in this stripe ($S_{\text{stripe}} = \lceil S / k \rceil$).
    pub token_rows: usize,
    /// Hidden dimension ($d$).
    pub hidden_dim: usize,
    /// Simulated or measured arrival latency (`ms`) over this worker's uplink.
    pub arrival_latency_ms: f64,
    /// True if this worker lagged (`p99` straggler) or yielded for gaming and was bypassed via `RAID-5` parity.
    pub bypassed_as_straggler: bool,
    /// Flattened payload values (`token_rows * hidden_dim`).
    pub values: Vec<f32>,
}

/// Telemetry and reconstruction report from a `RAID-0 / RAID-5` striped tensor pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaidStripedPassReport {
    pub data_stripes_k: usize,
    pub parity_stripes_p: usize,
    pub total_tokens: usize,
    pub hidden_dim: usize,
    pub total_payload_mb: f64,
    /// Time (`ms`) if sent sequentially over a single residential uplink (`~25 Mbps`).
    pub single_link_baseline_ms: f64,
    /// Time (`ms`) using naive `RAID-0` striping (bottlenecked by the slowest straggler `max_j(t_j)`).
    pub raid0_straggler_bound_ms: f64,
    /// Time (`ms`) using `RAID-5` parity striping (completes on the $k$-th fastest arrival, ignoring the slowest node).
    pub raid5_parity_completed_ms: f64,
    /// Effective aggregate upload bandwidth (`Mbps`) achieved by `RAID-5` swarming.
    pub effective_aggregate_uplink_mbps: f64,
    /// Speedup factor over single-link residential upload (`single_link_baseline_ms / raid5_parity_completed_ms`).
    pub speedup_vs_single_link: f64,
    ///Tail-latency reduction (`%`) of `RAID-5` vs. `RAID-0` when a residential node stutters.
    pub tail_latency_reduction_pct: f64,
    /// Which worker stripe index (if any) was bypassed and reconstructed via $Y_m = Y_{\text{parity}} - \sum_{j \neq m} Y_j$.
    pub reconstructed_straggler_index: Option<usize>,
    /// Maximum absolute numerical reconstruction error $\max |Y_{\text{exact}} - Y_{\text{raid5}}|$.
    pub max_reconstruction_error: f32,
}

/// Linear-algebraic `RAID-0 / RAID-5` Tensor & Activation Striper.
pub struct SwarmRaidStriper;

impl SwarmRaidStriper {
    /// Splits a flattened row-major activation matrix $X \in \mathbb{R}^{S \times d}$ into $k$ equal
    /// token-chunk data stripes plus $1$ `RAID-5` linear-algebraic parity stripe
    /// $X_{\text{parity}} = \sum_{j=0}^{k-1} X_j$.
    pub fn encode_input_stripes_with_parity(
        input_matrix: &[f32],
        total_tokens: usize,
        hidden_dim: usize,
        data_stripes_k: usize,
        worker_ids: &[NodeId],
    ) -> Vec<TensorStripePacket> {
        let k = data_stripes_k.max(2);
        assert_eq!(
            input_matrix.len(),
            total_tokens * hidden_dim,
            "input_matrix length must equal total_tokens * hidden_dim"
        );
        assert!(
            total_tokens.is_multiple_of(k),
            "total_tokens ({total_tokens}) must be divisible by data_stripes_k ({k})"
        );

        let rows_per_stripe = total_tokens / k;
        let elems_per_stripe = rows_per_stripe * hidden_dim;
        let mut packets = Vec::with_capacity(k + 1);
        let mut parity_values = vec![0.0f32; elems_per_stripe];

        for s in 0..k {
            let start = s * elems_per_stripe;
            let slice = &input_matrix[start..start + elems_per_stripe];
            for (p, &v) in parity_values.iter_mut().zip(slice.iter()) {
                *p += v;
            }
            let node_id = worker_ids
                .get(s % worker_ids.len().max(1))
                .cloned()
                .unwrap_or_else(|| NodeId(format!("swarm-seeder-{s:02}")));

            packets.push(TensorStripePacket {
                stripe_index: s,
                is_parity_stripe: false,
                worker_node_id: node_id,
                token_rows: rows_per_stripe,
                hidden_dim,
                arrival_latency_ms: 0.0,
                bypassed_as_straggler: false,
                values: slice.to_vec(),
            });
        }

        let parity_node_id = worker_ids
            .get(k % worker_ids.len().max(1))
            .cloned()
            .unwrap_or_else(|| NodeId(format!("swarm-parity-{k:02}")));

        packets.push(TensorStripePacket {
            stripe_index: k,
            is_parity_stripe: true,
            worker_node_id: parity_node_id,
            token_rows: rows_per_stripe,
            hidden_dim,
            arrival_latency_ms: 0.0,
            bypassed_as_straggler: false,
            values: parity_values,
        });

        packets
    }

    /// Executes a distributed linear projection $Y = X W$ ($W \in \mathbb{R}^{d \times d_{\text{out}}}$)
    /// across $k$ data workers + $1$ `RAID-5` parity worker, simulating asymmetric residential upload
    /// links (`25–45 Mbps`) where one worker experiences a `p99` straggler spike or gaming yield.
    ///
    /// Automatically reconstructs the straggler's output stripe from the parity worker via:
    /// $$Y_m = Y_{\text{parity}} - \sum_{j \neq m} Y_j$$
    /// and verifies bit-accurate recovery against the exact monolithic matrix product.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_raid5_coded_projection(
        input_matrix: &[f32],
        weight_matrix: &[f32],
        total_tokens: usize,
        hidden_dim: usize,
        out_dim: usize,
        data_stripes_k: usize,
        simulated_straggler_idx: Option<usize>,
        total_logical_payload_mb: f64,
    ) -> (Vec<f32>, RaidStripedPassReport) {
        let k = data_stripes_k.max(2);
        assert_eq!(weight_matrix.len(), hidden_dim * out_dim);

        let worker_ids: Vec<NodeId> = (0..=k)
            .map(|i| NodeId(format!("swarm-seeder-{i:02}")))
            .collect();
        let input_stripes = Self::encode_input_stripes_with_parity(
            input_matrix,
            total_tokens,
            hidden_dim,
            k,
            &worker_ids,
        );

        let rows_per_stripe = total_tokens / k;
        let out_elems_per_stripe = rows_per_stripe * out_dim;

        // Each worker j in 0..=k computes Y_j = X_j * W independently:
        let mut output_stripes: Vec<TensorStripePacket> = Vec::with_capacity(k + 1);
        let stripe_payload_mb = total_logical_payload_mb / (k as f64);
        let straggler_target = simulated_straggler_idx.unwrap_or(k / 2).min(k - 1);

        for pkt in &input_stripes {
            let mut y = vec![0.0f32; out_elems_per_stripe];
            for r in 0..rows_per_stripe {
                let x_row = &pkt.values[r * hidden_dim..(r + 1) * hidden_dim];
                let y_row = &mut y[r * out_dim..(r + 1) * out_dim];
                for c in 0..hidden_dim {
                    let xv = x_row[c];
                    let w_row = &weight_matrix[c * out_dim..(c + 1) * out_dim];
                    for o in 0..out_dim {
                        y_row[o] += xv * w_row[o];
                    }
                }
            }

            // Realistic residential seeder uplink (~28-36 Mbps upload, ~500 Mbps download):
            // Inbound takes ~65 ms over fast download; outbound takes (stripe_payload_mb * 8 / uplink_mbps) * 1000 ms.
            let uplink_mbps = 28.0 + ((pkt.stripe_index * 7) % 11) as f64;
            let base_transfer_ms = 65.0 + (stripe_payload_mb * 8.0 / uplink_mbps) * 1000.0;
            let is_straggler = !pkt.is_parity_stripe && pkt.stripe_index == straggler_target;
            let arrival_ms = if is_straggler {
                // Simulate a residential Wi-Fi spike / packet-loss retransmit or gaming preemption (+2,400 ms stall)
                base_transfer_ms + 2400.0
            } else {
                base_transfer_ms
            };

            output_stripes.push(TensorStripePacket {
                stripe_index: pkt.stripe_index,
                is_parity_stripe: pkt.is_parity_stripe,
                worker_node_id: pkt.worker_node_id.clone(),
                token_rows: rows_per_stripe,
                hidden_dim: out_dim,
                arrival_latency_ms: (arrival_ms * 10.0).round() / 10.0,
                bypassed_as_straggler: is_straggler,
                values: y,
            });
        }

        // Reconstruct missing/straggler data stripe `straggler_target` using the parity stripe (index `k`):
        // Y_{straggler} = Y_{parity} - \sum_{j != straggler} Y_j
        let parity_y = &output_stripes[k].values;
        let mut reconstructed_stripe = parity_y.clone();
        for (s, stripe) in output_stripes.iter().enumerate().take(k) {
            if s != straggler_target {
                for (acc, &val) in reconstructed_stripe.iter_mut().zip(stripe.values.iter()) {
                    *acc -= val;
                }
            }
        }

        // Assemble final full output matrix Y (using reconstructed_stripe for `straggler_target`):
        let mut assembled_y = Vec::with_capacity(total_tokens * out_dim);
        let mut max_err: f32 = 0.0;
        for (s, stripe) in output_stripes.iter().enumerate().take(k) {
            let chosen = if s == straggler_target {
                for (&exact, &rec) in stripe.values.iter().zip(reconstructed_stripe.iter()) {
                    let diff = (exact - rec).abs();
                    if diff > max_err {
                        max_err = diff;
                    }
                }
                &reconstructed_stripe
            } else {
                &stripe.values
            };
            assembled_y.extend_from_slice(chosen);
        }

        // Compute latency metrics:
        let single_link_baseline_ms = 65.0 + (total_logical_payload_mb * 8.0 / 28.0) * 1000.0;
        let raid0_straggler_bound_ms = output_stripes[0..k]
            .iter()
            .map(|p| p.arrival_latency_ms)
            .fold(0.0f64, f64::max);

        // RAID-5 completes as soon as the k-th fastest stripe among all (k+1) workers arrives!
        let mut all_latencies: Vec<f64> = output_stripes
            .iter()
            .map(|p| p.arrival_latency_ms)
            .collect();
        all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let raid5_parity_completed_ms = all_latencies[k - 1];

        let effective_aggregate_uplink_mbps =
            (total_logical_payload_mb * 8.0) / (raid5_parity_completed_ms / 1000.0);
        let speedup_vs_single_link = single_link_baseline_ms / raid5_parity_completed_ms;
        let tail_latency_reduction_pct = ((raid0_straggler_bound_ms - raid5_parity_completed_ms)
            / raid0_straggler_bound_ms)
            * 100.0;

        let report = RaidStripedPassReport {
            data_stripes_k: k,
            parity_stripes_p: 1,
            total_tokens,
            hidden_dim: out_dim,
            total_payload_mb: total_logical_payload_mb,
            single_link_baseline_ms: (single_link_baseline_ms * 10.0).round() / 10.0,
            raid0_straggler_bound_ms: (raid0_straggler_bound_ms * 10.0).round() / 10.0,
            raid5_parity_completed_ms: (raid5_parity_completed_ms * 10.0).round() / 10.0,
            effective_aggregate_uplink_mbps: (effective_aggregate_uplink_mbps * 10.0).round()
                / 10.0,
            speedup_vs_single_link: (speedup_vs_single_link * 100.0).round() / 100.0,
            tail_latency_reduction_pct: (tail_latency_reduction_pct * 10.0).round() / 10.0,
            reconstructed_straggler_index: Some(straggler_target),
            max_reconstruction_error: max_err,
        };

        (assembled_y, report)
    }

    /// Packet-level `RAID-5` XOR parity encoder/decoder for `64 KiB` `.ktb` weight pages and
    /// `KvPageBlock` state migrations across $k$ BitTorrent seeders + $1$ XOR parity seeder.
    pub fn encode_xor_parity_page(data_pages: &[Vec<u8>]) -> Vec<u8> {
        assert!(!data_pages.is_empty(), "data_pages must not be empty");
        let page_len = data_pages[0].len();
        let mut parity = vec![0u8; page_len];
        for page in data_pages {
            assert_eq!(
                page.len(),
                page_len,
                "all pages in stripe must have equal size"
            );
            for (p, &b) in parity.iter_mut().zip(page.iter()) {
                *p ^= b;
            }
        }
        parity
    }

    /// Reconstructs a missing `64 KiB` `.ktb` or KV-cache page at `missing_index` using the
    /// remaining $k-1$ data pages and the `RAID-5` XOR parity page.
    pub fn reconstruct_missing_page_xor(
        data_pages: &[Option<Vec<u8>>],
        parity_page: &[u8],
        missing_index: usize,
    ) -> Vec<u8> {
        let mut recovered = parity_page.to_vec();
        for (idx, maybe_page) in data_pages.iter().enumerate() {
            if idx != missing_index {
                let page = maybe_page
                    .as_ref()
                    .expect("all non-missing data pages must be present for RAID-5 XOR recovery");
                for (r, &b) in recovered.iter_mut().zip(page.iter()) {
                    *r ^= b;
                }
            }
        }
        recovered
    }
}

/// Multi-Homed Interface Connection Striper (`DDR Dual/Quad-Channel` style link bonding across
/// Ethernet LAN, Wi-Fi 6E, USB-ADB Phone Farm tunnels, and WAN Hole-Punched paths).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPathStripedSchedule {
    pub total_chunks: usize,
    pub chunk_size_kb: usize,
    pub bonded_bandwidth_gbps: f64,
    pub single_best_link_gbps: f64,
    pub multi_path_speedup: f64,
    pub estimated_transfer_ms: f64,
    pub channel_allocations: Vec<MultiPathChannelAllocation>,
}

/// Per-interface chunk allocation produced by `MultiPathInterfaceStriper`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPathChannelAllocation {
    pub interface_name: String,
    pub transport: TransportKind,
    pub bandwidth_gbps: f64,
    pub rtt_ms: f64,
    pub assigned_chunks: usize,
    pub share_pct: f64,
}

pub struct MultiPathInterfaceStriper;

impl MultiPathInterfaceStriper {
    /// Schedules `total_chunks` (`64 KiB` frames) across multiple parallel physical interfaces on
    /// a node or peer pair using Deficit-Round-Robin (`DRR`) weighted by effective goodput
    /// $\text{Weight}_i = \frac{\text{BW}_i}{1 + 0.02 \cdot \text{RTT}_i}$.
    pub fn schedule_multipath_stripes(
        links: &[PeerLinkMetrics],
        total_payload_mb: f64,
    ) -> MultiPathStripedSchedule {
        let chunk_size_kb = 64usize;
        let total_chunks = ((total_payload_mb * 1024.0) / (chunk_size_kb as f64))
            .ceil()
            .max(1.0) as usize;

        let active_links: Vec<&PeerLinkMetrics> = links.iter().take(4).collect();
        let weights: Vec<f64> = active_links
            .iter()
            .map(|l| l.bandwidth_gbps / (1.0 + 0.02 * l.rtt_latency_ms))
            .collect();
        let weight_sum: f64 = weights.iter().sum::<f64>().max(1e-6);

        let mut bonded_bw = 0.0f64;
        let mut single_best = 0.0f64;
        let mut assigned_so_far = 0usize;
        let mut allocations = Vec::with_capacity(active_links.len());

        for (i, link) in active_links.iter().enumerate() {
            bonded_bw += link.bandwidth_gbps;
            if link.bandwidth_gbps > single_best {
                single_best = link.bandwidth_gbps;
            }
            let share = weights[i] / weight_sum;
            let chunks = if i + 1 == active_links.len() {
                total_chunks.saturating_sub(assigned_so_far)
            } else {
                ((total_chunks as f64) * share).round() as usize
            };
            assigned_so_far += chunks;

            allocations.push(MultiPathChannelAllocation {
                interface_name: format!(
                    "{}->{}:{:?}",
                    link.from_node.0, link.to_node.0, link.transport
                ),
                transport: link.transport,
                bandwidth_gbps: link.bandwidth_gbps,
                rtt_ms: link.rtt_latency_ms,
                assigned_chunks: chunks,
                share_pct: (share * 1000.0).round() / 10.0,
            });
        }

        let estimated_transfer_ms =
            ((total_payload_mb * 8.0) / (bonded_bw.max(0.01) * 1000.0)) * 1000.0;
        let multi_path_speedup = (bonded_bw / single_best.max(0.01) * 100.0).round() / 100.0;

        MultiPathStripedSchedule {
            total_chunks,
            chunk_size_kb,
            bonded_bandwidth_gbps: (bonded_bw * 100.0).round() / 100.0,
            single_best_link_gbps: (single_best * 100.0).round() / 100.0,
            multi_path_speedup,
            estimated_transfer_ms: (estimated_transfer_ms * 100.0).round() / 100.0,
            channel_allocations: allocations,
        }
    }

    /// Convenience helper that builds a realistic multi-homed channel schedule for a node cluster.
    pub fn benchmark_cluster_multipath(
        nodes: &[HardwareNode],
        payload_mb: f64,
    ) -> MultiPathStripedSchedule {
        let matrix = crate::build_mesh_link_matrix(nodes);
        Self::schedule_multipath_stripes(&matrix, payload_mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kadmium_core::sample_personal_cluster;

    #[test]
    fn test_raid5_linear_algebraic_parity_striping_and_straggler_recovery() {
        let total_tokens = 64usize; // e.g., 8 stripes of 8 tokens each
        let hidden_dim = 16usize;
        let out_dim = 16usize;
        let k = 8usize;

        let input_matrix: Vec<f32> = (0..total_tokens * hidden_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
            .collect();
        let weight_matrix: Vec<f32> = (0..hidden_dim * out_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
            .collect();

        // Simulate a 33 MB INT8 prefill across k=8 residential seeders where seeder #3 stalls:
        let (y_recovered, report) = SwarmRaidStriper::execute_raid5_coded_projection(
            &input_matrix,
            &weight_matrix,
            total_tokens,
            hidden_dim,
            out_dim,
            k,
            Some(3),
            33.0,
        );

        assert_eq!(y_recovered.len(), total_tokens * out_dim);
        assert_eq!(report.reconstructed_straggler_index, Some(3));
        assert!(
            report.max_reconstruction_error < 1e-4,
            "Linear RAID-5 parity recovery must be numerically exact, got err={}",
            report.max_reconstruction_error
        );
        // Verify >6x speedup over single residential uplink and >50% tail-latency reduction over RAID-0:
        assert!(report.speedup_vs_single_link > 6.5);
        assert!(report.tail_latency_reduction_pct > 60.0);
    }

    #[test]
    fn test_raid5_xor_page_parity_recovery_and_multipath_striper() {
        let page0 = vec![0xAAu8; 65_536];
        let page1 = vec![0x55u8; 65_536];
        let page2 = vec![0xF0u8; 65_536];
        let page3 = vec![0x0Fu8; 65_536];
        let parity = SwarmRaidStriper::encode_xor_parity_page(&[
            page0.clone(),
            page1.clone(),
            page2.clone(),
            page3.clone(),
        ]);

        // Lose page #2 (e.g., seeder #2 disconnects mid-transfer):
        let partial = vec![Some(page0), Some(page1), None, Some(page3)];
        let recovered_page2 = SwarmRaidStriper::reconstruct_missing_page_xor(&partial, &parity, 2);
        assert_eq!(recovered_page2, page2);

        let cluster = sample_personal_cluster();
        let mp = MultiPathInterfaceStriper::benchmark_cluster_multipath(&cluster, 33.0);
        assert!(mp.bonded_bandwidth_gbps > mp.single_best_link_gbps);
        assert!(mp.multi_path_speedup > 1.2);
    }
}
