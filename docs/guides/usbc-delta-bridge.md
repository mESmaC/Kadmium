# Guide: USB-C Delta Bridges

A USB-C bridge ([`usbc_bridge.rs`](../../crates/kadmium-network/src/usbc_bridge.rs)) carries one ordered byte stream between two nodes over several lanes at once, typically a USB4 / Thunderbolt or USB-NCM cable plus a Wi-Fi or Ethernet path. Delta links run on top of it unchanged. Pulling the cable therefore costs only the segments that were in flight, not every chunk the peer already held.

---

## 1. Why a bridge

[Delta links](../ARCHITECTURE.md#93-transport) already send each frame as only the chunks the receiver does not hold. Their chunk stores belong to one TCP connection, though. When the USB-C cable carrying that connection is pulled:

1. The connection dies, and every in-flight call fails.
2. The pool redials over Wi-Fi, and the new connection starts with empty stores on both sides.
3. The next frames resend everything: a 4 MiB stage context, 16 MiB of weights, KV pages the peer already had.

InferSSD's device staging solves the same problem for source trees. Since InferSSD PR #91, each device keeps the commits it has received, and the next bundle leaves out everything that anchor already holds. For PR #76, the staged sources shrank from two archives of about 35 MB each to a 48 KB bundle. A bridge applies that idea to links. The receiver's anchor is the sequence number of the next segment it lacks. A lane that fails, or a cable replugged within the resume grace, resends only the segments after it. The delta stores never see a gap, so they stay valid.

## 2. How it works

| Piece | Behavior |
| :--- | :--- |
| **Segments** | The sender cuts the stream into segments of up to 64 KiB (the USB4 IP jumbo frame and the largest CDC-NCM transfer block). It coalesces small writes the way an NCM transfer block packs datagrams, without waiting for more. |
| **Lane choice** | Each segment goes to the live lane that would finish it first: `(queued bytes + segment) / weight`. Small, latency-bound frames stay on the fastest cable; bulk transfers spread across every lane in proportion to how fast each one drains. |
| **Reliability** | The receiver reorders segments and returns a cumulative ACK on every lane. The sender keeps unacknowledged segments, up to a window (16 MiB by default), and resends them on other lanes when their lane fails. The receiver drops duplicates. |
| **Unplug detection** | A lane fails on a write error, a reset, or 750 ms with no data and no heartbeat (every lane sends one each 100 ms). A pulled cable usually shows up as silence rather than a reset. |
| **Resume** | The dialing side redials failed lanes with backoff (25 ms to 1 s) into the same session. A session ends only after no lane has been up for the resume grace (30 s by default). |
| **Hot-plug** | `UsbCBridge::add_lane` adds a cable plugged in mid-session; `set_lane_enabled` takes one out of service. |
| **Joining** | The session id is 128 random bits, but it crosses every lane in plaintext. Joining also takes a 32-byte resume secret. The listener returns it only on the lane that creates the session, which the dialer picks as its heaviest lane (normally the cable). Every other lane and every redial proves it holds the secret with an HMAC-SHA256 challenge-response in both directions. Someone who sniffs a hello on Wi-Fi therefore cannot inject segments, draw replies, or take over a lane index, and a hello that tries to create a session under an existing id is refused. |
| **Safety** | A dialer sends the create flag only until it holds the secret, and drops a lane whose create succeeds after that. A listener that restarted therefore answers "unknown session" instead of silently starting an empty stream the peer's delta stores would no longer match. Lanes cannot attach to a session that has closed. Out-of-window sequence numbers, oversized segments and unknown tags close the session. A worker closes its bridge when the stream breaks, as does a `PeerChannel` that fails or is closed. |

The lane wire format is documented at the top of `usbc_bridge.rs`.

## 3. Using it

On the worker, accept bridge sessions alongside the normal TCP listener:

```sh
KADMIUM_USBC_BRIDGE_BIND=0.0.0.0:7871 kadmium join --ticket kdm1_...
```

From code, call `LiveNodeWorkerServer::spawn_usbc_bridge_listener("0.0.0.0:7871", true)`.

On the coordinator, address the worker with a `kdbr://` address anywhere a worker address goes through the `PeerChannelPool`. List one `host:port` per lane, each with an optional weight in Gbit/s:

```text
kdbr://169.254.78.11:7871@4.6,169.254.78.11:7871@4.6,169.254.78.11:7871@4.6,192.168.1.20:7871@1.2
```

List the USB4 address several times. Each lane is its own TCP connection, and one connection does not fill a USB4 link. On Windows loopback, one lane moved 2 to 6 Gbit/s while four lanes moved 18 to 24 Gbit/s, against 8 to 27 Gbit/s for a raw TCP socket on the same machine. `UsbCDirectInterconnectScanner::plan_delta_bridge` (reported as `delta_bridge_plan` by `GET /api/v1/network/usbc`) opens one lane per DMA stream and sizes the window. It sizes the window to cover the USB-C lanes while a segment crosses the slowest backup lane.

## 4. Measured results

`cargo run --release -p kadmium-network --example usbc_bridge_bench -- serve 0.0.0.0:7880` on the worker and `-- run <host>:7880 kdbr://...:7881,...:7881` on the coordinator. Each scenario runs 12 steps. Before step 4 the first path is lost: the plain channel is closed and redialed, and lane 0 of the bridge is disabled. Every reply carries the XXH64 of the rebuilt payload, and every one matched.

Over Wi-Fi to an M4 Mac mini on 2026-10-30, three runs. Both bridge lanes used the same Wi-Fi path; no USB-C cable connected the two machines, so this measures the unplug cost, not USB4 throughput.

| Scenario | Bytes sent after the loss (plain → bridge) | Time after the loss (plain → bridge) |
| :--- | :--- | :--- |
| Decode step: header, fresh 28 KiB activations, unchanged 4 MiB context | 4.33 MiB → 0.35 MiB (12.5× fewer) | 317–337 ms → 151–159 ms (2.0–2.2× faster) |
| Weights re-sent: 16 MiB with a changing header | 16.33 MiB → 0.37 MiB (44× fewer) | 845–920 ms → 283–348 ms (2.7–3.0× faster) |

The loss happened between calls, so nothing was in flight and nothing was retransmitted. The unit test `unplugging_a_lane_mid_transfer_loses_nothing` covers a lane pulled mid-transfer, and `delta_stores_survive_an_unplug_so_only_changes_cross_the_backup` covers the delta link across it.

## 5. USB-C transport research (October 2026)

| Path | Platforms | Status in Kadmium |
| :--- | :--- | :--- |
| USB4 / Thunderbolt IP networking | Linux `thunderbolt-net`, macOS Thunderbolt Bridge, Windows 11 `USB4P2PNetAdapter` | Bridge lanes run over it (`UsbCTransportPath::Usb4IpNetwork`). Linux patches in 2026 enlarge the `thunderbolt-net` rings from 256 to 2048 entries after NCCL all-reduce over a three-node chain dropped about 1% of packets. |
| CDC-NCM | USB 3.x gadgets (`usb_f_ncm`) to Linux, macOS and Windows hosts | Bridge lanes run over it (`CdcNcmIp`). Transfer blocks default to 16 KiB; 64 KiB needs the gadget to advertise it and the host to raise `cdc_ncm` `rx_max` / `tx_max`. |
| USB4STREAM | Linux 7.2 `thunderbolt_stream`: raw DMA tunnels at `/dev/tbstreamX`, configured under `/sys/kernel/config/thunderbolt/stream/` | Detected (`Usb4RawStream`). Intel reports about 2.8 GB/s on Thunderbolt 4 without the network stack. A lane type for it is future work. |
| RDMA over Thunderbolt 5 | macOS 26.2+, enabled with `rdma_ctl enable` from recovery; used by MLX's JACCL backend, which picks a mesh topology when latency dominates and a ring when bandwidth does | Detected when `rdma_ctl status` prints `enabled` (`RdmaThunderbolt5`); the M4 Mac mini used above reports `disabled`. A lane type for it is future work. |

## 6. Limits

- The figures in `UsbCDeltaBridgePlan` are model estimates from tier goodput and RTT, not measurements.
- Lanes are TCP. A bridge does not bypass the OS network stack; USB4STREAM and RDMA lane types would.
- The resume secret crosses the lane that creates the session. If the cable cannot connect at session start, that lane is Wi-Fi, and someone sniffing it then could join later.
- Lanes are not encrypted, and segment integrity rests on TCP checksums. An attacker on the path can read or alter traffic, as with plain TCP and delta links (whose XXH64 keys are not cryptographic). Use bridges between trusted nodes on trusted links.
- Each lane copies a segment a few more times than a raw socket. On a single lane that costs throughput on fast links, which is why the plan opens several lanes per cable.

Sources: [Apple WWDC26 session 233](https://developer.apple.com/videos/play/wwdc2026/233/), [MLX JACCL RDMA guide](https://upd.dev/ml-explore/mlx/issues/3207), [USB4STREAM in Linux 7.2](https://news.lavx.hu/article/intel-s-usb4stream-protocol-set-to-revolutionize-direct-host-to-host-communication-in-linux-7-2), [thunderbolt-net ring patch](https://lkml.iu.edu/hypermail/linux/kernel/2604.3/07551.html), [Windows USB4 connection manager](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/usb4-intro-to-connection-manager), [cdc_ncm 64 KiB NTB change](https://lkml.rescloud.iu.edu/2208.1/04047.html).
