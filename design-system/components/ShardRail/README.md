# ShardRail

The brand's mark at data scale: a rail with one rod per node, each rod's depth proportional to that node's share of the plan. It replaces pies and stacked bars wherever something is split across heterogeneous nodes — layers, experts, micro-batches, verifier work.

You provide `nodes` (`name`, `share` 0–1, optional `placement: "cloud"` for a cherenkov rod) in flow order, a `label` for assistive tech, and optionally `pitch` (a multiple of 8) and `depth`. The deepest rod reaches the full depth; a zero share draws a short graphite stub so the node is still counted. Labels are mono uppercase with the percentage beneath; hide it with `showShare={false}` when the numbers appear in a table beside the rail.

Don't: animate rods on load; sort by share (flow order carries meaning); use it for a single quantity (that is a `Meter`).
