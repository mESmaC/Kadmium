//! Monte Carlo Tree Search (MCTS) + PUCT Engine for Dream-RSI Discovery Trees (`kadmium-rl::mcts`).

use crate::DiscoveryTree;
use serde::{Deserialize, Serialize};

/// Per-node visit and Q-value statistics maintained by `DreamRsiMctsEngine`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsNodeStats {
    pub node_id: u32,
    pub visits: u32,
    pub total_value: f64,
    pub mean_q_value: f64,
}

/// Complete report from running PUCT / UCB1 Monte Carlo Tree Search over a `DiscoveryTree`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsSearchReport {
    pub iterations_run: u32,
    pub c_puct: f64,
    pub best_leaf_node_id: u32,
    pub best_leaf_score: f64,
    pub optimal_path_node_ids: Vec<u32>,
    pub node_stats: Vec<MctsNodeStats>,
}

/// Monte Carlo Tree Search (MCTS) + PUCT engine for Dream-RSI causal discovery trees.
///
/// Uses the PUCT selection rule:
/// $$\text{PUCT}(u) = Q(u) + c_{\text{puct}} \cdot P_{\text{novelty}}(u) \cdot \frac{\sqrt{N(\text{parent}(u))}}{1 + N(u)}$$
pub struct DreamRsiMctsEngine;

impl DreamRsiMctsEngine {
    pub fn run_puct_search(tree: &DiscoveryTree, iterations: u32, c_puct: f64) -> MctsSearchReport {
        let n = tree.nodes.len();
        let mut visits = vec![0u32; n];
        let mut total_val = vec![0.0f64; n];

        let children_of = |parent_id: u32| -> Vec<usize> {
            tree.nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| nd.parent_id == Some(parent_id))
                .map(|(idx, _)| idx)
                .collect()
        };

        for _iter in 0..iterations {
            // 1. Selection: walk from root (index 0) down to a leaf or unvisited child using PUCT
            let mut curr_idx = 0usize;
            let mut path = vec![0usize];

            loop {
                let curr_node_id = tree.nodes[curr_idx].node_id;
                let kids = children_of(curr_node_id);
                if kids.is_empty() {
                    break;
                }
                // If any child has 0 visits, expand the highest-novelty unvisited child first
                if let Some(&unvisited_idx) =
                    kids.iter().filter(|&&k| visits[k] == 0).max_by(|&&a, &&b| {
                        tree.nodes[a]
                            .prior_novelty
                            .partial_cmp(&tree.nodes[b].prior_novelty)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    curr_idx = unvisited_idx;
                    path.push(curr_idx);
                    break;
                }

                let parent_visits = (visits[curr_idx].max(1)) as f64;
                let best_kid = *kids
                    .iter()
                    .max_by(|&&a, &&b| {
                        let qa = total_val[a] / (visits[a].max(1) as f64);
                        let qb = total_val[b] / (visits[b].max(1) as f64);
                        let ua = qa
                            + c_puct
                                * tree.nodes[a].prior_novelty
                                * (parent_visits.sqrt() / (1.0 + visits[a] as f64));
                        let ub = qb
                            + c_puct
                                * tree.nodes[b].prior_novelty
                                * (parent_visits.sqrt() / (1.0 + visits[b] as f64));
                        ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(&kids[0]);

                curr_idx = best_kid;
                path.push(curr_idx);
            }

            // 2. Evaluation: read realized score at selected node
            let leaf_reward = tree.nodes[curr_idx].realized_score;

            // 3. Backpropagation: update visits and total_val along `path`
            for &idx in &path {
                visits[idx] += 1;
                total_val[idx] += leaf_reward;
            }
        }

        // Reconstruct greedy highest-visit / highest-Q trajectory from root
        let mut optimal_path = vec![tree.nodes[0].node_id];
        let mut curr_id = tree.nodes[0].node_id;
        let mut best_leaf_score = tree.nodes[0].realized_score;

        loop {
            let kids = children_of(curr_id);
            if kids.is_empty() {
                break;
            }
            let best_kid = *kids
                .iter()
                .max_by(|&&a, &&b| {
                    let qa = total_val[a] / (visits[a].max(1) as f64);
                    let qb = total_val[b] / (visits[b].max(1) as f64);
                    qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&kids[0]);
            curr_id = tree.nodes[best_kid].node_id;
            best_leaf_score = tree.nodes[best_kid].realized_score;
            optimal_path.push(curr_id);
        }

        let node_stats = tree
            .nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| {
                let q = if visits[i] > 0 {
                    total_val[i] / (visits[i] as f64)
                } else {
                    0.0
                };
                MctsNodeStats {
                    node_id: nd.node_id,
                    visits: visits[i],
                    total_value: (total_val[i] * 100.0).round() / 100.0,
                    mean_q_value: (q * 1000.0).round() / 1000.0,
                }
            })
            .collect();

        MctsSearchReport {
            iterations_run: iterations,
            c_puct,
            best_leaf_node_id: curr_id,
            best_leaf_score,
            optimal_path_node_ids: optimal_path,
            node_stats,
        }
    }
}
