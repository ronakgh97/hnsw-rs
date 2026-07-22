use crate::prelude::*;
use ahash::{HashMap, HashMapExt, HashSet};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use wincode::{SchemaRead, SchemaWrite};

// TODO; Store metadata separately from (item + graph) logic?
// like stored externally so then fetching metadata are not our concerns,
// also could be integrated with other storage backends :)

/// Simple bitset backed for tracking visited nodes during graph traversal instead of `HashSet`.
struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    #[inline(always)]
    fn with_capacity(n: usize) -> Self {
        let words = n.div_ceil(64);
        Self {
            bits: vec![0u64; words],
        }
    }

    /// Clears the bitset by resetting all bits to 0
    #[inline(always)]
    fn reset(&mut self) {
        self.bits.fill(0);
    }

    /// Returns true if the bit was newly set (not already set).
    #[inline(always)]
    fn insert(&mut self, idx: usize) -> bool {
        let word = idx >> 6;
        let bit_idx = idx & 63;
        let bit = 1u64 << bit_idx;

        unsafe {
            #[rustfmt::skip]
            let chunk =
                self.bits.get_unchecked_mut(word);

            let old = *chunk;
            *chunk = old | bit;
            (old & bit) == 0
        }
    }
}

/// Reusable scratch buffers for `search_layer_knn` to avoid per-call allocations.
struct SearchScratch {
    visited: BitSet,
    candidates: BinaryHeap<Candidate>,
    results: BinaryHeap<ScoredResult>,
    result_buf: Vec<(NodeIndex, f32)>,
}

impl SearchScratch {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            visited: BitSet::with_capacity(capacity),
            candidates: BinaryHeap::with_capacity(capacity),
            results: BinaryHeap::with_capacity(capacity),
            result_buf: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.visited.reset();
        self.candidates.clear();
        self.results.clear();
        self.result_buf.clear();
    }
}

#[derive(Clone, Copy)]
struct Candidate(NodeIndex, f32);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // pop() gives us the HIGHEST similarity candidate
        self.1.total_cmp(&other.1)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct ScoredResult(NodeIndex, f32);

impl PartialEq for ScoredResult {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Eq for ScoredResult {}

impl Ord for ScoredResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // peek() gives us the WORST result in our top-k
        other.1.total_cmp(&self.1)
    }
}

impl PartialOrd for ScoredResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub const DEFAULT_EF_MULTIPLIER: usize = 6;
pub const DEFAULT_EF_INC_FACTOR: f32 = 1.572;

/// Hierarchical Navigable Small World (HNSW) [Paper ref](https://arxiv.org/pdf/1603.09320)
///
/// Properties:
/// - **Hierarchical**: Multiple layers with exponentially decreasing nodes per layer
/// - **Navigable Small World**: Efficiently navigable graph structure at each layer
/// - **Logarithmic search complexity**: O(log N) by searching from top to bottom layers
/// - **Proper layer assignment**: Uses exponential distribution -ln(uniform) * mL
///
/// Algorithm highlights:
/// **Insert**: Search from top layer down, connect at each layer bidirectionally.
/// **Search**: Simple Greedy descent or Knn-search through upper layers (paper suggestion) and bounded Knn-search at bottom layer.
/// **Pruning**: Select neighbors using either simple top-M or heuristic for diversity (Alg. 3 vs Alg. 4).
/// **Tombstones**: Mark deleted nodes and ONLY skip during search end collection, periodic cleanup & reindexing.
///
/// # Vector HNSW Example
///
///```rust
///use hnsw_rs::prelude::*;
///
/// fn main() {
/// let mut hnsw = VectorHnsw::default(); // HNSW<FlatVectorStore>::default()
///
///    let (mut vectors, _seed) = gen_vec(10, 1024, 0); // vectors=10 d=1024 seed=0
///
///    for vector in vectors.iter_mut() {
///        let mut id = [0u8; 32];
///        fastrand::fill(&mut id); // elegant random 256-bit UUID
///        let level_asg = hnsw.get_random_level(); // random graph level for this node
///        let metadata = vec![];
///        hnsw.insert(id, vector, metadata, level_asg).unwrap(); // take &mut vector btw
///    }
///
///    assert_eq!(hnsw.size(), 10);
///    assert_eq!(hnsw.search(&mut vectors[5], 3, None).len(), 3);
/// }
///```
#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
pub struct HNSW<I: ItemBackend> {
    /// All nodes in the graph, not layer-wise
    node_list: Vec<Node>,
    /// Backend storage for the actual data space items (vectors, string, locations, images, AST, bioinformatics, any arbitrary quantity etc.)
    node_backend: I,
    /// ep; First node at the top layer, used as entry point for searches
    entry_point: Option<NodeIndex>,
    /// Lc; Total number of layers in the graph
    max_layers: usize,
    // TODO; Mmax, M which is the one? STUPID PARER!!!!!
    /// M; Degree of each node (max number of neighbors) per layer
    max_neighbors: usize,
    /// ef_const; More values explored during insertion means better chance of finding good neighbors
    ef_const: usize,
    /// mL; Level norm factor for exponential level distribution (paper section 4.1).
    /// Paper default is `1/ln(M)` where M is the max neighbors per layer (e.g. 0.36 for M=16).
    distribution_bias: f32,
    /// If true, `select_neighbors_heuristic` extends the candidate set with the neighbors of the
    /// candidates (paper Alg. 4). Useful for highly clustered data.
    extend_candidates: bool,
    /// If true, `select_neighbors_heuristic` adds discarded candidates back to reach `max_results`
    /// (paper Alg. 4). Ensures fixed number of connections per element.
    keep_pruned_connections: bool,
    /// If true (paper default), the upper-layer pass uses a simple greedy search
    /// (paper Alg. 5, ef=1: "avoid introduction of additional parameters").
    /// If false, the upper-layer pass uses `search_layer_knn` with ef=1 (BFS).
    use_simple_greedy_upper_layer: bool,
    /// If true, neighbor selection uses Alg. 4 (heuristic for diversity).
    /// If false (paper default), uses Alg. 3 (simple top-M by similarity).
    use_heuristic_selection: bool,
    /// Mapping from node uuid to array index
    id_mapper: HashMap<NodeUUID, NodeIndex>,
}

impl<I: ItemBackend + Default> Default for HNSW<I> {
    /// Default Config:
    /// - I::default()
    /// - max_neighbors=16
    /// - ef_construction=256
    /// - max_layers=18
    /// - distribution_bias=1/ln(16)
    /// - use_simple_greedy_search_upper_layer=true
    /// - use_heuristic_selection=true
    fn default() -> Self {
        let m = 16;
        let ml = 1.0 / (m as f32).ln();
        HNSW::new(I::default(), m, 256, 18, ml, 512_000, true, true)
    }
}

impl<I: ItemBackend> HNSW<I> {
    /// Creates a new HNSW instance with specified parameters.
    ///
    /// * `item_backend` — The storage backend that holds item data and computes distances (e.g. [`FlatVectorStore`]).
    /// * `max_neighbors` — M, the max number of neighbors per node per layer (paper default 16).
    /// * `ef_construction` — ef, size of the dynamic candidate list during insertion (paper default 200).
    /// * `max_layers` — Maximum number of layers in the graph.
    /// * `distribution_bias` — mL, level norm factor, controls how quickly layers thin out. Default is `1/ln(M)` ≈ 0.36 for M=16.
    /// * `with_capacity` — Pre-allocated capacity (expected number of nodes).
    /// * `use_simple_greedy_upper_layer` — true = simple greedy search (paper default), false = BFS with ef=1 for upper layers.
    /// * `use_heuristic_selection` — true = Alg. 4 heuristic neighbor selection (diversity), false = Alg. 3 top-M (paper default).
    ///
    /// Note: `extend_candidates` and `keep_pruned_connections` are hardcoded to `false` here; use [`HNSW::with_options`] to configure them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        item_backend: I,
        max_neighbors: usize,
        ef_construction: usize,
        max_layers: usize,
        distribution_bias: f32,
        with_capacity: usize,
        use_simple_greedy_upper_layer: bool,
        use_heuristic_selection: bool,
    ) -> Self {
        HNSW {
            node_list: Vec::with_capacity(with_capacity),
            node_backend: item_backend,
            entry_point: None,
            max_layers,
            max_neighbors,
            ef_const: ef_construction,
            distribution_bias,
            extend_candidates: false,
            keep_pruned_connections: false,
            use_simple_greedy_upper_layer,
            use_heuristic_selection,
            id_mapper: HashMap::with_capacity(with_capacity),
        }
    }

    /// Creates a new HNSW instance with full control over paper Alg. 4 flags.
    ///
    /// Same as [`HNSW::new`] but also lets you configure `extend_candidates` and `keep_pruned_connections`.
    ///
    /// * `extend_candidates` — If true, extends candidate set with neighbors of candidates (Alg. 4). Useful for highly clustered data.
    /// * `keep_pruned_connections` — If true, adds discarded candidates back to reach `max_results` (Alg. 4). Ensures fixed neighbor count.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        item_backend: I,
        max_neighbors: usize,
        ef_construction: usize,
        max_layers: usize,
        distribution_bias: f32,
        with_capacity: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
        use_simple_greedy_upper_layer: bool,
        use_heuristic_selection: bool,
    ) -> Self {
        HNSW {
            node_list: Vec::with_capacity(with_capacity),
            node_backend: item_backend,
            entry_point: None,
            max_layers,
            max_neighbors,
            ef_const: ef_construction,
            distribution_bias,
            extend_candidates,
            keep_pruned_connections,
            use_simple_greedy_upper_layer,
            use_heuristic_selection,
            id_mapper: HashMap::with_capacity(with_capacity),
        }
    }

    /// Generate a random graph level for a new node using an exponential distribution (paper section 4.1).
    /// Formula: `floor(-ln(U(0,1)) * mL)` where mL = `distribution_bias`.
    /// Returns a value in `0..max_layers`.
    ///
    /// Typically used to assign `max_level` when calling [`HNSW::insert`],
    /// or you can use your own curve to assign levels.
    #[inline(always)]
    pub fn get_random_level(&self) -> usize {
        let r: f32 = 1.0 - fastrand::f32();
        let level = (-r.ln() * self.distribution_bias).floor() as usize;
        level.min(self.max_layers - 1)
    }

    /// Insert a new node into the HNSW graph (paper Alg. 1).
    /// - If first node, just add it as entry point
    /// - Otherwise, search from top layer down to find nearest neighbors
    /// - Connect the new node to its neighbors at each layer
    ///
    /// * `id` — A unique 256-bit UUID for this node. Must not already exist in the graph.
    /// * `node` — The item data. Coerced to `&mut` so the [backend](ItemBackend) can potentially modify it if needed.
    /// * `metadata` — Opaque bytes stored alongside the node. Pass `vec![]` if unused.
    /// * `max_level` — The graph layer this node belongs to. Usually from [`HNSW::get_random_level`].
    ///
    /// Returns `Ok(NodeUUID)` — the same UUID that was passed in, on success, or `Err()` if
    /// [`validation`](ItemBackend::validate_item) rejects the item (wrong dimensions, format, etc.) and
    /// `id` is empty or already exist in the graph
    pub fn insert(
        &mut self,
        id: NodeUUID,
        node: &mut I::Item,
        metadata: Vec<u8>,
        max_level: usize,
    ) -> Result<NodeUUID> {
        if id.is_empty() {
            return Err(anyhow::anyhow!("NodeUUID cannot be empty"));
        }
        if self.id_mapper.contains_key(&id) {
            return Err(anyhow::anyhow!(
                "NodeUUID: {} already exists",
                hex::encode(id)
            ));
        }
        if !self.node_backend.validate_item(node) {
            return Err(anyhow::anyhow!("Node validation check failed"));
        }

        let node_id = self.node_list.len();

        // apply modification
        self.node_backend.insert_modify(node);

        // create node with empty neighbor lists
        // fill them in after finding neighbors
        let node = Node {
            uuid: id,
            metadata,
            neighbors: vec![Vec::with_capacity(self.max_neighbors); max_level + 1],
            max_level,
            tombstone: false,
        };

        // map id for reindexing helper
        self.id_mapper.insert(id, node_id);

        // set ep
        if self.entry_point.is_none() {
            self.node_list.push(node);
            self.entry_point = Some(node_id);
            return Ok(id);
        }

        // store the "node"
        self.node_list.push(node);

        // start search from entry point
        let mut current_nearest = self.entry_point.expect("Entry point is NONE");
        let entry_level = self.node_list[current_nearest].max_level;

        let mut scratch = SearchScratch::with_capacity(self.ef_const * 2);
        // Greedily traverse from top layer down to new node's level + 1
        // Just find the closest node, don't connect yet
        for layer in (max_level + 1..=entry_level).rev() {
            // Paper suggests simple greedy search for upper layers to avoid extra parameters,
            // but we can also do a quick BFS with ef=1 for potentially better navigation (especially for clustered data)
            if self.use_simple_greedy_upper_layer {
                current_nearest = self.search_layer_greedy(
                    self.node_backend.get(node_id),
                    current_nearest,
                    layer,
                );
            } else if let Some((nearest_id, _)) = self
                .search_layer_knn(
                    self.node_backend.get(node_id),
                    &[current_nearest],
                    1,
                    layer,
                    &mut scratch,
                )
                .first()
            {
                current_nearest = *nearest_id;
            }
        }

        // From new node's max_level down to 0, find neighbors and connect
        for layer in (0..=max_level).rev() {
            let max_n = self.max_neighbors_for_layer(layer);

            // Find ef_construction nearest neighbors at this layer
            let candidates = self.search_layer_knn(
                self.node_backend.get(node_id),
                &[current_nearest],
                self.ef_const,
                layer,
                &mut scratch,
            );

            // Apply neighbor selection (Alg. 3 simple or Alg. 4 heuristic)
            let selected = self.select_neighbors_dispatch(node_id, candidates, max_n, layer);

            // Connect new node to its neighbors (bidirectional)
            for &neighbor_id in &selected {
                if layer <= self.node_list[neighbor_id].max_level {
                    self.node_list[node_id].neighbors[layer].push(neighbor_id);
                    self.node_list[neighbor_id].neighbors[layer].push(node_id);
                    if self.node_list[neighbor_id].neighbors[layer].len()
                        > self.max_neighbors_for_layer(layer)
                    {
                        self.prune_connections(neighbor_id, layer);
                    }
                }
            }

            // Update current nearest for next layer
            if let Some((nearest_id, _)) = candidates.first() {
                current_nearest = *nearest_id;
            }
        }

        // If new node has higher max level than current entry point, update entry point
        if max_level > entry_level {
            self.node_list[current_nearest]
                .neighbors
                .resize(max_level + 1, Vec::new());
            self.node_list[current_nearest].max_level = max_level;
            for layer in (entry_level + 1)..=max_level {
                self.node_list[node_id].neighbors[layer].push(current_nearest);
                self.node_list[current_nearest].neighbors[layer].push(node_id);
            }
            self.entry_point = Some(node_id);
        }

        Ok(id)
    }

    /// Greedy search: find single closest node at a layer, used for navigating upper layers quickly
    /// `ALGORITHM 2 from paper suggestion`
    #[inline]
    fn search_layer_greedy(&self, query: &I::Item, entry: NodeIndex, layer: usize) -> NodeIndex {
        let mut current = entry;
        let mut current_sim = self
            .node_backend
            .similarity(query, self.node_backend.get(current));
        let mut improved = true;

        while improved {
            improved = false;

            // Check all neighbors at this layer
            if layer <= self.node_list[current].max_level {
                for &neighbor_id in &self.node_list[current].neighbors[layer] {
                    let neighbor_sim = self
                        .node_backend
                        .similarity(query, self.node_backend.get(neighbor_id));

                    if neighbor_sim > current_sim {
                        current = neighbor_id;
                        current_sim = neighbor_sim;
                        improved = true;
                    }
                }
            }
        }

        current
    }

    /// Selects the closest neighbors directly without applying any diversity heuristic.
    /// Just takes the top K the closest candidates.
    /// `ALGORITHM 3 from the paper suggestion`.
    #[inline]
    fn select_neighbors_simple(
        &self,
        candidates: &[(NodeIndex, f32)],
        max_results: usize,
    ) -> Vec<NodeIndex> {
        candidates
            .iter()
            .take(max_results)
            .map(|&(id, _)| id)
            .collect()
    }

    /// Iterates candidates sorted by similarity. For each candidate,
    /// checks if it's "redundant" — i.e., closer to an already-selected neighbor
    /// than the query node is to that neighbor. Skips redundant candidates to make spatial diversity in the neighbor set.
    /// `ALGORITHM 4 from the paper`.
    #[inline(always)]
    fn select_neighbors_heuristic(
        &self,
        query_id: NodeIndex,
        candidates: &[(NodeIndex, f32)],
        max_results: usize,
        layer: usize,
    ) -> Vec<NodeIndex> {
        let query_item = self.node_backend.get(query_id);
        let mut result: Vec<NodeIndex> = Vec::with_capacity(max_results);
        let mut discarded: Vec<NodeIndex> = Vec::new();
        let mut working: Vec<(NodeIndex, f32)> = candidates.to_vec();

        // TODO; this is bit unclear? with what paper says

        // Optionally extend candidates with neighbors of candidates (paper Alg. 4 lines 3-7)
        if self.extend_candidates {
            let original: Vec<NodeIndex> = candidates.iter().map(|&(id, _)| id).collect();
            for candidate_id in &original {
                if layer < self.node_list[*candidate_id].neighbors.len() {
                    for &adj in &self.node_list[*candidate_id].neighbors[layer] {
                        if self.node_list[adj].tombstone {
                            // should they be avoided? same logic as `search_layer_knn`
                            // which also skips tombstoned nodes when adding candidates
                            continue;
                        }
                        // avoid duplicates
                        if !working.iter().any(|(id, _)| *id == adj) {
                            // Compute similarity for the adjacent node and add to working set
                            let sim = self
                                .node_backend
                                .similarity(query_item, self.node_backend.get(adj));
                            working.push((adj, sim));
                        }
                    }
                }
            }
            // sort by similarity descending
            working.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }

        // iter candidates in order of similarity,
        // applying the heuristic to filter out "redundant" neighbors
        for &(candidate_id, _sim) in &working {
            if result.len() >= max_results {
                break;
            }
            let candidate_item = self.node_backend.get(candidate_id);

            // check if candidate is "redundant" with respect to already selected neighbors
            let mut is_good = true;
            for &selected_id in &result {
                let selected_item = self.node_backend.get(selected_id);
                let dist_candidate_to_selected =
                    self.node_backend.distance(candidate_item, selected_item);
                let dist_query_to_selected = self.node_backend.distance(query_item, selected_item);

                // redundant: if candidate is closer to selected neighbor than query is to that neighbor
                if dist_candidate_to_selected < dist_query_to_selected {
                    is_good = false;
                    break;
                }
            }

            if is_good {
                result.push(candidate_id);
            } else {
                discarded.push(candidate_id);
            }
        }

        // Optionally add back discarded candidates to maintain a fixed number of neighbors (paper Alg. 4 lines 15-17)
        if self.keep_pruned_connections && result.len() < max_results {
            for id in &discarded {
                if result.len() >= max_results {
                    break;
                }
                result.push(*id);
            }
        }

        result
    }

    /// Dispatches to either the heuristic or simple neighbor selection method based on configuration.
    #[inline(always)]
    fn select_neighbors_dispatch(
        &self,
        query_id: NodeIndex,
        candidates: &[(NodeIndex, f32)],
        max_results: usize,
        layer: usize,
    ) -> Vec<NodeIndex> {
        if self.use_heuristic_selection {
            self.select_neighbors_heuristic(query_id, candidates, max_results, layer)
        } else {
            self.select_neighbors_simple(candidates, max_results)
        }
    }

    /// K-NN search at a specific layer: find K nearest neighbors
    /// Uses a BOUNDED search with ef parameter (critical for performance)
    ///
    /// - Start with entry point in both candidates and results
    /// - Pop highest similarity candidate from heap (best-first)
    /// - If candidate is worse than our worst result, skip (prune)
    /// - Otherwise, explore all its neighbors
    /// - Add promising neighbors to candidates AND results (if better than worst)
    /// - Repeat until candidates empty
    /// - Return top-k results sorted by similarity with computed similarity
    ///
    /// COMPLEXITY: O(log n) per operation instead of O(n log n)
    /// Takes `&mut SearchScratch` to reuse allocations across calls.
    /// `ALGORITHM 5 from paper`
    #[inline]
    fn search_layer_knn<'scratch>(
        &self,
        query: &I::Item,
        entries: &[NodeIndex],
        ef: usize,
        layer: usize,
        scratch: &'scratch mut SearchScratch,
    ) -> &'scratch [(NodeIndex, f32)] {
        let ef = ef.max(1);
        let capacity = ef.saturating_mul(2).max(1);
        let max_nodes = self.node_list.len();

        scratch.clear();
        if scratch.visited.bits.len() * 64 < max_nodes {
            scratch.visited = BitSet::with_capacity(max_nodes);
        }
        if scratch.candidates.capacity() < capacity {
            scratch.candidates = BinaryHeap::with_capacity(capacity);
        }
        if scratch.results.capacity() < ef {
            scratch.results = BinaryHeap::with_capacity(ef);
        }

        // TODO; multiple eps_entries?
        // heaps with all entry points
        for &entry in entries {
            let entry_sim = self
                .node_backend
                .similarity(query, self.node_backend.get(entry));
            scratch.visited.insert(entry);
            scratch.candidates.push(Candidate(entry, entry_sim));
            scratch.results.push(ScoredResult(entry, entry_sim));
        }

        while let Some(Candidate(current_id, current_sim)) = scratch.candidates.pop() {
            if let Some(worst_result) = scratch.results.peek()
                && scratch.results.len() >= ef
                && current_sim < worst_result.1
            {
                break;
            }

            if layer <= self.node_list[current_id].max_level {
                for &neighbor_id in &self.node_list[current_id].neighbors[layer] {
                    if scratch.visited.insert(neighbor_id) {
                        let sim = self
                            .node_backend
                            .similarity(query, self.node_backend.get(neighbor_id));

                        let worst_if_full = scratch.results.peek().map(|r| r.1);
                        let should_add = match (scratch.results.len(), worst_if_full) {
                            (len, _) if len < ef => true,            // still filling
                            (_, Some(worst)) if sim > worst => true, // better than worst
                            _ => false,
                        };

                        if should_add {
                            scratch.candidates.push(Candidate(neighbor_id, sim));
                            scratch.results.push(ScoredResult(neighbor_id, sim));

                            if scratch.results.len() > ef {
                                scratch.results.pop();
                            }
                        }
                    }
                }
            }
        }

        // drain results into the reusable buffer, sort, and return
        scratch.result_buf.clear();
        for ScoredResult(id, sim) in scratch.results.drain() {
            scratch.result_buf.push((id, sim));
        }
        scratch
            .result_buf
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        &scratch.result_buf
    }

    /// Remove connections to keep only the M closest neighbors
    #[inline(always)]
    fn prune_connections(&mut self, node_id: NodeIndex, layer: usize) {
        let max_n = self.max_neighbors_for_layer(layer);
        if self.node_list[node_id].neighbors[layer].len() <= max_n {
            return;
        }

        // Store the old neighbor list to identify which edges to remove
        let old_neighbors: HashSet<NodeIndex> = self.node_list[node_id].neighbors[layer]
            .iter()
            .copied()
            .collect();

        // Calculate similarities to all neighbors, filtering out tombstoned nodes
        let node_item = self.node_backend.get(node_id);
        let mut neighbor_sims: Vec<(NodeIndex, f32)> = self.node_list[node_id].neighbors[layer]
            .iter()
            .filter(|&&n| !self.node_list[n].tombstone)
            .map(|&n| {
                let sim = self
                    .node_backend
                    .similarity(node_item, self.node_backend.get(n));
                (n, sim)
            })
            .collect();

        // Sort by similarity descending (best first)
        neighbor_sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Do it again here since selection is based on pruned sims, not original sims
        let new_neighbors = self.select_neighbors_dispatch(node_id, &neighbor_sims, max_n, layer);
        let new_neighbors_set: HashSet<NodeIndex> = new_neighbors.iter().copied().collect();

        let removed_neighbors: Vec<NodeIndex> = old_neighbors
            .difference(&new_neighbors_set)
            .copied()
            .collect();

        // Update this node's neighbor list
        self.node_list[node_id].neighbors[layer] = new_neighbors;

        // Remove reverse edges from pruned neighbors to maintain bidirectionality
        for removed_neighbor_id in removed_neighbors {
            if layer <= self.node_list[removed_neighbor_id].max_level {
                self.node_list[removed_neighbor_id].neighbors[layer].retain(|&n| n != node_id);
            }
        }
    }

    /// Returns the max number of neighbors for a given layer.
    /// Layer 0 uses M0 = 2*M per the paper (Section 4.1).
    #[inline(always)]
    fn max_neighbors_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.max_neighbors * 2
        } else {
            self.max_neighbors
        }
    }

    /// Internal search method that performs the actual HNSW search, other `[search_*]` fn uses this internally
    /// Returns (index, similarity) of candidates, including `tombstoned` nodes or empty vec otherwise
    /// `ef` is here `explorable nodes limit`, internally does ef =~ cK, because, we want to have a `'good'` chance of finding K if not more than K `'good'` nodes
    /// > Time ~ O(log n) + O(ef log ef) + O(k) <- for final truncation, but usually ef dominates
    #[inline(always)]
    pub fn search_kernel(&self, query: &mut I::Item, k: usize, ef: usize) -> Vec<(NodeIndex, f32)> {
        let entry = if let Some(entry) = self.entry_point {
            if self.node_list.get(entry).is_none()
                || self.node_list[entry].tombstone
                || !self.node_backend.validate_item(query)
            {
                return Vec::new();
            }
            entry
        } else {
            return Vec::new();
        };

        if k == 0 || ef == 0 {
            return Vec::new();
        }

        // apply modification
        self.node_backend.search_modify(query);

        let entry_level = self.node_list[entry].max_level;
        let mut current = entry;
        let mut scratch = SearchScratch::with_capacity(ef * 2);

        // Traverse from top to layer 1
        for layer in (1..=entry_level).rev() {
            if self.use_simple_greedy_upper_layer {
                current = self.search_layer_greedy(query, current, layer);
            } else if let Some((nearest_id, _)) = self
                .search_layer_knn(query, &[current], 1, layer, &mut scratch)
                .first()
            {
                current = *nearest_id;
            }
        }

        // search layer 0 thoroughly for K neighbors
        // `search_layer_knn` already returns sorted results
        // return full ef results, search() handles truncation after filtering tombstones
        self.search_layer_knn(query, &[current], ef, 0, &mut scratch)
            .to_vec()
    }

    /// Finds topK nearest neighbors to a query, if `ef_search` is None then, internally does a loop increase base ef for better odds.
    /// Returns results as (NodeIndex, f32) tuples sorted by similarity (highest first), empty if no entry point or k=0.
    /// - `query` - The &mut query item for optional pre-processing & stuffs
    /// - `k` - Number of nearest neighbors to return
    /// - `ef_search` - Optional bounded width for search. If None, uses k * DEFAULT_EF_MULTIPLIER
    pub fn search(
        &self,
        query: &mut I::Item,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeUUID, f32)> {
        self.search_helper(query, k, ef_search)
            .into_iter()
            .map(|(id, sim)| (self.node_list[id].uuid, sim))
            .collect()
    }

    /// Like [`HNSW::search`] but also returns the `"lazily collected"` metadata bytes stored with each node.
    /// Returns `Vec<(NodeUUID, f32, Vec<u8>)>` sorted by similarity (highest first).
    pub fn search_metadata(
        &self,
        query: &mut I::Item,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeUUID, f32, Vec<u8>)> {
        self.search_helper(query, k, ef_search)
            .into_iter()
            .map(|(id, sim)| {
                let node = &self.node_list[id];
                (node.uuid, sim, node.metadata.clone())
            })
            .collect()
    }

    /// Internal helper: loop that grows `ef` until we have `k` non-tombstoned results and the
    /// kth similarity "stabilizes" (or we hit `max_ef`).
    #[inline(always)]
    fn search_helper(
        &self,
        query: &mut I::Item,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeIndex, f32)> {
        if k == 0 || self.entry_point.is_none() {
            return Vec::new();
        }

        let ef = ef_search.unwrap_or(k * DEFAULT_EF_MULTIPLIER);
        let mut current_ef = ef.max(k).max(1);
        let max_ef = if ef_search.is_some() {
            current_ef
        } else {
            self.node_list.len().max(ef)
        };
        let mut prev_kth_sim: f32 = f32::NEG_INFINITY;

        loop {
            let results = self.search_kernel(query, k, current_ef);

            let active_indices: Vec<_> = results
                .into_iter()
                .filter(|(id, _)| !self.node_list[*id].tombstone)
                .collect();

            let have_k = active_indices.len() >= k;
            let kth_sim = if have_k {
                active_indices[k - 1].1
            } else if let Some((_, sim)) = active_indices.last() {
                *sim
            } else {
                f32::NEG_INFINITY
            };

            let stable_improved = have_k && kth_sim <= prev_kth_sim;
            if current_ef >= max_ef || stable_improved {
                return active_indices.into_iter().take(k).collect();
            }

            prev_kth_sim = kth_sim;
            let grown = ((current_ef as f32) * DEFAULT_EF_INC_FACTOR).ceil() as usize;
            current_ef = grown.max(current_ef + 1).min(max_ef);
        }
    }

    /// Brute-force parallel search for testing and validation.
    /// Scans *all* non-tombstoned nodes using rayon, so O(N). Use [`HNSW::search`] for real usage.
    #[inline]
    pub fn brute_search(&self, query: &mut I::Item, k: usize) -> Vec<(NodeUUID, f32)> {
        use rayon::iter::*;
        self.node_backend.search_modify(query);
        let mut results: Vec<(NodeUUID, f32)> = self
            .node_list
            .par_iter()
            .enumerate()
            .filter(|(_, node)| !node.tombstone)
            .map(|(i, node)| {
                (
                    node.uuid,
                    self.node_backend
                        .similarity(query, self.node_backend.get(i)),
                )
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        results
    }

    /// Brute-force parallel search with metadata included in results.
    /// Returns similar to [HNSW::search_metadata].
    #[inline]
    pub fn brute_search_metadata(
        &self,
        query: &mut I::Item,
        k: usize,
    ) -> Vec<(NodeUUID, f32, Vec<u8>)> {
        use rayon::iter::*;
        self.node_backend.search_modify(query);
        let mut results: Vec<(NodeUUID, f32, Vec<u8>)> = self
            .node_list
            .par_iter()
            .enumerate()
            .filter(|(_, node)| !node.tombstone)
            .map(|(i, node)| {
                (
                    node.uuid,
                    self.node_backend
                        .similarity(query, self.node_backend.get(i)),
                    node.metadata.clone(),
                )
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        results
    }

    /// Get entry point node, returns None if no entry point or if entry point is tombstoned
    #[inline]
    pub fn get_entry_point(&self) -> Option<&Node> {
        self.entry_point
            .and_then(|id| self.node_list.get(id))
            .filter(|&node| !node.tombstone)
    }

    /// Get node by node ID, returns None if not found or tombstoned
    #[inline]
    pub fn get_node(&self, uuid: &[u8; 32]) -> Option<&Node> {
        self.id_mapper
            .get(uuid)
            .and_then(|&id| self.node_list.get(id))
            .filter(|&node| !node.tombstone)
    }

    /// Convenience method to get node by index, returns None if not found
    #[inline]
    pub fn get_node_by_index(&self, idx: usize) -> Option<&Node> {
        self.node_list.get(idx)
    }

    /// Get all nodes present in a specific layer
    /// (i.e. nodes with max_level >= given level are present),
    /// including tombstoned ones, returns empty vec if level out of bounds
    #[inline]
    pub fn get_nodes_at_level(&self, level: usize) -> Vec<&Node> {
        if level > self.max_layers {
            return Vec::new();
        }
        self.node_list
            .iter()
            .filter(|node| node.max_level >= level)
            .collect()
    }

    /// Returns the item for the given node UUID, or None if not found or tombstoned
    #[inline]
    pub fn get_item(&self, uuid: &[u8; 32]) -> Option<&I::Item> {
        self.id_mapper
            .get(uuid)
            .and_then(|&id| self.node_list.get(id).map(|node| (id, node)))
            .filter(|(_, node)| !node.tombstone)
            .map(|(id, _)| self.node_backend.get(id))
    }

    /// Returns the item for the given node index, or None if not found
    #[inline]
    pub fn get_item_by_index(&self, idx: NodeIndex) -> Option<&I::Item> {
        if idx < self.node_list.len() {
            Some(self.node_backend.get(idx))
        } else {
            None
        }
    }

    /// Returns all items present in a specific layer (i.e. nodes with max_level >= given level are present),
    /// including tombstoned ones, returns empty vec if level out of bounds.
    #[inline]
    pub fn get_items_at_level(&self, level: usize) -> Vec<&I::Item> {
        if level > self.max_layers {
            return Vec::new();
        }
        self.node_list
            .iter()
            .enumerate()
            .filter(|(_, node)| node.max_level >= level)
            .map(|(idx, _)| self.node_backend.get(idx))
            .collect()
    }

    /// Soft-deletes (tombstones) a node by its UUID.
    /// The node's graph entry is kept but skipped in search end results. If the deleted node was the entry point, a replacement is found.
    /// Use [`HNSW::tombstone_ratio`] to monitor accumulated tombstones and trigger reindexing.
    /// Returns `Err` if the UUID does not exist in the graph.
    #[inline(always)]
    pub fn delete_node(&mut self, uuid: &[u8; 32]) -> Result<()> {
        let node_idx = if let Some(idx) = self.id_mapper.get(uuid).copied() {
            if let Some(node) = self.node_list.get_mut(idx) {
                // rm from id_mapper
                node.tombstone = true;
            }
            idx
        } else {
            return Err(anyhow::anyhow!("NodeUUID: {} not found", hex::encode(uuid)));
        };

        // find and sets new entry point when the current one is deleted,
        if let Some(entry) = self.entry_point
            && entry == node_idx
        {
            // searches from max_layer down to find the highest-level active node
            // should be fine, just take any active node from top layer right?...right?
            for layer in (0..self.max_layers).rev() {
                for (id, node) in self.node_list.iter().enumerate() {
                    if node.max_level == layer && !node.tombstone {
                        self.entry_point = Some(id);
                        return Ok(());
                    }
                }
            }
            // no active nodes found
            self.entry_point = None;
        }

        Ok(())
    }

    /// Get the index configuration parameters: (max_layers, max_neighbors, ef_construction)
    #[inline]
    pub fn index_config(&self) -> (usize, usize, usize) {
        (self.max_layers, self.max_neighbors, self.ef_const)
    }

    /// Returns the total count of nodes in the graph, including tombstoned ones
    #[inline]
    pub fn size(&self) -> usize {
        self.node_list.len()
    }

    /// Returns the total memory usage of the graph in bytes, including everything
    #[inline]
    pub fn mem_size(&self) -> usize {
        use rayon::iter::*;
        let node_size = size_of::<Node>();
        let neighbors_size: usize = self
            .node_list
            .par_iter()
            .map(|node| {
                node.neighbors
                    .iter()
                    .map(|layer| layer.len() * size_of::<NodeIndex>())
                    .sum::<usize>()
            })
            .sum();

        self.node_list.len() * node_size + neighbors_size + self.node_backend.mem_size()
    }

    #[inline]
    /// Returns the count of active (non-tombstoned) nodes
    pub fn active_count(&self) -> usize {
        self.node_list.iter().filter(|node| !node.tombstone).count()
    }

    #[inline]
    /// Returns the count of tombstoned (deleted) nodes
    pub fn tombstone_count(&self) -> usize {
        self.node_list.iter().filter(|node| node.tombstone).count()
    }

    #[inline]
    /// Returns the ratio of tombstoned nodes to total nodes (float division, 0..=1).
    /// Can be used in trigger when to clean up & reindex
    pub fn tombstone_ratio(&self) -> f64 {
        if self.node_list.is_empty() {
            0.0
        } else {
            self.tombstone_count() as f64 / self.node_list.len() as f64
        }
    }

    // // TODO; impl this correctly and nicely
    // /// Debug method to print summary of the graph,
    // /// including entry point info and layer distribution, quality,
    // /// Optionally takes a NodeUUID for deep recursively insight.
    // pub fn stats(&self, _uuid: Option<&[u8; 32]>) {
    //     println!("Layer distribution:");
    //     for layer in 0..=self.max_layers {
    //         let count = self.get_nodes_at_level(layer).len();
    //         if count > 0 {
    //             println!(
    //                 "-L{}: {} nodes ({}%)",
    //                 layer,
    //                 count,
    //                 if self.nodes.is_empty() {
    //                     0.0
    //                 } else {
    //                     (count as f32 / self.nodes.len() as f32) * 100.0
    //                 }
    //             );
    //         }
    //     }
    //     println!();
    //
    //     if let Some(ep_idx) = self.entry_point
    //         && let Some(ep) = self.nodes.get(ep_idx)
    //     {
    //         println!(
    //             "Entry point: uuid={} idx={} max_level={}",
    //             &hex::encode(ep.uuid),
    //             ep_idx,
    //             ep.max_level
    //         );
    //         for (level, neighbour) in ep.neighbors.iter().enumerate() {
    //             println!(
    //                 "-L{} count={} neightbour_idx={:?}",
    //                 level,
    //                 neighbour.len(),
    //                 neighbour
    //             );
    //         }
    //     } else {
    //         println!("No entry point");
    //     }
    //     println!();
    // }
}

/// Alias for node index type, used for indexing into the `node_list`.
pub type NodeIndex = usize;

/// Unique 256-bit identifier for a node, provided at insertion time.
/// Stable across reindexing — the same UUID always refers to the same "logical" item.
pub type NodeUUID = [u8; 32];

#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
/// A node in the HNSW graph: UUID, metadata, neighbor lists per layer, and a tombstone flag.
pub struct Node {
    /// Unique Stable Identifier for the node, provided during insertion
    pub uuid: NodeUUID,
    /// Metadata associated with the node
    pub metadata: Vec<u8>,
    /// Neighbors per layer, e.g `neighbors[0]` is the list of neighbors in layer 0
    pub neighbors: Vec<Vec<NodeIndex>>,
    /// The highest layer this node exists in
    pub max_level: usize,
    /// Flag for lazy deletion
    tombstone: bool,
}

impl Node {
    /// Returns true if this node has been soft-deleted (tombstoned).
    #[inline(always)]
    pub fn is_deleted(&self) -> bool {
        self.tombstone
    }
}
