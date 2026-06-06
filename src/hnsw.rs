use crate::maths::{Metrics, dot_product, euclidean_similarity, normalize_l2};
use crate::prelude::gen_vec;
use ahash::{HashMap, HashMapExt, HashSet};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use wincode::{SchemaRead, SchemaWrite};

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
        let word = idx / 64;
        let bit = 1u64 << (idx % 64);
        let chunk = &mut self.bits[word];
        let was_set = (*chunk & bit) != 0;
        *chunk |= bit;
        !was_set
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
/// **Insert**: Search from top layer down, connect at each layer bidirectionally
/// **Search**: Greedy descent through upper layers, bounded search at bottom layer
/// **Pruning**: Keep only M closest neighbors per node per layer
/// **Tombstones**: Mark deleted nodes and skip during search, periodic cleanup & reindexing
///
///```rust
///use hnsw_rs::prelude::*;
///use fastrand::fill;
///
/// fn main() {
/// let mut hnsw = HNSW::default();
///
///    let mut vectors = vec![
///                    vec![1.0, 0.0, 0.0],
///                    vec![1.41, 1.41, 0.0],
///                    vec![0.0, 1.0, 0.0],
///                    vec![0.0, 1.41, 1.41],
///                    vec![0.0, 0.0, 1.0]
///                    ];
///
///    for vector in vectors.iter_mut() { // need mut for l2 normalize
///        let mut id = [0u8; 32];
///        fastrand::fill(&mut id); // 256-bit random [u8; 32]
///        let level_asg = hnsw.get_random_level(); // compute mL from M
///        let metadata = vec![]; // metadata can be anything (std vec)
///        hnsw.insert(id, vector, metadata, level_asg).unwrap();
///    }
///
///    assert!(hnsw.size() == vectors.len());
///    assert!(hnsw.search(&mut [1.41, 1.41, 0.0], 3, None).len() == 3);
/// }
///```
#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
pub struct HNSW {
    /// All nodes in the graph, not layer-wise (vectors are in `flat_vectors`)
    nodes: Vec<Node>,
    /// SoA contiguous storage for all node vectors. `flat_vectors[node_id * dim ... (node_id + 1) * dim]`
    flat_vectors: Vec<f32>,
    /// Dimensionality of all vectors (set on first insert, must be consistent)
    dim: usize,
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
    /// candidates (paper Alg. 4 lines 3-7). Useful for highly clustered data.
    extend_candidates: bool,
    /// If true, `select_neighbors_heuristic` adds discarded candidates back to reach `max_results`
    /// (paper Alg. 4 lines 15-17). Ensures fixed number of connections per element.
    keep_pruned_connections: bool,
    /// If true (paper default), the upper-layer pass uses a simple greedy search
    /// (paper Alg. 5, ef=1: "avoid introduction of additional parameters").
    /// If false, the upper-layer pass uses `search_layer_knn` with ef=1 (BFS).
    use_simple_greedy_upper_layer: bool,
    /// Similarity metric to use for distance calculations (default: Cosine)
    metrics: Metrics,
    /// Mapping from node uuid to array index
    id_mapper: HashMap<NodeUUID, NodeIndex>,
}

impl Default for HNSW {
    /// Default Config: max_neighbors=16, ef_construction=256, max_layers=18, distribution_bias=1/ln(16),
    /// Cosine similarity, pre-allocate for 512k nodes, and use simple greedy search for upper layers.
    fn default() -> Self {
        let m = 16;
        let ml = 1.0 / (m as f32).ln();
        HNSW::new(m, 256, 18, ml, Some(Metrics::Cosine), 512_000, true)
    }
}

impl HNSW {
    /// Creates a new HNSW instance with specified parameters.
    /// `distribution_bias` is the level norm factor `mL` (paper 4.1, default is 1/ln(16) ≈ 0.36, where 16 is M).
    /// `use_simple_greedy_upper_layer` toggles between the paper's "simple greedy search" (true, default)
    /// and `search_layer_knn` with ef=1 (false) for the upper-layer pass.
    pub fn new(
        max_neighbors: usize,
        ef_construction: usize,
        max_layers: usize,
        distribution_bias: f32,
        metrics: Option<Metrics>,
        pre_allocate: usize,
        use_simple_greedy_upper_layer: bool,
    ) -> Self {
        HNSW {
            nodes: Vec::with_capacity(pre_allocate),
            flat_vectors: Vec::new(),
            dim: 0,
            entry_point: None,
            max_layers,
            max_neighbors,
            ef_const: ef_construction,
            distribution_bias,
            extend_candidates: false,
            keep_pruned_connections: false,
            use_simple_greedy_upper_layer,
            metrics: metrics.unwrap_or(Metrics::Cosine),
            id_mapper: HashMap::with_capacity(pre_allocate),
        }
    }

    /// Creates a new HNSW instance with options over paper Alg. 4 flags.
    /// `extend_candidates` and `keep_pruned_connections` correspond to Alg. 4's `extendCandidates` and `keepPrunedConnections` parameters.
    /// `use_simple_greedy_upper_layer` toggles between the paper's "simple greedy search" (true)
    /// and `search_layer_knn` with ef=1 (false) for the upper-layer pass.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        max_neighbors: usize,
        ef_construction: usize,
        max_layers: usize,
        distribution_bias: f32,
        metrics: Option<Metrics>,
        pre_allocate: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
        use_simple_greedy_upper_layer: bool,
    ) -> Self {
        HNSW {
            nodes: Vec::with_capacity(pre_allocate),
            flat_vectors: Vec::new(),
            dim: 0,
            entry_point: None,
            max_layers,
            max_neighbors,
            ef_const: ef_construction,
            distribution_bias,
            extend_candidates,
            keep_pruned_connections,
            use_simple_greedy_upper_layer,
            metrics: metrics.unwrap_or(Metrics::Cosine),
            id_mapper: HashMap::with_capacity(pre_allocate),
        }
    }

    /// Returns a slice of the vector for the given node index, the core SoA accessor — all vector reads should go through here.
    #[inline(always)]
    fn get_vector_slice(&self, idx: NodeIndex) -> &[f32] {
        let start = idx * self.dim;
        &self.flat_vectors[start..start + self.dim]
    }

    /// Generates a random level for a new node based on an exponential distribution uses the HNSW paper formula:
    /// floor(-ln(<rand(0...1)>) * mL) where mL is the level norm factor stored in `distribution_bias` (paper default: 1/ln(M), e.g. 0.36 for M=16).
    /// Used this in [`insert`](HNSW::insert), or you may use your own distribution curve
    #[inline(always)]
    pub fn get_random_level(&self) -> usize {
        let r: f32 = 1.0 - fastrand::f32();
        let level = (-r.ln() * self.distribution_bias).floor() as usize;
        level.min(self.max_layers - 1)
    }

    // /// Same as [`insert`](HNSW::insert), but [`level assignments`](HNSW::get_random_level) internally
    // pub fn insert_auto(
    //     &mut self,
    //     node_id: String,
    //     vector: &[f32],
    //     metadata: Vec<u8>,
    // ) -> Result<NodeIndex> {
    //     let level = self.get_random_level();
    //     self.insert(node_id, vector, metadata, level)
    // }

    /// EXPERIMENTAL: Fill the graph with random bullshit, good for testing and benchmarking,
    /// returns the final seed used for generation so result can reproduce the same data if needed
    pub fn fast_fill(&mut self, fill_count: usize, dim: usize) -> Result<usize> {
        let (mut vec, seed) = gen_vec(fill_count, dim, 198);
        let mut id = vec![0u8; fill_count * 32];
        fastrand::fill(&mut id);

        for (i, v) in vec.iter_mut().enumerate() {
            let id: [u8; 32] = id[i * 32..(i + 1) * 32].try_into()?;
            let level = self.get_random_level();
            self.insert(id, v, vec![], level)?;
        }

        Ok(seed)
    }

    /// Insert a new node into the HNSW graph
    /// This is the core HNSW algorithm:
    /// - If first node, just add it as entry point
    /// - Otherwise, search from top layer down to find nearest neighbors
    /// - Connect the new node to its neighbors at each layer
    ///
    /// Takes &mut slice for L2 normalize if metrics is COSINE
    /// Returns `Ok(NodeID)` UUID of the newly inserted node, If everything goes right!!
    /// `ALGORITHM 1 from paper`
    pub fn insert(
        &mut self,
        id: NodeUUID,
        vector: &mut [f32],
        metadata: Vec<u8>,
        max_level: usize,
    ) -> Result<NodeUUID> {
        if id.is_empty() || vector.is_empty() {
            return Err(anyhow::anyhow!("NodeUUID or Vector cannot be empty"));
        }

        if self.id_mapper.contains_key(&id) {
            return Err(anyhow::anyhow!(
                "Node uuid: {} already exists",
                hex::encode(id)
            ));
        }

        if self.entry_point.is_some() && self.dim != vector.len() {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch with existing nodes"
            ));
        }

        let node_id = self.nodes.len();

        if matches!(self.metrics, Metrics::Cosine) {
            normalize_l2(vector);
        }

        // set dim on first insert
        if self.dim == 0 {
            self.dim = vector.len();
        }

        // append vector to contiguous storage
        // TODO; extend_from_slice clones, perf issue, need unsafe?
        // let len = vector.len();
        // if len > 0 {
        //     let start = self.flat_vectors.len();
        //     self.flat_vectors.reserve(len);
        //     unsafe {
        //         let dst = self.flat_vectors.as_mut_ptr().add(start);
        //         std::ptr::copy_nonoverlapping(vector.as_ptr(), dst, len);
        //         self.flat_vectors.set_len(start + len);
        //     }
        // }
        //
        self.flat_vectors.extend_from_slice(vector);

        // create the node with empty neighbor lists
        // (we'll fill them in after finding neighbors)
        let node = Node {
            uuid: id,
            metadata,
            neighbors: vec![Vec::with_capacity(self.max_neighbors); max_level + 1],
            max_level,
            tombstone: false,
        };

        // put in the map for reindexing helper
        self.id_mapper.insert(id, node_id);

        if self.entry_point.is_none() {
            self.nodes.push(node);
            self.entry_point = Some(node_id);
            return Ok(id);
        }

        self.nodes.push(node);

        // Start search from entry point
        let mut current_nearest = self.entry_point.expect("Entry point is NONE");
        let entry_level = self.nodes[current_nearest].max_level;

        let mut scratch = SearchScratch::with_capacity(self.ef_const * 2);
        // Greedily traverse from top layer down to new node's level + 1
        // Just find the closest node, don't connect yet
        for layer in (max_level + 1..=entry_level).rev() {
            // paper suggests simple greedy search for upper layers to avoid extra parameters,
            // but we can also do a quick BFS with ef=1 for potentially better navigation (especially for clustered data)
            if self.use_simple_greedy_upper_layer {
                current_nearest = self.search_layer_greedy(
                    self.get_vector_slice(node_id),
                    current_nearest,
                    layer,
                );
            } else if let Some((nearest_id, _)) = self
                .search_layer_knn(
                    self.get_vector_slice(node_id),
                    current_nearest,
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
                self.get_vector_slice(node_id),
                current_nearest,
                self.ef_const,
                layer,
                &mut scratch,
            );

            // Apply heuristic (Algorithm 4) for diversity-aware neighbor selection
            let selected = self.select_neighbors_heuristic(node_id, &candidates, max_n, layer);

            // Connect new node to its neighbors (bidirectional)
            for &neighbor_id in &selected {
                if layer <= self.nodes[neighbor_id].max_level {
                    self.nodes[node_id].neighbors[layer].push(neighbor_id);
                    self.nodes[neighbor_id].neighbors[layer].push(node_id);
                    if self.nodes[neighbor_id].neighbors[layer].len()
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
            self.nodes[current_nearest]
                .neighbors
                .resize(max_level + 1, Vec::new());
            self.nodes[current_nearest].max_level = max_level;
            for layer in (entry_level + 1)..=max_level {
                self.nodes[node_id].neighbors[layer].push(current_nearest);
                self.nodes[current_nearest].neighbors[layer].push(node_id);
            }
            self.entry_point = Some(node_id);
        }

        Ok(id)
    }

    /// Greedy search: find single closest node at a layer, used for navigating upper layers quickly
    /// `ALGORITHM 2 from paper suggestion`
    fn search_layer_greedy(&self, query: &[f32], entry: NodeIndex, layer: usize) -> NodeIndex {
        let mut current = entry;
        let mut current_sim = self.similarity(query, self.get_vector_slice(current), &self.metrics);
        let mut improved = true;

        while improved {
            improved = false;

            // Check all neighbors at this layer
            if layer <= self.nodes[current].max_level {
                for &neighbor_id in &self.nodes[current].neighbors[layer] {
                    let neighbor_sim =
                        self.similarity(query, self.get_vector_slice(neighbor_id), &self.metrics);

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

    // /// Selects the closest neighbors directly without applying any diversity heuristic.
    // /// Just takes the top K the closest candidates.
    // /// `ALGORITHM 3 from the paper`.
    // #[allow(unused)]
    // #[inline]
    // fn select_neighbors_simple(
    //     &self,
    //     candidates: &[(NodeIndex, f32)],
    //     max_results: usize,
    // ) -> Vec<NodeIndex> {
    //     candidates
    //         .iter()
    //         .take(max_results)
    //         .map(|&(id, _)| id)
    //         .collect()
    // }

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
        let query_vec = self.get_vector_slice(query_id);
        let mut result: Vec<NodeIndex> = Vec::with_capacity(max_results);
        let mut discarded: Vec<NodeIndex> = Vec::new();

        let mut working: Vec<(NodeIndex, f32)> = candidates.to_vec();

        // TODO; this is bit unclear

        // Optionally extend candidates with neighbors of candidates (paper Alg. 4 lines 3-7)
        if self.extend_candidates {
            let original: Vec<NodeIndex> = candidates.iter().map(|&(id, _)| id).collect();
            for candidate_id in &original {
                if layer < self.nodes[*candidate_id].neighbors.len() {
                    for &adj in &self.nodes[*candidate_id].neighbors[layer] {
                        if self.nodes[adj].tombstone {
                            // should they be avoided? same logic as `search_layer_knn`
                            // which also skips tombstoned nodes when adding candidates
                            continue;
                        }
                        // Avoid adding duplicates
                        if !working.iter().any(|(id, _)| *id == adj) {
                            // Compute similarity for the adjacent node and add to working set
                            let sim = self.similarity(
                                query_vec,
                                self.get_vector_slice(adj),
                                &self.metrics,
                            );
                            working.push((adj, sim));
                        }
                    }
                }
            }
            // Sort the extended candidate list by similarity descending
            working.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }

        // Iterate candidates in order of similarity, applying the heuristic to filter out "redundant" neighbors
        for &(candidate_id, _sim) in &working {
            if result.len() >= max_results {
                break;
            }
            let candidate_vec = self.get_vector_slice(candidate_id);

            // Check if candidate is "redundant" with respect to already selected neighbors
            let mut is_good = true;
            for &selected_id in &result {
                let selected_vec = self.get_vector_slice(selected_id);
                let dist_candidate_to_selected = self.distance(candidate_vec, selected_vec);
                let dist_query_to_selected = self.distance(query_vec, selected_vec);

                // redundant if candidate is closer to selected neighbor than query is to that neighbor
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

    /// K-NN search at a specific layer: find K nearest neighbors
    /// Uses a BOUNDED search with ef parameter (critical for performance)
    ///
    /// ALGORITHM (5):
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
    fn search_layer_knn(
        &self,
        query: &[f32],
        entry: NodeIndex,
        ef: usize,
        layer: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<(NodeIndex, f32)> {
        let ef = ef.max(1);
        let capacity = ef.saturating_mul(2).max(1);
        let max_nodes = self.nodes.len();

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

        let entry_sim = self.similarity(query, self.get_vector_slice(entry), &self.metrics);
        scratch.visited.insert(entry);
        scratch.candidates.push(Candidate(entry, entry_sim));
        scratch.results.push(ScoredResult(entry, entry_sim));

        while let Some(Candidate(current_id, current_sim)) = scratch.candidates.pop() {
            if let Some(worst_result) = scratch.results.peek()
                && scratch.results.len() >= ef
                && current_sim < worst_result.1
            {
                break;
            }

            if layer <= self.nodes[current_id].max_level {
                for &neighbor_id in &self.nodes[current_id].neighbors[layer] {
                    if scratch.visited.insert(neighbor_id) {
                        let sim = self.similarity(
                            query,
                            self.get_vector_slice(neighbor_id),
                            &self.metrics,
                        );

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

        // maintain the capacity of the buffer for next call
        std::mem::replace(&mut scratch.result_buf, Vec::with_capacity(capacity))
    }

    /// Remove connections to keep only the M closest neighbors
    #[inline(always)]
    fn prune_connections(&mut self, node_id: NodeIndex, layer: usize) {
        let max_n = self.max_neighbors_for_layer(layer);
        if self.nodes[node_id].neighbors[layer].len() <= max_n {
            return;
        }

        // Store the old neighbor list to identify which edges to remove
        let old_neighbors: HashSet<NodeIndex> = self.nodes[node_id].neighbors[layer]
            .iter()
            .copied()
            .collect();

        // Calculate similarities to all neighbors, filtering out tombstoned nodes
        let node_vec = self.get_vector_slice(node_id);
        let mut neighbor_sims: Vec<(NodeIndex, f32)> = self.nodes[node_id].neighbors[layer]
            .iter()
            .filter(|&&n| !self.nodes[n].tombstone)
            .map(|&n| {
                let sim = self.similarity(node_vec, self.get_vector_slice(n), &self.metrics);
                (n, sim)
            })
            .collect();

        // Sort by similarity descending (best first)
        neighbor_sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Apply the paper's heuristic (Algorithm 4) for diversity-aware selection
        let new_neighbors = self.select_neighbors_heuristic(node_id, &neighbor_sims, max_n, layer);
        let new_neighbors_set: HashSet<NodeIndex> = new_neighbors.iter().copied().collect();

        let removed_neighbors: Vec<NodeIndex> = old_neighbors
            .difference(&new_neighbors_set)
            .copied()
            .collect();

        // Update this node's neighbor list
        self.nodes[node_id].neighbors[layer] = new_neighbors;

        // Remove reverse edges from pruned neighbors to maintain bidirectionality
        for removed_neighbor_id in removed_neighbors {
            if layer <= self.nodes[removed_neighbor_id].max_level {
                self.nodes[removed_neighbor_id].neighbors[layer].retain(|&n| n != node_id);
            }
        }
    }

    /// Returns the max number of neighbors for a given layer.
    /// Layer 0 uses M0 = 2*M per the HNSW paper (Section 4.1).
    #[inline(always)]
    fn max_neighbors_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.max_neighbors * 2
        } else {
            self.max_neighbors
        }
    }

    /// Compute distance between two vectors.
    /// For Cosine; distance = 1.0 - similarity (since vectors are pre-normalized).
    /// For Euclidean; distance = 1.0 / similarity - 1 (real_dist, lower = closer)
    /// For RawDot; distance = -dot_product (negated so lower = closer).
    #[inline(always)]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metrics {
            Metrics::Cosine => 1.0 - unsafe { dot_product(a, b) },
            Metrics::Euclidean => unsafe { 1.0 / (euclidean_similarity(a, b) - 1.0) },
            Metrics::RawDot => -unsafe { dot_product(a, b) },
        }
    }

    /// Internal fast-path similarity. For Cosine metric, both `a` and `b` MUST be L2-norm
    /// (this is done in `insert`, `search_kernel`, and `brute_search*`).
    /// When that's true, cosine similarity reduces to a raw dot product (~3x faster than the
    /// `cosine_similarity` SIMD path that recomputes the norms).
    #[inline]
    fn similarity(&self, a: &[f32], b: &[f32], metrics: &Metrics) -> f32 {
        unsafe {
            match metrics {
                Metrics::Cosine => dot_product(a, b),
                Metrics::Euclidean => euclidean_similarity(a, b),
                Metrics::RawDot => dot_product(a, b),
            }
        }
    }

    /// Internal search method that performs the actual HNSW search, other `[search_*]` fn uses this internally
    /// Takes &mut query for l2 norm
    /// Returns (index, similarity) of candidates, including `tombstoned` nodes or empty vec otherwise
    /// `ef` is here `explorable nodes limit`, internally does ef =~ cK, because, we want to have a `'good'` chance of finding K if not more than K `'good'` nodes
    /// > Time ~ O(log n) + O(ef log ef) + O(k) <- for final truncation, but usually ef dominates
    #[inline(always)]
    pub fn search_kernel(&self, query: &mut [f32], k: usize, ef: usize) -> Vec<(NodeIndex, f32)> {
        let entry = if let Some(entry) = self.entry_point {
            if self.nodes.get(entry).is_none()
                || self.nodes[entry].tombstone
                || self.dim != query.len()
            {
                return Vec::new();
            }
            entry
        } else {
            return Vec::new();
        };

        if k == 0 || ef == 0 || query.is_empty() {
            return Vec::new();
        }

        // norm query once for Cosine metric, stored vectors are already normalized
        if matches!(self.metrics, Metrics::Cosine) {
            normalize_l2(query);
        }

        let entry_level = self.nodes[entry].max_level;
        let mut current = entry;
        let mut scratch = SearchScratch::with_capacity(ef * 2);

        // Traverse from top to layer 1
        for layer in (1..=entry_level).rev() {
            if self.use_simple_greedy_upper_layer {
                current = self.search_layer_greedy(query, current, layer);
            } else if let Some((nearest_id, _)) = self
                .search_layer_knn(query, current, 1, layer, &mut scratch)
                .first()
            {
                current = *nearest_id;
            }
        }

        // search layer 0 thoroughly for K neighbors
        // `search_layer_knn` already returns sorted results
        // return full ef results, search() handles truncation after filtering tombstones
        self.search_layer_knn(query, current, ef, 0, &mut scratch)
    }

    /// Finds topK nearest neighbors to a query, if `ef_search` is None then, internally does a loop increase base ef for better odds
    /// Returns results as (node_index, similarity) tuples sorted by similarity (highest first), empty if no entry point or k=0
    ///
    /// - `query` - The query vector
    /// - `k` - Number of nearest neighbors to return
    /// - `ef_search` - Optional bounded width for search. If None, uses k * DEFAULT_EF_MULTIPLIER
    pub fn search(
        &self,
        query: &mut [f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeUUID, f32)> {
        self.search_helper(query, k, ef_search)
            .into_iter()
            .map(|(id, sim)| (self.nodes[id].uuid, sim))
            .collect()
    }

    #[inline]
    /// Search and return results with metadata, similiar to [search](HNSW::search), but collects metadata on return
    /// Returns results as (node_id, similarity, metadata_as_bytes) tuples sorted by similarity (highest first)
    pub fn search_metadata(
        &self,
        query: &mut [f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeUUID, f32, Vec<u8>)> {
        self.search_helper(query, k, ef_search)
            .into_iter()
            .map(|(id, sim)| {
                let node = &self.nodes[id];
                (node.uuid, sim, node.metadata.clone())
            })
            .collect()
    }

    /// Internal helper: loop that grows `ef` until we have `k` non-tombstoned results and the
    /// kth similarity "stabilizes" (or we hit `max_ef`).
    fn search_helper(
        &self,
        query: &mut [f32],
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
            self.nodes.len().max(ef)
        };
        let mut prev_kth_sim: f32 = f32::NEG_INFINITY;

        loop {
            let results = self.search_kernel(query, k, current_ef);

            let active_indices: Vec<_> = results
                .into_iter()
                .filter(|(id, _)| !self.nodes[*id].tombstone)
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

    #[inline]
    /// Brute-force parallel search for testing and validation.
    /// Returns similar to [`search`](HNSW::search)
    pub fn brute_search(&self, query: &mut [f32], k: usize) -> Vec<(NodeUUID, f32)> {
        use rayon::iter::*;
        let dim = self.dim;
        let flat = &self.flat_vectors;
        if self.metrics == Metrics::Cosine {
            normalize_l2(query);
        }
        let mut results: Vec<(NodeUUID, f32)> = self
            .nodes
            .par_iter()
            .enumerate()
            .filter(|(_, node)| !node.tombstone)
            .map(|(i, node)| {
                let vec_slice = &flat[i * dim..(i + 1) * dim];
                (node.uuid, self.similarity(query, vec_slice, &self.metrics))
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        results
    }

    #[inline]
    /// Brute-force search with metadata included in results.
    /// Returns similiar to [`search_with_metadata`](HNSW::search_metadata)
    pub fn brute_search_metadata(
        &self,
        query: &mut [f32],
        k: usize,
    ) -> Vec<(NodeUUID, f32, Vec<u8>)> {
        use rayon::iter::*;
        let dim = self.dim;
        let flat = &self.flat_vectors;

        if self.metrics == Metrics::Cosine {
            normalize_l2(query);
        }

        let mut results: Vec<(NodeUUID, f32, Vec<u8>)> = self
            .nodes
            .par_iter()
            .enumerate()
            .filter(|(_, node)| !node.tombstone)
            .map(|(i, node)| {
                let vec_slice = &flat[i * dim..(i + 1) * dim];
                (
                    node.uuid,
                    self.similarity(query, vec_slice, &self.metrics),
                    node.metadata.clone(),
                )
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        results
    }

    /// Get node by node ID, returns None if not found or tombstoned
    #[inline]
    pub fn get_node(&self, uuid: &[u8; 32]) -> Option<&Node> {
        self.id_mapper
            .get(uuid)
            .and_then(|&id| self.nodes.get(id))
            .and_then(|node| if node.tombstone { None } else { Some(node) })
    }

    /// Convenience method to get node by index, returns None if index out of bounds, internally does [`self.nodes.get(idx)`](Vec::get)
    #[inline]
    pub fn get_node_by_index(&self, idx: usize) -> Option<&Node> {
        self.nodes.get(idx)
    }

    /// Get all nodes present in a specific layer (i.e. nodes with max_level >= given level are present),
    /// including tombstoned ones, returns empty vec if level out of bounds
    #[inline]
    pub fn get_nodes_at_level(&self, level: usize) -> Vec<&Node> {
        if level > self.max_layers {
            return Vec::new();
        }
        self.nodes
            .iter()
            .filter(|node| node.max_level >= level)
            .collect()
    }

    /// Get entry point node, returns None if no entry point or if entry point is tombstoned
    #[inline]
    pub fn get_entry_point(&self) -> Option<&Node> {
        self.entry_point
            .and_then(|id| self.nodes.get(id))
            .and_then(|node| if node.tombstone { None } else { Some(node) })
    }

    /// Returns the vector for the given node UUID, or None if not found/tombstoned
    #[inline]
    pub fn get_vector(&self, uuid: &[u8; 32]) -> Option<&[f32]> {
        self.id_mapper
            .get(uuid)
            .map(|&idx| self.get_vector_slice(idx))
    }

    /// Returns the vector for the given node index, or None if out of bounds
    #[inline]
    pub fn get_vector_by_index(&self, idx: NodeIndex) -> Option<&[f32]> {
        if idx < self.nodes.len() {
            Some(self.get_vector_slice(idx))
        } else {
            None
        }
    }

    /// Lazy delete a node by node ID, if the deleted node is the entry point, finds a new entry point.
    /// id entry is removed so the same UUID can be re-inserted later.
    /// Returns err if node ID not found.
    #[inline(always)]
    pub fn delete_node(&mut self, uuid: &[u8; 32]) -> Result<()> {
        let node_idx = if let Some(idx) = self.id_mapper.get(uuid).copied() {
            if let Some(node) = self.nodes.get_mut(idx) {
                node.tombstone = true;
            }
            idx
        } else {
            return Err(anyhow::anyhow!("Node uuid {} not found", hex::encode(uuid)));
        };

        // Find and sets new entry point when the current one is deleted
        // Searches from max_layer down to find the highest-level active node
        if let Some(entry) = self.entry_point
            && entry == node_idx
        {
            for layer in (0..self.max_layers).rev() {
                for (id, node) in self.nodes.iter().enumerate() {
                    if node.max_level == layer && !node.tombstone {
                        self.entry_point = Some(id);
                        return Ok(());
                    }
                }
            }
            // No active nodes found
            self.entry_point = None;
        }

        Ok(())
    }

    /// Returns the dimensionality of all vectors in the index
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the similarity metric used by this HNSW instance (defaults is [Cosine](Metrics::Cosine))
    #[inline]
    pub fn get_metrics(&self) -> Metrics {
        self.metrics.clone()
    }

    /// Get the index configuration parameters: (max_layers, max_neighbors, ef_construction)
    #[inline]
    pub fn index_config(&self) -> (usize, usize, usize) {
        (self.max_layers, self.max_neighbors, self.ef_const)
    }

    /// Returns the total count of nodes in the graph, including tombstoned ones
    #[inline]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total memory usage of the graph in bytes, including everything
    #[inline]
    pub fn mem_size(&self) -> usize {
        let node_size = size_of::<Node>();
        let neighbors_size: usize = self
            .nodes
            .iter()
            .map(|node| {
                node.neighbors
                    .iter()
                    .map(|layer| layer.len() * size_of::<NodeIndex>())
                    .sum::<usize>()
            })
            .sum();

        self.nodes.len() * node_size + neighbors_size + self.flat_vectors.len() * size_of::<f32>()
    }

    #[inline]
    /// Returns the count of active (non-tombstoned) nodes
    pub fn active_count(&self) -> usize {
        self.nodes.iter().filter(|node| !node.tombstone).count()
    }

    #[inline]
    /// Returns the count of tombstoned (deleted) nodes
    pub fn tombstone_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.tombstone).count()
    }

    #[inline]
    /// Returns the ratio of tombstoned nodes to total nodes
    /// Can be used in trigger when to clean up & reindex
    pub fn tombstone_ratio(&self) -> f32 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.tombstone_count() as f32 / self.nodes.len() as f32
        }
    }

    // TODO; complete this
    /// Debug method to print summary of the graph, including entry point info and layer distribution, quality
    /// Optionally takes a NodeUUID for deep recursively insight.
    pub fn debug(&self, _uuid: Option<&[u8; 32]>) {
        println!("Total nodes: {}", self.nodes.len());
        if let Some(ep_idx) = self.entry_point
            && let Some(ep) = self.nodes.get(ep_idx)
        {
            println!(
                "Entry point UUID/Index: {}/{} (max_level {})",
                hex::encode(ep.uuid),
                ep_idx,
                ep.max_level
            );
        } else {
            println!("No entry point");
        }

        println!("Layer distribution:");
        for layer in 0..=self.max_layers {
            let count = self.get_nodes_at_level(layer).len();
            println!(
                "-L{}: {} nodes ({:.4}%)",
                layer,
                count,
                if !self.nodes.is_empty() {
                    (count as f32 / self.nodes.len() as f32) * 100.0
                } else {
                    0.0
                }
            );
        }
    }
}

/// Array index into nodes Vec
pub type NodeIndex = usize;

/// Unique identifier for a node. (Stable across reindexing)
/// A 256-bit (32-byte) uuid provided when inserting a node.
/// This helps for tracking node when rebuilding the graph.
pub type NodeUUID = [u8; 32];

#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
/// Represents a node in the HNSW graph (vector are stored in SoA layout).
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
