use crate::maths::{Metrics, cosine_similarity, dot_product, euclidean_similarity};
use crate::prelude::gen_vec;
use crate::utils::gen_bytes;
use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use wincode::{SchemaRead, SchemaWrite};
//TODO: rm clones, replace hashset wit bitset, optional parallel,
// introduce sim_cache and reduce sim calculation, fix sorting,
// reduce reallocation in search loops (use small buffer), proper preallocation, fix memory layout
// batch simd_calculation somehow?, add better checks for public api, imp error handling
// add pub, pri field to separate clean api, and provide clean abstraction for users
// i have came to know my impl differ by lot from the paper, this is lot of redundant code, and unnecessary overhead
// need to make this API idiomatic and clean, and also need to add more comments, docs

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
pub const DEFAULT_EF_INC_FACTOR: f32 = 1.57575;

/// Hierarchical Navigable Small World (HNSW) [Quick ref](https://arxiv.org/pdf/1603.09320)
///
/// Properties:
/// - **Hierarchical**: Multiple layers with exponentially decreasing nodes per layer
/// - **Navigable Small World**: Efficiently navigable graph structure at each layer
/// - **Logarithmic search complexity**: O(log N) by searching from top to bottom layers
/// - **Proper layer assignment**: Uses exponential distribution -ln(uniform) * 1/ln(M)
///
/// Algorithm highlights:
/// **Insert**: Search from top layer down, connect at each layer bidirectionally
/// **Search**: Greedy descent through upper layers, bounded search at bottom layer
/// **Pruning**: Keep only M closest neighbors per node per layer
/// **Tombstones**: Mark deleted nodes and skip during search, periodic cleanup & reindexing
///
///```
///use hnsw_rs::prelude::*;
///
/// fn main() {
///    let mut hnsw = HNSW::default();
///
///    let vectors = vec![
///                    vec![1.0, 0.0, 0.0],
///                    vec![1.41, 1.41, 0.0],
///                    vec![0.0, 1.0, 0.0],
///                    vec![0.0, 1.41, 1.41],
///                    vec![0.0, 0.0, 1.0]
///                    ];
///
///    for vector in vectors.iter() {
///        let id = hex::encode(gen_bytes(16)); // rand 16-byte ID for each node
///        let level_asg = hnsw.get_random_level(); // <-- (-rand.ln() * mL).floor(), where mL is 1/ln(M)
///        let metadata = format!("node-{}", i).to_bytes();
///        hnsw.insert(id, vector, metadata, level_asg).unwrap();
///    }
///
///    assert!(hnsw.count() == vectors.len());
///    assert!(hnsw.search(&[1.41, 1.41, 0.0], 3, None).len() == 3);
///
///    let res = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
///    assert!(res.contains(&("0".to_string(),
///            Metrics::calculate(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0],
///            Some(Metrics::Cosine)))));
///    assert!(res.contains(&("1".to_string(),
///            Metrics::calculate(&[1.0, 0.0, 0.0], &[1.41, 1.41, 0.0],
///            Some(Metrics::Cosine)))));
/// }
///```
#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
pub struct HNSW {
    /// All nodes in the graph, not layer-wise
    nodes: Vec<Node>,
    /// First node at the top layer, used as entry point for searches
    entry_point: Option<NodeIndex>,
    /// Total number of layers in the graph
    max_layers: usize,
    /// Degree of each node (max number of neighbors) per layer
    max_neighbors: usize,
    /// More values explored during insertion means better chance of finding good neighbors
    ef_const: usize,
    /// Controls the layer distribution of nodes (exponential distribution bias) CURRENTLY UNUSED
    pub distribution_bias: f32,
    /// Similarity metric to use for distance calculations (default: Cosine)
    metrics: Option<Metrics>,
    /// Mapping from node ID (set by params) to array index
    /// It's just a fucking HashMap for O(1) lookups, for keep tracking that's all it is, nothing fancy
    id_mapper: HashMap<NodeID, NodeIndex>,
}

impl Default for HNSW {
    fn default() -> Self {
        HNSW::new(18, 256, 16, 1.0, Some(Metrics::Cosine), 512_000)
    }
}

impl HNSW {
    /// Creates a new HNSW instance with specified parameters.
    pub fn new(
        max_neighbors: usize,
        ef_construction: usize,
        max_layers: usize,
        distribution_bias: f32,
        metrics: Option<Metrics>,
        pre_allocate: usize,
    ) -> Self {
        HNSW {
            nodes: Vec::with_capacity(pre_allocate),
            entry_point: None,
            max_layers,
            max_neighbors,
            ef_const: ef_construction,
            distribution_bias, // Currently unused
            metrics,
            id_mapper: HashMap::with_capacity(pre_allocate),
        }
    }

    /// Generates a random level for a new node based on an exponential distribution uses the HNSW paper formula: floor(-ln(rand) * 1/ln(M))
    /// Used in [`insert`](HNSW::insert), or you may use your own distribution curve
    #[inline(always)]
    pub fn get_random_level(&self) -> usize {
        let r: f32 = rand::random::<f32>().max(1e-9);
        // m_l = 1/ ln(M)
        let m_l = 1.0 / (self.max_neighbors as f32).ln();
        // l = [-ln(unif(0..1)) * m_l]
        let level = (-r.ln() * m_l).floor() as usize;
        level.min(self.max_layers - 1)

        // Alternative simpler version without precomputed bias
        // let r: f32 = rand::random();
        // let level = (-r.ln() / self.distribution_bias).floor() as usize;
        // // Clamp to [0, max_layers - 1]
        // level.min(self.max_layers - 1)
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
    /// returns the final seed used for generation so you can reproduce the same data if needed
    pub fn auto_fill(&mut self, count_fill: usize) -> Result<u64> {
        let (vec, seed) = gen_vec(count_fill, 128, 198);

        for v in vec.iter().take(count_fill) {
            let id = hex::encode(gen_bytes(32));
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
    /// Returns `Ok(NodeID)` UUID of the newly inserted node, If everything goes right!!
    pub fn insert(
        &mut self,
        id: NodeID,
        vector: &[f32],
        metadata: Vec<u8>,
        max_level: usize,
    ) -> Result<NodeID> {
        if id.is_empty() || vector.is_empty() {
            return Err(anyhow::anyhow!("Node ID and vector cannot be empty"));
        }

        if self.id_mapper.contains_key(&id) {
            return Err(anyhow::anyhow!("Node ID already exists"));
        }

        if let Some(ep_idx) = self.entry_point
            && let Some(entry_node) = self.nodes.get(ep_idx)
            && entry_node.vector.len() != vector.len()
        {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch with existing nodes"
            ));
        }

        let node_id = self.nodes.len();

        // Create the node with empty neighbor lists (we'll fill them in after finding neighbors)
        let node = Node {
            uuid: id.clone(),
            metadata,
            vector: vector.to_vec(),
            neighbors: vec![Vec::with_capacity(self.max_neighbors); max_level + 1],
            max_level,
            tombstone: false,
        };

        // Put in the map for reindexing helper
        self.id_mapper.insert(id.clone(), node_id);

        if self.entry_point.is_none() {
            self.nodes.push(node);
            self.entry_point = Some(node_id);
            return Ok(id);
        }

        self.nodes.push(node);

        // Start search from entry point
        let mut current_nearest = self.entry_point.expect("Ohh no...entry_point is None");
        let entry_level = self.nodes[current_nearest].max_level;

        // Greedily traverse from top layer down to new node's level + 1
        // Just find the closest node, don't connect yet
        for layer in (max_level + 1..=entry_level).rev() {
            current_nearest =
                self.search_layer_greedy(&self.nodes[node_id].vector, current_nearest, layer);
        }

        // From new node's max_level down to 0, find neighbors and connect
        for layer in (0..=max_level).rev() {
            // Find ef_construction nearest neighbors at this layer
            let candidates = self.search_layer_knn(
                &self.nodes[node_id].vector,
                current_nearest,
                self.ef_const,
                layer,
            );

            let selected_count = candidates.len().min(self.max_neighbors);

            // Connect new node to its neighbors (bidirectional!!)
            for &(neighbor_id, _) in &candidates[..selected_count] {
                if layer <= self.nodes[neighbor_id].max_level {
                    self.nodes[node_id].neighbors[layer].push(neighbor_id);
                    self.nodes[neighbor_id].neighbors[layer].push(node_id);
                    if self.nodes[neighbor_id].neighbors[layer].len() > self.max_neighbors {
                        self.prune_connections(neighbor_id, layer);
                    }
                }
            }

            // Update current nearest for next layer
            if let Some((nearest_id, _)) = candidates.first() {
                current_nearest = *nearest_id;
            }
        }

        if max_level > entry_level {
            self.entry_point = Some(node_id);
        }

        Ok(id)
    }

    /// Greedy search: find single closest node at a layer
    /// Used for navigating upper layers quickly
    fn search_layer_greedy(&self, query: &[f32], entry: NodeIndex, layer: usize) -> NodeIndex {
        let mut current = entry;
        let mut current_sim =
            self.similarity(query, &self.nodes[current].vector, self.metrics.as_ref());
        let mut improved = true;

        while improved {
            improved = false;

            // Check all neighbors at this layer
            if layer <= self.nodes[current].max_level {
                for &neighbor_id in &self.nodes[current].neighbors[layer] {
                    let neighbor_sim = self.similarity(
                        query,
                        &self.nodes[neighbor_id].vector,
                        self.metrics.as_ref(),
                    );

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

    /// K-NN search at a specific layer: find K nearest neighbors
    /// Uses a BOUNDED search with ef parameter (critical for performance)
    ///
    /// ALGORITHM:
    /// - Start with entry point in both candidates and results
    /// - Pop highest similarity candidate from heap (best-first)
    /// - If candidate is worse than our worst result, skip (prune)
    /// - Otherwise, explore all its neighbors
    /// - Add promising neighbors to candidates AND results (if better than worst)
    /// - Repeat until candidates empty
    /// - Return top-k results sorted by similarity with computed similarity
    ///
    /// COMPLEXITY: O(log n) per operation instead of O(n log n)
    fn search_layer_knn(
        &self,
        query: &[f32],
        entry: NodeIndex,
        ef: usize,
        layer: usize,
    ) -> Vec<(NodeIndex, f32)> {
        let ef = ef.max(1);
        let capacity = ef.saturating_mul(2).max(1);
        let mut visited = HashSet::with_capacity(capacity);

        // CANDIDATES Heap: explore highest similarity first
        // pop() gives us the most promising node to explore next
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(capacity);

        // RESULTS heap: track top-k results
        // peek() gives us the WORST result in our top-k (for pruning)
        let mut results: BinaryHeap<ScoredResult> = BinaryHeap::with_capacity(ef);

        let entry_sim = self.similarity(query, &self.nodes[entry].vector, self.metrics.as_ref());
        visited.insert(entry);
        candidates.push(Candidate(entry, entry_sim));
        results.push(ScoredResult(entry, entry_sim));

        while let Some(Candidate(current_id, current_sim)) = candidates.pop() {
            // If we've filled ef slots and current is worse than our worst result,
            // all remaining candidates are also worse (heap gives best first) -> break early
            if let Some(worst_result) = results.peek()
                && results.len() >= ef
                && current_sim < worst_result.1
            {
                // There's no point exploring it because all its neighbors will be even worse
                break;
            }

            // Explore neighbors of current candidate
            if layer <= self.nodes[current_id].max_level {
                for &neighbor_id in &self.nodes[current_id].neighbors[layer] {
                    if visited.insert(neighbor_id) {
                        let sim = self.similarity(
                            query,
                            &self.nodes[neighbor_id].vector,
                            self.metrics.as_ref(),
                        );

                        // WHATDAFAK: should we add this neighbor to our search frontier?
                        // Add if we haven't filled ef slots OR new node is better than our worst
                        let worst_if_full = results.peek().map(|r| r.1);
                        let should_add = match (results.len(), worst_if_full) {
                            (len, _) if len < ef => true,            // still filling
                            (_, Some(worst)) if sim > worst => true, // better than worst
                            _ => false,
                        };

                        if should_add {
                            candidates.push(Candidate(neighbor_id, sim));
                            results.push(ScoredResult(neighbor_id, sim));

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Final sort for highest similarity first for output consistency
        let mut sorted_results: Vec<ScoredResult> = results.into_vec();
        sorted_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        sorted_results
            .into_iter()
            .map(|ScoredResult(id, sim)| (id, sim))
            .collect()
    }

    /// Remove connections to keep only the M closest neighbors
    fn prune_connections(&mut self, node_id: NodeIndex, layer: usize) {
        if self.nodes[node_id].neighbors[layer].len() <= self.max_neighbors {
            return;
        }

        // Store the old neighbor list to identify which edges to remove
        let old_neighbors: HashSet<NodeIndex> = self.nodes[node_id].neighbors[layer]
            .iter()
            .copied()
            .collect();

        // Calculate similarities to all neighbors, filtering out tombstoned nodes
        let mut neighbor_sims: Vec<(NodeIndex, f32)> = self.nodes[node_id].neighbors[layer]
            .iter()
            .filter(|&&n| !self.nodes[n].tombstone) // Skip tombstoned neighbors
            .map(|&n| {
                let sim = self.similarity(
                    &self.nodes[node_id].vector,
                    &self.nodes[n].vector,
                    self.metrics.as_ref(),
                );
                (n, sim)
            })
            .collect();

        neighbor_sims.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        neighbor_sims.truncate(self.max_neighbors);

        let new_neighbors: Vec<NodeIndex> = neighbor_sims.into_iter().map(|(id, _)| id).collect();
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

    /// Similarity [`metric`](Metrics): cosine similarity, Euclidean similarity, or raw dot product etc
    #[inline]
    fn similarity(&self, a: &[f32], b: &[f32], metrics: Option<&Metrics>) -> f32 {
        match metrics {
            Some(Metrics::Cosine) | None => cosine_similarity(a, b),
            Some(Metrics::Euclidean) => euclidean_similarity(a, b),
            Some(Metrics::RawDot) => dot_product(a, b),
        }
    }

    /// Internal search method that performs the actual HNSW search, other `[search_*]` fn uses this internally
    /// Returns (index, similarity) of candidates, including `tombstoned` nodes or empty vec otherwise
    /// `ef` is here `explorable nodes limit`, internally does ef =~ cK, because, we want to have a `'good'` chance of finding K if not more than K `'good'` nodes
    /// > Time ~ O(log n) + O(ef log ef) + O(k) <- for final truncation, but usually ef dominates
    #[inline(always)]
    pub fn search_kernel(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeIndex, f32)> {
        let entry = if let Some(entry) = self.entry_point {
            if let Some(entry_node) = self.nodes.get(entry) {
                if entry_node.vector.len() != query.len() {
                    return Vec::new();
                }
                entry
            } else {
                return Vec::new();
            }
        } else {
            return Vec::new();
        };

        if k == 0 || ef == 0 || query.is_empty() {
            return Vec::new();
        }

        let entry_level = self.nodes[entry].max_level;
        let mut current = entry;

        // Traverse from top to layer 1
        for layer in (1..=entry_level).rev() {
            current = self.search_layer_greedy(query, current, layer);
        }

        // Search layer 0 thoroughly for K neighbors
        // `search_layer_knn` already returns sorted results
        // Return full ef results, search() handles truncation after filtering tombstones
        self.search_layer_knn(query, current, ef, 0)
    }

    /// Finds topK nearest neighbors to a query, if `ef_search` is None then, internally does a loop increase base ef for better odds
    /// Returns results as (node_index, similarity) tuples sorted by similarity (highest first), empty if no entry point or k=0
    ///
    /// - `query` - The query vector
    /// - `k` - Number of nearest neighbors to return
    /// - `ef_search` - Optional bounded width for search. If None, uses k * DEFAULT_EF_MULTIPLIER
    pub fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Vec<(NodeID, f32)> {
        if k == 0 || self.entry_point.is_none() {
            return Vec::new();
        }

        let ef = ef_search.unwrap_or(k * DEFAULT_EF_MULTIPLIER);
        let mut current_ef = ef.max(k).max(1); // We are caping at K
        let max_ef = self.nodes.len().max(ef);

        loop {
            let results = self.search_kernel(query, k, current_ef);

            // Filter out tombstoned nodes and convert to Node UUID
            let active_results: Vec<(NodeID, f32)> = results
                .into_iter()
                .filter(|(id, _)| !self.nodes[*id].tombstone)
                .map(|(id, sim)| (self.nodes[id].uuid.clone(), sim))
                .collect();

            // If we have enough results, or we've reached max ef, return
            if active_results.len() >= k || current_ef >= max_ef {
                return active_results.into_iter().take(k).collect();
            }

            // Not enough active results, perform DOMAIN EXPANSION
            let grown = ((current_ef as f32) * DEFAULT_EF_INC_FACTOR).ceil() as usize;
            current_ef = grown.max(current_ef + 1).min(max_ef);
        }
    }

    #[inline]
    /// Search and return results with metadata, similiar to [search](HNSW::search), but collects metadata on return
    /// Returns results as (node_id, similarity, metadata_as_bytes) tuples sorted by similarity (highest first)
    pub fn search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(NodeID, f32, Vec<u8>)> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let ef = ef_search.unwrap_or(k * DEFAULT_EF_MULTIPLIER).max(k).max(1);
        let mut current_ef = ef;
        let max_ef = self.nodes.len().max(ef);

        loop {
            let results = self.search_kernel(query, k, current_ef);

            let active_results: Vec<(NodeID, f32, Vec<u8>)> = results
                .into_iter()
                .filter(|(id, _)| !self.nodes[*id].tombstone)
                .map(|(id, sim)| {
                    let node = &self.nodes[id];
                    (node.uuid.clone(), sim, node.metadata.clone())
                })
                .collect();

            if active_results.len() >= k || current_ef >= max_ef {
                return active_results.into_iter().take(k).collect();
            }

            let grown = ((current_ef as f32) * DEFAULT_EF_INC_FACTOR).ceil() as usize;
            current_ef = grown.max(current_ef + 1).min(max_ef);
        }
    }

    #[inline]
    /// Brute-force parallel search for testing and validation.
    /// Returns similar to [`search`](HNSW::search)
    pub fn brute_search(&self, query: &[f32], k: usize) -> Vec<(NodeID, f32)> {
        use rayon::iter::*;
        let mut results: Vec<(NodeID, f32)> = self
            .nodes
            .par_iter()
            .filter(|node| !node.tombstone) // Filter out tombstoned nodes
            .map(|node| {
                (
                    node.uuid.clone(),
                    self.similarity(query, &node.vector, self.metrics.as_ref()),
                )
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        results
    }

    #[inline]
    /// Brute-force search with metadata included in results.
    /// Returns similiar to [`search_with_metadata`](HNSW::search_with_metadata)
    pub fn brute_search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<(NodeID, f32, Vec<u8>)> {
        use rayon::iter::*;
        let mut results: Vec<(NodeID, f32, Vec<u8>)> = self
            .nodes
            .par_iter()
            .filter(|node| !node.tombstone)
            .map(|node| {
                (
                    node.uuid.clone(),
                    self.similarity(query, &node.vector, self.metrics.as_ref()),
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
    pub fn get_node(&self, uuid: &str) -> Option<&Node> {
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

    /// Get an iterator over all nodes, with option to include tombstoned nodes
    #[inline]
    pub fn get_node_iter(&self, allow_tombstone: bool) -> impl Iterator<Item = &Node> {
        self.nodes
            .iter()
            .filter(move |node| allow_tombstone || !node.tombstone)
    }

    /// Get entry point node, returns None if no entry point or if entry point is tombstoned
    #[inline]
    pub fn get_entry_point(&self) -> Option<&Node> {
        self.entry_point
            .and_then(|id| self.nodes.get(id))
            .and_then(|node| if node.tombstone { None } else { Some(node) })
    }

    /// Get the similarity metric used by this HNSW instance, defaults is [Cosine](Metrics::Cosine)
    #[inline]
    pub fn get_metrics_used(&self) -> Metrics {
        self.metrics.clone().unwrap_or(Metrics::Cosine)
    }

    /// Get all nodes at a specific layer, including tombstoned ones, returns empty vec if level out of bounds
    #[inline]
    pub fn get_node_at_level(&self, level: usize) -> Vec<&Node> {
        if level == 0 || level > self.max_layers {
            return Vec::new();
        }

        use rayon::iter::*;
        self.nodes
            .par_iter()
            .filter(|node| node.max_level == level)
            .collect()
    }

    /// Get the index configuration parameters: (max_layers, max_neighbors, ef_construction)
    #[inline]
    pub fn index_config(&self) -> (usize, usize, usize) {
        (self.max_layers, self.max_neighbors, self.ef_const)
    }

    /// Lazy-delete a node by node ID, if the deleted node is the entry point, finds a new entry point
    /// Returns err if node ID not found
    #[inline]
    pub fn delete_node(&mut self, uuid: &str) -> Result<()> {
        let node_idx = if let Some(idx) = self.id_mapper.get_mut(uuid) {
            if let Some(node) = self.nodes.get_mut(*idx) {
                node.tombstone = true;
            }
            idx
        } else {
            return Err(anyhow::anyhow!("Node index '{}' not found", uuid));
        };

        // Find and sets new entry point when the current one is deleted
        // Searches from max_layer down to find the highest-level active node
        if let Some(entry) = self.entry_point
            && entry == *node_idx
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

    /// Returns the total count of nodes in the graph, including tombstoned ones
    #[inline]
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total memory usage of the graph in bytes, including everything
    #[inline]
    pub fn size_in_bytes(&self) -> usize {
        use rayon::iter::*;
        let node_size = size_of::<Node>();
        let neighbors_size: usize = self
            .nodes
            .par_iter()
            .map(|node| {
                node.neighbors
                    .par_iter()
                    .map(|layer| layer.len() * size_of::<NodeIndex>())
                    .sum::<usize>()
            })
            .sum();

        self.nodes.len() * node_size + neighbors_size
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
    /// Can used in trigger when to clean up & reindex
    pub fn tombstone_ratio(&self) -> f32 {
        if self.nodes.is_empty() {
            0.0
        } else {
            self.tombstone_count() as f32 / self.nodes.len() as f32
        }
    }
}

/// Array index into nodes Vec
pub type NodeIndex = usize;

/// Unique identifier for a node. (Stable across reindexing)
/// It's just a string that is provided when inserting a node
/// This helps for tracking node when rebuilding the graph
pub type NodeID = String;

#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
/// Represents a node in the HNSW graph.
pub struct Node {
    /// Unique Stable Identifier for the node, provided during insertion
    pub uuid: NodeID,
    /// Vector representation of the node, any dimensionality
    pub vector: Vec<f32>,
    /// Metadata associated with the node
    pub metadata: Vec<u8>, // TODO: Make it generic? or something else, but serializable can be overhead
    /// Neighbors per layer, e.g `neighbors[0]` is the list of neighbors in layer 0
    pub neighbors: Vec<Vec<NodeIndex>>,
    /// The highest layer this node exists in
    pub max_level: usize,
    /// Flag for lazy deletion
    tombstone: bool,
}

impl Node {
    /// Returns true if this node has been soft-deleted (tombstoned).
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.tombstone
    }
}
