use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use walkdir::WalkDir;

use crate::commands::vectorstore;
use crate::panic_guard::run_guarded_async;

const DEFAULT_RESULTS: usize = 20;
const MAX_RESULTS: usize = 50;
const RRF_K: f64 = 60.0;
const FILENAME_EXACT_BONUS: f64 = 200.0;
const PHRASE_IN_TITLE_BONUS: f64 = 50.0;
const PHRASE_IN_CONTENT_PER_OCC: f64 = 20.0;
const MAX_PHRASE_OCC_COUNTED: usize = 10;
const TITLE_TOKEN_WEIGHT: f64 = 5.0;
const CONTENT_TOKEN_WEIGHT: f64 = 1.0;
const SNIPPET_CONTEXT: usize = 80;
/// An exact frontmatter-alias hit is the strongest authoring signal a page can
/// carry. It outranks every phrase bonus so that a page whose alias the query
/// spells cannot be displaced out of the candidate set by pages that merely
/// carry more bulk text.
const ALIAS_EXACT_BONUS: f64 = 120.0;
const SEARCH_EMBEDDING_TIMEOUT_SECS: u64 = 8;
const MAX_SEARCH_FILES: usize = 10_000;
const MIN_GRAPH_RESULT_RATIO: f64 = 0.15;
const MAX_GRAPH_RESULT_RATIO: f64 = 0.30;
const MAX_GRAPH_SEEDS: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchImageRef {
    pub url: String,
    pub alt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSearchResult {
    pub path: String,
    pub title: String,
    pub snippet: String,
    pub title_match: bool,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    pub images: Vec<SearchImageRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub graph_related_to: Vec<String>,
}

/// One recorded step-down from the retrieval mode the request asked for.
/// Emitted instead of only writing to stderr, so a caller reading the response
/// can tell "keyword, because the corpus has no match" apart from "keyword,
/// because the embedding endpoint failed".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchDegradation {
    /// Where the step-down happened: `embedding`, `vectorSearch`, `vectorIndex`.
    pub stage: String,
    /// Mode that was requested or expected before this stage failed.
    pub from: String,
    /// Mode actually served after it.
    pub to: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSearchResponse {
    pub mode: String,
    pub results: Vec<ProjectSearchResult>,
    pub token_hits: usize,
    pub vector_hits: usize,
    pub graph_hits: usize,
    /// Mode the request was set up to serve, when that is not the mode served.
    /// Set only if a degradation was recorded, and cleared when the served mode
    /// ended up matching anyway. Omitted from the wire when absent, so existing
    /// consumers see the response shape unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degradations: Vec<SearchDegradation>,
}

/// Query embedding plus any degradation incurred while obtaining it.
#[derive(Debug, Clone, Default)]
pub struct QueryEmbeddingOutcome {
    pub embedding: Option<Vec<f32>>,
    pub degradations: Vec<SearchDegradation>,
}

#[derive(Debug, Clone)]
struct GraphPage {
    path: String,
    title: String,
    content: String,
    links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageLinkEntry {
    pub title: String,
    pub path: Option<String>,
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageLinksResponse {
    pub outgoing: Vec<PageLinkEntry>,
    pub backlinks: Vec<PageLinkEntry>,
    pub missing: Vec<PageLinkEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEmbeddingConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub output_dimensionality: Option<f64>,
    /// Extra HTTP headers to send with every embedding request, e.g.
    /// `X-Model-Provider-Id: siliconflow` for the mify gateway.
    /// Reserved names (Authorization, Content-Type, Host,
    /// Content-Length, x-goog-api-key) are skipped — they're managed
    /// by the client.
    #[serde(default)]
    pub extra_headers: Option<BTreeMap<String, String>>,
}

#[tauri::command]
pub async fn search_project(
    project_path: String,
    query: String,
    top_k: Option<usize>,
    include_content: Option<bool>,
    query_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
) -> Result<ProjectSearchResponse, String> {
    run_guarded_async("search_project", async move {
        let outcome = resolve_query_embedding(&query, query_embedding, embedding_config).await?;
        search_project_inner(
            project_path,
            query,
            top_k.unwrap_or(DEFAULT_RESULTS),
            include_content.unwrap_or(false),
            outcome.embedding,
            outcome.degradations,
        )
        .await
    })
    .await
}

#[tauri::command]
pub async fn embedding_fetch(
    text: String,
    cfg: SearchEmbeddingConfig,
    max_retries: Option<usize>,
) -> Result<Vec<f32>, String> {
    run_guarded_async("embedding_fetch", async move {
        fetch_embedding_with_retry(&text, &cfg, max_retries.unwrap_or(3)).await
    })
    .await
}

#[tauri::command]
pub async fn embedding_fetch_batch(
    texts: Vec<String>,
    cfg: SearchEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, String> {
    run_guarded_async("embedding_fetch_batch", async move {
        fetch_embedding_batch(&texts, &cfg).await
    })
    .await
}

#[tauri::command]
pub async fn get_page_links(
    project_path: String,
    file_path: String,
) -> Result<PageLinksResponse, String> {
    tokio::task::spawn_blocking(move || get_page_links_inner(&project_path, &file_path))
        .await
        .map_err(|err| format!("page links worker failed: {err}"))?
}

/// Build link relationships from Markdown source rather than the UI's lazy
/// file tree. Canonical paths enforce the project/wiki boundary, while the
/// returned paths stay project-relative so all desktop platforms share one
/// wire format and Windows separators never leak into frontend routing.
fn get_page_links_inner(project_path: &str, file_path: &str) -> Result<PageLinksResponse, String> {
    let project = fs::canonicalize(project_path)
        .map_err(|err| format!("Failed to resolve project path: {err}"))?;
    let file =
        fs::canonicalize(file_path).map_err(|err| format!("Failed to resolve page path: {err}"))?;
    let wiki_root = project.join("wiki");
    if !file.starts_with(&wiki_root)
        || !file.is_file()
        || file.extension().and_then(|value| value.to_str()) != Some("md")
    {
        return Err("Page links target must be an existing Markdown file under wiki/".to_string());
    }

    let mut pages = BTreeMap::<String, GraphPage>::new();
    let canonical_project = project.to_string_lossy();
    for entry in WalkDir::new(&wiki_root).into_iter().filter_map(Result::ok) {
        if pages.len() >= MAX_SEARCH_FILES
            || !entry.file_type().is_file()
            || entry.path().extension().and_then(|value| value.to_str()) != Some("md")
        {
            continue;
        }
        let Ok(content) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let path = relative_to_project(&canonical_project, entry.path());
        let title = extract_title(
            &content,
            entry
                .path()
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        );
        pages.insert(
            normalize_path(&path),
            GraphPage {
                path,
                title,
                links: extract_wikilinks(&content),
                content,
            },
        );
    }

    let current_path = normalize_path(&relative_to_project(&canonical_project, &file));
    let current = pages
        .get(&current_path)
        .ok_or_else(|| "Page is not available in the current wiki index".to_string())?;
    let mut outgoing = Vec::new();
    let mut missing = Vec::new();
    for link in &current.links {
        if let Some(target_path) = resolve_reader_wikilink(&pages, link) {
            if target_path.as_str() == current_path {
                continue;
            }
            if let Some(target) = pages.get(target_path) {
                outgoing.push(PageLinkEntry {
                    title: target.title.clone(),
                    path: Some(target.path.clone()),
                    snippet: None,
                });
            }
        } else {
            missing.push(PageLinkEntry {
                title: link.clone(),
                path: None,
                snippet: None,
            });
        }
    }

    let mut backlinks = Vec::new();
    for (path, page) in &pages {
        if path == &current_path {
            continue;
        }
        let links_here = page.links.iter().any(|link| {
            resolve_reader_wikilink(&pages, link)
                .is_some_and(|target| target.as_str() == current_path)
        });
        if links_here {
            backlinks.push(PageLinkEntry {
                title: page.title.clone(),
                path: Some(page.path.clone()),
                snippet: Some(build_snippet(&page.content, &current.title)),
            });
        }
    }

    outgoing.sort_by(|a, b| a.title.cmp(&b.title));
    outgoing.dedup_by(|a, b| a.path == b.path);
    backlinks.sort_by(|a, b| a.title.cmp(&b.title));
    missing.sort_by(|a, b| a.title.cmp(&b.title));
    missing.dedup_by(|a, b| a.title == b.title);
    Ok(PageLinksResponse {
        outgoing,
        backlinks,
        missing,
    })
}

/// Mirror `resolveRelatedSlug` in the frontend reader. Path-shaped links must
/// match their project-relative path exactly; bare links resolve by exact
/// filename after adding `.md`. Deliberately avoid title and fuzzy aliases so
/// the links panel never claims a target that the rendered page cannot open.
fn resolve_reader_wikilink<'a>(
    pages: &'a BTreeMap<String, GraphPage>,
    link: &str,
) -> Option<&'a String> {
    let link = link.trim().replace('\\', "/");
    if link.contains('/') {
        return pages.get_key_value(&link).map(|(path, _)| path);
    }
    let filename = if link.ends_with(".md") {
        link
    } else {
        format!("{link}.md")
    };
    pages.keys().find(|path| {
        std::path::Path::new(path.as_str())
            .file_name()
            .is_some_and(|name| name == filename.as_str())
    })
}

pub async fn resolve_query_embedding(
    query: &str,
    explicit_embedding: Option<Vec<f32>>,
    embedding_config: Option<SearchEmbeddingConfig>,
) -> Result<QueryEmbeddingOutcome, String> {
    if let Some(embedding) = explicit_embedding {
        return validate_query_embedding(embedding).map(|embedding| QueryEmbeddingOutcome {
            embedding: Some(embedding),
            degradations: Vec::new(),
        });
    }
    let Some(cfg) = embedding_config else {
        return Ok(QueryEmbeddingOutcome::default());
    };
    if !cfg.enabled || cfg.endpoint.trim().is_empty() || cfg.model.trim().is_empty() {
        return Ok(QueryEmbeddingOutcome::default());
    }
    match fetch_embedding_with_retry(query, &cfg, 0).await {
        Ok(embedding) => validate_query_embedding(embedding).map(|embedding| {
            QueryEmbeddingOutcome {
                embedding: Some(embedding),
                degradations: Vec::new(),
            }
        }),
        Err(err) => {
            // Previously this was stderr-only, so a caller saw `mode: "keyword"`
            // with no way to tell a configured-but-broken embedding endpoint
            // apart from one that was never enabled (F-7).
            eprintln!("[Search] embedding disabled for this request: {err}");
            Ok(QueryEmbeddingOutcome {
                embedding: None,
                degradations: vec![SearchDegradation {
                    stage: "embedding".to_string(),
                    from: "hybrid".to_string(),
                    to: "keyword".to_string(),
                    reason: format!("query embedding unavailable: {err}"),
                }],
            })
        }
    }
}

fn validate_query_embedding(embedding: Vec<f32>) -> Result<Vec<f32>, String> {
    if embedding.is_empty() {
        return Err("queryEmbedding must not be empty".to_string());
    }
    if embedding.iter().any(|v| !v.is_finite()) {
        return Err("queryEmbedding must contain only finite numbers".to_string());
    }
    Ok(embedding)
}

pub async fn search_project_inner(
    project_path: String,
    query: String,
    top_k: usize,
    include_content: bool,
    query_embedding: Option<Vec<f32>>,
    mut degradations: Vec<SearchDegradation>,
) -> Result<ProjectSearchResponse, String> {
    if query.trim().is_empty() {
        return Err("query is required".to_string());
    }
    let limit = top_k.clamp(1, MAX_RESULTS);
    let tokens = tokenize_query(&query);
    let effective_tokens = if tokens.is_empty() {
        vec![fold_for_match(query.trim())]
    } else {
        tokens
    };
    let phrases = phrase_candidates(&query);
    let mut results = Vec::new();
    let mut page_paths_by_stem = BTreeMap::new();
    let mut graph_pages = BTreeMap::new();

    let wiki_root = Path::new(&project_path).join("wiki");
    if wiki_root.exists() {
        let mut searched_files = 0usize;
        for entry in WalkDir::new(&wiki_root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file()
                || entry.path().extension().and_then(|s| s.to_str()) != Some("md")
            {
                continue;
            }
            searched_files += 1;
            if searched_files > MAX_SEARCH_FILES {
                eprintln!(
                    "[Search] stopped scanning wiki after {MAX_SEARCH_FILES} markdown files in {project_path}"
                );
                break;
            }
            let content = match fs::read_to_string(entry.path()) {
                Ok(content) => content,
                Err(_) => continue,
            };
            if let Some(stem) = entry.path().file_stem().and_then(|s| s.to_str()) {
                let previous = page_paths_by_stem.insert(
                    stem.to_string(),
                    relative_to_project(&project_path, entry.path()),
                );
                if let Some(previous) = previous {
                    eprintln!(
                        "[Search] duplicate wiki page stem '{stem}': '{previous}' and '{}' share one vector page_id",
                        relative_to_project(&project_path, entry.path())
                    );
                }
            }
            let relative_path = relative_to_project(&project_path, entry.path());
            let title = extract_title(
                &content,
                entry
                    .path()
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default(),
            );
            let hit = score_file(
                &project_path,
                entry.path(),
                &content,
                &effective_tokens,
                &phrases,
                &query,
                include_content,
            );
            graph_pages.insert(
                normalize_path(&relative_path),
                GraphPage {
                    path: relative_path,
                    title,
                    links: extract_wikilinks(&content),
                    content,
                },
            );
            if let Some(hit) = hit {
                results.push(hit);
            }
        }
    }

    let mut token_sorted = (0..results.len()).collect::<Vec<_>>();
    token_sorted.sort_by(|a, b| {
        results[*b]
            .score
            .partial_cmp(&results[*a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| results[*a].path.cmp(&results[*b].path))
    });
    let mut token_rank = BTreeMap::new();
    for (idx, result_idx) in token_sorted.iter().enumerate() {
        let result = &results[*result_idx];
        token_rank.insert(normalize_path(&result.path), idx + 1);
    }

    let mut vector_rank: BTreeMap<String, usize> = BTreeMap::new();
    let mut vector_score: BTreeMap<String, f32> = BTreeMap::new();
    let mut vector_hits = 0;
    if let Some(embedding) = query_embedding {
        if !embedding.is_empty() {
            match search_by_embedding(&project_path, embedding, limit.max(10)).await {
                Ok(vector_results) => {
                    vector_hits = vector_results.len();
                    if vector_hits == 0 {
                        degradations.push(SearchDegradation {
                            stage: "vectorIndex".to_string(),
                            from: "hybrid".to_string(),
                            to: "keyword".to_string(),
                            reason:
                                "vector index returned no chunks; the project may not be embedded"
                                    .to_string(),
                        });
                    }
                    for (idx, vr) in vector_results.iter().enumerate() {
                        vector_rank.insert(vr.id.clone(), idx + 1);
                        vector_score.insert(vr.id.clone(), vr.score);
                    }
                    materialize_vector_only_results(
                        &vector_results,
                        &page_paths_by_stem,
                        &project_path,
                        &mut results,
                        include_content,
                    );
                }
                Err(err) => {
                    eprintln!(
                        "[Search] vector search failed; falling back to keyword results: {err}"
                    );
                    degradations.push(SearchDegradation {
                        stage: "vectorSearch".to_string(),
                        from: "hybrid".to_string(),
                        to: "keyword".to_string(),
                        reason: format!("vector search failed: {err}"),
                    });
                }
            }
        }
    }

    if vector_hits > 0 {
        apply_rrf_scores(&mut results, &token_rank, &vector_rank, &vector_score);
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    let graph_hits = blend_graph_results(
        &mut results,
        &graph_pages,
        limit,
        vector_hits,
        include_content,
    );

    let mode = search_mode(token_rank.is_empty(), vector_hits, graph_hits).to_string();
    // What the request was set up to serve, before anything fell over. Derived
    // from a recorded degradation rather than guessed, and dropped when the
    // served mode matched it anyway (graph expansion can restore `hybrid` even
    // after the vector leg failed). A degraded run is then legible in the
    // response instead of only in stderr (F-7).
    let requested_mode = degradations
        .first()
        .map(|degradation| degradation.from.clone())
        .filter(|requested| requested != &mode);

    Ok(ProjectSearchResponse {
        mode,
        token_hits: token_rank.len(),
        vector_hits,
        graph_hits,
        requested_mode,
        degradations,
        results,
    })
}

fn apply_rrf_scores(
    results: &mut [ProjectSearchResult],
    token_rank: &BTreeMap<String, usize>,
    vector_rank: &BTreeMap<String, usize>,
    vector_score: &BTreeMap<String, f32>,
) {
    for result in results {
        let token = token_rank.get(&normalize_path(&result.path)).copied();
        let vector = vector_rank.get(&file_stem(&result.path)).copied();
        let mut rrf = 0.0;
        if let Some(rank) = token {
            rrf += 1.0 / (RRF_K + rank as f64);
        }
        if let Some(rank) = vector {
            rrf += 1.0 / (RRF_K + rank as f64);
        }
        if let Some(score) = vector_score.get(&file_stem(&result.path)).copied() {
            result.vector_score = Some(score);
        }
        result.score = rrf;
    }
}

/// Reserve 15-30% of the final window for one-hop graph expansion. A full
/// vector window leaves the minimum graph share; sparse vector retrieval moves
/// progressively toward the maximum. Missing graph candidates automatically
/// return their slots to keyword/vector results.
fn graph_result_quota(limit: usize, vector_hits: usize) -> usize {
    if limit < 2 {
        return 0;
    }
    let vector_coverage = vector_hits.min(limit) as f64 / limit as f64;
    let ratio = MAX_GRAPH_RESULT_RATIO
        - (MAX_GRAPH_RESULT_RATIO - MIN_GRAPH_RESULT_RATIO) * vector_coverage;
    ((limit as f64 * ratio).ceil() as usize).clamp(1, limit - 1)
}

fn blend_graph_results(
    ranked_results: &mut Vec<ProjectSearchResult>,
    pages: &BTreeMap<String, GraphPage>,
    limit: usize,
    vector_hits: usize,
    include_content: bool,
) -> usize {
    if ranked_results.is_empty() || pages.is_empty() {
        ranked_results.truncate(limit);
        return 0;
    }

    let mut aliases = BTreeMap::<String, String>::new();
    for (normalized_path, page) in pages {
        let wiki_relative = page.path.strip_prefix("wiki/").unwrap_or(&page.path);
        let stem = file_stem(&page.path);
        for alias in [
            page.path.as_str(),
            wiki_relative,
            stem.as_str(),
            page.title.as_str(),
        ] {
            aliases.insert(normalize_graph_alias(alias), normalized_path.clone());
        }
    }

    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for (source, page) in pages {
        for link in &page.links {
            let Some(target) = aliases.get(&normalize_graph_alias(link)) else {
                continue;
            };
            if source == target {
                continue;
            }
            adjacency
                .entry(source.clone())
                .or_default()
                .insert(target.clone());
            adjacency
                .entry(target.clone())
                .or_default()
                .insert(source.clone());
        }
    }

    let seed_paths: Vec<String> = ranked_results
        .iter()
        .take(limit.min(MAX_GRAPH_SEEDS))
        .map(|result| normalize_path(&result.path))
        .collect();
    let seed_set: BTreeSet<String> = seed_paths.iter().cloned().collect();
    let mut candidate_scores = BTreeMap::<String, f64>::new();
    let mut candidate_seeds = BTreeMap::<String, BTreeSet<String>>::new();
    for (rank, seed) in seed_paths.iter().enumerate() {
        let Some(neighbors) = adjacency.get(seed) else {
            continue;
        };
        for neighbor in neighbors {
            if seed_set.contains(neighbor) {
                continue;
            }
            *candidate_scores.entry(neighbor.clone()).or_default() += 1.0 / (rank + 1) as f64;
            if let Some(seed_page) = pages.get(seed) {
                candidate_seeds
                    .entry(neighbor.clone())
                    .or_default()
                    .insert(seed_page.title.clone());
            }
        }
    }

    let mut candidates: Vec<(String, f64)> = candidate_scores.into_iter().collect();
    candidates.sort_by(|(path_a, score_a), (path_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| path_a.cmp(path_b))
    });
    candidates.truncate(graph_result_quota(limit, vector_hits));
    if candidates.is_empty() {
        ranked_results.truncate(limit);
        return 0;
    }

    let selected_paths: BTreeSet<String> =
        candidates.iter().map(|(path, _)| path.clone()).collect();
    let mut existing = BTreeMap::<String, ProjectSearchResult>::new();
    let mut ranked_paths = Vec::new();
    for result in ranked_results.drain(..) {
        let path = normalize_path(&result.path);
        ranked_paths.push(path.clone());
        existing.insert(path, result);
    }

    let graph_count = candidates.len();
    let base_limit = limit.saturating_sub(graph_count);
    let mut base_results: Vec<ProjectSearchResult> = ranked_paths
        .iter()
        .filter(|path| !selected_paths.contains(*path))
        .filter_map(|path| existing.get(path).cloned())
        .take(base_limit)
        .collect();

    for (path, graph_score) in candidates {
        if let Some(result) = existing.remove(&path) {
            let mut result = result;
            result.graph_related_to = candidate_seeds
                .remove(&path)
                .unwrap_or_default()
                .into_iter()
                .collect();
            base_results.push(result);
            continue;
        }
        let Some(page) = pages.get(&path) else {
            continue;
        };
        let related_titles = candidate_seeds
            .remove(&path)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let related = related_titles.join(", ");
        base_results.push(ProjectSearchResult {
            path: page.path.clone(),
            title: page.title.clone(),
            snippet: format!("Graph neighbor of {related}"),
            title_match: false,
            score: graph_score / (RRF_K + 1.0),
            vector_score: None,
            images: extract_image_refs(&page.content),
            content: include_content.then(|| page.content.clone()),
            graph_related_to: related_titles,
        });
    }
    *ranked_results = base_results;
    graph_count
}

fn normalize_graph_alias(value: &str) -> String {
    value
        .split('#')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches(".md")
        .replace('\\', "/")
        .replace(' ', "-")
        .to_lowercase()
}

fn extract_wikilinks(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find("[[") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("]]") else {
            break;
        };
        let target = rest[..end].split('|').next().unwrap_or_default().trim();
        if !target.is_empty() {
            links.push(target.to_string());
        }
        rest = &rest[end + 2..];
    }
    links
}

/// Human-readable suffix naming any degradation, for the `wiki.search` progress
/// line. Empty when the served mode is the requested one — the common case must
/// stay silent, or the signal is lost in noise.
pub fn degradation_suffix(
    requested_mode: Option<&str>,
    degradations: &[SearchDegradation],
) -> String {
    if degradations.is_empty() && requested_mode.is_none() {
        return String::new();
    }
    let reasons = degradations
        .iter()
        .map(|degradation| format!("{}: {}", degradation.stage, degradation.reason))
        .collect::<Vec<_>>()
        .join("; ");
    match (requested_mode, reasons.is_empty()) {
        (Some(requested), false) => format!(", DEGRADED from {requested} ({reasons})"),
        (Some(requested), true) => format!(", DEGRADED from {requested}"),
        (None, false) => format!(", DEGRADED ({reasons})"),
        (None, true) => String::new(),
    }
}

fn search_mode(token_rank_empty: bool, vector_hits: usize, graph_hits: usize) -> &'static str {
    if graph_hits > 0 {
        "hybrid"
    } else if vector_hits == 0 {
        "keyword"
    } else if token_rank_empty {
        "vector"
    } else {
        "hybrid"
    }
}

#[derive(Debug, Clone)]
struct PageVectorResult {
    id: String,
    score: f32,
    chunk_text: String,
    heading_path: String,
}

async fn search_by_embedding(
    project_path: &str,
    query_embedding: Vec<f32>,
    top_k: usize,
) -> Result<Vec<PageVectorResult>, String> {
    let raw_chunks = vectorstore::vector_search_chunks(
        project_path.to_string(),
        query_embedding,
        (top_k * 3).max(30),
    )
    .await?;
    if raw_chunks.is_empty() {
        return Ok(vec![]);
    }

    let mut by_page: BTreeMap<String, Vec<vectorstore::ChunkSearchResult>> = BTreeMap::new();
    for chunk in raw_chunks {
        by_page
            .entry(chunk.page_id.clone())
            .or_default()
            .push(chunk);
    }

    let mut ranked = Vec::new();
    for (id, mut chunks) in by_page {
        chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });
        let top_chunk = chunks[0].clone();
        let top = top_chunk.score;
        let tail: f32 = chunks.iter().skip(1).map(|chunk| chunk.score).sum();
        let blended = top + (tail * 0.3).min((1.0 - top).max(0.0));
        ranked.push(PageVectorResult {
            id,
            score: blended,
            chunk_text: top_chunk.chunk_text,
            heading_path: top_chunk.heading_path,
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    ranked.truncate(top_k);
    Ok(ranked)
}

fn materialize_vector_only_results(
    vector_results: &[PageVectorResult],
    page_paths_by_stem: &BTreeMap<String, String>,
    project_path: &str,
    results: &mut Vec<ProjectSearchResult>,
    include_content: bool,
) {
    let mut known: BTreeSet<String> = results.iter().map(|r| file_stem(&r.path)).collect();
    for vr in vector_results {
        if known.contains(&vr.id) {
            continue;
        }
        if let Some(rel) = page_paths_by_stem.get(&vr.id) {
            let path = Path::new(project_path).join(rel);
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let file_name = Path::new(&rel)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let title = extract_title(&content, file_name);
            let snippet = build_vector_snippet(vr);
            results.push(ProjectSearchResult {
                path: rel.clone(),
                title,
                snippet,
                title_match: false,
                score: 0.0,
                vector_score: Some(vr.score),
                images: extract_image_refs(&content),
                content: include_content.then_some(content),
                graph_related_to: Vec::new(),
            });
            known.insert(vr.id.clone());
        }
    }
}

fn build_vector_snippet(result: &PageVectorResult) -> String {
    let mut text = result.chunk_text.trim().replace('\n', " ");
    if text.is_empty() {
        return String::new();
    }
    if text.chars().count() > SNIPPET_CONTEXT * 2 {
        text = text.chars().take(SNIPPET_CONTEXT * 2).collect::<String>();
        text.push_str("...");
    }
    let heading = result.heading_path.trim();
    if heading.is_empty() {
        text
    } else {
        format!("{heading}: {text}")
    }
}

fn score_file(
    project_path: &str,
    path: &Path,
    content: &str,
    tokens: &[String],
    phrases: &[String],
    query: &str,
    include_content: bool,
) -> Option<ProjectSearchResult> {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let title = extract_title(content, file_name);
    let title_text = format!("{title} {file_name}");
    // Everything below matches on folded text so that an unaccented query and
    // an accented page (and the reverse) meet on the same spelling.
    let title_folded = fold_for_match(&title_text);
    let content_folded = fold_for_match(content);
    let stem = fold_for_match(file_name.trim_end_matches(".md"));
    let aliases = frontmatter_aliases(content);

    let filename_exact = phrases.iter().any(|phrase| stem == *phrase);
    let alias_exact = phrases
        .iter()
        .any(|phrase| aliases.iter().any(|alias| alias == phrase));
    let title_has_phrase = phrases
        .iter()
        .any(|phrase| title_folded.contains(phrase.as_str()));
    // `phrases` is longest-first, so this picks the most specific phrase the
    // page actually contains.
    let content_phrase = phrases
        .iter()
        .find(|phrase| content_folded.contains(phrase.as_str()));
    let content_phrase_occ = content_phrase
        .map(|phrase| count_occurrences(&content_folded, phrase).min(MAX_PHRASE_OCC_COUNTED))
        .unwrap_or(0);
    let title_token_score = token_match_score(&title_folded, tokens);
    let content_token_score = token_match_score(&content_folded, tokens);

    if !filename_exact
        && !alias_exact
        && !title_has_phrase
        && content_phrase_occ == 0
        && title_token_score == 0
        && content_token_score == 0
    {
        return None;
    }

    let score = (if filename_exact {
        FILENAME_EXACT_BONUS
    } else {
        0.0
    }) + (if alias_exact { ALIAS_EXACT_BONUS } else { 0.0 })
        + (if title_has_phrase {
            PHRASE_IN_TITLE_BONUS
        } else {
            0.0
        })
        + content_phrase_occ as f64 * PHRASE_IN_CONTENT_PER_OCC
        + title_token_score as f64 * TITLE_TOKEN_WEIGHT
        + content_token_score as f64 * CONTENT_TOKEN_WEIGHT;

    // Priority order for the excerpt window: the matched phrase, then whichever
    // query tokens the page actually carries, then the raw query as a last
    // resort. `build_snippet_from_anchors` confines all of them to the body.
    let folded_query = fold_for_match(query);
    let mut anchors: Vec<&str> = Vec::new();
    if let Some(phrase) = content_phrase {
        anchors.push(phrase.as_str());
    }
    anchors.extend(
        tokens
            .iter()
            .filter(|token| content_folded.contains(token.as_str()))
            .map(String::as_str),
    );
    anchors.push(folded_query.as_str());

    Some(ProjectSearchResult {
        path: relative_to_project(project_path, path),
        title,
        snippet: build_snippet_from_anchors(content, &anchors),
        title_match: title_token_score > 0 || title_has_phrase,
        score,
        vector_score: None,
        images: extract_image_refs(content),
        content: include_content.then_some(content.to_string()),
        graph_related_to: Vec::new(),
    })
}

/// Lowercase and strip the diacritic from one character.
///
/// Deliberately **1:1** — every input char yields exactly one output char, so a
/// char offset into the folded string addresses the same char in the source and
/// snippet windows stay aligned. That rules out the expanding folds (`ß`→`ss`,
/// `æ`→`ae`, `œ`→`oe`), which are left untouched; none of them appear in the
/// Spanish legal vocabulary this exists to serve. `char::to_lowercase` is taken
/// one char at a time for the same reason.
fn fold_char(c: char) -> char {
    let lower = c.to_lowercase().next().unwrap_or(c);
    match lower {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'ď' | 'đ' => 'd',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'ĥ' | 'ħ' => 'h',
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'ĵ' => 'j',
        'ķ' => 'k',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'ţ' | 'ť' | 'ŧ' => 't',
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ŵ' => 'w',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'ź' | 'ż' | 'ž' => 'z',
        other => other,
    }
}

/// Diacritic-folded, lowercased copy used for *all* matching, on both sides.
///
/// Applying the identical fold to page text and to the query is what makes
/// `cedula` and `cédula` retrieve each other; folding only one side would make
/// the relation one-way. Char count is preserved (see [`fold_char`]).
pub fn fold_for_match(value: &str) -> String {
    value.chars().map(fold_char).collect()
}

/// Contiguous runs of content words, in query order, with interrogative and
/// function words removed.
///
/// `"Who is defendant counsel?"` yields `["defendant counsel"]` — the same
/// phrase the bare keyword query produces — so wrapping a query in question
/// form no longer costs it the phrase bonus that kept its page in the
/// candidate set. A stop word breaks a run rather than being spliced out, so
/// only words that really were adjacent in the query form a phrase.
fn content_phrases(query: &str) -> Vec<String> {
    let folded = fold_for_match(query);
    let mut phrases = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    for token in folded.split(is_query_separator) {
        if token.chars().count() > 1 && !is_stop_word(token) {
            run.push(token);
            continue;
        }
        if run.len() > 1 {
            phrases.push(run.join(" "));
        }
        run.clear();
    }
    if run.len() > 1 {
        phrases.push(run.join(" "));
    }
    phrases
}

/// Every phrase worth scoring for this query, longest first.
///
/// The whole trimmed query stays at the front so a query carrying no stop
/// words scores exactly as it did before this change; the content-word runs
/// are additive fallbacks for the question forms.
fn phrase_candidates(query: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let whole = trim_query_punctuation(&fold_for_match(query));
    if !whole.is_empty() {
        candidates.push(whole);
    }
    for phrase in content_phrases(query) {
        if !candidates.contains(&phrase) {
            candidates.push(phrase);
        }
    }
    candidates.sort_by_key(|phrase| std::cmp::Reverse(phrase.split(' ').count()));
    candidates
}

/// Aliases declared in a page's YAML frontmatter, folded for matching.
///
/// Handles both the inline (`aliases: ["a", "b"]`) and block (`- "a"`) forms.
/// Parsed by hand rather than with a YAML crate because a malformed page must
/// degrade to "no aliases" instead of failing the whole search.
fn frontmatter_aliases(content: &str) -> Vec<String> {
    let Some(block) = frontmatter_block(content) else {
        return Vec::new();
    };
    let mut aliases = Vec::new();
    let mut in_aliases = false;
    for line in block.lines() {
        let trimmed = line.trim();
        if in_aliases {
            if let Some(item) = trimmed.strip_prefix("- ") {
                aliases.push(item.to_string());
                continue;
            }
            // Any non-item line ends the sequence.
            if !trimmed.is_empty() {
                in_aliases = false;
            }
        }
        let Some(rest) = trimmed
            .strip_prefix("aliases:")
            .or_else(|| trimmed.strip_prefix("alias:"))
        else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() {
            in_aliases = true;
            continue;
        }
        let inline = rest.trim_start_matches('[').trim_end_matches(']');
        aliases.extend(inline.split(',').map(ToOwned::to_owned));
    }
    aliases
        .into_iter()
        .map(|alias| fold_for_match(alias.trim().trim_matches('"').trim_matches('\'').trim()))
        .filter(|alias| !alias.is_empty())
        .collect()
}

pub fn tokenize_query(query: &str) -> Vec<String> {
    let raw = fold_for_match(query)
        .split(is_query_separator)
        .filter(|token| token.chars().count() > 1)
        .filter(|token| !is_stop_word(token))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    for token in raw {
        let chars = token.chars().collect::<Vec<_>>();
        let has_cjk = chars.iter().any(|c| ('\u{3400}'..='\u{9fff}').contains(c));
        if has_cjk && chars.len() > 2 {
            for pair in chars.windows(2) {
                out.push(pair.iter().collect());
            }
            for ch in &chars {
                let s = ch.to_string();
                if !is_stop_word(&s) {
                    out.push(s);
                }
            }
            out.push(token);
        } else {
            out.push(token);
        }
    }
    out.into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn is_query_separator(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_punctuation()
        || matches!(
            c,
            '，' | '。'
                | '！'
                | '？'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '·'
                | '～'
                | '…'
        )
}

/// Function words that carry no retrieval signal.
///
/// Entries are written **unaccented** on purpose: callers fold before testing,
/// so `qué`/`que` and `cuándo`/`cuando` are both covered by one entry.
///
/// The interrogative and auxiliary sets (`who`, `which`, `where`, `can`,
/// `quien`, `cual`, `donde`, …) were added for F-3: without them a question
/// like "Who is defendant counsel?" scored no phrase at all, and the asserting
/// page lost the candidate set to pages with more bulk text.
fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        // Interrogatives and auxiliaries (English)
        "who" | "whom"
            | "whose"
            | "which"
            | "when"
            | "where"
            | "why"
            | "whether"
            | "can"
            | "could"
            | "should"
            | "would"
            | "will"
            | "shall"
            | "may"
            | "might"
            | "must"
            | "am"
            // Interrogatives and function words (Spanish; written unaccented)
            | "quien"
            | "quienes"
            | "que"
            | "cual"
            | "cuales"
            | "cuando"
            | "donde"
            | "adonde"
            | "como"
            | "cuanto"
            | "cuantos"
            | "cuanta"
            | "cuantas"
            | "el"
            | "la"
            | "lo"
            | "los"
            | "las"
            | "un"
            | "una"
            | "unos"
            | "unas"
            | "del"
            | "de"
            | "en"
            | "por"
            | "para"
            | "con"
            | "es"
            | "son"
            | "era"
            | "fue"
            | "esta"
            | "este"
            | "esto"
            | "estos"
            | "estas"
            | "se"
            | "su"
            | "sus"
            | "al"
            | "ha"
            | "han"
        | "的"
            | "是"
            | "了"
            | "什么"
            | "在"
            | "有"
            | "和"
            | "与"
            | "对"
            | "从"
            | "the"
            | "is"
            | "a"
            | "an"
            | "what"
            | "how"
            | "are"
            | "was"
            | "were"
            | "do"
            | "does"
            | "did"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "it"
            | "its"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "with"
            | "by"
            | "this"
            | "that"
            | "these"
            | "those"
    )
}

fn trim_query_punctuation(value: &str) -> String {
    value.trim_matches(is_query_separator).to_string()
}

/// Count how many query tokens appear in `folded_text`. The caller folds; this
/// must not fold again or the two sides could disagree about what was compared.
fn token_match_score(folded_text: &str, tokens: &[String]) -> usize {
    tokens
        .iter()
        .filter(|token| folded_text.contains(token.as_str()))
        .count()
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.match_indices(needle).count()
}

pub fn extract_title(content: &str, file_name: &str) -> String {
    let has_frontmatter = content.starts_with("---");
    let mut in_frontmatter = has_frontmatter;
    let mut frontmatter_closed = false;
    for line in content.lines().skip(if has_frontmatter { 1 } else { 0 }) {
        let trimmed = line.trim();
        if in_frontmatter && trimmed == "---" {
            in_frontmatter = false;
            frontmatter_closed = true;
            continue;
        }
        if in_frontmatter && trimmed.starts_with("title:") {
            return trimmed
                .trim_start_matches("title:")
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        }
        if has_frontmatter && !frontmatter_closed {
            continue;
        }
        if let Some(title) = trimmed.strip_prefix("# ") {
            return title.trim().to_string();
        }
    }
    file_name.trim_end_matches(".md").replace('-', " ")
}

pub fn extract_image_refs(content: &str) -> Vec<SearchImageRef> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut rest = content;
    while let Some(start) = rest.find("![") {
        rest = &rest[start + 2..];
        let Some(alt_end) = rest.find("](") else {
            break;
        };
        let alt = &rest[..alt_end];
        rest = &rest[alt_end + 2..];
        let Some(url_end) = rest.find(')') else {
            break;
        };
        let url = &rest[..url_end];
        if !url.trim().is_empty()
            && !url.contains(char::is_whitespace)
            && seen.insert(url.to_string())
        {
            out.push(SearchImageRef {
                url: url.to_string(),
                alt: alt.to_string(),
            });
        }
        rest = &rest[url_end + 1..];
    }
    out
}

async fn fetch_embedding_with_retry(
    text: &str,
    cfg: &SearchEmbeddingConfig,
    max_retries: usize,
) -> Result<Vec<f32>, String> {
    let mut current = text.to_string();
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        match fetch_embedding_once(&current, cfg).await {
            Ok(embedding) => return Ok(embedding),
            Err(EmbeddingFetchError::Oversize(message)) => {
                if attempts <= max_retries
                    && current.len() > 64
                    && halve_text_on_char_boundary(&mut current)
                {
                    eprintln!(
                        "[Embedding] auto-halving after oversize error at {} chars; retrying at {} chars ({attempts}/{})",
                        text.chars().count(),
                        current.chars().count(),
                        max_retries + 1
                    );
                    continue;
                }
                return Err(format!(
                    "Endpoint rejected input even at {} chars. Lower Settings -> Embedding -> Max Chunk Chars. {message}",
                    current.len()
                ));
            }
            Err(EmbeddingFetchError::Other(message)) => return Err(message),
        }
    }
}

async fn fetch_embedding_batch(
    texts: &[String],
    cfg: &SearchEmbeddingConfig,
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() || texts.len() > 64 {
        return Err("Embedding batch must contain between 1 and 64 inputs".to_string());
    }
    if is_google_embedding_config(cfg) || is_doubao_multimodal_embedding_config(cfg) {
        return Err(
            "This embedding provider does not use the OpenAI-compatible batch format".to_string(),
        );
    }

    let endpoint = volcengine_embedding_endpoint(cfg);
    let mut req = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            SEARCH_EMBEDDING_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| format!("Embedding HTTP client error: {e}"))?
        .post(&endpoint)
        .header("Content-Type", "application/json");
    if is_local_or_private_http_endpoint(&endpoint) {
        req = req.header("Origin", "http://localhost");
    }
    if !cfg.api_key.trim().is_empty() {
        req = req.bearer_auth(cfg.api_key.trim());
    }
    if let Some(extra) = cfg.extra_headers.as_ref() {
        for (name, value) in extra {
            let name = name.trim();
            let value = value.trim();
            if !name.is_empty()
                && !value.is_empty()
                && is_safe_extra_header_name(name)
                && !is_reserved_extra_header_name(name)
            {
                req = req.header(name, value);
            }
        }
    }
    let response = req
        .json(&json!({ "model": cfg.model, "input": texts }))
        .send()
        .await
        .map_err(|e| format!("Embedding batch request failed: {e}"))?;
    let status = response.status();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Embedding batch response read failed: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "Embedding batch API HTTP {status}: {}",
            response_text.chars().take(200).collect::<String>()
        ));
    }
    let data: Value = serde_json::from_str(&response_text).map_err(|e| {
        format!(
            "Embedding batch response parse failed: {e}: {}",
            response_text.chars().take(200).collect::<String>()
        )
    })?;
    parse_embedding_batch_values(&data, texts.len())
}

fn parse_embedding_batch_values(data: &Value, expected: usize) -> Result<Vec<Vec<f32>>, String> {
    let entries = data
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "Embedding batch response missing data array".to_string())?;
    if entries.len() != expected {
        return Err(format!(
            "Embedding batch returned {} vectors for {expected} inputs",
            entries.len()
        ));
    }
    let mut indexed = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let index = entry
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(position);
        if index >= expected {
            return Err("Embedding batch response contains an out-of-range index".to_string());
        }
        let values = entry
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| "Embedding batch response missing vector".to_string())?;
        let mut vector = Vec::with_capacity(values.len());
        for value in values {
            let number = value
                .as_f64()
                .ok_or_else(|| "Embedding batch response contains non-number values".to_string())?;
            if !number.is_finite() {
                return Err("Embedding batch response contains non-finite values".to_string());
            }
            vector.push(number as f32);
        }
        if vector.is_empty() {
            return Err("Embedding batch response vector is empty".to_string());
        }
        indexed.push((index, vector));
    }
    indexed.sort_by_key(|(index, _)| *index);
    if indexed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("Embedding batch response contains duplicate indexes".to_string());
    }
    let dimension = indexed.first().map(|(_, vector)| vector.len()).unwrap_or(0);
    if indexed.iter().any(|(_, vector)| vector.len() != dimension) {
        return Err("Embedding batch response contains inconsistent vector dimensions".to_string());
    }
    Ok(indexed.into_iter().map(|(_, vector)| vector).collect())
}

fn halve_text_on_char_boundary(text: &mut String) -> bool {
    let char_count = text.chars().count();
    if char_count <= 1 {
        return false;
    }
    let keep = (char_count / 2).max(1);
    *text = text.chars().take(keep).collect();
    true
}

enum EmbeddingFetchError {
    Oversize(String),
    Other(String),
}

async fn fetch_embedding_once(
    text: &str,
    cfg: &SearchEmbeddingConfig,
) -> Result<Vec<f32>, EmbeddingFetchError> {
    let is_google = is_google_embedding_config(cfg);
    let is_doubao_multimodal = is_doubao_multimodal_embedding_config(cfg);
    let endpoint = if is_google {
        google_embedding_endpoint(cfg)
    } else {
        volcengine_embedding_endpoint(cfg)
    };
    let mut req = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            SEARCH_EMBEDDING_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding HTTP client error: {e}")))?
        .post(&endpoint)
        .header("Content-Type", "application/json");
    // Browser-based local model servers often require a browser-like
    // Origin even when the request is routed through Rust. Keep this
    // reserved so user-supplied extra headers cannot override it.
    if is_local_or_private_http_endpoint(&endpoint) {
        req = req.header("Origin", "http://localhost");
    }
    if !cfg.api_key.trim().is_empty() {
        if is_google {
            req = req.header("x-goog-api-key", cfg.api_key.trim());
        } else {
            req = req.bearer_auth(cfg.api_key.trim());
        }
    }
    if let Some(extra) = cfg.extra_headers.as_ref() {
        for (name, value) in extra {
            let trimmed = name.trim();
            let value = value.trim();
            if trimmed.is_empty() || value.is_empty() || !is_safe_extra_header_name(trimmed) {
                continue;
            }
            if is_reserved_extra_header_name(trimmed) {
                continue;
            }
            req = req.header(trimmed, value);
        }
    }
    let body = if is_google {
        google_embedding_body(&cfg.model, text, cfg.output_dimensionality)
    } else if is_doubao_multimodal {
        doubao_multimodal_embedding_body(&cfg.model, text)
    } else {
        json!({ "model": cfg.model, "input": text })
    };
    let resp = req
        .json(&body)
        .send()
        .await
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding request failed: {e}")))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| EmbeddingFetchError::Other(format!("Embedding response read failed: {e}")))?;
    if !status.is_success() {
        let preview = text.chars().take(200).collect::<String>();
        if looks_like_oversize_error(status.as_u16(), &text) {
            return Err(EmbeddingFetchError::Oversize(format!(
                "Embedding API HTTP {status}: {preview}"
            )));
        }
        return Err(EmbeddingFetchError::Other(format!(
            "Embedding API HTTP {status}: {preview}"
        )));
    }
    let data: Value = serde_json::from_str(&text).map_err(|e| {
        EmbeddingFetchError::Other(format!(
            "Embedding response parse failed: {e}: {}",
            text.chars().take(200).collect::<String>()
        ))
    })?;
    parse_embedding_values(&data, is_google, is_doubao_multimodal)
        .map_err(EmbeddingFetchError::Other)
}

fn looks_like_oversize_error(status: u16, body: &str) -> bool {
    if status == 413 {
        return true;
    }
    let lower = body.to_lowercase();
    lower.contains("too long")
        || lower.contains("maximum context")
        || lower.contains("max_tokens")
        || lower.contains("max tokens")
        || lower.contains("context length")
        || lower.contains("token limit")
        || lower.contains("exceeds")
        || lower.contains("input length")
}

fn parse_embedding_values(
    data: &Value,
    is_google: bool,
    is_doubao_multimodal: bool,
) -> Result<Vec<f32>, String> {
    let values = if is_google {
        data.get("embedding")
            .and_then(|v| v.get("values"))
            .and_then(Value::as_array)
    } else if is_doubao_multimodal {
        data.get("data")
            .and_then(|v| v.get("embedding"))
            .and_then(Value::as_array)
    } else {
        data.get("data")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|v| v.get("embedding"))
            .and_then(Value::as_array)
    }
    .ok_or_else(|| "Embedding response missing vector".to_string())?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let n = value
            .as_f64()
            .ok_or_else(|| "Embedding response contains non-number values".to_string())?;
        if !n.is_finite() {
            return Err("Embedding response contains non-finite values".to_string());
        }
        out.push(n as f32);
    }
    if out.is_empty() {
        return Err("Embedding response vector is empty".to_string());
    }
    Ok(out)
}

fn is_safe_extra_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            matches!(
                b,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
                    | b'0'..=b'9'
                    | b'A'..=b'Z'
                    | b'a'..=b'z'
            )
        })
}

fn is_reserved_extra_header_name(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "content-type" | "host" | "content-length" | "origin" | "x-goog-api-key"
    )
}

fn is_google_embedding_config(cfg: &SearchEmbeddingConfig) -> bool {
    let endpoint = cfg.endpoint.to_lowercase();
    endpoint.contains("generativelanguage.googleapis.com") || endpoint.contains(":embedcontent")
}

fn is_local_or_private_http_endpoint(endpoint: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .trim_matches('[')
        .trim_matches(']')
        .to_ascii_lowercase();
    if matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return true;
    }
    let octets = host
        .split('.')
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>();
    let Ok(octets) = octets else {
        return false;
    };
    if octets.len() != 4 {
        return false;
    }
    octets[0] == 10
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 168)
        || octets[0] == 127
}

fn is_volcengine_embedding_endpoint(endpoint: &str) -> bool {
    let host = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_lowercase()))
        .unwrap_or_else(|| {
            let trimmed = endpoint.trim();
            trimmed
                .split_once("://")
                .map(|(_, rest)| rest)
                .unwrap_or(trimmed)
                .split(['/', '?', '#'])
                .next()
                .unwrap_or("")
                .to_lowercase()
        });
    host == "volces.com" || host.ends_with(".volces.com") || host.contains("volcengine")
}

fn is_doubao_multimodal_embedding_config(cfg: &SearchEmbeddingConfig) -> bool {
    cfg.model
        .trim()
        .to_lowercase()
        .contains("doubao-embedding-vision")
}

fn volcengine_embedding_endpoint(cfg: &SearchEmbeddingConfig) -> String {
    let raw = cfg.endpoint.trim();
    if !is_volcengine_embedding_endpoint(raw) {
        return raw.to_string();
    }
    let suffix = if is_doubao_multimodal_embedding_config(cfg) {
        "/embeddings/multimodal"
    } else {
        "/embeddings"
    };
    append_endpoint_path(raw, suffix)
}

fn append_endpoint_path(endpoint: &str, target_suffix: &str) -> String {
    let suffix = target_suffix.trim_start_matches('/');
    match reqwest::Url::parse(endpoint) {
        Ok(mut url) => {
            let path = url.path().trim_end_matches('/').to_string();
            let lower_path = path.to_lowercase();
            let lower_suffix = format!("/{}", suffix.to_lowercase());
            if lower_path.ends_with(&lower_suffix) {
                url.set_path(if path.is_empty() { "/" } else { &path });
                return url.to_string();
            }
            if lower_path.ends_with("/embeddings/multimodal") && lower_suffix == "/embeddings" {
                let base = path.trim_end_matches("/multimodal");
                url.set_path(if base.is_empty() { "/" } else { base });
                return url.to_string();
            }
            if lower_path.ends_with("/embeddings") && lower_suffix == "/embeddings/multimodal" {
                url.set_path(&format!("{path}/multimodal"));
                return url.to_string();
            }
            let next = format!("{path}/{suffix}").replace("//", "/");
            url.set_path(&next);
            url.to_string()
        }
        Err(_) => {
            let (base, query) = endpoint.split_once('?').unwrap_or((endpoint, ""));
            let trimmed = base.trim_end_matches('/');
            let lower = trimmed.to_lowercase();
            let lower_suffix = format!("/{}", suffix.to_lowercase());
            let next = if lower.ends_with(&lower_suffix) {
                trimmed.to_string()
            } else if lower.ends_with("/embeddings/multimodal") && lower_suffix == "/embeddings" {
                trimmed.trim_end_matches("/multimodal").to_string()
            } else if lower.ends_with("/embeddings") && lower_suffix == "/embeddings/multimodal" {
                format!("{trimmed}/multimodal")
            } else {
                format!("{trimmed}/{suffix}")
            };
            if query.is_empty() {
                next
            } else {
                format!("{next}?{query}")
            }
        }
    }
}

fn google_embedding_endpoint(cfg: &SearchEmbeddingConfig) -> String {
    let raw = strip_google_api_key_query(cfg.endpoint.trim())
        .trim_end_matches('/')
        .to_string();
    if raw.to_lowercase().contains(":batchembedcontents") {
        return raw
            .replace(":batchEmbedContents", ":embedContent")
            .replace(":batchembedcontents", ":embedContent");
    }
    if raw.to_lowercase().contains(":embedcontent") {
        return raw;
    }
    let model = cfg.model.trim().trim_start_matches("models/");
    if raw.to_lowercase().contains("/models/") {
        format!("{raw}:embedContent")
    } else {
        format!("{raw}/models/{model}:embedContent")
    }
}

fn strip_google_api_key_query(endpoint: &str) -> String {
    if !endpoint.contains('?') {
        return endpoint.to_string();
    }
    match reqwest::Url::parse(endpoint) {
        Ok(mut url) => {
            let kept = url
                .query_pairs()
                .filter(|(key, _)| !key.eq_ignore_ascii_case("key"))
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect::<Vec<_>>();
            url.query_pairs_mut().clear().extend_pairs(kept);
            url.to_string().trim_end_matches('?').to_string()
        }
        Err(_) => endpoint
            .split_once('?')
            .map(|(base, query)| {
                let kept = query
                    .split('&')
                    .filter(|pair| {
                        let key = pair.split_once('=').map(|(k, _)| k).unwrap_or(*pair);
                        !key.eq_ignore_ascii_case("key")
                    })
                    .collect::<Vec<_>>();
                if kept.is_empty() {
                    base.to_string()
                } else {
                    format!("{base}?{}", kept.join("&"))
                }
            })
            .unwrap_or_else(|| endpoint.to_string()),
    }
}

fn google_embedding_body(model: &str, text: &str, output_dimensionality: Option<f64>) -> Value {
    let model_path = if model.trim().starts_with("models/") {
        model.trim().to_string()
    } else {
        format!("models/{}", model.trim())
    };
    let mut body = json!({
        "model": model_path,
        "content": { "parts": [{ "text": text }] },
    });
    if let Some(dim) = output_dimensionality
        .filter(|dim| dim.is_finite() && *dim >= 1.0)
        .map(|dim| dim.floor() as u32)
    {
        body["output_dimensionality"] = json!(dim);
    }
    body
}

fn doubao_multimodal_embedding_body(model: &str, text: &str) -> Value {
    json!({
        "model": model,
        "encoding_format": "float",
        "input": [{ "type": "text", "text": text }],
    })
}

/// The YAML frontmatter block's inner text, if the page opens with one.
///
/// An unterminated block yields `None`: a malformed page is treated as all
/// body, which is the safe direction — we would rather show YAML than show
/// nothing.
fn frontmatter_span(content: &str) -> Option<(usize, usize, usize)> {
    let after_fence = if content.starts_with("---\n") {
        4
    } else if content.starts_with("---\r\n") {
        5
    } else {
        return None;
    };
    let mut cursor = after_fence;
    for line in content[after_fence..].split_inclusive('\n') {
        if line.trim_end() == "---" {
            return Some((after_fence, cursor, cursor + line.len()));
        }
        cursor += line.len();
    }
    None
}

fn frontmatter_block(content: &str) -> Option<&str> {
    frontmatter_span(content).map(|(start, end, _)| &content[start..end])
}

/// Byte offset of the first character of the page body — i.e. past any leading
/// YAML frontmatter. `0` when the page has none.
fn body_offset(content: &str) -> usize {
    frontmatter_span(content)
        .map(|(_, _, body)| body.min(content.len()))
        .unwrap_or(0)
}

/// First index at or after `from` where `needle` occurs in `haystack`.
/// Char-slice search, because snippet offsets must be char-indexed (see
/// [`build_snippet_from_anchors`]).
fn find_chars(haystack: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() || needle.len() > haystack.len() - from {
        return None;
    }
    (from..=haystack.len() - needle.len())
        .find(|start| &haystack[*start..start + needle.len()] == needle)
}

/// Byte offset of the first `##`-or-deeper heading in the body.
///
/// Everything above it is the H1 title and any preamble — text that repeats the
/// page's identifiers and asserts nothing. A match inside a titled section is
/// preferred over one there, which is what "the first substantive section"
/// means for these pages.
fn first_section_offset(content: &str, body_start: usize) -> Option<usize> {
    let mut cursor = body_start;
    for line in content[body_start..].split_inclusive('\n') {
        if line.trim_start().starts_with("##") {
            return Some(cursor);
        }
        cursor += line.len();
    }
    None
}

/// The nearest Markdown heading at or above `offset`, for labelling a snippet
/// with the section it came from. Mirrors the `heading: text` shape that vector
/// snippets already use, so both retrieval paths read the same.
fn enclosing_heading(content: &str, offset: usize) -> Option<String> {
    let mut heading = None;
    let mut cursor = 0usize;
    for line in content.split_inclusive('\n') {
        if cursor > offset {
            break;
        }
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let text = trimmed.trim_start_matches('#').trim();
            if !text.is_empty() {
                heading = Some(text.to_string());
            }
        }
        cursor += line.len();
    }
    heading
}

pub fn build_snippet(content: &str, query: &str) -> String {
    build_snippet_from_anchors(content, std::slice::from_ref(&query))
}

/// Pick the excerpt window.
///
/// Two rules, both from F-5 (`12_QA/e2e_results_2026-07-28.md`): every mirrored
/// page opens with a YAML block whose `aliases` are engineered to match, so the
/// first token hit almost always lands in the frontmatter and the window closes
/// before the section that actually asserts the fact.
///
/// 1. Anchors are sought in the **body only**, and the window is clamped so it
///    can never open inside the frontmatter.
/// 2. When no anchor occurs in the body, the window opens at the start of the
///    body rather than at byte 0 — substance beats YAML even with no match.
///
/// Frontmatter-only pages fall back to the whole content so they still yield
/// something.
pub fn build_snippet_from_anchors(content: &str, anchors: &[&str]) -> String {
    let char_positions: Vec<usize> = content.char_indices().map(|(idx, _)| idx).collect();
    if char_positions.is_empty() {
        return String::new();
    }
    let last_char = char_positions.len() - 1;
    // Matching happens in *char* space. Folding is 1:1 by char but not by byte
    // (`é` is two bytes, `e` is one), so a byte offset into the folded text
    // would not address the same place in the source once a page carries any
    // accent — which every page in this corpus does.
    let folded: Vec<char> = content.chars().map(fold_char).collect();
    let byte_to_char = |byte: usize| {
        char_positions
            .iter()
            .position(|candidate| *candidate >= byte)
            .unwrap_or(last_char)
    };

    let body_byte = body_offset(content);
    // A page that is nothing but frontmatter still has to return an excerpt.
    let body_byte = if content[body_byte..].trim().is_empty() {
        0
    } else {
        body_byte
    };
    let body_start_char = byte_to_char(body_byte);

    let first_section_char = first_section_offset(content, body_byte).map(byte_to_char);

    // Two passes. The first only considers matches inside a titled section, so
    // "## What it proves" wins over the H1 line that merely repeats the docid.
    // The second falls back to anywhere in the body.
    let mut search_starts = Vec::with_capacity(2);
    if let Some(section) = first_section_char {
        search_starts.push(section);
    }
    search_starts.push(body_start_char);

    // Within a pass, anchors are tried in priority order, not merged: the
    // caller passes the matched phrase first, then single tokens. Taking the
    // earliest match across all of them would let a common token outrank the
    // phrase that actually explains why the page ranked.
    let match_idx = search_starts.into_iter().find_map(|from| {
        anchors
            .iter()
            .filter(|anchor| !anchor.trim().is_empty())
            .find_map(|anchor| {
                let needle: Vec<char> = anchor.chars().map(fold_char).collect();
                find_chars(&folded, &needle, from).map(|found| (found, needle.len()))
            })
    });
    let (match_char, anchor_chars) = match_idx.unwrap_or((body_start_char, 1));

    // A match above the first titled section is in the H1 or preamble — text
    // that restates the page's own identifiers. Spend the whole window forward
    // from it, because whatever asserts the fact is below, not above. Matches
    // inside a section (and pages with no sections) keep the centred window.
    let matched_in_preamble = first_section_char.is_some_and(|section| match_char < section);
    let (lead, trail) = if matched_in_preamble {
        (0, SNIPPET_CONTEXT * 2)
    } else {
        (SNIPPET_CONTEXT, SNIPPET_CONTEXT)
    };
    let start_char = match_char
        .saturating_sub(lead)
        .max(body_start_char)
        .min(last_char);
    let end_char = (match_char + anchor_chars.max(1) + trail).min(char_positions.len());
    let start = char_positions[start_char];
    let end = if end_char < char_positions.len() {
        char_positions[end_char]
    } else {
        content.len()
    };
    let mut snippet = content[start..end.max(start)].replace('\n', " ");
    if start > 0 {
        snippet = format!("...{snippet}");
    }
    if end < content.len() {
        snippet.push_str("...");
    }
    // Name the section when the window does not already contain its heading,
    // so "What it proves" is legible even on a mid-section window.
    match enclosing_heading(content, start) {
        Some(heading) if !snippet.contains(heading.as_str()) => format!("{heading}: {snippet}"),
        _ => snippet,
    }
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn relative_to_project(project_path: &str, path: &Path) -> String {
    let root = Path::new(project_path);
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_project() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("llm-wiki-search-test-{id}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join("wiki/concepts")).unwrap();
        path
    }

    fn write_page(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    fn result(path: &str) -> ProjectSearchResult {
        ProjectSearchResult {
            path: path.to_string(),
            title: path.to_string(),
            snippet: String::new(),
            title_match: false,
            score: 0.0,
            vector_score: None,
            images: vec![],
            content: None,
            graph_related_to: Vec::new(),
        }
    }

    #[test]
    fn tokenizes_cjk_bigrams_and_chars() {
        let tokens = tokenize_query("默会知识");
        assert!(tokens.contains(&"默会".to_string()));
        assert!(tokens.contains(&"知识".to_string()));
        assert!(tokens.contains(&"默".to_string()));
    }

    #[test]
    fn extracts_image_refs_without_duplicates() {
        let refs = extract_image_refs("![a](wiki/media/x.png)\n![b](wiki/media/x.png)");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].alt, "a");
    }

    #[test]
    fn extract_title_uses_frontmatter_or_heading_not_body_title_lines() {
        let with_frontmatter = "---\ntitle: Real Title\n---\n\ntitle: Body Label\n# Heading";
        assert_eq!(
            extract_title(with_frontmatter, "fallback-name.md"),
            "Real Title"
        );

        let without_frontmatter = "intro\ntitle: Body Label\n# Real Heading";
        assert_eq!(
            extract_title(without_frontmatter, "fallback-name.md"),
            "Real Heading"
        );

        assert_eq!(
            extract_title("plain body", "vector-database.md"),
            "vector database"
        );
    }

    #[test]
    fn explicit_query_embedding_is_validated() {
        assert!(validate_query_embedding(vec![0.1, 0.2]).is_ok());
        assert!(validate_query_embedding(vec![]).is_err());
        assert!(validate_query_embedding(vec![f32::NAN]).is_err());
        assert!(validate_query_embedding(vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn google_embedding_endpoint_strips_key_and_normalizes_batch_endpoint() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=URL_KEY&alt=json".to_string(),
            api_key: "HEADER_KEY".to_string(),
            model: "gemini-embedding-001".to_string(),
            output_dimensionality: Some(768.0),
            extra_headers: None,
        };

        let endpoint = google_embedding_endpoint(&cfg);
        assert!(endpoint.contains(":embedContent"));
        assert!(!endpoint.contains(":batchEmbedContents"));
        assert!(!endpoint.contains("URL_KEY"));
        assert!(endpoint.contains("alt=json"));

        let body = google_embedding_body("gemini-embedding-001", "hello", Some(768.0));
        assert_eq!(body["model"], "models/gemini-embedding-001");
        assert_eq!(body["output_dimensionality"], 768);
    }

    #[test]
    fn volcengine_embedding_endpoint_only_appends_for_volcengine_hosts() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-text-240715".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );

        let custom = SearchEmbeddingConfig {
            endpoint: "https://gateway.example.com/proxy/volcengine?upstream=volces.com"
                .to_string(),
            ..cfg.clone()
        };
        assert_eq!(
            volcengine_embedding_endpoint(&custom),
            "https://gateway.example.com/proxy/volcengine?upstream=volces.com"
        );
    }

    #[test]
    fn doubao_multimodal_endpoint_and_body_use_vision_shape() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ark.cn-beijing.volces.com/api/v3/embeddings?trace=1".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-vision".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };

        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal?trace=1"
        );

        let body = doubao_multimodal_embedding_body("doubao-embedding-vision", "hello");
        assert_eq!(body["model"], "doubao-embedding-vision");
        assert_eq!(body["encoding_format"], "float");
        assert_eq!(body["input"][0]["type"], "text");
        assert_eq!(body["input"][0]["text"], "hello");
    }

    #[test]
    fn volcengine_embedding_endpoint_does_not_duplicate_existing_suffixes() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://ARK.cn-beijing.volces.com/api/v3/embeddings".to_string(),
            api_key: "ARK_KEY".to_string(),
            model: "doubao-embedding-text-240715".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );

        let vision = SearchEmbeddingConfig {
            endpoint: "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal".to_string(),
            model: "doubao-embedding-vision".to_string(),
            ..cfg.clone()
        };
        assert_eq!(
            volcengine_embedding_endpoint(&vision),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
        );

        let text_on_multimodal_endpoint = SearchEmbeddingConfig {
            model: "doubao-embedding-text-240715".to_string(),
            ..vision
        };
        assert_eq!(
            volcengine_embedding_endpoint(&text_on_multimodal_endpoint),
            "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        );
    }

    #[test]
    fn doubao_multimodal_detection_is_model_driven_for_custom_gateways() {
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "https://gateway.example.com/ark/embeddings/multimodal".to_string(),
            api_key: "KEY".to_string(),
            model: "doubao-embedding-vision".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };

        assert!(is_doubao_multimodal_embedding_config(&cfg));
        assert_eq!(
            volcengine_embedding_endpoint(&cfg),
            "https://gateway.example.com/ark/embeddings/multimodal"
        );

        let body = doubao_multimodal_embedding_body(&cfg.model, "hello");
        assert_eq!(body["input"][0]["type"], "text");
        assert_eq!(body["input"][0]["text"], "hello");
    }

    #[test]
    fn doubao_multimodal_response_shape_is_distinct_from_openai_shape() {
        let values =
            parse_embedding_values(&json!({ "data": { "embedding": [0.1, 0.2] } }), false, true)
                .expect("vision response should parse");
        assert_eq!(values, vec![0.1, 0.2]);

        assert!(
            parse_embedding_values(&json!({ "data": [{ "embedding": [0.1] }] }), false, true,)
                .is_err()
        );
    }

    #[test]
    fn extra_embedding_header_names_are_validated_and_reserved_names_are_skipped() {
        assert!(is_safe_extra_header_name("X-Model-Provider-Id"));
        assert!(is_safe_extra_header_name("x_trace.id"));
        assert!(!is_safe_extra_header_name(""));
        assert!(!is_safe_extra_header_name("Bad Header"));
        assert!(!is_safe_extra_header_name("中文"));

        assert!(is_reserved_extra_header_name("Authorization"));
        assert!(is_reserved_extra_header_name("content-type"));
        assert!(is_reserved_extra_header_name("Origin"));
        assert!(is_reserved_extra_header_name("X-Goog-Api-Key"));
        assert!(!is_reserved_extra_header_name("X-Model-Provider-Id"));
    }

    #[test]
    fn embedding_origin_header_is_limited_to_local_or_private_endpoints() {
        assert!(is_local_or_private_http_endpoint(
            "http://127.0.0.1:1234/v1/embeddings"
        ));
        assert!(is_local_or_private_http_endpoint(
            "http://192.168.1.20:11434/v1/embeddings"
        ));
        assert!(is_local_or_private_http_endpoint(
            "http://172.16.0.5/v1/embeddings"
        ));
        assert!(!is_local_or_private_http_endpoint(
            "https://api.openai.com/v1/embeddings"
        ));
    }

    #[test]
    fn embedding_halving_never_splits_cjk_codepoints() {
        let mut text = "默会知识库".to_string();
        assert!(halve_text_on_char_boundary(&mut text));
        assert_eq!(text, "默会");
    }

    #[test]
    fn google_embedding_body_floors_positive_dimensions_and_omits_invalid_values() {
        let body = google_embedding_body("gemini-embedding-001", "hello", Some(1.9));
        assert_eq!(body["output_dimensionality"], 1);

        let zero = google_embedding_body("gemini-embedding-001", "hello", Some(0.0));
        assert!(zero.get("output_dimensionality").is_none());

        let negative = google_embedding_body("gemini-embedding-001", "hello", Some(-4.0));
        assert!(negative.get("output_dimensionality").is_none());
    }

    #[test]
    fn rrf_combines_token_and_vector_ranks_and_keeps_vector_score() {
        let mut results = vec![
            result("wiki/concepts/both.md"),
            result("wiki/concepts/token-only.md"),
            result("wiki/concepts/vector-only.md"),
        ];
        let token_rank = BTreeMap::from([
            ("wiki/concepts/both.md".to_string(), 1),
            ("wiki/concepts/token-only.md".to_string(), 2),
        ]);
        let vector_rank = BTreeMap::from([("both".to_string(), 1), ("vector-only".to_string(), 2)]);
        let vector_score =
            BTreeMap::from([("both".to_string(), 0.95), ("vector-only".to_string(), 0.8)]);

        apply_rrf_scores(&mut results, &token_rank, &vector_rank, &vector_score);
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        assert_eq!(results[0].path, "wiki/concepts/both.md");
        assert!((results[0].score - (1.0 / 61.0 + 1.0 / 61.0)).abs() < 0.000001);
        assert_eq!(results[0].vector_score, Some(0.95));
        assert!((results[1].score - (1.0 / 62.0)).abs() < 0.000001);
        assert!((results[2].score - (1.0 / 62.0)).abs() < 0.000001);
    }

    #[test]
    fn search_mode_distinguishes_keyword_vector_and_hybrid() {
        assert_eq!(search_mode(false, 0, 0), "keyword");
        assert_eq!(search_mode(true, 3, 0), "vector");
        assert_eq!(search_mode(false, 3, 0), "hybrid");
        assert_eq!(search_mode(false, 0, 1), "hybrid");
    }

    #[test]
    fn graph_quota_scales_from_thirty_to_fifteen_percent() {
        assert_eq!(graph_result_quota(1, 0), 0);
        assert_eq!(graph_result_quota(20, 0), 6);
        assert_eq!(graph_result_quota(20, 10), 5);
        assert_eq!(graph_result_quota(20, 20), 3);
        assert_eq!(graph_result_quota(10, 100), 2);
    }

    #[test]
    fn vector_only_materialization_uses_chunk_snippet_and_any_wiki_subdir() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/custom/deep-page.md",
            "---\ntitle: Deep Page\n---\n\n# Deep Page\n\nThe literal query is absent here.",
        );
        let vector_results = vec![PageVectorResult {
            id: "deep-page".to_string(),
            score: 0.91,
            chunk_text: "A semantic chunk explains the actual reason for retrieval.".to_string(),
            heading_path: "Section > Detail".to_string(),
        }];
        let mut results = Vec::new();
        let pages = BTreeMap::from([(
            "deep-page".to_string(),
            "wiki/custom/deep-page.md".to_string(),
        )]);

        materialize_vector_only_results(
            &vector_results,
            &pages,
            &root.to_string_lossy(),
            &mut results,
            false,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "wiki/custom/deep-page.md");
        assert_eq!(results[0].title, "Deep Page");
        assert_eq!(results[0].vector_score, Some(0.91));
        assert!(results[0].snippet.contains("Section > Detail"));
        assert!(results[0].snippet.contains("semantic chunk"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn vector_snippet_empty_chunk_does_not_echo_query() {
        let vector = PageVectorResult {
            id: "empty".to_string(),
            score: 0.5,
            chunk_text: "  ".to_string(),
            heading_path: "Heading".to_string(),
        };

        assert_eq!(build_vector_snippet(&vector), "");
    }

    #[tokio::test]
    async fn keyword_search_prefers_filename_exact_match() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/attention.md",
            "---\ntitle: Attention\n---\n\n# Attention\n\nbody about attention.",
        );
        write_page(
            &root,
            "wiki/concepts/random.md",
            "---\ntitle: Random\n---\n\n# Random\n\nattention is mentioned briefly.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "attention".into(),
            20,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "keyword");
        assert_eq!(out.results[0].title, "Attention");
        assert!(out.results[0].title_match);
        assert!(out.results[0].score > 100.0);
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_always_blends_available_graph_neighbors() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/agent.md",
            "---\ntitle: Agent Runtime\n---\n\n# Agent Runtime\n\nagent runtime details. [[Tool Registry]]",
        );
        write_page(
            &root,
            "wiki/concepts/tool-registry.md",
            "---\ntitle: Tool Registry\n---\n\n# Tool Registry\n\nDefines callable tools.",
        );
        write_page(
            &root,
            "wiki/concepts/unrelated.md",
            "---\ntitle: Unrelated\n---\n\n# Unrelated\n\nNo graph connection.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "agent runtime".into(),
            10,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "hybrid");
        assert_eq!(out.graph_hits, 1);
        assert!(out
            .results
            .iter()
            .any(|result| result.title == "Agent Runtime"));
        assert!(out.results.iter().any(|result| {
            result.title == "Tool Registry"
                && result.snippet.contains("Graph neighbor")
                && result.graph_related_to == vec!["Agent Runtime"]
        }));
        assert!(!out.results.iter().any(|result| result.title == "Unrelated"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn page_links_resolve_outgoing_backlinks_and_missing_targets() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/Alpha.md",
            "---\ntitle: Alpha\n---\n# Alpha\n\n[[Beta|label]], [[Alpha]], [[Title Only]], and [[Missing Page]].",
        );
        write_page(
            &root,
            "wiki/concepts/Beta.md",
            "---\ntitle: Beta\n---\n# Beta\n\nBeta details.",
        );
        write_page(
            &root,
            "wiki/concepts/different-filename.md",
            "---\ntitle: Title Only\n---\n# Title Only\n",
        );
        write_page(
            &root,
            "wiki/concepts/gamma.md",
            "---\ntitle: Gamma\n---\n# Gamma\n\nGamma references [[Alpha]] here.",
        );

        let links = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/Alpha.md").to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(links.outgoing.len(), 1);
        assert_eq!(links.outgoing[0].title, "Beta");
        assert_eq!(links.backlinks.len(), 1);
        assert_eq!(links.backlinks[0].title, "Gamma");
        assert!(links.backlinks[0]
            .snippet
            .as_deref()
            .unwrap()
            .contains("Alpha"));
        assert_eq!(links.missing.len(), 2);
        assert!(links
            .missing
            .iter()
            .any(|entry| entry.title == "Missing Page"));
        assert!(links
            .missing
            .iter()
            .any(|entry| entry.title == "Title Only"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn page_links_require_markdown_input_and_exact_reader_paths() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/current.md",
            "[[wiki/concepts/target.md]] [[target#section]]",
        );
        write_page(&root, "wiki/concepts/target.md", "# Target");
        write_page(&root, "wiki/concepts/not-markdown.txt", "text");

        let links = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/current.md").to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(links.outgoing.len(), 1);
        assert_eq!(
            links.outgoing[0].path.as_deref(),
            Some("wiki/concepts/target.md")
        );
        assert_eq!(links.missing.len(), 1);
        assert_eq!(links.missing[0].title, "target#section");

        let error = get_page_links_inner(
            root.to_str().unwrap(),
            root.join("wiki/concepts/not-markdown.txt")
                .to_str()
                .unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("Markdown"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_handles_cjk_bigram_queries() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/tacit.md",
            "---\ntitle: 默会知识\n---\n\n# 默会知识\n\n默会知识强调难以言明的实践经验。",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "默会知识".into(),
            20,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(out.results[0].title, "默会知识");
        assert!(out.results[0].snippet.contains("默会知识"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn keyword_search_phrase_in_content_beats_scattered_tokens() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/phrase.md",
            "---\ntitle: Phrase\n---\n\n# Phrase\n\nThe phrase vector database appears together.",
        );
        write_page(
            &root,
            "wiki/concepts/scattered.md",
            "---\ntitle: Scattered\n---\n\n# Scattered\n\nvector appears here. database appears later.",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "vector database".into(),
            20,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(out.results[0].title, "Phrase");
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn parses_openai_batch_vectors_in_input_order() {
        let response = json!({
            "data": [
                { "index": 1, "embedding": [2.0, 2.5] },
                { "index": 0, "embedding": [1.0, 1.5] }
            ]
        });
        let vectors = parse_embedding_batch_values(&response, 2).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 1.5], vec![2.0, 2.5]]);
    }

    #[test]
    fn rejects_duplicate_openai_batch_indexes() {
        let response = json!({
            "data": [
                { "index": 0, "embedding": [1.0] },
                { "index": 0, "embedding": [2.0] }
            ]
        });
        assert!(parse_embedding_batch_values(&response, 2)
            .unwrap_err()
            .contains("duplicate"));
    }

    // ---------------------------------------------------------------------
    // Regressions for the retrieval defects recorded in bella-casefile
    // `12_QA/e2e_results_2026-07-28.md` (SARAH-E2E-RUN, Surface A NO-GO):
    // F-5 frontmatter-led snippets, F-3 question-hostile ranking,
    // F-7 diacritic-sensitive matching, and silent mode degradation.
    // ---------------------------------------------------------------------

    /// The EVD-0019 page shape from F-5: a YAML head whose aliases are
    /// engineered to match, then the section that actually asserts the fact.
    const FRONTMATTER_LED_PAGE: &str = "---\n\
type: \"document\"\n\
docid: \"EVD-0019\"\n\
privilege: \"Public\"\n\
production: \"producible (Public)\"\n\
category: \"Evidence\"\n\
source: \"01_Catalog/master_catalog.csv:56\"\n\
aliases:\n  \
- \"EVD-0019\"\n  \
- \"Chapel Royal title\"\n\
---\n\
\n\
# EVD-0019 — HM Land Registry Chapel Royal ESX146745 OFFICIAL COPY\n\
\n\
## What it proves\n\
\n\
Certified Chapel Royal title — UPGRADES EVD-0008 from G2 to G1.\n";

    fn write_counsel_corpus(root: &Path) {
        // The asserting page. Carries the literal frontmatter alias
        // "defendant counsel" but almost no bulk text.
        write_page(
            root,
            "wiki/people/arnoldo-acuna-alvarado.md",
            "---\n\
type: \"person\"\n\
group: \"Counsel & experts\"\n\
source: \"04_People/cast_of_characters.md:28\"\n\
aliases:\n  \
- \"Arnoldo Acuña Alvarado\"\n  \
- \"defendant counsel\"\n  \
- \"defense counsel\"\n\
---\n\
\n\
# Lic. Arnoldo Esteban Acuña Alvarado\n\
\n\
## Key facts\n\
\n\
Marcus's sole counsel of record.\n",
        );
        // The work-product page that displaced it in the QA run: no alias, but
        // far more textual matches on the common word.
        write_page(
            root,
            "wiki/documents/WP-0005.md",
            "---\n\
type: \"document\"\n\
docid: \"WP-0005\"\n\
privilege: \"WorkProduct\"\n\
production: \"EXCLUDED — DO NOT PRODUCE\"\n\
---\n\
\n\
# WP-0005 — Strategy chats\n\
\n\
## Summary\n\
\n\
Strategy chats with counsel. Counsel and investigators reviewed counsel notes.\n\
Counsel, counsel, counsel, counsel, counsel in this case.\n",
        );
        // Bulk-text decoys. Each carries every content word of the question
        // form ("who", "defendant", "counsel") and so outscores the asserting
        // page once the alias stops carrying — which is how a page that is
        // indexed and highly rankable leaves the window entirely.
        for index in 1..=10 {
            write_page(
                root,
                &format!("wiki/documents/TRN-{index:04}.md"),
                &format!(
                    "---\n\
type: \"document\"\n\
docid: \"TRN-{index:04}\"\n\
privilege: \"WorkProduct\"\n\
production: \"EXCLUDED — DO NOT PRODUCE\"\n\
---\n\
\n\
# TRN-{index:04} — Transcript\n\
\n\
## Summary\n\
\n\
Notes on who attended, which counsel spoke for the defendant, and what the \
defendant's counsel said in this case.\n"
                ),
            );
        }
    }

    fn rank_of(out: &ProjectSearchResponse, path: &str) -> Option<usize> {
        out.results
            .iter()
            .position(|result| normalize_path(&result.path) == path)
            .map(|index| index + 1)
    }

    // --- Task 1 / F-5 -----------------------------------------------------

    #[test]
    fn frontmatter_span_handles_absent_and_unterminated_blocks() {
        assert_eq!(body_offset("# No frontmatter\n\nbody"), 0);
        // An unterminated block must not swallow the whole page.
        assert_eq!(body_offset("---\ntype: \"document\"\nstill open"), 0);
        let closed = "---\ntitle: X\n---\nbody";
        assert_eq!(&closed[body_offset(closed)..], "body");
        let crlf = "---\r\ntitle: X\r\n---\r\nbody";
        assert_eq!(&crlf[body_offset(crlf)..], "body");
    }

    #[test]
    fn snippet_skips_frontmatter_and_returns_the_asserting_section() {
        // "EVD-0019" occurs first inside the YAML head. Before the fix the
        // window opened there and closed before "## What it proves".
        let snippet = build_snippet(FRONTMATTER_LED_PAGE, "EVD-0019");

        assert!(
            snippet.contains("Certified Chapel Royal title"),
            "snippet must carry the asserting sentence, got: {snippet}"
        );
        assert!(
            snippet.contains("UPGRADES EVD-0008 from G2 to G1"),
            "snippet must carry the G2 to G1 upgrade, got: {snippet}"
        );
        assert!(
            !snippet.contains("privilege:"),
            "snippet must not open on the YAML head, got: {snippet}"
        );
        assert!(
            !snippet.contains("aliases:"),
            "snippet must not open on the YAML head, got: {snippet}"
        );
    }

    #[test]
    fn snippet_prefers_a_titled_section_over_the_h1_preamble() {
        // The anchor occurs twice: once in the preamble, once in a titled
        // section, far enough apart that one window cannot contain both.
        let page = "---\ntitle: Alpha\n---\n\n\
# Alpha identifier\n\n\
Routing preamble that exists only to restate the identifier and to push the \
asserting section past the end of any window opened on the preamble copy of \
the anchor, which is exactly the shape that made Q2 fail.\n\n\
## Key facts\n\n\
Alpha is asserted here with the controlling fact.\n";

        let snippet = build_snippet_from_anchors(page, &["alpha"]);

        assert!(
            snippet.contains("asserted here with the controlling fact"),
            "expected the titled section, got: {snippet}"
        );
        assert!(
            !snippet.contains("Routing preamble"),
            "window should not have opened on the preamble, got: {snippet}"
        );
    }

    #[test]
    fn snippet_offsets_survive_accents_before_the_match() {
        // Folding is 1:1 by char but not by byte, so every accent ahead of the
        // anchor drifts a byte-indexed window further right. Enough of them and
        // the window slides clean past the fact it was supposed to frame.
        let mut page = String::from("---\ntitle: Expediente\n---\n\n# Página\n\n## Datos\n\n");
        page.push_str(&"áéíóúñ ".repeat(40));
        page.push_str("la cédula 9-0110-0855 consta aquí.\n\n");
        page.push_str(&"filler text that is plain ascii. ".repeat(15));

        let snippet = build_snippet_from_anchors(&page, &["9-0110-0855"]);

        assert!(
            snippet.contains("9-0110-0855"),
            "window drifted off the anchor, got: {snippet}"
        );
    }

    #[test]
    fn snippet_falls_back_to_the_body_not_the_yaml_when_nothing_matches() {
        let snippet = build_snippet(FRONTMATTER_LED_PAGE, "term-that-is-absent");
        assert!(
            !snippet.contains("privilege:"),
            "a no-match page must still return body text, got: {snippet}"
        );
        assert!(
            snippet.contains("HM Land Registry"),
            "expected the body head, got: {snippet}"
        );
    }

    #[test]
    fn snippet_uses_the_whole_page_when_there_is_no_body() {
        let only_frontmatter = "---\ntype: \"document\"\ndocid: \"EVD-0019\"\n---\n";
        let snippet = build_snippet(only_frontmatter, "EVD-0019");
        assert!(
            snippet.contains("EVD-0019"),
            "a frontmatter-only page must still yield an excerpt, got: {snippet}"
        );
    }

    #[tokio::test]
    async fn search_returns_the_asserting_section_for_a_frontmatter_led_page() {
        let root = tmp_project();
        write_page(&root, "wiki/documents/EVD-0019.md", FRONTMATTER_LED_PAGE);

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "What does EVD-0019 prove?".into(),
            20,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        let snippet = &out.results[0].snippet;
        assert!(
            snippet.contains("Certified Chapel Royal title"),
            "got: {snippet}"
        );
        assert!(!snippet.contains("privilege:"), "got: {snippet}");
        let _ = fs::remove_dir_all(root);
    }

    // --- Task 2 / F-3 -----------------------------------------------------

    #[test]
    fn content_phrases_survive_interrogative_wrappers() {
        // All three question forms reduce to the same content phrase as the
        // bare keyword query, which is what restores the phrase bonus.
        assert_eq!(
            content_phrases("Who is defendant counsel?"),
            vec!["defendant counsel".to_string()]
        );
        assert_eq!(
            content_phrases("Who is the defendant counsel in this case?"),
            vec!["defendant counsel".to_string()]
        );
        assert_eq!(
            content_phrases("Who is Marcus's sole counsel of record?"),
            vec!["sole counsel".to_string()]
        );
        // A stop word breaks a run rather than being spliced out, so words that
        // were not adjacent never form a phrase — and a run of one is not one.
        assert!(content_phrases("counsel of record").is_empty());
    }

    #[test]
    fn frontmatter_aliases_parse_block_and_inline_forms() {
        let block = "---\naliases:\n  - \"defendant counsel\"\n  - \"defense counsel\"\n---\nbody";
        assert_eq!(
            frontmatter_aliases(block),
            vec![
                "defendant counsel".to_string(),
                "defense counsel".to_string()
            ]
        );
        let inline = "---\naliases: [\"cédula 9-0110-0855\", \"Verónica Pierce\"]\n---\nbody";
        // Aliases are folded, so an unaccented query reaches them too.
        assert_eq!(
            frontmatter_aliases(inline),
            vec![
                "cedula 9-0110-0855".to_string(),
                "veronica pierce".to_string()
            ]
        );
        assert!(frontmatter_aliases("# No frontmatter").is_empty());
    }

    #[tokio::test]
    async fn question_form_keeps_the_alias_page_in_the_candidate_set() {
        let root = tmp_project();
        write_counsel_corpus(&root);
        const COUNSEL: &str = "wiki/people/arnoldo-acuna-alvarado.md";

        // Every phrasing recorded in the F-3 table, with the rank bound the
        // report itself justifies. The three interrogative forms are the ones
        // that were ABSENT from all candidates; the bare single token ranked
        // 3rd even in the passing run, so presence is its bar — F-3's claim is
        // that question form must not remove the page, not that a one-word
        // query must rank it first.
        // topK 8, as the QA run used, so "fell out of the window" is a real
        // failure mode here and not an artifact of an oversized window.
        for (probe, worst_acceptable_rank) in [
            ("Arnoldo Acuña Alvarado", 1),
            ("defendant counsel", 1),
            ("counsel", 8),
            ("Who is defendant counsel?", 1),
            ("Who is the defendant counsel in this case?", 1),
            ("Who is Marcus's sole counsel of record?", 1),
        ] {
            let out = search_project_inner(
                root.to_string_lossy().to_string(),
                probe.into(),
                8,
                false,
                None,
                Vec::new(),
            )
            .await
            .unwrap();

            let rank = rank_of(&out, COUNSEL);
            let listed = out
                .results
                .iter()
                .map(|result| result.path.clone())
                .collect::<Vec<_>>();
            assert!(
                rank.is_some_and(|rank| rank <= worst_acceptable_rank),
                "probe {probe:?} expected the counsel page at rank <= {worst_acceptable_rank}, \
                 got {rank:?}; candidates: {listed:?}"
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn unaccented_question_still_reaches_the_alias_page() {
        // F-3 and F-7 compounded: question form *and* a stripped accent.
        let root = tmp_project();
        write_counsel_corpus(&root);

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "Who is Arnoldo Acuna Alvarado?".into(),
            25,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(
            rank_of(&out, "wiki/people/arnoldo-acuna-alvarado.md"),
            Some(1)
        );
        let _ = fs::remove_dir_all(root);
    }

    // --- Task 3 / F-7 -----------------------------------------------------

    #[test]
    fn folding_is_one_to_one_and_symmetric() {
        assert_eq!(fold_for_match("Cédula"), "cedula");
        assert_eq!(fold_for_match("PENSIÓN"), "pension");
        assert_eq!(fold_for_match("Dictámen"), "dictamen");
        assert_eq!(fold_for_match("Verónica Sáenz Monge"), "veronica saenz monge");
        assert_eq!(fold_for_match("niño"), "nino");
        // Folding an already-folded string is a no-op, so callers may fold
        // defensively without changing the result.
        assert_eq!(fold_for_match("cedula"), "cedula");
        // Char count is preserved — snippet offsets depend on it.
        for sample in ["cédula", "pensión", "dictámen", "默会知识", "Ünïcödé"] {
            assert_eq!(
                fold_for_match(sample).chars().count(),
                sample.chars().count(),
                "fold must be 1:1 for {sample:?}"
            );
        }
        // Expanding folds are deliberately not applied.
        assert_eq!(fold_for_match("straße"), "straße");
    }

    #[tokio::test]
    async fn diacritic_folding_matches_both_directions() {
        let root = tmp_project();
        // Each term exists in exactly ONE spelling, and the filenames are
        // deliberately neutral: a filename carrying the unaccented form would
        // satisfy the query by itself and hide the defect. `cédula` and
        // `pensión` are accented-only; `dictamen` is unaccented-only, so both
        // directions of the fold are under test.
        write_page(
            &root,
            "wiki/glossary/termino-01.md",
            "---\ntitle: Cédula\n---\n\n# Cédula\n\n## Key facts\n\nLa cédula 9-0110-0855 identifica a la actora.\n",
        );
        write_page(
            &root,
            "wiki/glossary/termino-02.md",
            "---\ntitle: Pensión alimentaria\n---\n\n# Pensión\n\n## Key facts\n\nLa pensión alimentaria se fija por el juzgado.\n",
        );
        write_page(
            &root,
            "wiki/glossary/termino-03.md",
            "---\ntitle: Dictamen pericial\n---\n\n# Dictamen\n\n## Key facts\n\nEl dictamen pericial consta en el expediente.\n",
        );

        for (query, expected) in [
            ("cedula", "wiki/glossary/termino-01.md"),
            ("cédula", "wiki/glossary/termino-01.md"),
            ("pension", "wiki/glossary/termino-02.md"),
            ("pensión", "wiki/glossary/termino-02.md"),
            ("dictamen", "wiki/glossary/termino-03.md"),
            ("dictámen", "wiki/glossary/termino-03.md"),
        ] {
            let out = search_project_inner(
                root.to_string_lossy().to_string(),
                query.into(),
                20,
                false,
                None,
                Vec::new(),
            )
            .await
            .unwrap();

            assert!(
                !out.results.is_empty(),
                "query {query:?} returned nothing; before the fix the unaccented form returned 0 results"
            );
            assert_eq!(
                rank_of(&out, expected),
                Some(1),
                "query {query:?} should rank {expected} first"
            );
            assert!(
                out.token_hits > 0,
                "query {query:?} should register token hits"
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    // --- Task 4 / mode degradation ---------------------------------------

    #[tokio::test]
    async fn a_clean_keyword_run_reports_no_degradation() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/attention.md",
            "---\ntitle: Attention\n---\n\n# Attention\n\n## Key facts\n\nAttention is all you need.\n",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "attention".into(),
            20,
            false,
            None,
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "keyword");
        assert_eq!(out.requested_mode, None);
        assert!(out.degradations.is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn embedding_failure_is_reported_in_the_response_not_only_stderr() {
        let root = tmp_project();
        write_page(
            &root,
            "wiki/concepts/attention.md",
            "---\ntitle: Attention\n---\n\n# Attention\n\n## Key facts\n\nAttention is all you need.\n",
        );

        let out = search_project_inner(
            root.to_string_lossy().to_string(),
            "attention".into(),
            20,
            false,
            None,
            vec![SearchDegradation {
                stage: "embedding".to_string(),
                from: "hybrid".to_string(),
                to: "keyword".to_string(),
                reason: "query embedding unavailable: connection refused".to_string(),
            }],
        )
        .await
        .unwrap();

        assert_eq!(out.mode, "keyword");
        assert_eq!(out.requested_mode.as_deref(), Some("hybrid"));
        assert_eq!(out.degradations.len(), 1);
        assert_eq!(out.degradations[0].stage, "embedding");
        assert!(out.degradations[0].reason.contains("connection refused"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn resolve_query_embedding_records_why_it_gave_up() {
        // An enabled config pointing at a dead endpoint used to return
        // Ok(None) with nothing but an eprintln to show for it.
        let cfg = SearchEmbeddingConfig {
            enabled: true,
            endpoint: "http://127.0.0.1:1/v1/embeddings".to_string(),
            api_key: String::new(),
            model: "test-embed".to_string(),
            output_dimensionality: None,
            extra_headers: None,
        };

        let outcome = resolve_query_embedding("cedula", None, Some(cfg))
            .await
            .unwrap();

        assert!(outcome.embedding.is_none());
        assert_eq!(outcome.degradations.len(), 1);
        assert_eq!(outcome.degradations[0].stage, "embedding");
        assert_eq!(outcome.degradations[0].to, "keyword");
    }

    #[tokio::test]
    async fn a_disabled_embedding_config_is_not_a_degradation() {
        // Never configured is not the same as configured and broken.
        let cfg = SearchEmbeddingConfig {
            enabled: false,
            endpoint: String::new(),
            api_key: String::new(),
            model: String::new(),
            output_dimensionality: None,
            extra_headers: None,
        };

        let outcome = resolve_query_embedding("cedula", None, Some(cfg))
            .await
            .unwrap();

        assert!(outcome.embedding.is_none());
        assert!(outcome.degradations.is_empty());
    }

    #[test]
    fn degradation_suffix_is_silent_unless_something_degraded() {
        assert_eq!(degradation_suffix(None, &[]), "");
        let degradations = vec![SearchDegradation {
            stage: "vectorIndex".to_string(),
            from: "hybrid".to_string(),
            to: "keyword".to_string(),
            reason: "vector index returned no chunks".to_string(),
        }];
        let suffix = degradation_suffix(Some("hybrid"), &degradations);
        assert!(suffix.contains("DEGRADED from hybrid"));
        assert!(suffix.contains("vectorIndex"));
        assert!(suffix.contains("no chunks"));
    }
}
