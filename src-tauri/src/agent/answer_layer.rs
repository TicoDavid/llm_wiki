//! Privilege-aware answer layer for the backend Agent.
//!
//! Three behaviours, all of which must hold in *both* generated and
//! fallback (ranked-listing) modes, because a control that only exists on
//! the generated path is a control that disappears the moment the provider
//! is unreachable:
//!
//!   1. **Production warning.** When retrieved pages carry
//!      production-excluded markers, or the question is about producing /
//!      disclosing / filing material, the answer opens with an explicit
//!      warning block naming the exclusion predicate and the marked
//!      documents — before any content. (QA finding F-4: the markings were
//!      correct in the data and completely inert in the surface, so a
//!      production question returned six DO-NOT-PRODUCE documents with no
//!      signal at all.)
//!
//!   2. **Absence.** When retrieval finds nothing relevant, the answer says
//!      the fact is not in the wiki instead of presenting unrelated pages
//!      as though they answered the question.
//!
//!   3. **Citations.** Generated prose is instructed to cite page paths,
//!      and the pages it drew on are listed underneath.
//!
//! ## Perimeter doctrine
//!
//! This layer **warns; it never hides**. No document is dropped, demoted,
//! or withheld from the response — internal users see the full result set,
//! exactly as before, with the marked items named up front. Redaction is a
//! production-time concern and does not belong in an internal retrieval
//! surface: an analyst who cannot see the privileged document cannot tell
//! that it *is* privileged.

use std::collections::BTreeSet;

use super::types::AgentReference;

/// Frontmatter keys inspected for a production/privilege determination.
const PRIVILEGE_KEY: &str = "privilege";
const PRODUCTION_KEY: &str = "production";

/// Privilege values that exclude a document from production. Matched
/// case-insensitively against the whole frontmatter value. Mirrors rule
/// SEC-01 in the casefile catalog: `Privilege ∈ {WorkProduct, Privileged}`.
const EXCLUDING_PRIVILEGE_VALUES: &[&str] = &["workproduct", "work product", "privileged"];

/// DocID prefix that is production-excluded by construction — the second
/// limb of SEC-01. A `WP-` document is work product whether or not the
/// frontmatter says so.
const EXCLUDED_DOCID_PREFIX: &str = "WP-";

/// Substrings in a `production:` value that mark exclusion. Kept as
/// substrings rather than exact values because the catalog writes a
/// human sentence ("EXCLUDED — DO NOT PRODUCE"), not an enum token.
const EXCLUDING_PRODUCTION_MARKERS: &[&str] = &["do not produce", "excluded", "no producir"];

/// Query verbs that put the user in a production posture. English and
/// Spanish, because the corpus and its users are bilingual and an
/// unaccented or Spanish-language question must not slip the control.
const PRODUCTION_INTENT_TERMS: &[&str] = &[
    "produce",
    "producing",
    "production",
    "disclose",
    "disclosing",
    "disclosure",
    "turn over",
    "hand over",
    "opposing counsel",
    "other side",
    "discovery request",
    "discovery response",
    "file with the court",
    "filing",
    "serve on",
    "share with",
    "send to counsel",
    "aportar",
    "presentar",
    "divulgar",
    "entregar",
    "producir",
    "contraparte",
];

/// A retrieved page that must not be produced, with the reason why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedDocument {
    pub title: String,
    pub path: String,
    /// Human-readable predicate, e.g. `privilege: WorkProduct`. This is the
    /// thing a reader needs in order to check the call, so it is quoted
    /// from the record rather than summarised.
    pub predicate: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrivilegeAssessment {
    pub excluded: Vec<ExcludedDocument>,
    pub query_asks_about_production: bool,
}

impl PrivilegeAssessment {
    /// Warn when the *question* is about production, even if nothing
    /// retrieved is marked — the user is about to make a production
    /// decision and needs to know the surface did not clear anything.
    /// Warn when documents *are* marked, even if the question was neutral —
    /// the copy-paste risk F-4 describes does not require the user to have
    /// announced their intent.
    pub fn requires_warning(&self) -> bool {
        !self.excluded.is_empty() || self.query_asks_about_production
    }
}

/// Does this question put the user in a production/disclosure posture?
pub fn query_asks_about_production(query: &str) -> bool {
    let lowered = normalize_for_match(query);
    PRODUCTION_INTENT_TERMS
        .iter()
        .any(|term| lowered.contains(term))
}

/// Fold accents and lowercase so `producción` and `produccion` match the
/// same term. Diacritic sensitivity already cost this surface a retrieval
/// finding (F-7); it must not also cost it a privilege warning.
fn normalize_for_match(value: &str) -> String {
    value
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' | 'â' | 'Á' | 'À' | 'Ä' | 'Â' => 'a',
            'é' | 'è' | 'ë' | 'ê' | 'É' | 'È' | 'Ë' | 'Ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' | 'Í' | 'Ì' | 'Ï' | 'Î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' | 'Ó' | 'Ò' | 'Ö' | 'Ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' | 'Ú' | 'Ù' | 'Ü' | 'Û' => 'u',
            'ñ' | 'Ñ' => 'n',
            other => other.to_ascii_lowercase(),
        })
        .collect()
}

/// Read the production predicate off one page body plus its reference.
///
/// `body` may be empty when the page could not be read; detection then
/// falls back to the reference snippet and the DocID prefix, so a read
/// failure downgrades precision but never silently clears a document.
pub fn assess_document(reference: &AgentReference, body: &str) -> Option<ExcludedDocument> {
    let mut predicates: Vec<String> = Vec::new();

    if let Some(value) = frontmatter_value(body, PRIVILEGE_KEY) {
        let normalized = value.to_ascii_lowercase();
        if EXCLUDING_PRIVILEGE_VALUES
            .iter()
            .any(|marker| normalized.contains(marker))
        {
            predicates.push(format!("privilege: {value}"));
        }
    }
    if let Some(value) = frontmatter_value(body, PRODUCTION_KEY) {
        let normalized = value.to_ascii_lowercase();
        if EXCLUDING_PRODUCTION_MARKERS
            .iter()
            .any(|marker| normalized.contains(marker))
        {
            predicates.push(format!("production: {value}"));
        }
    }
    if docid_is_excluded(&reference.title) || docid_is_excluded(&reference.path) {
        predicates.push(format!("DocID prefix `{EXCLUDED_DOCID_PREFIX}` (work product)"));
    }
    // Snippet fallback: when the body is unavailable the top excerpt is
    // frequently the frontmatter block itself (F-5), so the marker is
    // often still visible there.
    if predicates.is_empty() {
        if let Some(snippet) = reference.snippet.as_deref() {
            let normalized = snippet.to_ascii_lowercase();
            if EXCLUDING_PRODUCTION_MARKERS
                .iter()
                .any(|marker| normalized.contains(marker))
            {
                predicates.push("production marker in retrieved excerpt".to_string());
            }
        }
    }

    if predicates.is_empty() {
        return None;
    }
    Some(ExcludedDocument {
        title: reference.title.clone(),
        path: reference.path.clone(),
        predicate: predicates.join(" · "),
    })
}

/// `WP-0004`, `wp-0004.md`, `documents/WP-0001` all count. The check is on
/// a path/title *segment* so an unrelated word ending in "wp" cannot match.
fn docid_is_excluded(value: &str) -> bool {
    let prefix = EXCLUDED_DOCID_PREFIX.to_ascii_lowercase();
    value
        .split(['/', '\\', ' ', '(', ')', ','])
        .any(|segment| segment.trim().to_ascii_lowercase().starts_with(&prefix))
}

/// Extract a top-level YAML frontmatter scalar. Deliberately minimal: the
/// mirrored pages are machine-written with flat `key: value` frontmatter,
/// and pulling a YAML parser in to read two keys would be a larger
/// dependency surface than the job needs.
fn frontmatter_value(content: &str, key: &str) -> Option<String> {
    let rest = content.strip_prefix("---")?;
    let rest = rest.strip_prefix('\n').or_else(|| rest.strip_prefix("\r\n"))?;
    for line in rest.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim() == "---" {
            break;
        }
        let Some((raw_key, raw_value)) = trimmed.split_once(':') else {
            continue;
        };
        if !raw_key.trim().eq_ignore_ascii_case(key) {
            continue;
        }
        let value = raw_value
            .trim()
            .trim_matches(|c| c == '"' || c == '\'')
            .trim();
        if value.is_empty() {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

/// Assess the whole retrieval set. `bodies` is parallel to `references`;
/// a missing entry is treated as an unreadable body, not a clear document.
pub fn assess(
    query: &str,
    references: &[AgentReference],
    bodies: &[Option<String>],
) -> PrivilegeAssessment {
    let mut excluded = Vec::new();
    let mut seen = BTreeSet::new();
    for (idx, reference) in references.iter().enumerate() {
        let body = bodies.get(idx).and_then(Option::as_deref).unwrap_or("");
        if let Some(document) = assess_document(reference, body) {
            if seen.insert(document.path.clone()) {
                excluded.push(document);
            }
        }
    }
    PrivilegeAssessment {
        excluded,
        query_asks_about_production: query_asks_about_production(query),
    }
}

/// Render the warning block that opens the response. Returns `None` when
/// no warning is warranted, so callers can prepend unconditionally.
pub fn render_warning_block(assessment: &PrivilegeAssessment) -> Option<String> {
    if !assessment.requires_warning() {
        return None;
    }
    let mut out = String::from("⚠️ **PRODUCTION WARNING — READ BEFORE USING THIS ANSWER**\n\n");

    if assessment.excluded.is_empty() {
        out.push_str(
            "This question concerns producing, disclosing, or filing material. No retrieved page \
             carries a production-excluded marker, but **this surface has not cleared anything \
             for production**: absence of a marker is not a production authorisation. Confirm \
             against the privilege log before releasing any document.\n",
        );
        return Some(out);
    }

    out.push_str(&format!(
        "{} of the retrieved document(s) below are marked **EXCLUDED FROM PRODUCTION**. \
         Do not produce, disclose, file, or forward them.\n\n",
        assessment.excluded.len()
    ));
    out.push_str("| Document | Path | Exclusion predicate |\n");
    out.push_str("|---|---|---|\n");
    for document in &assessment.excluded {
        out.push_str(&format!(
            "| {} | `{}` | {} |\n",
            escape_table_cell(&document.title),
            document.path,
            escape_table_cell(&document.predicate)
        ));
    }
    if assessment.query_asks_about_production {
        out.push_str(
            "\nYour question asks about production or disclosure. The answer below is a \
             **retrieval result, not a production authorisation** — it lists what the wiki holds, \
             including material that must be withheld.\n",
        );
    } else {
        out.push_str(
            "\nThe answer below lists what the wiki holds, including material that must be \
             withheld from production.\n",
        );
    }
    Some(out)
}

/// A pipe inside a cell would break the markdown table and could split one
/// document's row into two, misattributing a predicate.
fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

/// The response when retrieval found nothing to answer from. Stated as an
/// absence claim about the wiki, not about the world.
pub fn absence_answer(query: &str) -> String {
    format!(
        "**Not in the wiki.** I searched this LLM Wiki project for \"{}\" and found no page that \
         states it.\n\nThis is an absence in the indexed wiki, not a finding that the fact is \
         false — the source may exist but be un-ingested, outside the watched folders, or indexed \
         under different wording. No page is cited below because none of the retrieved pages \
         address the question.",
        collapse(query)
    )
}

/// Citation footer for generated prose. The model is asked to cite inline;
/// this block is the deterministic backstop so the set of pages the answer
/// was built from is always visible even if the model cites nothing.
pub fn render_citations(references: &[AgentReference]) -> Option<String> {
    if references.is_empty() {
        return None;
    }
    let mut out = String::from("\n\n---\n**Pages consulted**\n");
    for (idx, reference) in references.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} — `{}`\n",
            idx + 1,
            reference.title,
            reference.path
        ));
    }
    Some(out)
}

/// Instructions appended to the system prompt when a provider will
/// synthesize the answer. Absence honesty is stated as a hard rule
/// because the failure it prevents — dressing unrelated pages as an
/// answer — is exactly what a helpful-sounding model does by default.
pub fn generation_directives() -> &'static str {
    "\n\nAnswer construction rules (these override any conflicting instruction above):\n\
     - Answer in prose, directly, from the retrieved pages below. Do not emit a list of search \
     results as your answer.\n\
     - Cite the wiki page path in backticks immediately after each claim it supports, e.g. \
     `wiki/people/example.md`.\n\
     - Use ONLY the retrieved pages. Do not add facts from your own knowledge, and do not infer \
     a fact that no retrieved page states.\n\
     - If the retrieved pages do not state the answer, begin your reply with the exact text \
     'Not in the wiki.' and then say what the pages do cover. Never present a page that does not \
     answer the question as though it does.\n\
     - If a retrieved page is marked as work product, privileged, or excluded from production, \
     say so when you cite it."
}

fn collapse(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(title: &str, path: &str) -> AgentReference {
        AgentReference {
            title: title.to_string(),
            path: path.to_string(),
            kind: "wiki".to_string(),
            snippet: None,
            score: Some(1.0),
            knowledge_context: None,
        }
    }

    const WORK_PRODUCT_PAGE: &str = "---\ntitle: Trial Notes\nprivilege: WorkProduct\nproduction: \"EXCLUDED — DO NOT PRODUCE\"\n---\n\n# Trial Notes\n";
    const CLEAN_PAGE: &str =
        "---\ntitle: Cast of Characters\nprivilege: None\nproduction: Producible\n---\n\n# Cast\n";

    #[test]
    fn work_product_frontmatter_is_flagged_with_both_predicates() {
        let document = assess_document(&reference("Trial Notes", "wiki/documents/TRN-0004.md"), WORK_PRODUCT_PAGE)
            .expect("work product page must be flagged");
        assert!(document.predicate.contains("privilege: WorkProduct"));
        assert!(document.predicate.contains("EXCLUDED"));
    }

    #[test]
    fn producible_page_is_not_flagged() {
        assert_eq!(
            assess_document(&reference("Cast", "wiki/people/cast.md"), CLEAN_PAGE),
            None
        );
    }

    #[test]
    fn wp_docid_is_flagged_even_without_frontmatter() {
        // Second limb of SEC-01: the prefix alone is dispositive. A page
        // whose frontmatter was never written must not read as clean.
        let document = assess_document(&reference("WP-0004", "wiki/documents/WP-0004.md"), "")
            .expect("WP- DocID must be flagged with no frontmatter at all");
        assert!(document.predicate.contains("WP-"));
    }

    #[test]
    fn a_word_merely_starting_with_wp_does_not_match() {
        assert_eq!(
            assess_document(&reference("Whitepaper", "wiki/notes/whitepaper.md"), CLEAN_PAGE),
            None
        );
    }

    #[test]
    fn unreadable_body_falls_back_to_the_snippet_marker() {
        let mut reference = reference("Trial Notes", "wiki/documents/TRN-0001.md");
        reference.snippet = Some("privilege: WorkProduct production: EXCLUDED — DO NOT PRODUCE".to_string());
        let document = assess_document(&reference, "").expect("snippet marker must be honoured");
        assert!(document.predicate.contains("retrieved excerpt"));
    }

    #[test]
    fn production_questions_are_detected_in_both_languages_and_without_accents() {
        assert!(query_asks_about_production("Can we produce EVD-0019 to opposing counsel?"));
        assert!(query_asks_about_production("¿Podemos aportar el dictamen?"));
        assert!(query_asks_about_production("podemos producir esto"));
        assert!(query_asks_about_production("What are we disclosing in discovery response 2?"));
        assert!(!query_asks_about_production("Who is defendant counsel?"));
    }

    #[test]
    fn warning_block_opens_with_the_marked_documents_and_names_the_predicate() {
        let references = vec![
            reference("Trial Notes", "wiki/documents/TRN-0004.md"),
            reference("Cast", "wiki/people/cast.md"),
        ];
        let bodies = vec![Some(WORK_PRODUCT_PAGE.to_string()), Some(CLEAN_PAGE.to_string())];
        let assessment = assess("can we produce this to opposing counsel", &references, &bodies);
        assert_eq!(assessment.excluded.len(), 1);
        assert!(assessment.query_asks_about_production);

        let block = render_warning_block(&assessment).expect("warning required");
        assert!(block.starts_with("⚠️ **PRODUCTION WARNING"));
        assert!(block.contains("wiki/documents/TRN-0004.md"));
        assert!(block.contains("privilege: WorkProduct"));
        assert!(block.contains("not a production authorisation"));
        // Perimeter doctrine: the clean page must not be named as excluded,
        // and nothing may be removed from the caller's result set.
        assert!(!block.contains("wiki/people/cast.md"));
    }

    #[test]
    fn production_question_with_no_marked_results_still_warns() {
        let references = vec![reference("Cast", "wiki/people/cast.md")];
        let bodies = vec![Some(CLEAN_PAGE.to_string())];
        let assessment = assess("can we produce this", &references, &bodies);
        assert!(assessment.excluded.is_empty());
        let block = render_warning_block(&assessment).expect("posture alone must warn");
        assert!(block.contains("has not cleared anything"));
    }

    #[test]
    fn marked_results_warn_even_when_the_question_was_neutral() {
        let references = vec![reference("Trial Notes", "wiki/documents/TRN-0004.md")];
        let bodies = vec![Some(WORK_PRODUCT_PAGE.to_string())];
        let assessment = assess("who wrote the trial notes", &references, &bodies);
        assert!(!assessment.query_asks_about_production);
        assert!(render_warning_block(&assessment).is_some());
    }

    #[test]
    fn no_warning_when_nothing_is_marked_and_nothing_is_asked() {
        let references = vec![reference("Cast", "wiki/people/cast.md")];
        let bodies = vec![Some(CLEAN_PAGE.to_string())];
        let assessment = assess("who is defendant counsel", &references, &bodies);
        assert!(!assessment.requires_warning());
        assert_eq!(render_warning_block(&assessment), None);
    }

    #[test]
    fn duplicate_paths_are_reported_once() {
        let references = vec![
            reference("Trial Notes", "wiki/documents/TRN-0004.md"),
            reference("Trial Notes", "wiki/documents/TRN-0004.md"),
        ];
        let bodies = vec![
            Some(WORK_PRODUCT_PAGE.to_string()),
            Some(WORK_PRODUCT_PAGE.to_string()),
        ];
        assert_eq!(assess("produce", &references, &bodies).excluded.len(), 1);
    }

    #[test]
    fn table_cells_escape_pipes_so_one_document_cannot_become_two_rows() {
        let mut reference = reference("Notes | Draft", "wiki/documents/WP-0009.md");
        reference.snippet = None;
        let assessment = assess("x", std::slice::from_ref(&reference), &[Some(String::new())]);
        let block = render_warning_block(&assessment).unwrap();
        let rows = block.lines().filter(|l| l.starts_with("| ")).count();
        // Header row + one document row. A raw pipe would make it three.
        assert_eq!(rows, 2);
        assert!(block.contains("Notes \\| Draft"));
    }

    #[test]
    fn absence_answer_states_absence_and_cites_nothing() {
        let answer = absence_answer("what colour is the sky");
        assert!(answer.starts_with("**Not in the wiki.**"));
        assert!(answer.contains("no page that states it"));
        assert!(!answer.contains(".md"));
    }

    #[test]
    fn citations_list_every_page_and_are_absent_when_nothing_was_retrieved() {
        let references = vec![reference("Cast", "wiki/people/cast.md")];
        let block = render_citations(&references).unwrap();
        assert!(block.contains("`wiki/people/cast.md`"));
        assert_eq!(render_citations(&[]), None);
    }

    #[test]
    fn generation_directives_forbid_outside_knowledge_and_mandate_the_absence_phrase() {
        let directives = generation_directives();
        assert!(directives.contains("Not in the wiki."));
        assert!(directives.contains("Use ONLY the retrieved pages"));
        assert!(directives.contains("excluded from production"));
    }

    #[test]
    fn frontmatter_parsing_ignores_body_lines_that_look_like_keys() {
        let page = "---\ntitle: Real\n---\n\nprivilege: WorkProduct\n";
        assert_eq!(frontmatter_value(page, PRIVILEGE_KEY), None);
        // ...and a page with no frontmatter at all must not panic or match.
        assert_eq!(frontmatter_value("privilege: WorkProduct\n", PRIVILEGE_KEY), None);
    }

    #[test]
    fn frontmatter_parsing_handles_crlf_and_quoted_values() {
        let page = "---\r\nprivilege: \"WorkProduct\"\r\n---\r\n\r\n# Body\r\n";
        assert_eq!(
            frontmatter_value(page, PRIVILEGE_KEY).as_deref(),
            Some("WorkProduct")
        );
    }
}
