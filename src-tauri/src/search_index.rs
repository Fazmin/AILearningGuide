#![allow(dead_code)]

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchIndexFile {
    #[serde(default)]
    fingerprint: Option<String>,
    documents: Vec<IndexedDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IndexedDocument {
    id: String,
    kind: String,
    title: String,
    subtitle: String,
    body: String,
    module_id: Option<String>,
    module_title: Option<String>,
    module_slug: Option<String>,
    group_id: Option<String>,
    mode: Option<String>,
    section: Option<String>,
    step_index: Option<i64>,
    term: Option<String>,
    card_label: Option<String>,
    screen: Option<String>,
    settings_tab: Option<String>,
    rank_boost: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub subtitle: String,
    pub body: String,
    pub module_id: Option<String>,
    pub module_title: Option<String>,
    pub module_slug: Option<String>,
    pub group_id: Option<String>,
    pub mode: Option<String>,
    pub section: Option<String>,
    pub step_index: Option<i64>,
    pub term: Option<String>,
    pub card_label: Option<String>,
    pub screen: Option<String>,
    pub settings_tab: Option<String>,
    pub rank_boost: f64,
    pub score: f64,
    pub snippet: String,
}

pub fn create_schema(connection: &Connection) -> rusqlite::Result<()> {
    connection.execute_batch(
        "
        PRAGMA journal_mode = DELETE;
        CREATE TABLE search_documents (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            subtitle TEXT NOT NULL,
            body TEXT NOT NULL,
            module_id TEXT,
            module_title TEXT,
            module_slug TEXT,
            group_id TEXT,
            mode TEXT,
            section TEXT,
            step_index INTEGER,
            term TEXT,
            card_label TEXT,
            screen TEXT,
            settings_tab TEXT,
            rank_boost REAL NOT NULL
        );
        CREATE VIRTUAL TABLE search_fts USING fts5(
            title,
            subtitle,
            body,
            content='search_documents',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        ",
    )
}

pub fn populate_from_json(connection: &Connection, json: &str) -> Result<usize, String> {
    let index: SearchIndexFile =
        serde_json::from_str(json).map_err(|error| format!("Search index JSON is invalid: {error}"))?;
    create_schema(connection).map_err(|error| error.to_string())?;

    {
        let mut insert = connection
            .prepare(
                "INSERT INTO search_documents (
                    id, kind, title, subtitle, body, module_id, module_title, module_slug,
                    group_id, mode, section, step_index, term, card_label, screen, settings_tab, rank_boost
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            )
            .map_err(|error| error.to_string())?;

        for document in &index.documents {
            insert
                .execute(params![
                    document.id,
                    document.kind,
                    document.title,
                    document.subtitle,
                    document.body,
                    document.module_id,
                    document.module_title,
                    document.module_slug,
                    document.group_id,
                    document.mode,
                    document.section,
                    document.step_index,
                    document.term,
                    document.card_label,
                    document.screen,
                    document.settings_tab,
                    document.rank_boost
                ])
                .map_err(|error| error.to_string())?;
        }
    }

    connection
        .execute("INSERT INTO search_fts(search_fts) VALUES('rebuild')", [])
        .map_err(|error| error.to_string())?;
    Ok(index.documents.len())
}

pub fn fts_match_query(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.len() >= 2)
        .map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .collect();
            format!("{}*", cleaned.to_ascii_lowercase())
        })
        .filter(|token| token.len() > 1)
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" AND "))
    }
}

pub fn search_documents(
    connection: &Connection,
    query: &str,
    limit: i64,
) -> Result<Vec<SearchHit>, String> {
    let Some(match_query) = fts_match_query(query) else {
        return Ok(Vec::new());
    };

    let mut statement = connection
        .prepare(
            "SELECT
                d.id, d.kind, d.title, d.subtitle, d.body,
                d.module_id, d.module_title, d.module_slug, d.group_id,
                d.mode, d.section, d.step_index, d.term, d.card_label,
                d.screen, d.settings_tab, d.rank_boost,
                snippet(search_fts, 2, '«', '»', '…', 16) AS snippet,
                bm25(search_fts) AS score
             FROM search_fts
             JOIN search_documents d ON d.rowid = search_fts.rowid
             WHERE search_fts MATCH ?1
             ORDER BY (bm25(search_fts) / MAX(d.rank_boost, 0.1)) ASC, d.title ASC
             LIMIT ?2",
        )
        .map_err(|error| error.to_string())?;

    let rows = statement
        .query_map(params![match_query, limit], |row| {
            Ok(SearchHit {
                id: row.get(0)?,
                kind: row.get(1)?,
                title: row.get(2)?,
                subtitle: row.get(3)?,
                body: row.get(4)?,
                module_id: row.get(5)?,
                module_title: row.get(6)?,
                module_slug: row.get(7)?,
                group_id: row.get(8)?,
                mode: row.get(9)?,
                section: row.get(10)?,
                step_index: row.get(11)?,
                term: row.get(12)?,
                card_label: row.get(13)?,
                screen: row.get(14)?,
                settings_tab: row.get(15)?,
                rank_boost: row.get(16)?,
                snippet: row.get(17)?,
                score: row.get(18)?,
            })
        })
        .map_err(|error| error.to_string())?;

    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

pub fn write_index_file(json: &str, destination: &std::path::Path) -> Result<usize, String> {
    let index: SearchIndexFile =
        serde_json::from_str(json).map_err(|error| format!("Search index JSON is invalid: {error}"))?;
    let stamp_path = destination.with_extension("sqlite.stamp");
    if destination.exists() {
        if let (Some(fingerprint), Ok(existing)) = (&index.fingerprint, std::fs::read_to_string(&stamp_path)) {
            if existing.trim() == fingerprint {
                return Ok(index.documents.len());
            }
        }
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    if destination.exists() {
        std::fs::remove_file(destination).map_err(|error| error.to_string())?;
    }
    let connection = Connection::open(destination).map_err(|error| error.to_string())?;
    let count = populate_from_json(&connection, json)?;
    connection
        .execute_batch("PRAGMA journal_mode = DELETE; VACUUM;")
        .map_err(|error| error.to_string())?;
    if let Some(fingerprint) = index.fingerprint {
        std::fs::write(&stamp_path, fingerprint).map_err(|error| error.to_string())?;
    }
    Ok(count)
}
