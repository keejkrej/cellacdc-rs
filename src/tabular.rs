use anyhow::{anyhow, bail, Context, Result};
use calamine::{open_workbook_auto, Data, Reader};
use csv::{ReaderBuilder, WriterBuilder};
use rust_xlsxwriter::{Workbook, Worksheet};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableFormat {
    Csv,
    Xlsx,
}

impl TableFormat {
    pub fn from_extension(path: &Path) -> Result<Self> {
        match path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("csv") => Ok(Self::Csv),
            Some("xlsx") | Some("xlsm") | Some("xls") => Ok(Self::Xlsx),
            other => bail!(
                "Unsupported table format {:?} for {}. Supported formats are CSV and XLSX.",
                other,
                path.display()
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableValue {
    Empty,
    Number(f64),
    Text(String),
    Bool(bool),
}

impl TableValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
            Self::Text(value) => value.parse::<f64>().ok(),
            Self::Empty => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|value| value.round() as i64)
    }

    pub fn as_string_lossy(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Number(value) => {
                if value.fract() == 0.0 {
                    (*value as i64).to_string()
                } else {
                    value.to_string()
                }
            }
            Self::Text(value) => value.clone(),
            Self::Bool(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<TableValue>>,
}

impl Table {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
        }
    }

    pub fn push_row(&mut self, row: Vec<TableValue>) -> Result<()> {
        if row.len() != self.headers.len() {
            bail!(
                "Table row length mismatch: got {}, expected {}",
                row.len(),
                self.headers.len()
            );
        }
        self.rows.push(row);
        Ok(())
    }

    pub fn header_index(&self, name: &str) -> Result<usize> {
        self.headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| anyhow!("Missing table column {name:?}"))
    }

    pub fn maybe_header_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|header| header == name)
    }

    pub fn row_map(&self, row_index: usize) -> BTreeMap<String, TableValue> {
        self.headers
            .iter()
            .cloned()
            .zip(self.rows[row_index].iter().cloned())
            .collect()
    }

    pub fn with_column(&mut self, name: impl Into<String>, values: Vec<TableValue>) -> Result<()> {
        let name = name.into();
        if values.len() != self.rows.len() {
            bail!(
                "Column length mismatch for {:?}: got {}, expected {}",
                name,
                values.len(),
                self.rows.len()
            );
        }
        self.headers.push(name);
        for (row, value) in self.rows.iter_mut().zip(values) {
            row.push(value);
        }
        Ok(())
    }

    pub fn select_columns(&self, columns: &[String]) -> Self {
        let indices = columns
            .iter()
            .filter_map(|column| self.maybe_header_index(column).map(|idx| (column, idx)))
            .collect::<Vec<_>>();
        let rows = self
            .rows
            .iter()
            .map(|row| indices.iter().map(|(_, idx)| row[*idx].clone()).collect())
            .collect();
        Self {
            headers: indices
                .into_iter()
                .map(|(column, _)| column.clone())
                .collect(),
            rows,
        }
    }

    pub fn ensure_unique_headers(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for header in &self.headers {
            if !seen.insert(header) {
                bail!("Duplicate table header {header:?}");
            }
        }
        Ok(())
    }
}

pub fn read_table(path: &Path) -> Result<Table> {
    match TableFormat::from_extension(path)? {
        TableFormat::Csv => read_csv(path),
        TableFormat::Xlsx => read_xlsx(path),
    }
}

pub fn write_table(path: &Path, table: &Table) -> Result<()> {
    match TableFormat::from_extension(path)? {
        TableFormat::Csv => write_csv(path, table),
        TableFormat::Xlsx => write_xlsx(path, table),
    }
}

fn read_csv(path: &Path) -> Result<Table> {
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open CSV {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("Failed to read CSV headers in {}", path.display()))?
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    let mut table = Table::new(headers);
    for record in reader.records() {
        let record = record?;
        table.push_row(record.iter().map(parse_string_value).collect())?;
    }
    table.ensure_unique_headers()?;
    Ok(table)
}

fn write_csv(path: &Path, table: &Table) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let mut writer = WriterBuilder::new()
        .from_path(path)
        .with_context(|| format!("Failed to create CSV {}", path.display()))?;
    writer.write_record(&table.headers)?;
    for row in &table.rows {
        writer.write_record(row.iter().map(TableValue::as_string_lossy))?;
    }
    writer.flush()?;
    Ok(())
}

fn read_xlsx(path: &Path) -> Result<Table> {
    let mut workbook = open_workbook_auto(path)
        .with_context(|| format!("Failed to open workbook {}", path.display()))?;
    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("Workbook {} does not contain any sheets", path.display()))?;
    let range = workbook.worksheet_range(&sheet_name).with_context(|| {
        format!(
            "Failed to read worksheet {sheet_name:?} in {}",
            path.display()
        )
    })?;
    let mut rows = range.rows();
    let headers = rows
        .next()
        .ok_or_else(|| anyhow!("Worksheet {sheet_name:?} in {} is empty", path.display()))?
        .iter()
        .map(data_to_string)
        .collect::<Vec<_>>();
    let mut table = Table::new(headers);
    for row in rows {
        let mut values = row.iter().map(data_to_value).collect::<Vec<_>>();
        while values.len() < table.headers.len() {
            values.push(TableValue::Empty);
        }
        if values.len() > table.headers.len() {
            values.truncate(table.headers.len());
        }
        table.push_row(values)?;
    }
    table.ensure_unique_headers()?;
    Ok(table)
}

fn write_xlsx(path: &Path, table: &Table) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Output path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();
    write_worksheet(worksheet, table)?;
    workbook
        .save(path)
        .with_context(|| format!("Failed to save workbook {}", path.display()))?;
    Ok(())
}

fn write_worksheet(sheet: &mut Worksheet, table: &Table) -> Result<()> {
    for (col, header) in table.headers.iter().enumerate() {
        sheet.write_string(0, col as u16, header)?;
    }
    for (row_idx, row) in table.rows.iter().enumerate() {
        let sheet_row = (row_idx + 1) as u32;
        for (col_idx, value) in row.iter().enumerate() {
            let sheet_col = col_idx as u16;
            match value {
                TableValue::Empty => {}
                TableValue::Number(number) => {
                    sheet.write_number(sheet_row, sheet_col, *number)?;
                }
                TableValue::Text(text) => {
                    sheet.write_string(sheet_row, sheet_col, text)?;
                }
                TableValue::Bool(value) => {
                    sheet.write_boolean(sheet_row, sheet_col, *value)?;
                }
            }
        }
    }
    Ok(())
}

fn parse_string_value(value: &str) -> TableValue {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return TableValue::Empty;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower == "true" {
        return TableValue::Bool(true);
    }
    if lower == "false" {
        return TableValue::Bool(false);
    }
    if let Ok(number) = trimmed.parse::<f64>() {
        return TableValue::Number(number);
    }
    TableValue::Text(trimmed.to_string())
}

fn data_to_value(value: &Data) -> TableValue {
    match value {
        Data::Empty => TableValue::Empty,
        Data::String(text) => parse_string_value(text),
        Data::Float(number) => TableValue::Number(*number),
        Data::Int(number) => TableValue::Number(*number as f64),
        Data::Bool(value) => TableValue::Bool(*value),
        Data::DateTime(number) => TableValue::Text(number.to_string()),
        Data::DateTimeIso(text) => TableValue::Text(text.clone()),
        Data::DurationIso(text) => TableValue::Text(text.clone()),
        Data::Error(err) => TableValue::Text(err.to_string()),
    }
}

fn data_to_string(value: &Data) -> String {
    data_to_value(value).as_string_lossy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrips_csv_and_xlsx_tables() -> Result<()> {
        let temp = tempdir()?;
        let table = Table {
            headers: vec![
                "frame_i".into(),
                "Cell_ID".into(),
                "value".into(),
                "flag".into(),
            ],
            rows: vec![
                vec![
                    TableValue::Number(0.0),
                    TableValue::Number(1.0),
                    TableValue::Text("alpha".into()),
                    TableValue::Bool(true),
                ],
                vec![
                    TableValue::Number(1.0),
                    TableValue::Number(2.0),
                    TableValue::Empty,
                    TableValue::Bool(false),
                ],
            ],
        };

        let csv_path = temp.path().join("table.csv");
        write_table(&csv_path, &table)?;
        assert_eq!(read_table(&csv_path)?, table);

        let xlsx_path = temp.path().join("table.xlsx");
        write_table(&xlsx_path, &table)?;
        assert_eq!(read_table(&xlsx_path)?, table);
        Ok(())
    }
}
