import pandas as pd
import os
import sys
import numpy as np
from collections import Counter
import re

def detect_sheet_format(df):
	scores = {
		"uniform_columns": 0,
		"short_cells": 0,
		"no_newlines": 0,
		"non_sentence_endings": 0,
	}

	rows = df.values.tolist()
	col_counts = [len([c for c in row if pd.notnull(c)]) for row in rows]
	
	if max(col_counts) - min(col_counts) <= 2:
		scores["uniform_columns"] = 1

	cell_lengths = [len(str(cell)) for row in rows for cell in row if pd.notnull(cell)]
	avg_len = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0
	if avg_len <= 50:
		scores["short_cells"] = 1

	newline_count = sum(str(cell).count("\n") for row in rows for cell in row if pd.notnull(cell))
	if newline_count / len(cell_lengths) < 0.05:
		scores["no_newlines"] = 1

	sentence_endings = sum(str(cell).count("。") + str(cell).count("?") + str(cell).count("！") for row in rows for cell in row if pd.notnull(cell))
	if sentence_endings / len(cell_lengths) < 0.05:
		scores["non_sentence_endings"] = 1

	total_score = sum(scores.values())
	is_table = total_score >= 3
	return is_table

def load_csv_as_text(df):
	text = "\n".join([
		str(cell) for row in df.values for cell in row if pd.notnull(cell)
	])
	return text

def get_column_name_or_fallback(col_idx, df, header_row_idx):
	header_candidate = df.iloc[header_row_idx, col_idx]
	if pd.isnull(header_candidate) or str(header_candidate).strip() == "":
		return f"col_{col_idx}"
	return str(header_candidate).strip()

def split_column_content(column_name, values, chunk_size=800, overlap=100):
	text = f"【{column_name}】\n" + "\n".join(values)
	chunks = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end]
		chunks.append(chunk)
		start += chunk_size - overlap
	return chunks

def row_to_text(row, header):
	fields = []
	for i, cell in enumerate(row):
		if pd.notnull(cell):
			col_name = header[i] if pd.notnull(header[i]) else f"col_{i}"
			fields.append(f"{col_name}: {cell}")
	return " / ".join(fields)

def process_table_format(df, chunk_size=800, overlap=100):
	header_row_index = detect_header_row(df)
	print(f"✅ ヘッダー行 index={header_row_index} で処理")
	header = df.iloc[header_row_index]
	data = df.iloc[header_row_index + 1:]

	all_chunks = []
	for idx, row in data.iterrows():
		text = row_to_text(row, header)
		if text.strip():
			all_chunks.append(text)

	return all_chunks

def detect_header_row(df, max_rows_to_check=20, return_score=False):
	import numpy as np

	def is_short_text(text):
		return len(text) <= 15 and "\n" not in text

	def is_likely_field_name(text):
		keywords = ["名", "日", "番号", "氏名", "区分", "タイプ", "状況", "業種", "金額", "継続", "種別", "状態"]
		return any(k in text for k in keywords)

	best_score = -np.inf
	best_index = 0

	for idx in range(min(max_rows_to_check, len(df) - 1)):
		row = df.iloc[idx].dropna().astype(str).tolist()
		next_row = df.iloc[idx + 1].dropna().astype(str).tolist()

		if not row or not next_row:
			continue

		avg_len = np.mean([len(cell) for cell in row])
		short_ratio = np.mean([is_short_text(cell) for cell in row])
		keyword_ratio = np.mean([is_likely_field_name(cell) for cell in row])
		uniqueness = len(set(row)) / len(row)
		digit_ratio = sum(c.isdigit() for cell in row for c in cell) / sum(len(cell) for cell in row if cell)

		# next row に数字が多く含まれていれば「データ行らしい」と判定
		next_row_digit_ratio = sum(c.isdigit() for cell in next_row for c in cell) / max(
			sum(len(cell) for cell in next_row if cell), 1
		)
		next_row_is_data_like = next_row_digit_ratio > 0.2

		index_bonus = 1.0 if idx <= 5 else 0.0

		if is_natural_text_line(row):
			memo_penalty = -1.0
		else:
			memo_penalty = 0

		this_row_is_data_like = (
			short_ratio >= 0.9 and
			digit_ratio >= 0.4 and
			keyword_ratio < 0.25
		)
		data_penalty = -0.75 if this_row_is_data_like else 0

		score = (
			(1 / (1 + avg_len)) * 2 +
			short_ratio * 1.5 +
			keyword_ratio +
			uniqueness * 1 -
			digit_ratio * 1.0 +
			(2 if next_row_is_data_like else 0) +
			index_bonus +
			data_penalty
		)

		inconsistency_score = column_inconsistency_score(df, idx)
		score += inconsistency_score * 1.2

		column_ratio = non_empty_cell_ratio(df.iloc[idx], total_expected_columns=len(df.columns))
		if column_ratio < 0.5:
			score -= 1.0
		dominant_ratio = dominant_cell_ratio(df.iloc[idx])
		if dominant_ratio < 0.7:
			score -= 1.0

		if score > best_score:
			print(f"score: {score}, idx: {idx}, short_ratio: {short_ratio}, keyword_ratio: {keyword_ratio}, uniqueness: {uniqueness}, digit_ratio: {digit_ratio}, next_row_digit_ratio: {next_row_digit_ratio}, next_row_is_data_like: {next_row_is_data_like}, index_bonus: {index_bonus}, data_penalty: {data_penalty}, memo_penalty: {memo_penalty}, inconsistency_score: {inconsistency_score}, column_ratio: {column_ratio}, dominant_ratio: {dominant_ratio}")
			best_score = score
			best_index = idx

	return (best_index, best_score) if return_score else best_index

def non_empty_cell_ratio(row, total_expected_columns):
	count = row.count()
	return count / total_expected_columns if total_expected_columns else 0

def dominant_cell_ratio(row):
	cell_lengths = [len(str(c)) for c in row if pd.notnull(c)]
	if not cell_lengths:
		return 0
	max_len = max(cell_lengths)
	total_len = sum(cell_lengths)
	return max_len / total_len if total_len > 0 else 0

def classify_value(val):
	if isinstance(val, str):
		val = val.strip()
		if val == "":
			return "empty"
		elif re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", val):
			return "date"
		elif re.fullmatch(r"\d+", val):
			return "number"
		elif "株式会社" in val:
			return "company"
		elif re.fullmatch(r"(単発|継続|新規)", val):
			return "status"
		elif any("\u4e00" <= c <= "\u9fff" for c in val):  # 漢字含む
			return "japanese"
		else:
			return "other"
	return "unknown"

def column_inconsistency_score(df, row_index, compare_rows=10):
	"""
	指定した row_index が他の下位行と比べてどれだけ"浮いているか"をスコアリング
	"""
	if row_index + 1 >= len(df):
		return 0

	score = 0
	row = df.iloc[row_index]
	for col_idx in range(len(row)):
		cell_value = row[col_idx]
		if pd.isnull(cell_value):
			continue
		cell_type = classify_value(str(cell_value))

		# この列の compare_rows 行分の典型値を取得
		col_values = df.iloc[row_index+1:row_index+1+compare_rows, col_idx].dropna().astype(str)
		col_types = [classify_value(val) for val in col_values if val != ""]
		if not col_types:
			continue

		# 最頻タイプを取得
		most_common_type = Counter(col_types).most_common(1)[0][0]

		# 一貫性が崩れていれば加点
		if cell_type != most_common_type:
			score += 1

	return score / len(row.dropna())  # 逸脱率（高いほどヘッダーらしい）

def is_natural_text_line(cells):
	punctuations = "。！？,.、"
	total_chars = sum(len(c) for c in cells)
	punctuation_count = sum(c.count(p) for c in cells for p in punctuations)
	if total_chars == 0:
		return False
	return (punctuation_count / total_chars) > 0.03


def split_column_content(column_name, values, chunk_size=800, overlap=100):
	text = f"【{column_name}】\n" + "\n".join(values)
	chunks = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end]
		chunks.append(chunk)
		start += chunk_size - overlap
	return chunks

def split_text_with_overlap(text, chunk_size=800, overlap=100):
	chunks = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		chunks.append(text[start:end])
		start += chunk_size - overlap
	return chunks

def process_csv_file(df, chunk_size=800, overlap=100):
    if detect_sheet_format(df):
        print("🟦 テーブル形式として処理")
        chunks = process_table_format(df, chunk_size=chunk_size, overlap=overlap)
    else:
        print("🟨 自由記述形式として処理")
        text = load_csv_as_text(df)
        chunks = split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)

    return chunks

def clean_dataframe_and_save(df, output_path):
    # ヘッダー行を検出
    header_row_index = detect_header_row(df)
    print(f"✅ ヘッダー行 index={header_row_index} で処理")

    header = df.iloc[header_row_index]
    data = df.iloc[header_row_index + 1:]

    # ヘッダーをちゃんと列名にする
    data.columns = [col if pd.notnull(col) else f"col_{i}" for i, col in enumerate(header)]
    data = data.reset_index(drop=True)

    # CSV形式で保存
    data.to_csv(output_path, index=False)
    return output_path


def process_csv_from_file(csv_path, chunk_size=800, overlap=100):
    df = pd.read_csv(csv_path, header=None)
    return process_csv_file(df, chunk_size=chunk_size, overlap=overlap)

# CLI実行用
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        exit()

    chunks = process_csv_from_file(csv_path)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n{chunk}")
