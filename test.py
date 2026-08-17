import json


def check_chunk_lengths(
        jsonl_path: str = "data/intern_output/chunk_data.jsonl", max_limit: int = 2000):
    overflow_chunks = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            data = json.loads(line)

            # Récupère les index (gère start_index/end_index ou
            # first/last_character_index)
            start = data.get(
                "start_index", data.get(
                    "first_character_index", 0))
            end = data.get("end_index", data.get("last_character_index", 0))
            length = end - start

            # Récupère le chemin du fichier s'il existe dans les métadonnées
            file_path = data.get(
                "metadata",
                {}).get(
                "file_path",
                data.get(
                    "file_path",
                    "inconnu"))

            if length > max_limit:
                overflow_chunks.append({
                    "line": line_idx,
                    "file_path": file_path,
                    "start": start,
                    "end": end,
                    "length": length
                })

    return overflow_chunks


# Exécution
trop_longs = check_chunk_lengths(
    "data/intern_output/chunk_data.jsonl",
    max_limit=2000)

print(f"Total de chunks hors limite (> 2000) : {len(trop_longs)}")
if trop_longs:
    print("\nExemple des premiers chunks problématiques :")
    print(json.dumps(trop_longs[:5], indent=2))
