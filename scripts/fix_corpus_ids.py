input_path = "data/processed/pubmed_corpus.tsv"
output_path = "data/processed/pubmed_corpus_fixed.tsv"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for idx, line in enumerate(fin):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) != 2:
            continue  

        text = parts[1]
        fout.write(f"{idx}\t{text}\n")

print("DONE. Fixed corpus written to:", output_path)
