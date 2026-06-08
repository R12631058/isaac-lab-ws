from pypdf import PdfReader

reader = PdfReader("C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4.pdf")
full_text = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    full_text.append(f"--- PAGE {i+1} ---\n{text}")

with open("C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4_extracted.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(full_text))

print(f"Extracted {len(reader.pages)} pages of text successfully!")
