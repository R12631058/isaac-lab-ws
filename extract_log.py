import sys

try:
    with open('ndi_debug.txt', 'r', encoding='utf-16le', errors='ignore') as f:
        lines = f.readlines()
        
    out_lines = []
    for i, line in enumerate(lines):
        if 'DIAG]' in line:
            for j in range(i, min(i+7, len(lines))):
                out_lines.append(lines[j])
            out_lines.append("-" * 40 + "\n")
            
    with open('ndi_diag_extracted.txt', 'w', encoding='utf-8') as out:
        out.writelines(out_lines)
except Exception as e:
    with open('ndi_diag_extracted.txt', 'w', encoding='utf-8') as out:
        out.write(str(e))
