import json
import codecs

log_path = r"C:\Users\RMML\.gemini\antigravity\brain\8be363e1-b8f3-4638-a024-66add84b4c2d\.system_generated\logs\transcript.jsonl"
original_code = None

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'tool_calls' in data:
                for call in data['tool_calls']:
                    if call.get('name') == 'write_to_file':
                        args = call.get('args', {})
                        target = args.get('TargetFile', '')
                        if 'Chapter4.tex' in target:
                            original_code = args.get('CodeContent')
        except Exception as e:
            continue

if original_code:
    # Remove outer quotes if present
    if original_code.startswith('"') and original_code.endswith('"'):
        original_code = original_code[1:-1]
    
    # Strip any leading escaped quotes or literal quotes
    if original_code.startswith('\\"'):
        original_code = original_code[2:]
    if original_code.endswith('\\"'):
        original_code = original_code[:-2]
        
    try:
        # Use unicode_escape to resolve all backslash-n, backslash-t, etc.
        decoded = codecs.escape_decode(bytes(original_code, "utf-8"))[0].decode("utf-8")
        original_code = decoded
    except Exception as e:
        print(f"unicode_escape decode failed: {e}")
        # Fallback to simple replace if unicode_escape fails
        original_code = original_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
    
    with open('C:/Users/RMML/IsaacLab/scripts/isaaclab_ws/reachability_map/tex/Chapter4_restored.tex', 'w', encoding='utf-8') as out:
        out.write(original_code)
    print("Successfully restored Chapter4.tex to Chapter4_restored.tex!")
    
    # Count the lines in the restored file
    num_lines = len(original_code.splitlines())
    print(f"Total lines in restored file: {num_lines}")
else:
    print("Could not find Chapter4.tex in logs.")
