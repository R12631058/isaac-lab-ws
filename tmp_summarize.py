import json

with open("scripts/isaaclab_ws/final/occlusion_output/ndi_multipose_score_20260326_104951.json", "r") as f:
    d = json.load(f)

print("---")
for env, data in d["matrix_summary"].items():
    print(f"{env}: X={data['x_pos']:.2f}m | Cost={data['mean_cost']:.2f}")
print("---")
best = min(d["matrix_summary"].items(), key=lambda x: x[1]["mean_cost"])
print(f"BEST: {best[0]} with X={best[1]['x_pos']:.2f}m (Cost={best[1]['mean_cost']:.2f})")
