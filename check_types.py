import json

data = json.load(open("catalog.json", "r", encoding="utf-8"))

types = set()
for d in data:
    types.add(d.get("test_type", "?"))

print("Types in catalog:", types)
print()

# Show some samples with different test types
for t in sorted(types):
    items = [d for d in data if d.get("test_type") == t]
    print(f"Type '{t}': {len(items)} items")
    for item in items[:2]:
        name = item["name"][:50]
        print(f"  - {name}")
    print()
