from nimem.pipelines.ingest import extract_triplets, process_text_pipeline
from nimem.nlp.spacy import extract_entities as extract_entities_spacy
from nimem.nlp.spacy import extract_relations as extract_relations_spacy
from nimem.storage.conceptnet_store import extract_triplets_conceptnet

text = "John works for Apple Inc. and lives in San Francisco. Alice founded SpaceX. Sarah works for Google."

# Method 1: spaCy (default)
print("=== spaCy (default) ===")
result = extract_triplets(text)
if hasattr(result, "unwrap"):
    triplets = result.unwrap()
else:
    triplets = result
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Method 2: Explicit composition - spaCy
print("\n=== Explicit: spaCy entities + relations ===")
entities = extract_entities_spacy(text)
print(f"Entities: {entities}")
triplets = extract_relations_spacy(text, entities)
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Method 3: spaCy + ConceptNet (hybrid - uses semantic similarity for disambiguation)
print("\n=== spaCy + ConceptNet (hybrid) ===")
result = extract_triplets(text, use_conceptnet=True)
if hasattr(result, "unwrap"):
    triplets = result.unwrap()
else:
    triplets = result
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Method 4: ConceptNet-only (dependency pairs + ConceptNet disambiguation)
print("\n=== ConceptNet-only (dependency pairs) ===")
triplets = extract_triplets_conceptnet(text)
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")


# Method 5: Full pipeline with coreference
print("\n=== Pipeline with coreference ===")
#long_text = "John works for Apple Inc. He lives in San Francisco."
long_text = "John works for Apple Inc. and he lives in San Francisco. Alice founded SpaceX. Sarah works for Google."
resolved, triplets = process_text_pipeline(long_text, use_coref=True).unwrap()
print(f"Resolved: {resolved}")
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")
