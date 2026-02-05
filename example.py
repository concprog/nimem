import logging
import time
from nimem import memory
from returns.result import Success, Failure

# Configure logging to see what's happening under the hood
logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean, update to INFO for details

def section(title):
    print(f"\n{'-'*60}")
    print(f"| {title}")
    print(f"{'-'*60}")

def main():
    print("Welcome to the Nimem System Demo!")
    
    section("1. INGESTION")
    print("We will start by teaching the system about Alice.")
    
    text_1 = "Alice works for Google. She lives in London. Bob is Alice's friend."
    print(f"\n[Input Text]: \"{text_1}\"")
    
    start = time.time()
    result = memory.ingest_text(text_1)
    
    if isinstance(result, Success):
        print(f"[Success]: {result.unwrap()}")
    else:
        print(f"[Error]: {result.failure()}")
        exit(1)
        
    print(f"Time taken: {time.time() - start:.2f}s")

    section("2. MEMORY RECALL")
    print("Let's ask the system what it knows about 'Alice'.")
    
    facts = memory.recall_memory("Alice")
    if isinstance(facts, Success):
        for fact in facts.unwrap():
             print(f"  -> Alice [{fact['relation']}] {fact['object']}")
    
    section("3. UPDATING FACT (Cardinality ONE)")
    print("According to schema.py, 'located_in' is a ONE-to-ONE relation.")
    print("If we tell the system Alice lives somewhere else, the old fact should be invalidated.")
    
    text_2 = "Alice has moved. Alice lives in Paris."
    print(f"\n[Input Text]: \"{text_2}\"")
    memory.ingest_text(text_2)
    
    print("\nQuerying Alice again:")
    facts = memory.recall_memory("Alice")
    if isinstance(facts, Success):
        for fact in facts.unwrap():
             print(f"  -> Alice [{fact['relation']}] {fact['object']}")
        
        # Validation
        locs = [f['object'] for f in facts.unwrap() if f['relation'] == 'LOCATED_IN']
        if "Paris" in locs and "London" not in locs:
            print("\n[Checked]: System correctly updated location to Paris!")
        else:
            print(f"\n[Note]: Current locations: {locs}")

    section("4. TOPIC CONSOLIDATION")
    print("Trying to find shared topics among entities...")
    
    cons = memory.consolidate_topics()
    
    if isinstance(cons, Success):
        print(f"[Result]: {cons.unwrap()}")
        
        # Let's see if any new 'BELONGS_TO' relations were added
        print("\nChecking for discovered topics for Alice:")
        facts = memory.recall_memory("Alice")
        for fact in facts.unwrap():
            if fact['relation'] == 'BELONGS_TO':
                print(f"  -> Alice inferred topic: {fact['object']}")
    else:
        print(f"[Note]: Consolidation skipped or failed: {cons.failure()}")

if __name__ == "__main__":
    main()
