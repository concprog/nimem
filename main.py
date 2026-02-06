import logging
import sys
from nimem import memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("--- Nimem Memory System CLI ---")
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest \"text\"] | [add sub rel obj] | [query sub]")
        return

    command = sys.argv[1]
    
    if command == "ingest":
        text = sys.argv[2]
        print(f"Ingesting: {text}")
        result = memory.ingest_text(text)
        print(f"Result: {result}")
        
    elif command == "add":
        if len(sys.argv) != 5:
            print("Usage: python main.py add <subject> <relation> <object>")
            return
        sub, rel, obj = sys.argv[2], sys.argv[3], sys.argv[4]
        result = memory.add_memory(sub, rel, obj)
        print(f"Added: {result}")
        
    elif command == "query":
        sub = sys.argv[2]
        result = memory.recall_memory(sub)
        print(f"Facts about {sub}:")
        if isinstance(result, memory.Success):
            for fact in result.unwrap():
                print(f"  - {fact['relation']} -> {fact['object']}")
        else:
            print(f"Error: {result}")
            
    elif command == "consolidate":
        print("Consolidating memory topics...")
        result = memory.consolidate_topics()
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
