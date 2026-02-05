# Default Schema for Nimem Memory System
# Defines the entities and relations we want to extract.

ENTITIES = {
    "person": "People, characters, or user names",
    "organization": "Companies, groups, and institutions",
    "location": "Cities, countries, places, and physical locations",
    "event": "Specific events, incidents, or occasions",
    "date": "Date references",
    "concept": "Abstract concepts or ideas"
}

RELATIONS = {
    "works_for": "Employment or professional affiliation",
    "located_in": "Geographic containment",
    "knows": "Social or professional relationship between people",
    "founded": "Organization creation",
    "participated_in": "Involvement in an event",
    "happened_at": "Temporal or spatial occurrence of an event",
    "related_to": "General relationship when strictly defined ones don't fit"
}

THRESHOLDS = {
    "works_for": 0.6,
    "located_in": 0.6,
    "knows": 0.5,
    "default": 0.5
}

# Cardinality Constraints
# "ONE": A subject can only have one active relation of this type (Implicitly invalidates previous).
# "MANY": A subject can have multiple active relations (Accumulates).
CARDINALITY = {
    "works_for": "MANY",   # Can work multiple jobs
    "located_in": "ONE",   # Usually one location at a time (simplified)
    "knows": "MANY",
    "founded": "MANY",
    "participated_in": "MANY",
    "happened_at": "ONE",  # Event usually happens at one place
    "related_to": "MANY"
}
