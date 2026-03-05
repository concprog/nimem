ENTITIES = {
    "person": "People, characters, or user names",
    "organization": "Companies, groups, and institutions",
    "location": "Cities, countries, places, and physical locations",
    "event": "Specific events, incidents, or occasions",
    "date": "Date references",
    "concept": "Abstract concepts or ideas",
}

RELATIONS = {
    "works_for": "Employment or professional affiliation",
    "located_in": "Geographic containment",
    "knows": "Social or professional relationship between people",
    "founded": "Organization creation",
    "participated_in": "Involvement in an event",
    "happened_at": "Temporal or spatial occurrence of an event",
    "related_to": "General relationship when strictly defined ones don't fit",
}

THRESHOLDS = {
    "works_for": 0.6,
    "located_in": 0.6,
    "knows": 0.5,
    "default": 0.5,
}

CARDINALITY = {
    "works_for": "MANY",
    "located_in": "ONE",
    "knows": "MANY",
    "founded": "MANY",
    "participated_in": "MANY",
    "happened_at": "ONE",
    "related_to": "MANY",
}
