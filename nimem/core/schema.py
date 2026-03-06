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
    "founded": "Organization creation or establishment",
    "owns": "Ownership relationship",
    "manages": "Management or leadership relationship",
    "participated_in": "Involvement in an event",
    "happened_at": "Temporal or spatial occurrence of an event",
    "related_to": "General relationship when strictly defined ones don't fit",
    "created": "Creation of something (product, work, etc.)",
    "worked_with": "Professional collaboration between people",
}

VERB_TO_RELATION = {
    "work": "works_for",
    "employ": "works_for",
    "hire": "works_for",
    "join": "works_for",
    "found": "founded",
    "establish": "founded",
    "start": "founded",
    "create": "created",
    "build": "created",
    "develop": "created",
    "own": "owns",
    "acquire": "owns",
    "purchase": "owns",
    "buy": "owns",
    "manage": "manages",
    "lead": "manages",
    "direct": "manages",
    "run": "manages",
    "head": "manages",
    "live": "located_in",
    "reside": "located_in",
    "move": "located_in",
    "relocate": "located_in",
    "know": "knows",
    "meet": "knows",
    "befriend": "knows",
    "collaborate": "worked_with",
    "partner": "worked_with",
}

WITH_PREPOSITIONS = {"with"}

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
    "owns": "MANY",
    "manages": "MANY",
    "participated_in": "MANY",
    "happened_at": "ONE",
    "related_to": "MANY",
    "created": "MANY",
    "worked_with": "MANY",
}
