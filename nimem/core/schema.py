from typing import NamedTuple


class Triple(NamedTuple):
    subject: str
    relation: str
    object: str


class Entity(NamedTuple):
    text: str
    label: str
    start: int = 0
    end: int = 0
    confidence: float = 1.0


SPACY_MODEL = "en_core_web_md"

SPACY_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "DATE": "date",
    "NORP": "organization",
}

ENTITIES = {
    "person": "People, characters, or user names",
    "organization": "Companies, groups, and institutions",
    "location": "Cities, countries, places, and physical locations",
    "event": "Specific events, incidents, or occasions",
    "date": "Date references",
    "concept": "Abstract concepts or ideas",
}

ENTITY_RELATION_MAP = {
    ("person", "organization"): "works_for",
    ("person", "location"): "located_in",
    ("person", "person"): "knows",
    ("organization", "location"): "located_in",
    ("event", "location"): "happened_at",
}


RELATIONS = {
    # Core relations
    "works_for": "Employment or professional affiliation",
    "located_in": "Geographic containment",
    "knows": "Social or professional relationship between people",
    "founded": "Organization creation or establishment",
    "owns": "Ownership relationship",
    "manages": "Management or leadership relationship",
    "participated_in": "Involvement in an event",
    "happened_at": "Temporal or spatial occurrence of an event",
    "related_to": "General relationship when two people or two organizations or organization and person are vaguely related and the strictly defined ones don't fit",
    "created": "Creation of something (product, work, etc.)",
    "worked_with": "Professional collaboration between people",
    # Extended relations from ConceptNet
    "requires": "Something is needed as prerequisite",
    "capable_of": "Something has the ability to do something",
    "causes": "Something leads to or results in something",
    "used_for": "Something is utilized for a purpose",
}

VERB_TO_RELATION = {
    # PartOf / works_for
    "work": "works_for",
    "employ": "works_for",
    "hire": "works_for",
    "join": "works_for",
    "serve": "works_for",
    "act": "works_for",
    # CreatedBy / founded, created
    "found": "founded",
    "establish": "founded",
    "start": "founded",
    "create": "created",
    "build": "created",
    "develop": "created",
    "make": "created",
    "found": "founded",
    # HasA / owns
    "own": "owns",
    "acquire": "owns",
    "purchase": "owns",
    "buy": "owns",
    "possess": "owns",
    "hold": "owns",
    # PartOf / manages
    "manage": "manages",
    "lead": "manages",
    "direct": "manages",
    "run": "manages",
    "head": "manages",
    "supervise": "manages",
    # AtLocation / located_in
    "live": "located_in",
    "reside": "located_in",
    "stay": "located_in",
    "locate": "located_in",
    "situate": "located_in",
    "move": "located_in",
    "relocate": "located_in",
    "place": "located_in",
    # RelatedTo / knows
    "know": "knows",
    "meet": "knows",
    "befriend": "knows",
    "recognize": "knows",
    "remember": "knows",
    # RelatedTo / worked_with
    "collaborate": "worked_with",
    "partner": "worked_with",
    "cooperate": "worked_with",
    "team_up": "worked_with",
    # HasPrerequisite / requires
    "require": "requires",
    "need": "requires",
    "depend": "requires",
    # CapableOf / can
    "can": "capable_of",
    "able": "capable_of",
    # Causes / causes
    "cause": "causes",
    "lead_to": "causes",
    "result_in": "causes",
    "trigger": "causes",
    # UsedFor / used_for
    "use": "used_for",
    "utilize": "used_for",
    "apply": "used_for",
}

WITH_PREPOSITIONS = {"with"}

CONCEPTNET_RELATION_TEMPLATES = {
    "/r/PartOf": [
        "{head} is part of {tail}",
        "{head} belongs to {tail}",
        "{head} works for {tail}",
        "{head} is a member of {tail}",
    ],
    "/r/HasA": [
        "{head} has {tail}",
        "{head} owns {tail}",
        "{head} has a {tail}",
    ],
    "/r/UsedFor": [
        "{head} is used for {tail}",
        "{head} is used to {tail}",
        "{head} helps {tail}",
    ],
    "/r/CapableOf": [
        "{head} can {tail}",
        "{head} is capable of {tail}",
        "{head} is able to {tail}",
    ],
    "/r/AtLocation": [
        "{head} is at {tail}",
        "{head} is in {tail}",
        "{head} is located in {tail}",
        "{head} is at {tail}",
    ],
    "/r/Causes": [
        "{head} causes {tail}",
        "{head} leads to {tail}",
        "{head} results in {tail}",
    ],
    "/r/HasSubevent": [
        "{head} involves {tail}",
        "{head} includes {tail}",
    ],
    "/r/HasPrerequisite": [
        "{head} requires {tail}",
        "{head} needs {tail}",
        "{head} needs to {tail}",
    ],
    "/r/HasProperty": [
        "{head} is {tail}",
        "{head} has property {tail}",
    ],
    "/r/MotivatedByGoal": [
        "{head} is done for {tail}",
        "{head} is motivated by {tail}",
    ],
    "/r/Desires": [
        "{head} wants {tail}",
        "{head} desires {tail}",
    ],
    "/r/CreatedBy": [
        "{head} is created by {tail}",
        "{head} was created by {tail}",
        "{head} was made by {tail}",
    ],
    "/r/Synonym": [
        "{head} is similar to {tail}",
        "{head} is synonymous with {tail}",
    ],
    "/r/IsA": [
        "{head} is a {tail}",
        "{head} is a type of {tail}",
    ],
    "/r/DefinedAs": [
        "{head} is defined as {tail}",
    ],
    "/r/MannerOf": [
        "{head} is a way to {tail}",
        "{head} is how you {tail}",
    ],
    "/r/LocatedNear": [
        "{head} is near {tail}",
        "{head} is close to {tail}",
    ],
    "/r/MadeOf": [
        "{head} is made of {tail}",
        "{head} is made from {tail}",
    ],
    "/r/ReceivesAction": [
        "{tail} can be done to {head}",
        "{head} receives {tail}",
    ],
    "/r/SymbolOf": [
        "{head} symbolizes {tail}",
        "{head} represents {tail}",
    ],
    "/r/RelatedTo": [
        "{head} is related to {tail}",
        "{head} relates to {tail}",
    ],
}

RELATION_TO_CONCEPTNET = {
    "works_for": "/r/PartOf",
    "located_in": "/r/AtLocation",
    "knows": "/r/RelatedTo",
    "founded": "/r/CreatedBy",
    "owns": "/r/HasA",
    "manages": "/r/PartOf",
    "participated_in": "/r/AtLocation",
    "happened_at": "/r/AtLocation",
    "related_to": "/r/RelatedTo",
    "created": "/r/CreatedBy",
    "worked_with": "/r/RelatedTo",
}

CONCEPTNET_TO_RELATION = {v: k for k, v in RELATION_TO_CONCEPTNET.items()}

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
