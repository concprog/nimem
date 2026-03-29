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
