import json

def load_cases(path='data/cases.json'):
    with open(path) as f:
        return json.load(f)

def load_rules(path='data/rules.json'):
    with open(path) as f:
        return json.load(f)

def case_based_reasoning(location, cases_db):
    for case in cases_db:
        if case['location'].lower() == location.lower():
            return case['likely_diseases']
    return []

def rule_based_reasoning(symptoms, rules_db):
    triggered = []
    for rule in rules_db:
        if set(rule['if']).issubset(set(symptoms)):
            triggered.append(rule['then'])
    return triggered or ["No rule-based match"]
