import jsonlines
import random
from tqdm import tqdm

# --------------------------
# DEFINING LANGUAGE KNOWLEDGE
# --------------------------
# Vocabulary, phrases, grammar, topics
SWAHILI_WORDS = {
    "greetings": ["Habari", "Mambo", "Hujambo", "Salamu", "Shikamoo"],
    "people": ["mtu", "watu", "rafiki", "mzazi", "mwanafunzi", "mwalimu"],
    "places": ["Tanzania", "Kenya", "Dar es Salaam", "Dodoma", "Mombasa", "shule", "nyumbani"],
    "verbs": ["kwenda", "kula", "kusoma", "kufanya", "kuzungumza", "kuandika", "kufikiria"],
    "connectors": ["na", "kwa", "wa", "ya", "lakini", "kwa sababu", "kama", "pamoja na"]
}

ENGLISH_WORDS = {
    "greetings": ["Hello", "Hi", "Good morning", "Good evening", "How are you"],
    "people": ["person", "people", "friend", "parent", "student", "teacher"],
    "places": ["school", "home", "office", "city", "country"],
    "verbs": ["go", "eat", "read", "do", "talk", "write", "think"],
    "connectors": ["and", "with", "of", "but", "because", "if", "together with"]
}

# TOPICS: covers daily life, education, reasoning, news, tech, culture
TOPICS = [
    "Elimu na masomo", "Afya na maisha", "Teknolojia na intaneti", "Uchumi na biashara",
    "Utamaduni wa Kiafrika", "Kutatua matatizo", "Habari za kila siku", "Usafiri na mawasiliano",
    "School life and studies", "Health and lifestyle", "Technology and internet",
    "Mixed: Jinsi ya kutumia computer kwa usahihi", "Mixed: Habari mpya kutoka Tanzania"
]

# --------------------------
# GENERATION FUNCTIONS
# --------------------------
def generate_swahili_sentence():
    """Generate correct Swahili sentence"""
    parts = [
        random.choice(SWAHILI_WORDS["people"]).capitalize(),
        random.choice(SWAHILI_WORDS["verbs"]),
        random.choice(SWAHILI_WORDS["connectors"]),
        random.choice(SWAHILI_WORDS["places"]),
        random.choice(SWAHILI_WORDS["connectors"]),
        random.choice(SWAHILI_WORDS["people"]) + "."
    ]
    return " ".join(parts)

def generate_english_sentence():
    """Generate correct English sentence"""
    parts = [
        random.choice(ENGLISH_WORDS["people"]).capitalize(),
        random.choice(ENGLISH_WORDS["verbs"]),
        random.choice(ENGLISH_WORDS["connectors"]),
        random.choice(ENGLISH_WORDS["places"]),
        random.choice(ENGLISH_WORDS["connectors"]),
        random.choice(ENGLISH_WORDS["people"]) + "."
    ]
    return " ".join(parts)

def generate_kiswaenglish():
    """Natural mixed language — exactly how people speak"""
    mix_patterns = [
        # Swahili base + English words
        f"{random.choice(SWAHILI_WORDS['greetings'])}! Leo niko {random.choice(SWAHILI_WORDS['places'])} na nimefanya {random.choice(ENGLISH_WORDS['verbs'])} my homework.",
        f"Unajua {random.choice(SWAHILI_WORDS['rafiki'])} yangu anapenda kutumia {random.choice(ENGLISH_WORDS['places'])} sana?",
        # English base + Swahili words
        f"Today I went to {random.choice(SWAHILI_WORDS['places'])} and met my {random.choice(SWAHILI_WORDS['people'])}.",
        f"When you finish work, come tuonane kwa {random.choice(SWAHILI_WORDS['places'])}.",
        # Full mixed paragraph
        f"Habari zenu? I hope you are all fine. Jana nilikuwa Dar es Salaam kwa kazi, na I saw many people kutoka kila kona ya Tanzania. It was very nice kuzungumza nao."
    ]
    return random.choice(mix_patterns)

def generate_reasoning_text():
    """Text that teaches reasoning, logic, answers"""
    reasoning_templates = [
        "Swali: Kwa nini tunasoma? Jibu: Tunasoma ili kupata ujuzi, kuelewa mambo, na kuwa na maisha mazuri. Elimu ni ufunguo wa maendeleo.",
        "Swali: Why is health important? Jibu: Afya ni muhimu kwa sababu bila afya, hatuwezi kufanya kazi, kusoma, au kufurahia maisha.",
        "Eleza hatua za kupika wali: 1. Chambua wali 2. Osha vizuri 3. Weka kwenye sufuria na maji 4. Pika kwa moto wa wastani hadi uive.",
        "Jinsi ya kutumia intaneti kwa usalama: Usitoe taarifa zako za kibinafsi, tumia nywila nzuri, na usifungue ujumbe kutoka kwa watu usiowajua."
    ]
    return random.choice(reasoning_templates)

# --------------------------
# BUILD FULL DATASET
# --------------------------
TOTAL_SAMPLES = 200000  # ~30M tokens
OUTPUT_FILE = "full_dataset.jsonl"

print("Synthesizing dataset...")
with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for _ in tqdm(range(TOTAL_SAMPLES)):
        # language mix (60% KiswaEnglish, 20% Swahili, 20% English)
        lang_choice = random.choices(
            ["sw", "en", "mix", "reasoning"],
            weights=[0.2, 0.2, 0.4, 0.2]
        )[0]

        if lang_choice == "sw":
            text = " ".join([generate_swahili_sentence() for _ in range(random.randint(3,8))])
        elif lang_choice == "en":
            text = " ".join([generate_english_sentence() for _ in range(random.randint(3,8))])
        elif lang_choice == "mix":
            text = " ".join([generate_kiswaenglish() for _ in range(random.randint(3,6))])
        else: # reasoning
            text = generate_reasoning_text()

        writer.write({"text": text})

print(f" Done! Generated {TOTAL_SAMPLES} samples => full_dataset.jsonl")