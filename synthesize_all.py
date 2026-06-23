"""
Synthetic Text Generator
Author: Shadrackovsky
"""

import random


SW_SENTENCES = [
    "Elimu ni ufunguo wa maisha bora.",
    "Tanzania ina mazingira mazuri ya kilimo.",
    "Afya njema ni utajiri mkubwa sana.",
    "Kujifunza kila siku kunajenga uwezo wa akili.",
    "Watu wanaweza kushirikiana kutatua changamoto za jamii.",
    "Maji ni rasilimali muhimu kwa kila mtu.",
    "Miti husaidia kudumisha usawa wa hali ya hewa.",
    "Ukifanya kazi kwa bidii MUNGU atakupa riziki.",
    "Ukarimu wetu watanzania umetufanya tupendwe na dunia yote.",
    "Usalama wa jamii unategemea ushirikiano wa kila mtu."
]

EN_SENTENCES = [
    "Education opens doors to better opportunities.",
    "Agriculture plays a key role in the economy.",
    "Good health is more valuable than wealth.",
    "Continuous learning improves knowledge and skills.",
    "Working together solves community problems.",
    "Water is essential for all living things.",
    "Forests help maintain a stable climate.",
    "If you work hard GOD will bless you.",
    "Our traditions should be preserved for future generations.",
    "Safety depends on cooperation among everyone."
]

SW_WORDS = ["mwanafunzi", "shule", "maji", "ardhi", "afya", "elimu", "kazi", "jamii", "mazao", "mazingira"]
EN_WORDS = ["student", "school", "water", "land", "health", "education", "work", "community", "crops", "environment"]


def generate_swahili_sentence():
    return random.choice(SW_SENTENCES)


def generate_english_sentence():
    return random.choice(EN_SENTENCES)


def generate_kiswaenglish():
    parts = []
    for _ in range(random.randint(4, 8)):
        if random.random() < 0.5:
            parts.append(random.choice(SW_WORDS))
        else:
            parts.append(random.choice(EN_WORDS))
    return " ".join(parts).capitalize() + "."


def generate_reasoning_text():
    topics = [
        "Kwa nini ni muhimu kusoma?",
        "Jinsi ya kuboresha kilimo?",
        "Umuhimu wa usafi wa mazingira.",
        "Faida za ushirikiano kati ya jamii.",
        "Sababu za uhifadhi wa misitu."
    ]
    sentences = [
        "Hili ni jambo linalohitaji uangalifu maalumu.",
        "Kwa kuzingatia mazingira, tunaweza kufanya maendeleo.",
        "Kila hatua ndogo inachangia matokeo makubwa.",
        "Uwezo wa kutatua changamoto unajitokeza kwa elimu na uzoefu.",
        "Ushirikiano hupunguza gharama na huongeza ufanisi."
    ]
    lines = [random.choice(topics)]
    lines.extend(random.choices(sentences, k=random.randint(2, 4)))
    return " ".join(lines)