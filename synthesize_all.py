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
    "Usalama wa jamii unategemea ushirikiano wa kila mtu.",
    "Kilimo ndio msingi wa uchumi wa nchi yetu.",
    "Usafi huzuia magonjwa na kuleta afya njema.",
    "Teknolojia inasaidia kuongeza uzalishaji wa mazao.",
    "Kuhifadhi misitu ni jukumu la kila mwananchi.",
    "Kusoma huongeza uwezo wa kufikiri na kutatua matatizo.",
    "Biashara ndogo ndogo huboresha maisha ya familia.",
    "Upendo na amani hujenga jamii yenye maendeleo.",
    "Maji safi ni haki ya kila mtu duniani.",
    "Kufanya kazi kwa pamoja hufanya kazi iwe rahisi.",
    "Tamaduni zetu zinapaswa kuhifadhiwa kwa vizazi vijavyo.",
    "Serikali inajitahidi kutoa huduma bora kwa wananchi.",
    "Barabara nzuri hurahisisha usafirishaji wa bidhaa.",
    "Mazao ya kutosha huhakikisha usalama wa chakula.",
    "Kujituma katika kazi huleta matokeo mazuri kila wakati.",
    "Mazingira safi hufanya maisha kuwa ya furaha na afya.",
    "Kila mtu ana jukumu la kulinda rasilimali za nchi.",
    "Ujuzi mpya humsaidia mtu kukabiliana na mabadiliko.",
    "Uwekezaji katika elimu huleta faida kwa muda mrefu.",
    "Ushirikiano kati ya jamii na serikali hufanikisha miradi.",
    "Kuepuka makosa hujifunza mtu kuwa bora zaidi.",
    "Ardhi yenye rutuba hutoa mazao mengi kwa wakati mzuri.",
    "Mito na maziwa ni vyanzo muhimu vya maji safi.",
    "Watu wanaishi kwa amani wanapokuwa na uelewa mzuri.",
    "Kila kazi ina heshima yake na inastahili heshima.",
    "Kuzungumza kwa upole hupunguza migogoro kati ya watu.",
    "Uzalendo unamaanisha kumpenda na kuitumikia nchi.",
    "Elimu ya juu huwafungulia vijana milango ya maendeleo.",
    "Mazingira ya shule yanapaswa kuwa salama na rafiki.",
    "Kutoa msaada kwa wengine huleta furaha kubwa moyoni.",
    "Mabadiliko ya tabianchi yanahitaji uangalifu wa kila mtu.",
    "Kilimo cha kisasa huongeza pato la mkulima kwa kila msimu.",
    "Kujua kusoma na kuandika ni msingi wa maisha ya kisasa.",
    "Uamuzi mzuri huokoa muda na rasilimali za jamii.",
    "Wazazi wana jukumu la kuwaelimisha watoto vizuri.",
    "Utulivu wa akili humwezesha mtu kufanya kazi kwa ufanisi.",
    "Haki za binadamu zinapaswa kuheshimiwa kila mahali.",
    "Mawasiliano mazuri huzuia makosa na kutoelewana.",
    "Kila jamii ina utamaduni wake wa kipekee na wa thamani.",
    "Kufanya kazi kwa uaminifu huleta heshima kubwa kwa mtu.",
    "Maendeleo hayaji peke yake, yanahitaji juhudi za kila mtu."
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
    "Safety depends on cooperation among everyone.",
    "Farming ensures food security for the nation.",
    "Cleanliness prevents diseases and promotes health.",
    "Technology helps increase production and efficiency.",
    "Protecting nature benefits everyone in the long run.",
    "Reading improves thinking and problem‑solving ability.",
    "Small businesses improve family income and livelihood.",
    "Peace and unity build a strong and progressive society.",
    "Clean water is a basic right for every person.",
    "Teamwork makes hard tasks easier and faster.",
    "Respect and kindness bring harmony to the community.",
    "Good governance delivers quality services to citizens.",
    "Well‑built roads make transport of goods much easier.",
    "Enough crops ensure no one goes hungry in the country.",
    "Dedication and effort always produce great results.",
    "A clean environment makes life happy and healthy.",
    "Everyone has a duty to protect the country’s resources.",
    "New skills help people adapt to changing times.",
    "Investing in education brings long‑term benefits.",
    "Cooperation between people and government succeeds projects.",
    "Learning from mistakes makes a person better and stronger.",
    "Fertile land gives abundant harvests every season.",
    "Rivers and lakes are important sources of fresh water.",
    "People live in peace when they understand each other well.",
    "Every job has its own dignity and deserves respect.",
    "Speaking gently reduces conflicts and misunderstandings.",
    "Patriotism means loving and serving your nation faithfully.",
    "Higher education opens greater opportunities for young people.",
    "Schools should provide a safe and friendly learning space.",
    "Helping others brings true joy to your heart and soul.",
    "Climate change requires careful attention from everyone.",
    "Modern farming increases income for farmers every year.",
    "Literacy is the foundation of modern daily life.",
    "Good decisions save time and resources for the community.",
    "Parents have the duty to raise and teach their children well.",
    "A calm mind helps you work more efficiently and clearly.",
    "Human rights should be respected everywhere and by all.",
    "Clear communication prevents mistakes and confusion.",
    "Every community has its own unique and valuable culture.",
    "Working with honesty brings great respect and reputation.",
    "Development does not come easily — it requires effort from all."
]

SW_WORDS = [
    "mwanafunzi", "shule", "maji", "ardhi", "afya", "elimu", "kazi", "jamii", "mazao", "mazingira",
    "daktari", "walimu", "kijiji", "mji", "soko", "familia", "rafiki", "barabara", "chakula", "nyumba",
    "dawa", "msitu", "mto", "ziwa", "kilimo", "biashara", "serikali", "haki", "amani", "upendo",
    "utamaduni", "maendeleo", "usalama", "teknolojia", "ujuzi", "pato", "mazingira", "chama", "mkulima", "mgeni",
    "mawasiliano", "usafiri", "chama", "shughuli", "ndugu", "kazi", "uzalendo", "huduma", "mazingira", "maisha"
]

EN_WORDS = [
    "student", "school", "water", "land", "health", "education", "work", "community", "crops", "environment",
    "doctor", "teacher", "village", "town", "market", "family", "friend", "road", "food", "house",
    "medicine", "forest", "river", "lake", "farming", "business", "government", "rights", "peace", "love",
    "culture", "development", "safety", "technology", "skill", "income", "nature", "group", "farmer", "visitor",
    "communication", "transport", "activity", "relative", "patriotism", "service", "life", "progress", "growth", "knowledge"
]

# Track used items to reduce repetition
used_sw = []
used_en = []
used_kiswa = []
used_reason = []


def generate_swahili_sentence():
    global used_sw
    available = [s for s in SW_SENTENCES if s not in used_sw]
    if not available:
        used_sw = []
        available = SW_SENTENCES
    sent = random.choice(available)
    used_sw.append(sent)
    if len(used_sw) > 20:
        used_sw.pop(0)
    return sent


def generate_english_sentence():
    global used_en
    available = [s for s in EN_SENTENCES if s not in used_en]
    if not available:
        used_en = []
        available = EN_SENTENCES
    sent = random.choice(available)
    used_en.append(sent)
    if len(used_en) > 20:
        used_en.pop(0)
    return sent


def generate_kiswaenglish():
    global used_kiswa
    patterns = [
        f"Kwa masomo, {random.choice(SW_WORDS)} and {random.choice(EN_WORDS)} are very useful.",
        f"Katika {random.choice(SW_WORDS)}, we use good {random.choice(EN_WORDS)} methods.",
        f"Nilitembelea {random.choice(SW_WORDS)} last week and learned about {random.choice(EN_WORDS)}.",
        f"For better {random.choice(EN_WORDS)}, tunahitaji kutunza {random.choice(SW_WORDS)}.",
        f"Maendeleo ya {random.choice(SW_WORDS)} depend on proper {random.choice(EN_WORDS)}.",
        f"Unapofanya kazi kwenye {random.choice(SW_WORDS)}, you need good {random.choice(EN_WORDS)}.",
        f"Our {random.choice(EN_WORDS)} helps build a strong {random.choice(SW_WORDS)}.",
        f"Tunahitaji uhifadhi wa {random.choice(SW_WORDS)} so that future {random.choice(EN_WORDS)} benefit.",
        f"Kwa afya njema, {random.choice(SW_WORDS)} and {random.choice(EN_WORDS)} go hand in hand.",
        f"Katika {random.choice(SW_WORDS)}, we always follow best {random.choice(EN_WORDS)} practices."
    ]
    available = [p for p in patterns if p not in used_kiswa]
    if not available:
        used_kiswa = []
        available = patterns
    sent = random.choice(available)
    used_kiswa.append(sent)
    if len(used_kiswa) > 12:
        used_kiswa.pop(0)
    return sent


def generate_reasoning_text():
    global used_reason
    topics = [
        "Kwa nini ni muhimu kusoma?",
        "Jinsi ya kuboresha kilimo?",
        "Umuhimu wa usafi wa mazingira.",
        "Faida za ushirikiano kati ya jamii.",
        "Sababu za uhifadhi wa misitu.",
        "Jinsi ya kuanzisha biashara ndogo.",
        "Umuhimu wa teknolojia katika maisha.",
        "Jinsi ya kuzuia magonjwa kwa urahisi.",
        "Kwa nini tunahitaji amani katika jamii?",
        "Jinsi ya kuweka akiba ya fedha kwa ufanisi.",
        "Umuhimu wa lugha katika maisha ya kila siku.",
        "Faida za kutumia maji kwa uangalifu.",
        "Jinsi ya kuepuka mabadiliko ya tabianchi.",
        "Umuhimu wa kutoa msaada kwa wengine.",
        "Kwa nini uaminifu ni muhimu katika kazi?"
    ]
    explanations = [
        "Hili ni jambo linalohitaji uangalifu maalumu.",
        "Kwa kuzingatia mazingira, tunaweza kufanya maendeleo.",
        "Kila hatua ndogo inachangia matokeo makubwa.",
        "Uwezo wa kutatua changamoto unajitokeza kwa elimu na uzoefu.",
        "Ushirikiano hupunguza gharama na huongeza ufanisi.",
        "Mpango mzuri husaidia kufikia malengo haraka.",
        "Watu wanaofanya kazi pamoja hufanikiwa zaidi.",
        "Kujifunza kutokana na makosa kunajenga uwezo.",
        "Kwa juhudi na uvumilivu, kila jambo linawezekana.",
        "Uelewa mzuri huzuia migogoro na kutoelewana.",
        "Kila mtu ana jukumu la kuchangia maendeleo ya jamii.",
        "Mazoezi mazuri huleta matokeo yanayotegemewa.",
        "Kufanya kazi kwa makini hupunguza makosa ya kazi.",
        "Ujuzi mpya humsaidia mtu kukabiliana na changamoto.",
        "Kwa utulivu wa akili, mtu hufanya maamuzi bora."
    ]
    available_topics = [t for t in topics if t not in used_reason]
    if not available_topics:
        used_reason = []
        available_topics = topics
    chosen_topic = random.choice(available_topics)
    used_reason.append(chosen_topic)
    if len(used_reason) > 10:
        used_reason.pop(0)
    lines = [chosen_topic]
    lines.extend(random.choices(explanations, k=random.randint(2, 4)))
    return " ".join(lines)