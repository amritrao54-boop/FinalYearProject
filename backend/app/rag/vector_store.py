import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Veterinary Knowledge Base Documents
KNOWLEDGE_BASE = [
    {
        "content": "Lumpy Skin Disease (LSD) is a viral disease of cattle causing high fever, nodular skin lesions, enlarged superficial lymph nodes, and reduced milk production. Prevention includes vector control and homologous vaccination.",
        "disease": "Lumpy Skin Disease"
    },
    {
        "content": "Foot and Mouth Disease (FMD) is an acute, highly infectious viral disease of cloven-hoofed animals. Manifestations include fever, vesicular lesions (blisters) in mouth, tongue, and feet, leading to profuse salivation and lameness.",
        "disease": "Foot and Mouth"
    },
    {
        "content": "Mastitis is an inflammation of the mammary gland and udder tissue, predominantly caused by bacterial pathogens. Key signs include udder swelling, redness, pain, fever, and abnormal milk clots.",
        "disease": "Mastitis"
    },
    {
        "content": "Blackleg is an acute, fatal disease caused by Clostridium chauvoei. It causes emphysematous swelling in muscular areas (thighs, shoulders), severe lameness, and sudden high fever in cattle.",
        "disease": "Blackleg"
    },
    {
        "content": "Foot Rot (interdigital phlegmon) is an infectious subacute disease characterized by necrotic lesions in the interdigital skin, causing severe lameness, foul odor, and swelling.",
        "disease": "Foot Rot"
    },
    {
        "content": "Schmallenberg virus (SBV) and Bovine Diarrhea Virus causes fever, reduced milk yield, diarrhoea (loose motion), loss of appetite, and abortion or congenital malformations in calves.",
        "disease": "Schmallenberg Virus / Bovine Viral Diarrhea"
    }
]

_vectorizer = None
_tfidf_matrix = None

def init_vector_store():
    global _vectorizer, _tfidf_matrix
    if _vectorizer is None:
        docs = [item["content"] for item in KNOWLEDGE_BASE]
        _vectorizer = TfidfVectorizer(stop_words='english')
        _tfidf_matrix = _vectorizer.fit_transform(docs)

def retrieve_knowledge(query: str, k: int = 2):
    """
    Fast, reliable TF-IDF vector retrieval with zero PyTorch DLL dependency issues.
    """
    init_vector_store()
    if not query:
        return []
    
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    
    # Sort top indices
    top_indices = similarities.argsort()[::-1][:k]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            doc = KNOWLEDGE_BASE[idx]
            results.append(f"[{doc['disease']}]: {doc['content']}")
            
    if not results:
        results.append("General Veterinary Advice: Ensure proper biosecurity, clean hydration, and monitor temperature.")
        
    return results
