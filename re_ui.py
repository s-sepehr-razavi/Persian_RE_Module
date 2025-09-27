import streamlit as st
import json
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components
from inference import RelationExtractor


# --- Cached model loading ---
@st.cache_resource
def load_model():
    rel2id = json.load(open('meta/rel2id.json', 'r')) 
    id2rel = {value: key for key, value in rel2id.items()}
    rel_info = json.load(open('meta/rel_info.json', 'r'))
    id2rel = {value: key for key, value in rel2id.items()}
    RE = RelationExtractor("HooshvareLab/bert-fa-base-uncased", "models_weight/pretrain_state_dict.pth", "api_key")
    return RE, id2rel, rel_info

RE, id2rel, rel_info = load_model()

# --- Relation extraction ---
def extract_relations(text, entities=None):
    labels, entities, hts = RE.predict(text, entities or [])
    labels_np = np.array(labels)
    res = []
    for i in range(labels_np.shape[0]):
        pred = labels_np[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'head': entities[hts[i][0]],
                        'tail': entities[hts[i][1]],
                        'relation': rel_info[id2rel[p]],
                    }
                )
    return res

# --- PyVis visualization ---
def draw_graph_pyvis(triplets):
    net = Network(height="500px", width="100%", directed=True)

    for t in triplets:
        net.add_node(t['head'], label=t['head'], color="lightblue")
        net.add_node(t['tail'], label=t['tail'], color="lightgreen")
        net.add_edge(t['head'], t['tail'], label=t['relation'], color="gray")

    net.set_options("""
    var options = {
    "edges": {
        "font": {"size": 12, "align": "middle"},
        "smooth": false
    },
    "nodes": {
        "font": {"size": 14},
        "shape": "dot",
        "scaling": {"min": 20, "max": 40}
    },
    "physics": {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -20000,
        "centralGravity": 0.1,
        "springLength": 180,
        "springConstant": 0.02
        }
    }
    }
    """)

    
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=600, scrolling=True)

# --- Streamlit UI ---
st.title("Relation Extraction UI")

text_input = st.text_area(
    "Enter text:", 
    "Barack Obama was born in Hawaii. He served as the 44th President of the United States."
)
entities_input = st.text_input("Optional: Enter entities (comma separated)", "")

if st.button("Extract Relations"):
    entities = [e.strip() for e in entities_input.split(",")] if entities_input else []
    triplets = extract_relations(text_input, entities)

    st.subheader("Extracted Triplets")
    st.json(triplets)

    if triplets:
        st.subheader("Interactive Graph")
        draw_graph_pyvis(triplets)
