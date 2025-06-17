import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="나노융합기술 유사도 네트워크", layout="wide")
st.title("🔬 나노융합기술 100선 – 유사도 네트워크 시각화")

# ✅ GitHub CSV URL
csv_url = "https://raw.githubusercontent.com/gpig0702/20025.06.02/main/kimm_nano_100.csv"

try:
    df = pd.read_csv(csv_url, encoding="utf-8")
    st.success("✅ CSV 파일을 성공적으로 불러왔습니다.")
except Exception as e:
    st.error(f"❌ CSV 불러오기 실패: {e}")
    st.stop()

# ✅ 기술 설명 컬럼 선택
text_col = st.selectbox("기술 설명이 포함된 열을 선택하세요", df.columns)

# ✅ 유사도 임계값 슬라이더
threshold = st.slider("유사도 임계값 (0.0 ~ 1.0)", 0.0, 1.0, 0.3, 0.05)

# ✅ 검색어 입력
search_query = st.text_input("🔍 기술 검색어를 입력하세요 (예: 센서)", "").strip()

# ✅ TF-IDF 벡터화
texts = df[text_col].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer()
try:
    tfidf_matrix = vectorizer.fit_transform(texts)
except Exception as e:
    st.error(f"TF-IDF 처리 실패: {e}")
    st.stop()

similarity_matrix = cosine_similarity(tfidf_matrix)

# ✅ 네트워크 생성
G = nx.Graph()
for i, txt in enumerate(texts):
    G.add_node(i, label=txt[:6] + "…", full_text=txt)  # 라벨 더 짧게

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if similarity_matrix[i][j] >= threshold:
            G.add_edge(i, j, weight=float(similarity_matrix[i][j]))

if G.number_of_nodes() == 0:
    st.warning("데이터가 존재하지 않습니다.")
    st.stop()
if G.number_of_edges() == 0:
    st.warning("간선이 없습니다. 임계값을 낮춰보세요.")
    st.stop()

# ✅ 좌표 및 시각화 데이터 생성
pos = nx.spring_layout(G, seed=42)
node_x, node_y, hover_texts, short_labels, node_sizes, node_colors = [], [], [], [], [], []

highlighted_nodes = []

for n in G.nodes():
    x, y = pos[n]
    full_text = G.nodes[n]["full_text"]
    label = G.nodes[n]["label"]

    node_x.append(x)
    node_y.append(y)
    hover_texts.append(full_text)
    short_labels.append(label)

    if search_query and search_query.lower() in full_text.lower():
        node_colors.append("red")
        node_sizes.append(20)
        highlighted_nodes.append((n, full_text))
    else:
        node_colors.append("#8dbbf2")
        node_sizes.append(8)

edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# ✅ 시각화 (글자 크기 조절 포함)
edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.5, color="#ccc"), hoverinfo="none"
)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    text=short_labels, textposition="top center",
    hovertext=hover_texts, hoverinfo="text",
    textfont=dict(size=9),  # ✅ 글자 크기 줄임
    marker=dict(
        color=node_colors,
        size=node_sizes,
        line=dict(width=1, color="black")
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(text="기술 유사도 기반 네트워크", font=dict(size=20)),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

st.plotly_chart(fig, use_container_width=True)

# ✅ 보조 분석 - 검색 결과
if search_query:
    st.subheader("🔍 검색 결과 목록")
    if highlighted_nodes:
        matched_df = pd.DataFrame(highlighted_nodes, columns=["Index", "기술 설명"])
        st.dataframe(matched_df.set_index("Index"))
    else:
        st.info("검색 결과가 없습니다.")

# ✅ [신규 기능] 유사도 상위 기술쌍 + 공통 키워드
st.subheader("📊 유사도 상위 기술 쌍 + 유사 키워드")

similarities = []
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarities.append((i, j, similarity_matrix[i, j]))

top_similar_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:10]

feature_names = vectorizer.get_feature_names_out()

def get_top_shared_keywords(i, j, top_n=5):
    vec_i = tfidf_matrix[i].toarray().flatten()
    vec_j = tfidf_matrix[j].toarray().flatten()
    avg_scores = (vec_i + vec_j) / 2
    shared_keywords_idx = np.argsort(avg_scores)[::-1][:top_n]
    return [feature_names[k] for k in shared_keywords_idx]

for idx1, idx2, sim in top_similar_pairs:
    keywords = get_top_shared_keywords(idx1, idx2)
    st.markdown(f"""
    ✅ **기술 A:** {df[text_col][idx1]}  
    ✅ **기술 B:** {df[text_col][idx2]}  
    📌 **유사도:** {sim:.3f}  
    🔑 **공통 키워드:** _{', '.join(keywords)}_  
    ---
    """)
