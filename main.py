import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="나노융합기술 유사도 네트워크", layout="wide")
st.title("🔬 나노융합기술 100선 – 유사도 네트워크 시각화")

# ✅ GitHub Raw URL 정확하게 반영
csv_url = "https://raw.githubusercontent.com/gpig0702/20025.06.02/main/kimm_nano_100.csv"

try:
    df = pd.read_csv(csv_url, encoding="utf-8")
    st.success("✅ CSV 파일을 성공적으로 불러왔습니다.")
except Exception as e:
    st.error(f"❌ CSV 불러오기 실패: {e}")
    st.stop()

# 사용자에게 컬럼 선택 기능 제공
text_col = st.selectbox("기술 설명이 포함된 열을 선택하세요", df.columns)

threshold = st.slider(
    "유사도 임계값 (0.0 ~ 1.0)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# 벡터화와 유사도 계산
texts = df[text_col].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer()
try:
    tfidf_matrix = vectorizer.fit_transform(texts)
except Exception as e:
    st.error(f"TF-IDF 처리 실패: {e}")
    st.stop()

similarity_matrix = cosine_similarity(tfidf_matrix)

# 네트워크 생성
G = nx.Graph()
for i, txt in enumerate(texts):
    G.add_node(i, label=txt[:30] + "...")

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if similarity_matrix[i][j] >= threshold:
            G.add_edge(i, j, weight=float(similarity_matrix[i][j]))

if G.number_of_nodes() == 0:
    st.warning("데이터가 존재하지 않습니다. 컬럼과 CSV 내용을 확인하세요.")
    st.stop()
if G.number_of_edges() == 0:
    st.warning("간선이 없습니다. 임계값을 낮춰보세요.")
    st.stop()

# 레이아웃 계산
pos = nx.spring_layout(G, seed=42)

node_x, node_y, node_text, node_degrees = [], [], [], []
for n in G.nodes():
    x, y = pos[n]
    node_x.append(x); node_y.append(y)
    node_text.append(G.nodes[n]["label"])
    node_degrees.append(len(list(G.neighbors(n))))

edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.5, color="#888"), hoverinfo="none"
)

node_trace = go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    text=node_text, textposition="top center", hoverinfo="text",
    marker=dict(
        showscale=True, colorscale="YlGnBu",
        reversescale=True, color=node_degrees,
        size=10, line_width=2,
        colorbar=dict(title="연결 수", thickness=15, xanchor="left")
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="기술 유사도 기반 네트워크",
                    titlefont_size=20, showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))
st.plotly_chart(fig, use_container_width=True)
