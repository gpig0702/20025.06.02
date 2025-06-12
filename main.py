import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# CSV URL 직접 읽기
csv_url = "https://raw.githubusercontent.com/gpig0702/20025.06.02/main/한국기계연구원_나노융합기술100선_20230731.csv"

try:
    df = pd.read_csv(csv_url, encoding="utf-8")
except Exception as e:
    st.error("❌ CSV 파일을 불러오는 데 실패했습니다. URL 경로를 확인해주세요.")
    st.stop()

st.title("🔬 나노융합기술 100선 유사도 네트워크 시각화")

column = st.selectbox("기술 내용 기준 열 선택", df.columns)

threshold = st.slider("유사도 임계값 설정 (0 ~ 1)", 0.0, 1.0, 0.3, 0.05)

# 1. 전처리 및 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = df[column].fillna("").astype(str).tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

similarity_matrix = cosine_similarity(tfidf_matrix)

# 2. 그래프 생성
G = nx.Graph()

for i, title in enumerate(df["기술명"]):
    G.add_node(i, label=title)

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] >= threshold:
            G.add_edge(i, j, weight=similarity_matrix[i][j])

if G.number_of_edges() == 0:
    st.warning("⚠️ 설정한 임계값으로는 연결된 노드가 없습니다. 임계값을 낮춰보세요.")
    st.stop()

# 3. 레이아웃
pos = nx.spring_layout(G, seed=42)

node_x = []
node_y = []
node_text = []
node_adjacencies = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(G.nodes[node]["label"])
    node_adjacencies.append(len(list(G.neighbors(node))))

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="YlGnBu",
        reversescale=True,
        color=node_adjacencies,
        size=10,
        colorbar=dict(
            thickness=15,
            title="연결 수",
            xanchor="left",
            titleside="right"
        ),
        line_width=2
    )
)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines"
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="유사도 네트워크 그래프",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

st.plotly_chart(fig, use_container_width=True)
