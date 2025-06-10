import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="나노융합기술 유사도 분석", layout="wide")
st.title("🔬 나노융합기술 100선 - 유사도 기반 네트워크 분석")
st.markdown("기술 설명 텍스트를 기반으로 기술 간의 연관성과 클러스터를 시각화합니다.")

# ✅ GitHub의 raw CSV 파일 경로
csv_url = "https://raw.githubusercontent.com/gpig0702/20025.06.02/main/kimm_nano_100.csv"

# CSV 불러오기
try:
    df = pd.read_csv(csv_url)
    st.success("📂 CSV 파일을 성공적으로 불러왔습니다.")
except Exception as e:
    st.error(f"❌ CSV 파일을 불러오는 데 실패했습니다.\n\n{e}")
    st.stop()

# 사용자에게 설명 컬럼 선택하도록
text_col = st.selectbox("기술 설명이 포함된 컬럼을 선택하세요", df.columns)

# TF-IDF 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df[text_col].fillna(""))

# 코사인 유사도 계산
similarity_matrix = cosine_similarity(tfidf_matrix)

# 네트워크 생성
threshold = st.slider("유사도 임계값 (간선 생성 기준)", 0.1, 1.0, 0.3, 0.05)
G = nx.Graph()

for i in range(len(df)):
    G.add_node(i, label=df.iloc[i][text_col][:25] + "...")

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if similarity_matrix[i, j] > threshold:
            G.add_edge(i, j, weight=similarity_matrix[i, j])

# 위치 계산
pos = nx.spring_layout(G, seed=42)

# Edge 좌표
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Node 좌표 및 정보
node_x = []
node_y = []
labels = []
node_degrees = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    labels.append(G.nodes[node]['label'])
    node_degrees.append(len(list(G.neighbors(node))))

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=labels,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        color=node_degrees,
        size=12,
        colorbar=dict(
            thickness=15,
            title=dict(text='연결된 기술 수'),
            xanchor='left',
            titleside='right'
        )
    )
)

# 그래프 시각화
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='기술 간 유사도 네트워크',
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("기술 간 유사도가 높은 경우 더 많은 연결선이 보입니다. 유사도 임계값을 조정해보세요!")
