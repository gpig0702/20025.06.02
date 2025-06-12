import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objs as go

# ✅ 깃허브에 업로드된 CSV 파일 (raw URL로 사용)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/chohjinwoo/job_similarity_streamlit/main/job_similarity.csv"

# 제목
st.title("기술 유사도 기반 네트워크 시각화")

# GitHub에서 CSV 불러오기
@st.cache_data
def load_data_from_github(url):
    return pd.read_csv(url)

try:
    df = load_data_from_github(GITHUB_CSV_URL)
    st.success("✅ GitHub에서 CSV 파일을 성공적으로 불러왔습니다.")
except Exception as e:
    st.error(f"❌ CSV 불러오기 실패: {e}")
    st.stop()

# 유사도 임계값 설정
threshold = st.slider("유사도 임계값 (0.0~1.0)", 0.0, 1.0, 0.05)

# 기술 설명 선택
if '기술내용' not in df.columns or '기술명' not in df.columns or '유사도' not in df.columns:
    st.error("CSV 파일에 '기술명', '기술내용', '유사도' 열이 모두 있어야 합니다.")
    st.stop()

selected_techs = st.multiselect("기술 설명이 포함된 항목 선택", df['기술내용'].unique())

# 그래프 생성
G = nx.Graph()

for _, row in df.iterrows():
    if row['유사도'] >= threshold:
        G.add_node(row['기술명'], content=row['기술내용'])
        G.add_node(row['대상기술명'], content=row['대상기술내용'])
        G.add_edge(row['기술명'], row['대상기술명'], weight=row['유사도'])

# 레이아웃 설정
pos = nx.spring_layout(G, seed=42)

# 노드 위치
node_x, node_y, node_text = [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>{G.nodes[node]['content']}")

# 엣지 위치
edge_x, edge_y = [], []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# 시각화
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color='blue',
        size=10,
        line_width=2
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="기술 유사도 기반 네트워크",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
)

st.plotly_chart(fig, use_container_width=True)
