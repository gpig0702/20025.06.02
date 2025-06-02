# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="나노융합기술 유사도 분석", layout="wide")

st.title("🔬 나노융합기술 100선 - 유사도 기반 네트워크 분석")
st.markdown("기술 설명 텍스트를 기반으로 각 기술 간의 연관성과 클러스터를 시각화합니다.")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 2. 기술 설명 컬럼 확인
    text_col = st.selectbox("기술 설명 텍스트 컬럼을 선택하세요", df.columns)

    # 3. TF-IDF 벡터화
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_col].fillna(""))

    # 4. 코사인 유사도 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 5. 네트워크 그래프 생성
    threshold = st.slider("유사도 임계값 (간선 생성 기준)", 0.1, 1.0, 0.3, 0.05)
    G = nx.Graph()

    for i in range(len(df)):
        G.add_node(i, label=df.iloc[i][text_col][:20] + "...")

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    pos = nx.spring_layout(G, seed=42)

    # 6. Plotly 그래프 시각화
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

    node_x = []
    node_y = []
    labels = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(G.nodes[node]['label'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=labels,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[len(list(G.neighbors(n))) for n in G.nodes()],
            size=12,
            colorbar=dict(
                thickness=15,
                title='연결된 기술 수',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='기술 간 유사도 네트워크',
                        titlefont_size=20,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    st.plotly_chart(fig, use_container_width=True)

    st.success("분석 완료! 유사도가 높은 기술들끼리 연결되어 있습니다.")

