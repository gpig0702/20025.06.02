import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objs as go

st.title("기술 유사도 네트워크 시각화")

# CSV 업로드 받기
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 컬럼 선택
    text_col = st.selectbox("기술 설명이 포함된 컬럼을 선택하세요", df.columns)

    # 유사도 임계값 선택
    threshold = st.slider("유사도 임계값을 선택하세요 (0.0~1.0)", 0.0, 1.0, 0.3, 0.05)

    if st.button("시각화 시작"):
        texts = df[text_col].fillna("").astype(str).tolist()

        # TF-IDF 벡터화 및 코사인 유사도 계산
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # 네트워크 그래프 생성
        G = nx.Graph()
        for i in range(len(texts)):
            G.add_node(i, label=f"{i+1}")

        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                sim = similarity_matrix[i][j]
                if sim >= threshold:
                    G.add_edge(i, j, weight=sim)

        # 좌표 설정
        pos = nx.spring_layout(G, seed=42)

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node+1}: {texts[node][:50]}...")

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # 시각화
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

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
                color=[len(list(G.neighbors(n))) for n in G.nodes()],
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

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="기술 유사도 네트워크",
                            titlefont_size=16,
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                        ))

        st.plotly_chart(fig, use_container_width=True)

