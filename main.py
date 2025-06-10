import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go

st.title("기술 유사도 네트워크 시각화")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    # 텍스트 컬럼 선택
    text_col = st.selectbox("기술 설명이 포함된 컬럼을 선택하세요", df.columns)

    if st.button("시각화 시작"):
        with st.spinner("처리 중..."):
            texts = df[text_col].fillna("").astype(str).tolist()

            if not any(texts):
                st.error("선택한 컬럼에 유효한 텍스트가 없습니다.")
                st.stop()

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)

            similarity_matrix = cosine_similarity(tfidf_matrix)

            threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.3, 0.05)

            G = nx.Graph()
            for i in range(len(texts)):
                G.add_node(i, label=texts[i])
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if similarity_matrix[i, j] > threshold:
                        G.add_edge(i, j, weight=similarity_matrix[i, j])

            if len(G.nodes) == 0:
                st.warning("노드가 없습니다. 컬럼이나 임계값을 다시 확인해주세요.")
                st.stop()

            if len(G.edges) == 0:
                st.warning("간선이 없습니다. 임계값을 낮춰보세요.")
                st.stop()

            pos = nx.spring_layout(G, seed=42)

            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines"
            )

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                textposition="top center",
                marker=dict(
                    showscale=False,
                    size=10,
                    color="#FF5733",
                    line_width=2
                ),
                text=[str(G.nodes[n]["label"])[:20] + "..." for n in G.nodes()]
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title="<br>기술 유사도 네트워크",
                                titlefont_size=16,
                                showlegend=False,
                                hovermode="closest",
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False)
                            ))

            st.plotly_chart(fig, use_container_width=True)
