import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Streamlit 제목
st.title("기술 유사도 기반 네트워크 시각화")

# CSV 파일 업로드
uploaded_file = st.file_uploader("📁 CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.success("✅ CSV 파일을 성공적으로 불러왔습니다.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949')  # 한글 인코딩 대비
            st.success("✅ CSV 파일을 성공적으로 불러왔습니다.")
        except Exception as e:
            st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()
    
    # 유사도 임계값 슬라이더
    threshold = st.slider("유사도 임계값 (0.0 ~ 1.0)", 0.0, 1.0, 0.05, 0.01)

    # 기술 설명 포함 여부 선택
    기술선택 = st.selectbox("기술 설명이 포함된 열을 선택하세요", df.columns)

    # 유사도가 포함된 열 필터링 (0~1 값 가진 열)
    유사도_열들 = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and df[col].max() <= 1.0 and df[col].min() >= 0.0]

    if not 유사도_열들:
        st.warning("⚠️ 유사도 수치가 포함된 열이 없습니다.")
        st.stop()

    # 유사도 열 선택
    유사도열 = st.selectbox("유사도 점수가 포함된 열을 선택하세요", 유사도_열들)

    # 그래프 생성
    G = nx.Graph()

    # 노드 추가
    for 기술 in df[기술선택].unique():
        G.add_node(기술)

    # 엣지 추가 (유사도 조건 만족하는 경우만)
    for i, row in df.iterrows():
        source = row[기술선택]
        for j, row2 in df.iterrows():
            target = row2[기술선택]
            if source != target:
                similarity = row2[유사도열]
                if similarity >= threshold:
                    G.add_edge(source, target, weight=similarity)

    if len(G.nodes) == 0 or len(G.edges) == 0:
        st.warning("⚠️ 조건에 맞는 네트워크가 없습니다. 임계값을 낮춰보세요.")
        st.stop()

    # 노드 위치 설정
    pos = nx.spring_layout(G, seed=42)

    # 엣지 위치
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

    # 노드 위치 및 텍스트
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#00BFFF',
            size=20,
            line_width=2
        )
    )

    # 레이아웃 및 그래프 그리기
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="기술 유사도 기반 네트워크", font=dict(size=20)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    st.plotly_chart(fig, use_container_width=True)
