#!/bin/bash
# 림프칩 PINN 에이전트 실행 스크립트

cd "$(dirname "$0")"

echo "🧬 림프칩 PINN 에이전트 시작..."
echo ""
echo "브라우저에서 http://localhost:8501 으로 접속하세요"
echo "종료하려면 Ctrl+C를 누르세요"
echo ""

streamlit run app.py --server.headless true
